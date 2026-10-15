"""The live worker against the stream simulator (principle P13 / P17): same engine objects,
same audio, same seed -> the worker's output ring must carry the simulator's stream.

The audio device is replaced by ``FileInput`` (real-time paced blocks into the input ring)
and a reader thread that drains the output ring every 10 ms exactly like an output callback.
"""

import dataclasses
import threading
import time

import numpy as np
import pytest
import torch
from tests.conftest import MODELS, TEST_MODEL

from vox.audio.rings import SPSCRing
from vox.audio.streams import FileInput
from vox.config import defaults
from vox.config.schemas import RuntimeParams
from vox.engine.worker import InferenceWorker, ShapeRequest
from vox.offline.metrics import compare
from vox.offline.simulate import StreamSimulator, load_engine

pytestmark = [pytest.mark.assets, pytest.mark.gpu]
SR = 48000
CALLBACK = 480  # 10 ms


class OutputReader(threading.Thread):
    """Pulls ``CALLBACK`` samples every 10 ms, like a PortAudio output callback."""

    def __init__(self, ring: SPSCRing, seconds: float) -> None:
        super().__init__(daemon=True)
        self.ring = ring
        self.chunks = []
        self.seconds = seconds

    def run(self) -> None:
        period = CALLBACK / SR
        start = time.perf_counter()
        n = 0
        buf = np.zeros(CALLBACK, dtype=np.float32)
        while time.perf_counter() - start < self.seconds:
            due = start + (n + 1) * period
            while (rem := due - time.perf_counter()) > 0:
                time.sleep(min(rem, 0.002))
            self.ring.read(buf)
            self.chunks.append(buf.copy())
            n += 1


@pytest.fixture(scope="module")
def engine(device):
    return load_engine(MODELS, TEST_MODEL, defaults.DEFAULT_SHAPE, RuntimeParams(),
                       device=device, cache={})


def run_live(engine, audio: np.ndarray, gate: bool, seed: int = 0, fill_ms: float = 50.0,
             recorder=None):
    plan = engine.plan
    in_ring = SPSCRing(2 * SR, guard=4 * CALLBACK)
    out_ring = SPSCRing(SR, guard=2 * plan.block)
    hop_event = threading.Event()
    torch.manual_seed(seed)  # before the reset: it draws the window's first noise
    engine.pipeline.reset()
    engine.sola.reset()
    engine.gate.reset()
    engine.gate.set_config(dataclasses.replace(engine.gate.config, enabled=gate))
    engine.conditioner.reset()
    engine.timers.reset()
    fill = int(SR * fill_ms / 1000)
    # The same input chain the simulator runs, as LiveSession wires it.
    worker = InferenceWorker(engine.pipeline, engine.sola, engine.gate, engine.graphs,
                             engine.timers, in_ring, out_ring, hop_event, fill, log=lambda *_: None,
                             conditioner=engine.conditioner)
    worker.recorder = recorder
    seconds = len(audio) / SR + 0.5
    feeder = FileInput(audio, SR, CALLBACK, in_ring, plan.block, hop_event)
    reader = OutputReader(out_ring, seconds)
    worker.start()
    worker.ready.wait(5.0)
    feeder.start()
    worker.first_hop.wait(5.0)  # like LiveSession: the output starts on a primed ring
    reader.start()
    reader.join()
    feeder.stop()
    worker.stop()
    return np.concatenate(reader.chunks), worker, out_ring, in_ring


def test_worker_reproduces_the_simulator_stream_without_underruns(engine, clip_48k):
    audio = clip_48k[: 5 * SR]
    plan = engine.plan
    torch.manual_seed(0)
    sim = StreamSimulator(engine).run(audio, seed=0)
    live, worker, out_ring, in_ring = run_live(engine, audio, gate=False, seed=0)
    assert worker.stats.errors == []
    assert out_ring.underruns == 0 and in_ring.overruns == 0
    assert worker.stats.underrun_refills == 0
    assert worker.stats.hops >= len(audio) // plan.block - 1
    # The live stream is the simulator's stream delayed by the output prefill; find the lag
    # and compare the overlapping, settled region.
    sim_audio = sim.audio
    n = min(len(sim_audio), len(live))
    best = None
    for lag in range(0, int(0.2 * SR), plan.zc // 2):
        err = np.mean(np.abs(live[lag:n] - sim_audio[: n - lag]))
        if best is None or err < best[0]:
            best = (err, lag)
    err, lag = best
    aligned = live[lag:n]
    reference = sim_audio[: n - lag]
    start = int(1.5 * SR)
    quality = compare(reference[start:], aligned[start:], SR)
    # identical seed and state: the only differences are the two threads' noise draws
    # interleaving differently is impossible (one consumer), so this should be tiny
    assert quality["l1"] < 0.08, (quality, lag)
    assert worker.stats.out_fill_ms <= 60.0, worker.stats


def test_worker_gate_closes_and_output_is_silent_in_pauses(engine, clip_48k):
    speech = clip_48k[: 2 * SR]
    audio = np.concatenate([speech, np.zeros(2 * SR, np.float32), speech])
    live, worker, out_ring, _ = run_live(engine, audio, gate=True)
    assert out_ring.underruns == 0 and worker.stats.errors == []
    assert worker.stats.gate_transitions >= 2
    assert worker.stats.gated_hops >= 40
    # the middle of the pause (allowing hangover + drain + prefill delay) is digital silence
    mid = int(3.2 * SR)
    assert np.abs(live[mid:mid + SR // 2]).max() == 0.0


def test_worker_records_aligned_mic_and_output_takes(engine, clip_48k, tmp_path):
    """The Record button's capture: what the worker read, conditioned and wrote, hop by hop."""
    import soundfile as sf

    from vox.audio.recorder import SessionRecorder

    audio = clip_48k[: 3 * SR]
    recorder = SessionRecorder(SR)
    live, worker, out_ring, _ = run_live(engine, audio, gate=True, recorder=recorder)
    assert worker.stats.errors == [] and out_ring.underruns == 0
    paths = recorder.save(tmp_path, "take", {})
    raw, _ = sf.read(str(paths["mic_raw"]), dtype="float32")
    cond, _ = sf.read(str(paths["mic_conditioned"]), dtype="float32")
    out, _ = sf.read(str(paths["output"]), dtype="float32")
    assert len(raw) == len(cond) == len(out) >= len(audio) - engine.plan.block
    np.testing.assert_array_equal(raw[: len(audio)], audio[: len(raw)])  # the mic, untouched
    assert np.abs(cond).max() > np.abs(raw).max() * 0.5  # the input chain's output
    # the output track is the played stream, ahead of it by the output prefill only
    n = min(len(out), len(live))
    best = min(range(0, int(0.2 * SR), engine.plan.zc // 2),
               key=lambda lag: np.mean(np.abs(live[lag:n] - out[: n - lag])))
    assert np.mean(np.abs(live[best:n] - out[: n - best])) < 1e-6


def test_worker_rebuilds_shape_live_and_keeps_running(engine, clip_48k):
    """A restart-class change (hop 20 -> 30 ms) mid-stream: rebuild between hops, no crash,
    audio keeps flowing after the pause."""
    audio = clip_48k[: 8 * SR]
    plan = engine.plan
    in_ring = SPSCRing(2 * SR, guard=4 * CALLBACK)
    out_ring = SPSCRing(SR, guard=2 * plan.block)
    hop_event = threading.Event()
    engine.pipeline.reset()
    engine.sola.reset()
    engine.gate.reset()
    worker = InferenceWorker(engine.pipeline, engine.sola, engine.gate, engine.graphs,
                             engine.timers, in_ring, out_ring, hop_event, int(0.05 * SR),
                             log=lambda *_: None, conditioner=engine.conditioner)
    feeder = FileInput(audio, SR, CALLBACK, in_ring, plan.block, hop_event)
    reader = OutputReader(out_ring, len(audio) / SR + 0.5)
    worker.start()
    worker.ready.wait(5.0)
    feeder.start()
    worker.first_hop.wait(5.0)  # like LiveSession: the output starts on a primed ring
    reader.start()
    time.sleep(2.0)
    worker.apply_shape(ShapeRequest(dataclasses.replace(defaults.DEFAULT_SHAPE, block_ms=30)))
    hop_event.set()
    reader.join()
    feeder.stop()
    worker.stop()
    try:
        assert worker.stats.errors == []
        assert worker.stats.rebuilds == 1 and worker.plan.block_ms == 30.0
        assert engine.graphs.active and engine.graphs.sola is worker.sola
        assert engine.conditioner.block == worker.plan.block  # input chain resized in place
        live = np.concatenate(reader.chunks)
        tail = live[-int(1.0 * SR):]
        assert np.sqrt(np.mean(tail ** 2)) > 0.005  # audio flowing after the rebuild
        # 1.8-2.4 s at decoder_context_ms 0; 2.9-3.0 s at the 80 ms default (8.8)
        assert worker.stats.last_rebuild_ms < 3500, worker.stats.last_rebuild_ms
    finally:
        # restore the module-scoped engine for the other tests
        worker2 = worker
        worker2.apply_shape(ShapeRequest(defaults.DEFAULT_SHAPE))
        worker2._apply_pending()
        engine.sola = worker2.sola
        engine.gate = worker2.gate


def test_auto_pitch_readout_never_syncs_during_a_live_rebuild(engine, clip_48k):
    """The GUI reads the auto-pitch offset from another thread at 2 Hz. A device read there
    during a graph capture poisoned the CUDA context (take 2026-10-15 11:40: a hop change
    with auto pitch on killed the session). The readout must be a host mirror."""
    audio = clip_48k[: 6 * SR]
    plan = engine.plan
    engine.pipeline.set_params(dataclasses.replace(engine.pipeline.params, pitch_mode="auto"))
    in_ring = SPSCRing(2 * SR, guard=4 * CALLBACK)
    out_ring = SPSCRing(SR, guard=2 * plan.block)
    hop_event = threading.Event()
    torch.manual_seed(0)
    engine.pipeline.reset()
    engine.sola.reset()
    engine.gate.reset()
    worker = InferenceWorker(engine.pipeline, engine.sola, engine.gate, engine.graphs,
                             engine.timers, in_ring, out_ring, hop_event, int(0.05 * SR),
                             log=lambda *_: None, conditioner=engine.conditioner)
    feeder = FileInput(audio, SR, CALLBACK, in_ring, plan.block, hop_event)
    reader = OutputReader(out_ring, len(audio) / SR + 0.5)
    seen, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            seen.append(engine.pipeline.register.offset_st)
            time.sleep(0.02)

    poller = threading.Thread(target=poll, daemon=True)
    worker.start()
    worker.ready.wait(5.0)
    feeder.start()
    worker.first_hop.wait(5.0)
    reader.start()
    poller.start()
    time.sleep(1.5)
    worker.apply_shape(ShapeRequest(dataclasses.replace(defaults.DEFAULT_SHAPE, block_ms=40)))
    hop_event.set()
    reader.join()
    stop.set()
    poller.join()
    feeder.stop()
    worker.stop()
    try:
        assert worker.stats.errors == [], worker.stats.errors
        assert worker.stats.rebuilds == 1 and engine.graphs.active
        finite = [x for x in seen if x == x]
        assert finite and 6.0 < finite[-1] < 14.0, finite[-5:]  # a low male voice -> 200 Hz
    finally:
        worker.apply_shape(ShapeRequest(defaults.DEFAULT_SHAPE))
        worker._apply_pending()
        engine.pipeline.set_params(dataclasses.replace(engine.pipeline.params,
                                                       pitch_mode="fixed"))
        engine.graphs.capture()

