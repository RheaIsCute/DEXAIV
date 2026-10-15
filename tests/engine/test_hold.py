"""The worker's hold (the live voice closed while a typed clip renders and plays), driven
hop by hop without audio devices.

Closing mid-speech must be a fade, never a cut: the saved SOLA tail fades out, silence
follows, and the silence cushion only ever goes in behind audio that has already faded.
"""

import threading

import numpy as np
import pytest
from tests.conftest import MODELS, TEST_MODEL

from vox.audio.rings import SPSCRing
from vox.config import defaults
from vox.config.schemas import RuntimeParams
from vox.engine.gate import GateConfig
from vox.engine.worker import HOLD_CUSHION_S, InferenceWorker
from vox.offline.simulate import load_engine

pytestmark = [pytest.mark.assets, pytest.mark.gpu]
SR = 48000
HOLD_AT = 150  # 3 s into the clip: mid-sentence


@pytest.fixture(scope="module")
def engine(device):
    # The energy gate alone keeps the test about the hold, not about the VAD's timing.
    return load_engine(MODELS, TEST_MODEL, defaults.DEFAULT_SHAPE, RuntimeParams(),
                       device=device, cache={}, gate=GateConfig(use_vad=False),
                       log=lambda *_: None)


def drive(engine, audio, hops, before_hop=None):
    """Run ``hops`` hops by hand, reading one hop's worth after each like an output device.

    Returns the worker, the stream the device would have played, and the ring fill left
    after each hop. Hop ``k`` plays at ``fill_target + k * block`` in the stream until a
    cushion shifts it.
    """
    engine.pipeline.reset()
    engine.sola.reset()
    engine.gate.reset()
    block = engine.plan.block
    in_ring, out_ring = SPSCRing(2 * SR), SPSCRing(8 * SR)
    worker = InferenceWorker(engine.pipeline, engine.sola, engine.gate, engine.graphs,
                             engine.timers, in_ring, out_ring, threading.Event(),
                             int(0.05 * SR), log=lambda *_: None)
    played, fills = [], []
    for n in range(hops):
        in_ring.write(audio[n * block:(n + 1) * block])
        if before_hop is not None:
            before_hop(worker, n)
        worker._hop()
        got = np.zeros(block, dtype=np.float32)
        out_ring.read(got)  # also applies a requested drop, like an output callback
        played.append(got)
        fills.append(out_ring.available())
    assert out_ring.underruns == 0
    return worker, np.concatenate(played), fills


def hop_of(stream, worker, k, block):
    start = worker.fill_target + k * block
    return stream[start:start + block]


def test_hold_fades_the_live_voice_out_and_queues_silence_behind_it(engine, clip_48k):
    block = engine.plan.block
    cushion = int(HOLD_CUSHION_S * SR)

    def hold(worker, n):
        if n == HOLD_AT:
            worker.hold = True

    worker, stream, fills = drive(engine, clip_48k, HOLD_AT + 40, hold)
    drain = -(-engine.plan.sola_buffer // block)  # hops the saved tail takes to fade out
    live = hop_of(stream, worker, HOLD_AT - 1, block)
    tail = [hop_of(stream, worker, HOLD_AT + k, block) for k in range(drain)]
    after = hop_of(stream, worker, HOLD_AT + drain, block)
    assert np.abs(live).max() > 0.02  # speech was sounding
    assert not worker.stats.gate_open
    # continuous into the hold, then decaying to digital silence within the overlap
    seam = np.abs(np.diff(np.concatenate([live[-64:], tail[0][:64]]))).max()
    assert seam <= 1.5 * np.abs(np.diff(live)).max() + 0.01  # no bigger than the voice's own steps
    peaks = [np.abs(t).max() for t in tail]
    assert all(p > 0.0 for p in peaks) and peaks == sorted(peaks, reverse=True)
    assert np.abs(tail[-1][-32:]).max() < 0.1 * peaks[0]  # faded out by the overlap's end
    assert np.abs(after).max() == 0.0
    # silence from there to the end of what was played: nothing audible behind the fade
    assert np.abs(stream[worker.fill_target + (HOLD_AT + drain) * block:]).max() == 0.0
    # the cushion went in once, after the fade was out
    assert fills[HOLD_AT - 1:HOLD_AT + drain] == [worker.fill_target] * (drain + 1)
    assert max(fills) == worker.fill_target + cushion and fills[-1] == max(fills)


def test_hold_set_in_the_middle_of_a_hop_takes_effect_on_the_next_one(engine, clip_48k):
    """The control thread can raise the flag at any point of a hop. A hop that already
    decided to speak must stay a plain speech hop: no silence may be queued in front of it
    (that was an audible cut straight into half a second of silence)."""
    block = engine.plan.block
    process = engine.pipeline.process_hop
    state = {"n": 0, "worker": None}

    def hold_mid_hop():
        if state["n"] == HOLD_AT:
            state["worker"].hold = True  # after the gate decision, before the output write
        return process()

    def before(worker, n):
        state.update(n=n, worker=worker)

    engine.pipeline.process_hop = hold_mid_hop
    try:
        worker, stream, fills = drive(engine, clip_48k, HOLD_AT + 40, before)
    finally:
        engine.pipeline.process_hop = process
    assert fills[HOLD_AT] == worker.fill_target  # nothing but the hop itself was queued
    assert np.abs(hop_of(stream, worker, HOLD_AT, block)).max() > 0.02  # still speech
    fade = hop_of(stream, worker, HOLD_AT + 1, block)
    assert np.abs(fade).max() > 0.0
    assert np.abs(stream[worker.fill_target + (HOLD_AT + 3) * block:]).max() == 0.0


def test_release_trims_the_cushion_and_speech_fades_back_in(engine, clip_48k):
    block = engine.plan.block
    cushion = int(HOLD_CUSHION_S * SR)
    release_at = HOLD_AT + 20

    def hold(worker, n):
        if n == HOLD_AT:
            worker.hold = True
        if n == release_at:
            worker.hold = False

    worker, stream, fills = drive(engine, clip_48k, release_at + 40, hold)
    assert fills[release_at - 1] == worker.fill_target + cushion
    # the reader took the cushion back out on its first read after the release
    assert fills[release_at] <= worker.fill_target and not worker.out_ring.drop_pending
    assert fills[-1] <= worker.fill_target
    start = worker.fill_target + (HOLD_AT + 2) * block
    held = stream[start:start + (release_at - HOLD_AT - 2) * block]
    assert np.abs(held).max() == 0.0  # only silence was dropped or played meanwhile
    resumed = stream[start + len(held):]
    assert np.abs(resumed).max() > 0.02  # live speech is back
    onset = int(np.argmax(np.abs(resumed) > 1e-4))
    assert abs(float(resumed[onset])) < 0.02  # from silence, not mid-waveform
    assert np.abs(np.diff(resumed[onset:onset + block])).max() < 0.2
