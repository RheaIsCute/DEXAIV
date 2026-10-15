"""LiveSession reports what ran: after a live rebuild the take and the session line carry
the worker's plan and shape, never the plan the engine was built with (no audio, no GPU)."""

import dataclasses
import time
from types import SimpleNamespace

from vox.api.control import LiveConfig, LiveSession
from vox.config.schemas import RuntimeParams, ShapeParams
from vox.engine import chunker
from vox.engine.gate import GateConfig
from vox.engine.pipeline import STAGES
from vox.engine.worker import _applied
from vox.profiling.timers import StageTimers


class Recorder:
    def save(self, folder, stem, meta):
        return meta


def test_take_report_pairs_the_running_shape_with_the_running_plan(tmp_path):
    """take_20261015_004247 said shape.block_ms 60 but plan.hop_ms 30: the session had
    started at 30 ms and been rebuilt to 60; the plan came from the engine (stale)."""
    started = chunker.plan(30, 20, 2200, 10, 48000, 32000)
    rebuilt = chunker.plan(60, 20, 2200, 10, 48000, 32000)
    requested = ShapeParams(block_ms=30.0, extra_ms=2200.0)
    session = LiveSession(LiveConfig(model="Mom", models_dir=tmp_path, shape=requested))
    session.engine = SimpleNamespace(plan=started, mode="graphs", params=RuntimeParams(),
                                     timers=StageTimers(STAGES, "cpu", enabled=False))
    session.worker = SimpleNamespace(
        recorder=Recorder(), plan=rebuilt, params=RuntimeParams(f0_method="rmvpe"),
        shape=_applied(dataclasses.replace(requested, block_ms=60.0), rebuilt),
        gate=SimpleNamespace(config=GateConfig()), fill_target=6720)
    session.out_ring = SimpleNamespace(underruns=0)
    session.audio = SimpleNamespace(health=SimpleNamespace(input_latency_ms=22.0,
                                                           output_latency_ms=10.7))
    session.in_device = SimpleNamespace(index=28, name="Microphone (NVIDIA Broadcast)")
    session.out_device = SimpleNamespace(index=23, name="Line 1 (Virtual Audio Cable)")
    session._record_started = time.time()
    meta = session.stop_recording(tmp_path)
    assert meta["shape"]["block_ms"] == meta["plan"]["hop_ms"] == 60.0
    assert meta["shape"]["extra_ms"] == meta["plan"]["extra_ms"] == 2200.0
    assert "hop 60 ms" in meta["input"] and "hop 30" not in meta["input"]


def test_applied_shape_takes_the_chunker_rounding():
    plan = chunker.plan(34, 26, 1234, 9, 48000, 40000)
    shape = _applied(ShapeParams(block_ms=34, crossfade_ms=26, sola_search_ms=9,
                                 extra_ms=1234, embed_pad_ms=20.0), plan)
    assert (shape.block_ms, shape.crossfade_ms, shape.sola_search_ms, shape.extra_ms) == \
        (30.0, 30.0, 10.0, 1230.0)
    assert shape.embed_pad_ms == 20.0  # everything the chunker does not round is kept
