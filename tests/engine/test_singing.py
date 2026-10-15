"""Singing support and the onset fix on real weights: graph safety of the new flags, the
onset ramp in a gated stream (docs/DEVIATIONS.md §8)."""

import dataclasses

import numpy as np
import pytest
import torch
from tests.conftest import MODELS, TEST_MODEL

from vox.config import defaults
from vox.config.schemas import RuntimeParams
from vox.engine.gate import GateConfig
from vox.offline.simulate import StreamSimulator, load_engine

pytestmark = [pytest.mark.assets, pytest.mark.gpu]
SR = 48000


@pytest.fixture(scope="module")
def cache():
    return {}


def test_phase_carry_and_ceiling_replay_exactly_like_eager(device, cache):
    """The carried phase is written inside the synth graph and read by the next replay;
    the ceiling is a device scalar. Warmup verifies replay against eager from the same
    state and noise: both must still agree to kernel rounding."""
    params = RuntimeParams(sine_phase_carry=True, f0_ceiling_hz=300.0, f0_up_key=6.0)
    engine = load_engine(MODELS, TEST_MODEL, defaults.DEFAULT_SHAPE, params, device=device,
                         cache=cache, graphs="on")
    assert engine.mode == "graphs"
    assert engine.graphs.report.verify["log_mel_l1"] < 0.01
    pipe = engine.pipeline
    assert pipe.sine_phase.abs().item() == 0.0  # warmup leaves the stream reset
    t = np.arange(SR) / SR
    voiced = (0.1 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)))
    StreamSimulator(engine).run(voiced.astype(np.float32), seed=0)
    assert pipe.sine_phase.abs().item() > 0.0  # the replays advanced it
    pipe.on_onset()
    assert pipe.sine_phase.abs().item() == 0.0
    # switching either op-sequence flag makes the graphs stale (the worker recaptures)
    assert pipe.set_params(dataclasses.replace(params, sine_phase_carry=False))
    assert engine.graphs.stale()
    engine.graphs.capture()
    assert params.f0_trim_tail == 2  # the default since section 8
    assert pipe.set_params(dataclasses.replace(params, sine_phase_carry=False,
                                               f0_trim_tail=1))
    assert pipe.pitch_cache.trim_tail == 1 and engine.graphs.stale()
    assert not pipe.set_params(dataclasses.replace(params, sine_phase_carry=False,
                                                   f0_trim_tail=1, f0_ceiling_hz=0.0))
    engine.graphs.release()


def test_gate_onset_hop_keeps_its_head_with_the_short_ramp(device, cache, clip_48k):
    """Same audio, seed and gate; only the onset fade differs. The hop the gate opens on
    must carry clearly more of the entrance with the 5 ms ramp than with the overlap fade."""
    lead = np.zeros(SR // 2, dtype=np.float32)
    audio = np.concatenate([lead, clip_48k[:2 * SR]])
    gate = GateConfig(threshold_dbfs=-45.0)
    energies = {}
    for ramp in (0.0, 5.0):
        shape = dataclasses.replace(defaults.DEFAULT_SHAPE, onset_ramp_ms=ramp)
        engine = load_engine(MODELS, TEST_MODEL, shape, RuntimeParams(), device=device,
                             cache=cache, gate=gate)
        result = StreamSimulator(engine).run(audio, gate=True, seed=0)
        opens = [e["hop"] for e in result.events if e["kind"] == "gate_open"]
        assert opens, "the gate never opened"
        block = engine.plan.block
        n = opens[0]
        energies[ramp] = float(np.sum(result.audio[n * block:(n + 1) * block] ** 2))
        engine.graphs.release()
        torch.cuda.empty_cache()
    assert energies[5.0] > 1.3 * energies[0.0]
