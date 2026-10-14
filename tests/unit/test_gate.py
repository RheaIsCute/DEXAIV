"""Silence gate: state machine, hangover, and click-free SOLA drain / onset (no models)."""

import math

import numpy as np
import pytest
import torch

from vox.engine import chunker
from vox.engine.gate import GateConfig, SilenceGate, _Decimator, rms_dbfs
from vox.engine.sola import Sola

SR = 48000


def make_gate(block_ms=20, hangover_ms=200, use_vad=False, threshold_dbfs=-48.0):
    plan = chunker.plan(block_ms, 30, 500, 10, SR, SR)
    cfg = GateConfig(threshold_dbfs=threshold_dbfs, hangover_ms=hangover_ms, use_vad=use_vad)
    return plan, SilenceGate(plan, cfg)


def tone(n, start=0, freq=200.0, amp=0.1):
    t = np.arange(start, start + n) / SR
    return (amp * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def test_energy_gate_opens_closes_with_hangover_and_marks_transitions():
    plan, gate = make_gate(hangover_ms=200)
    block = plan.block
    loud, quiet = tone(block), np.zeros(block, np.float32)
    d = gate.decide(quiet)
    assert not d.speech and not d.onset and not d.drain
    d = gate.decide(loud)
    assert d.speech and d.onset and not d.drain
    assert gate.decide(loud).speech and not gate.decide(loud).onset
    # 200 ms hangover at a 20 ms hop: 10 quiet hops stay open, the 11th closes with drain.
    for _ in range(10):
        d = gate.decide(quiet)
        assert d.speech and not d.drain
    d = gate.decide(quiet)
    assert not d.speech and d.drain
    assert not gate.decide(quiet).drain
    assert gate.transitions == 2
    assert rms_dbfs(loud) == pytest.approx(20 * math.log10(0.1 / math.sqrt(2)), abs=0.1)


def test_disabled_gate_is_always_open():
    _, gate = make_gate()
    gate.set_config(GateConfig(enabled=False))
    d = gate.decide(np.zeros(gate.plan.block, np.float32))
    assert d.speech and d.onset
    assert gate.decide(np.zeros(gate.plan.block, np.float32)).speech


def test_decimator_is_stateful_and_preserves_a_low_tone():
    plan = chunker.plan(20, 30, 500, 10, SR, SR)
    dec = _Decimator(SR, plan.block)
    full = tone(10 * plan.block, freq=300.0)
    out = np.concatenate([dec(full[i * plan.block:(i + 1) * plan.block]) for i in range(10)])
    assert len(out) == 10 * plan.block // 3
    ref = full[::3]
    # Linear-phase FIR delays by (taps-1)/2 input samples = 31 -> ~10.3 output samples.
    lag = int(np.argmax([np.dot(out[20:-20], np.roll(ref, k)[20:-20]) for k in range(20)]))
    aligned = np.roll(ref, lag)[40:-40]
    assert np.abs(out[40:-40] - aligned).max() < 0.01  # no block-boundary seams


@pytest.mark.parametrize("block_ms, crossfade_ms", [(20, 30), (50, 20), (10, 40)])
def test_drain_and_onset_are_click_free(block_ms, crossfade_ms):
    """Speech -> close (drain) -> silence -> open: no step anywhere in the emitted stream."""
    plan = chunker.plan(block_ms, crossfade_ms, 500, 10, SR, SR)
    sola = Sola(plan, "cpu")
    freq = 200.0
    out, position = [], 0
    rng = np.random.default_rng(1)

    def speech_hop():
        nonlocal position
        jitter = int(rng.integers(0, 240))
        chunk = torch.from_numpy(tone(plan.out_device, position + jitter, freq, 1.0))
        out.append(sola.stitch(chunk).clone())
        position += plan.block

    for _ in range(20):
        speech_hop()
    # gate closes: drain the tail exactly once, then silence
    sola.begin_drain()
    drained = 0
    while sola.draining:
        out.append(sola.drain_step().clone())
        drained += 1
    assert drained == -(-sola.overlap // plan.block)
    assert sola.buffer.abs().sum() == 0
    for _ in range(5):
        out.append(torch.zeros(plan.block))
    # gate opens: the zero tail makes the first hop fade in through the crossfade
    for _ in range(20):
        speech_hop()
    stream = torch.cat(out).numpy()
    settle = (-(-sola.overlap // plan.block) + 1) * plan.block
    stream = stream[settle:]
    max_step = 2 * math.pi * freq / SR  # a unit sine's own steepest sample-to-sample step
    steps = np.abs(np.diff(stream))
    assert steps.max() <= max_step * 1.02, f"step {steps.max():.4f} vs sine max {max_step:.4f}"
    # and the stream really went silent in between before coming back
    silent_from = 20 * plan.block + drained * plan.block - settle
    assert np.abs(stream[silent_from:silent_from + 5 * plan.block]).max() == 0.0
    assert np.abs(stream[-plan.block:]).max() > 0.9


def test_hard_cut_would_fail_the_step_check():
    plan = chunker.plan(20, 30, 500, 10, SR, SR)
    # cut a quarter period in, where the sine is at its peak
    stream = np.concatenate([tone(plan.block * 5 + 60, 0, 200.0, 1.0), np.zeros(plan.block)])
    assert np.abs(np.diff(stream)).max() > 10 * (2 * math.pi * 200.0 / SR)


def test_silero_vad_tells_speech_like_from_silence():
    """Silero on the host: silence scores ~0; a loud synthetic voiced tone scores higher.
    (Real speech is what it is trained on; this only checks the plumbing.)"""
    from vox.engine.gate import SileroVad

    vad = SileroVad()
    vad.load()
    plan, _ = make_gate()
    gate = SilenceGate(plan, GateConfig(threshold_dbfs=-60.0, use_vad=True), vad)
    for _ in range(10):
        d = gate.decide(np.zeros(plan.block, np.float32))
    assert not d.speech and d.vad_prob < 0.2
