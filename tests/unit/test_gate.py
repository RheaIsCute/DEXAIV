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


class ScriptedVad:
    """Stands in for Silero: answers ``level`` for every chunk and counts the chunks."""

    def __init__(self, level: float = 0.0) -> None:
        self.level = level
        self.chunks = 0
        self.threshold = 0.5

    def prob(self, chunk) -> float:
        assert len(chunk) == 512
        self.chunks += 1
        return self.level

    def reset(self) -> None:
        pass


def vad_gate(level=0.0, threshold_dbfs=-48.0, hangover_ms=200.0, **kw):
    plan = chunker.plan(20, 20, 500, 10, SR, SR)
    vad = ScriptedVad(level)
    cfg = GateConfig(threshold_dbfs=threshold_dbfs, hangover_ms=hangover_ms, **kw)
    return plan, SilenceGate(plan, cfg, vad), vad


def at_dbfs(n, dbfs):
    """A 200 Hz tone whose block RMS is ``dbfs``."""
    return tone(n, amp=math.sqrt(2) * 10 ** (dbfs / 20))


def test_vad_sees_every_block_including_quiet_ones():
    """The decimator and Silero run on quiet blocks too (they used to be skipped), so their
    state never jumps: 20 blocks of 320 samples at 16 kHz are 12 whole 512-chunks."""
    plan, gate, vad = vad_gate()
    for _ in range(20):
        gate.decide(np.zeros(plan.block, np.float32))
    assert vad.chunks == 20 * 320 // 512


def test_strong_onset_opens_before_the_vad_and_lapses_without_it():
    plan, gate, _ = vad_gate(level=0.0)  # the VAD never says speech
    loud = at_dbfs(plan.block, -48.0 + 7.0)  # 7 dB over the threshold
    d = gate.decide(loud)
    assert d.speech and d.onset  # opened on energy, in the first hop
    # on probation only: closes after ONSET_PROBATION_MS (5 hops at 20 ms), not 200 ms
    states = [gate.decide(np.zeros(plan.block, np.float32)).speech for _ in range(8)]
    assert states == [True] * (gate.probation_hops - 1) + [False] * (9 - gate.probation_hops)


def open_with(gate, block, limit: int = 4):
    """Decide ``block`` until the gate opens (the VAD answers once a 512-sample chunk is
    complete: up to two 320-sample hops); return the opening decision."""
    for _ in range(limit):
        d = gate.decide(block)
        if d.speech:
            return d
    raise AssertionError("the gate never opened")


def test_steady_lifted_room_tone_does_not_trigger_the_onset_rule():
    """After auto level a quiet mic's room tone can sit above threshold + margin on its own;
    only a rise out of it counts as an onset."""
    plan, gate, _ = vad_gate(level=0.0)
    room = at_dbfs(plan.block, -48.0 + 8.0)
    states = [gate.decide(room).speech for _ in range(30)]
    assert states[0] and not any(states[gate.probation_hops + 1:])  # only the very first
    rise = at_dbfs(plan.block, -48.0 + 8.0 + 13.0)
    assert gate.decide(rise).onset  # 13 dB out of the room tone: a real onset


def test_weak_onset_waits_for_the_vad():
    plan, gate, vad = vad_gate(level=0.0)
    soft = at_dbfs(plan.block, -48.0 + 3.0)  # over the threshold, under the onset margin
    assert not any(gate.decide(soft).speech for _ in range(4))
    vad.level = 0.9
    d = open_with(gate, soft)
    assert d.onset


def test_level_hysteresis_holds_the_gate_through_a_soft_syllable():
    plan, gate, _ = vad_gate(level=0.9, hangover_ms=20.0)  # hangover of one hop
    open_with(gate, at_dbfs(plan.block, -45.0))
    for _ in range(10):  # 5 dB under the open threshold: still inside the hysteresis
        assert gate.decide(at_dbfs(plan.block, -53.0)).speech
    gate.decide(at_dbfs(plan.block, -56.0))  # below threshold - 6: the hangover hop
    assert not gate.decide(at_dbfs(plan.block, -56.0)).speech
    assert not gate.decide(at_dbfs(plan.block, -53.0)).speech  # and -53 cannot reopen it


def test_vad_hysteresis_holds_the_gate_while_the_probability_dips():
    plan, gate, vad = vad_gate(level=0.9, hangover_ms=20.0)
    block = at_dbfs(plan.block, -40.0)
    open_with(gate, block)
    vad.level = 0.4  # under the 0.5 to open, over 0.5 - 0.15 to stay open
    assert all(gate.decide(block).speech for _ in range(10))
    vad.level = 0.3
    states = [gate.decide(block).speech for _ in range(6)]
    assert states[-1] is False  # closes once the dip outlasts the one-hop hangover


def test_default_hangover_is_350_ms():
    assert GateConfig().hangover_ms == 350.0
    _, gate, _ = vad_gate(hangover_ms=GateConfig().hangover_ms)
    assert gate.hangover_hops == 18  # ceil(350 / 20)
