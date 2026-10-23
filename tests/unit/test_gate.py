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
    plan, gate, _ = vad_gate(level=0.9, hangover_ms=20.0)  # asks for one hop...
    assert gate.hangover_hops == 2  # ...but the 30 ms stream delay needs two (the guard)
    open_with(gate, at_dbfs(plan.block, -45.0))
    for _ in range(10):  # 5 dB under the open threshold: still inside the hysteresis
        assert gate.decide(at_dbfs(plan.block, -53.0)).speech
    gate.decide(at_dbfs(plan.block, -56.0))  # below threshold - 6: the hangover hops
    assert gate.decide(at_dbfs(plan.block, -56.0)).speech
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


def test_default_hangover_is_500_ms():
    assert GateConfig().hangover_ms == 500.0  # chosen by ear, docs/DEVIATIONS.md 14.7
    _, gate, _ = vad_gate(hangover_ms=GateConfig().hangover_ms)
    assert gate.hangover_hops == 25  # ceil(500 / 20)


def test_hangover_never_undercuts_the_stream_delay():
    """The output trails the input by overlap + search + right context: a hangover shorter
    than that would drain the SOLA tail while the end of the sentence is still inside the
    lookahead (docs/DEVIATIONS.md §14.1)."""
    plan = chunker.plan(20, 20, 500, 10, SR, SR, extra_right_ms=40.0)  # delay 70 ms
    gate = SilenceGate(plan, GateConfig(hangover_ms=20.0, use_vad=False))
    assert gate.hangover_hops == 4  # ceil(70 / 20)
    gate = SilenceGate(plan, GateConfig(hangover_ms=350.0, use_vad=False))
    assert gate.hangover_hops == 18  # the asked-for hangover when it is the longer one
    gate.set_config(GateConfig(hangover_ms=0.0, use_vad=False))
    assert gate.hangover_hops == 4


def test_energy_only_onset_waits_for_the_rise_to_persist_when_asked():
    """``onset_persist_chunks = 2``: a one-chunk burst (a desk tap) no longer opens the
    gate on energy alone; a rise that lasts two chunks still does, and the decision says
    it was an energy-only onset."""
    plan, gate, vad = vad_gate(level=0.0, onset_persist_chunks=2)
    quiet, loud = at_dbfs(plan.block, -70.0), at_dbfs(plan.block, -30.0)
    for _ in range(12):
        assert not gate.decide(quiet).speech
    # one loud 20 ms block: its chunk is loud but the one before it was quiet
    assert not gate.decide(loud).speech
    assert not gate.decide(quiet).speech
    for _ in range(10):
        gate.decide(quiet)
    # a rise that keeps going opens once two loud 32 ms chunks have completed: never on
    # the first 20 ms block, within 64 ms (four blocks) at the latest
    decisions = [gate.decide(loud) for _ in range(5)]
    opened = next(i for i, d in enumerate(decisions) if d.speech)
    assert 1 <= opened <= 3, opened
    assert decisions[opened].onset and decisions[opened].energy_only
    assert all(not d.speech for d in decisions[:opened])
    # and with the default (1 chunk) the single block opens at once, as before
    plan, gate, vad = vad_gate(level=0.0)
    for _ in range(12):
        gate.decide(quiet)
    d = gate.decide(loud)
    assert d.speech and d.onset and d.energy_only
    # a VAD-confirmed onset is never energy-only
    plan, gate, vad = vad_gate(level=0.9, onset_persist_chunks=2)
    for _ in range(12):
        gate.decide(quiet)
    d = gate.decide(at_dbfs(plan.block, -40.0))
    assert d.speech and d.onset and not d.energy_only


class SequenceVad(ScriptedVad):
    """Answers ``probs`` chunk by chunk (the last one repeats)."""

    def __init__(self, probs) -> None:
        super().__init__()
        self.probs = list(probs)

    def prob(self, chunk) -> float:
        p = self.probs[min(self.chunks, len(self.probs) - 1)]
        super().prob(chunk)
        return p


@pytest.mark.parametrize("reduce, opens", [("max", True), ("last", False)])
def test_a_long_hop_decides_on_the_most_speechlike_vad_chunk(reduce, opens):
    """A 60 ms hop is 960 samples at 16 kHz: the second hop completes two 512-sample chunks.
    Speech in the first of them (an onset late in the chunk before) must count; with
    ``vad_reduce="last"`` (the pre-fix behaviour) only the newest chunk did."""
    plan = chunker.plan(60, 20, 500, 10, SR, SR)
    vad = SequenceVad([0.0, 0.9, 0.1])
    gate = SilenceGate(plan, GateConfig(threshold_dbfs=-48.0, vad_reduce=reduce), vad)
    soft = at_dbfs(plan.block, -45.0)  # over the threshold, under the energy-onset margin
    assert not gate.decide(soft).speech and vad.chunks == 1
    d = gate.decide(soft)
    assert vad.chunks == 3 and d.speech is opens and d.onset is opens
    assert d.vad_prob == pytest.approx(0.9 if opens else 0.1)


def test_a_hop_with_no_finished_vad_chunk_keeps_the_last_probability():
    plan, gate, vad = vad_gate(level=0.9)  # 320 samples per hop at 16 kHz
    gate.decide(at_dbfs(plan.block, -40.0))  # 320 < 512: no chunk yet
    assert vad.chunks == 0 and gate._vad_prob == 0.0
    gate.decide(at_dbfs(plan.block, -40.0))
    assert vad.chunks == 1 and gate._vad_prob == pytest.approx(0.9)
    vad.level = 0.2
    gate.decide(at_dbfs(plan.block, -40.0))  # 128 + 320 < 512: still the 0.9
    assert vad.chunks == 1 and gate._vad_prob == pytest.approx(0.9)


def test_context_probation_rejects_a_click_and_confirms_a_held_onset():
    """``onset_confirm_ms`` (docs/DEVIATIONS.md §15): a 10 ms click in a pause flags an
    onset that decays at once, so the gate stays closed and the log says why; a rise that
    holds opens energy-only within the 30 ms confirmation. Off by default."""
    assert GateConfig().onset_confirm_ms == 0.0
    plan, gate, vad = vad_gate(level=0.0, onset_confirm_ms=30.0)
    sub = SR // 100
    quiet = at_dbfs(plan.block, -70.0)
    for _ in range(12):
        assert not gate.decide(quiet).speech
    clicked = quiet.copy()
    rng = np.random.default_rng(0)
    clicked[:sub] = (0.3 * rng.standard_normal(sub)).astype(np.float32)
    decisions = [gate.decide(clicked)] + [gate.decide(quiet) for _ in range(6)]
    assert not any(d.speech for d in decisions)
    entry = gate.onset_log[-1]
    assert entry["verdict"] == "rejected"
    assert entry["crest_db"] > 6.0 and 0.0 < entry["hf_ratio"] < 1.0
    for _ in range(12):
        gate.decide(quiet)
    loud = at_dbfs(plan.block, -30.0)
    decisions = [gate.decide(loud) for _ in range(4)]
    opened = next(i for i, d in enumerate(decisions) if d.speech)
    assert opened <= 2  # 30 ms of confirmation is at most two 20 ms hops
    assert decisions[opened].onset and decisions[opened].energy_only
    assert gate.onset_log[-1]["verdict"] == "held"


def test_context_probation_opens_on_a_vad_rise_after_a_burst():
    """A burst that decays but is followed by Silero's probability rising (a plosive the
    VAD recognises) is confirmed by the VAD rule."""
    plan, gate, vad = vad_gate(level=0.0, onset_confirm_ms=30.0)
    sub = SR // 100
    quiet = at_dbfs(plan.block, -70.0)
    for _ in range(12):
        gate.decide(quiet)
    burst = quiet.copy()
    burst[sub:2 * sub] = (0.3 * np.random.default_rng(1).standard_normal(sub)).astype(np.float32)
    gate.decide(burst)
    vad.level = 0.4  # a rise of 0.4 >= 0.25, still under the 0.5 open threshold
    decisions = [gate.decide(quiet) for _ in range(3)]
    d = next(d for d in decisions if d.speech)  # once 30 ms after the flag have arrived
    assert d.onset and d.energy_only
    assert gate.onset_log[-1]["verdict"] == "vad"
