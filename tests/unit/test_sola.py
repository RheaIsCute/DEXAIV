"""SOLA: known-offset recovery and stream continuity on synthetic signals."""

import math

import numpy as np
import pytest
import torch

from vox.engine import chunker
from vox.engine.sola import Sola

SR = 48000


def make(mode="plain", block_ms=20, crossfade_ms=30, device="cpu"):
    plan = chunker.plan(block_ms, crossfade_ms, 500, 10, SR, SR)
    return plan, Sola(plan, device, mode)


def sine(start: int, n: int, freq: float = 220.0) -> torch.Tensor:
    t = torch.arange(start, start + n, dtype=torch.float64)
    return torch.sin(2 * math.pi * freq * t / SR).float()


def test_windows_are_complementary_sin2_cos2():
    _, sola = make()
    assert torch.allclose(sola.fade_in + sola.fade_out, torch.ones_like(sola.fade_in))
    assert sola.fade_in[0] == 0 and sola.fade_in[-1] == pytest.approx(1.0)
    mid = sola.overlap // 2
    assert sola.fade_in[mid] == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("true_offset", [0, 37, 123, 240, 401, 480])
def test_recovers_a_known_offset(true_offset):
    """Plant the previous tail inside the new chunk at a known offset; SOLA must find it."""
    plan, sola = make()
    gen = torch.Generator().manual_seed(true_offset)
    chunk = torch.randn(plan.out_device, generator=gen) * 0.1
    sola.buffer.copy_(chunk[true_offset:true_offset + sola.overlap])
    assert sola.find_offset(chunk).item() == true_offset


def test_offset_tracks_sine_phase():
    """For a periodic signal the chosen offset must line the phases up exactly."""
    plan, sola = make()
    freq = 200.0  # period 240 samples: at most one period inside the 480-sample search
    sola.buffer.copy_(sine(1000, sola.overlap, freq))
    chunk = sine(1000 - 100, plan.out_device, freq)  # buffer content starts 100 samples in
    assert sola.find_offset(chunk).item() % 240 == 100


@pytest.mark.parametrize("mode", ["plain", "pv"])
@pytest.mark.parametrize("block_ms, crossfade_ms", [(20, 30), (10, 30), (50, 20), (30, 40)])
def test_stream_of_misaligned_chunks_is_continuous(mode, block_ms, crossfade_ms):
    """Chunks cut from one sine with a random misalignment each hop stitch seamlessly."""
    plan, sola = make(mode, block_ms, crossfade_ms)
    rng = np.random.default_rng(0)
    freq = 200.0  # period 240 samples
    out = []
    position = 0
    for hop in range(40):
        jitter = int(rng.integers(0, 240))  # the model output is never phase-coherent
        out.append(sola.stitch(sine(position + jitter, plan.out_device, freq)).clone())
        position += plan.block
    settle = -(-sola.overlap // plan.block) + 1  # hops until the initial fade-in has passed
    stream = torch.cat(out[settle:]).numpy()

    # A pure, constant-amplitude 200 Hz sine: the envelope is flat and no sample-to-sample
    # step exceeds a sine's own maximum slope.
    max_step = 2 * math.pi * freq / SR
    assert np.abs(np.diff(stream)).max() <= max_step * 1.02
    frames = stream[: len(stream) // 240 * 240].reshape(-1, 240)
    peaks = np.abs(frames).max(axis=1)
    assert peaks.min() > 0.97 and peaks.max() < 1.02


def sweep(start: int, n: int, f_start: float = 150.0, rate: float = 400.0) -> torch.Tensor:
    """A linear chirp from ``f_start`` rising by ``rate`` Hz per second (unique alignment)."""
    t = torch.arange(start, start + n, dtype=torch.float64) / SR
    return torch.sin(2 * math.pi * (f_start * t + 0.5 * rate * t * t)).float()


class SaveTailFirst(Sola):
    """The order the external report proposed: copy the next tail, *then* crossfade."""

    def _stitch_eager(self, infer):
        offset = self.find_offset(infer)
        torch.add(self._base, offset, out=self._index)
        torch.index_select(infer, 0, self._index, out=self._aligned)
        tail = self._aligned[self.block:self.block + self.overlap].clone()
        self._aligned[:self.overlap].mul_(self.fade_in).addcmul_(self.buffer, self.fade_out)
        self.buffer.copy_(tail)
        self.offset_hist.index_add_(0, offset.view(1), self._one)
        return self.out


def stitched_envelope(plan, hops: int = 60, cls=Sola) -> tuple[np.ndarray, np.ndarray]:
    """Stitch renders of one sweep whose level alternates 1.0 / 0.5 from hop to hop (two
    renders of the same audio never match); return (envelope, valid) per output sample,
    the envelope being output / sweep where the sweep is far from a zero crossing."""
    sola = cls(plan, "cpu")
    out, position = [], 0
    for hop in range(hops):
        gain = 1.0 if hop % 2 == 0 else 0.5
        # the render of hop n starts where the previous hop's emitted block ended
        out.append(sola.stitch(gain * sweep(position, plan.out_device)).clone())
        assert sola.offset_hist[0] == hop + 1, "renders are coherent: SOLA must pick offset 0"
        position += plan.block
    stream = torch.cat(out).numpy()
    reference = sweep(0, len(stream)).numpy()
    valid = np.abs(reference) > 0.5
    envelope = np.where(valid, stream / np.where(valid, reference, 1.0), np.nan)
    return envelope, valid


def max_envelope_step(plan, cls=Sola) -> float:
    """Largest envelope move between neighbouring valid samples, relative to the most the
    sin^2 crossfade itself can move it over that distance (gain difference 0.5, slope at
    most pi / (2 (overlap - 1)) per sample). <= 1: only ever the crossfade; a step: >> 1."""
    envelope, valid = stitched_envelope(plan, cls=cls)
    idx = np.flatnonzero(valid)
    idx = idx[idx > 3 * plan.block]  # past the initial fade-in from silence
    gaps = np.diff(idx)
    steps = np.abs(np.diff(envelope[idx]))
    near = gaps < 40  # neighbours across one zero crossing at most
    slope = 0.5 * math.pi / (2 * (plan.sola_buffer - 1))
    return float((steps[near] / (gaps[near] * slope)).max())


@pytest.mark.parametrize("block_ms, crossfade_ms", [(20, 20), (20, 30), (20, 10), (10, 40)])
def test_stitched_sweep_has_no_amplitude_step_at_hop_seams(block_ms, crossfade_ms):
    """overlap == block, the clamped case (30 and 40 requested at shorter hops), and overlap
    < block: the envelope only ever moves along the sin^2 crossfade."""
    plan = chunker.plan(block_ms, crossfade_ms, 500, 10, SR, SR)
    assert plan.sola_buffer <= plan.block
    assert max_envelope_step(plan) <= 1.05


def test_overlap_longer_than_the_hop_is_a_continuous_three_render_blend():
    """The pre-clamp geometry (30 ms overlap at a 20 ms hop, ``overlap_beyond_hop``): SOLA
    saves its tail after the crossfade, so the next hop starts on the mixed sample the last
    one ended on. No step: the weights of three renders still sum to one."""
    plan = chunker.plan(20, 30, 500, 10, SR, SR, overlap_beyond_hop=True)
    assert plan.sola_buffer > plan.block
    assert max_envelope_step(plan) <= 1.05


def test_saving_the_tail_before_the_crossfade_would_step_at_every_seam():
    """The external report's patch (copy the tail first): harmless while overlap <= block,
    where the ranges are disjoint, but with overlap > block the emitted head ends on a mix
    and the next hop restarts on the raw render. Guards against reintroducing it."""
    clamped = chunker.plan(20, 20, 500, 10, SR, SR)
    assert max_envelope_step(clamped, SaveTailFirst) <= 1.05
    long = chunker.plan(20, 30, 500, 10, SR, SR, overlap_beyond_hop=True)
    assert max_envelope_step(long, SaveTailFirst) > 20  # 25% of the gain difference at once


def test_hard_splice_without_sola_would_fail_the_same_check():
    """Sanity check of the test itself: naive concatenation does produce steps."""
    plan, _ = make()
    rng = np.random.default_rng(0)
    out, position = [], 0
    for _ in range(40):
        jitter = int(rng.integers(0, 240))
        out.append(sine(position + jitter, plan.block, 200.0))
        position += plan.block
    stream = torch.cat(out).numpy()
    assert np.abs(np.diff(stream)).max() > 10 * (2 * math.pi * 200.0 / SR)


def test_first_hop_fades_in_from_silence_and_state_resets():
    plan, sola = make(block_ms=50)  # hop longer than the overlap: the fade completes in it
    first = sola.stitch(torch.ones(plan.out_device)).clone()
    assert first[0] == 0 and first[-1] == pytest.approx(1.0)
    assert torch.all(torch.diff(first[: sola.overlap]) >= 0)
    assert sola.buffer.abs().sum() > 0
    sola.reset()
    assert sola.buffer.abs().sum() == 0 and sola.offset_hist.sum() == 0


def onset_sola(onset_ramp_ms: float):
    plan = chunker.plan(20, 20, 500, 10, SR, SR, onset_ramp_ms=onset_ramp_ms)
    return plan, Sola(plan, "cpu")


def test_gate_onset_keeps_the_decoded_head_with_a_short_ramp():
    """On an onset the tail is silent and the head of the hop is decoded pre-onset audio
    (the consonant the gate opened on): it fades in over 5 ms, not the whole overlap."""
    plan, sola = onset_sola(5.0)
    ramp = plan.onset_ramp
    assert ramp == 240
    sola.mark_onset()
    first = sola.stitch(torch.ones(plan.out_device)).clone()
    assert first[0] == 0 and first[ramp - 1] == pytest.approx(1.0)
    assert torch.all(torch.diff(first[:ramp]) >= 0) and torch.all(first[ramp:] == 1.0)
    # one hop only: the next stitch is the ordinary crossfade against the saved tail
    assert sola._onset.item() == 0.0
    second = sola.stitch(torch.ones(plan.out_device))
    assert torch.allclose(second, torch.ones_like(second))
    # without the mark (or with the ramp off) the silent tail fades in over the overlap
    for plan, sola in (onset_sola(5.0), onset_sola(0.0)):
        if plan.onset_ramp:
            sola.reset()
        else:
            sola.mark_onset()
        plain = sola.stitch(torch.ones(plan.out_device)).clone()
        assert torch.allclose(plain[:sola.overlap], sola.fade_in)
        assert plain[ramp] < 0.2  # 5 ms into a 20 ms sin^2 fade


def test_onset_ramp_keeps_more_of_a_consonant_burst():
    """A noise burst at the head of the onset hop (a /p/ or /t/ decoded just before the
    gate opened): the short ramp passes most of its energy, the plain fade very little."""
    gen = torch.Generator().manual_seed(1)
    plan, short = onset_sola(5.0)
    _, plain = onset_sola(0.0)
    chunk = torch.zeros(plan.out_device)
    chunk[:plan.block // 2] = 0.3 * torch.randn(plan.block // 2, generator=gen)  # 10 ms
    short.mark_onset()
    kept_short = short.stitch(chunk).pow(2).sum() / chunk.pow(2).sum()
    plain.mark_onset()
    kept_plain = plain.stitch(chunk).pow(2).sum() / chunk.pow(2).sum()
    # sin^2 ramps pass 3/8 of the energy under them: 5 of the burst's 10 ms vs all of it
    assert kept_short > 0.65 and kept_plain < 0.15


def test_silence_is_stable():
    plan, sola = make()
    for _ in range(3):
        out = sola.stitch(torch.zeros(plan.out_device))
    assert torch.isfinite(out).all() and out.abs().sum() == 0


def test_offset_histogram_and_stable_output_view():
    plan, sola = make()
    pointer = sola.out.data_ptr()
    for _ in range(5):
        sola.stitch(torch.randn(plan.out_device))
    assert sola.offset_hist.sum().item() == 5
    assert len(sola.offsets_ms(SR)) == 5 and sola.offsets_ms(SR).max() <= 10.0
    assert sola.out.data_ptr() == pointer and sola.out.shape[0] == plan.block


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["plain", "pv"])
def test_stitch_never_synchronises_with_the_host(mode):
    plan, sola = make(mode, device="cuda")
    chunks = [torch.randn(plan.out_device, device="cuda") for _ in range(4)]
    sola.stitch(chunks[0])
    torch.cuda.set_sync_debug_mode("error")
    try:
        for chunk in chunks[1:]:
            sola.stitch(chunk)
    finally:
        torch.cuda.set_sync_debug_mode("default")


# --------------------------------------------------------------- offset damping (hold)


def test_hold_eps_zero_is_the_plain_argmax():
    plan, sola = make()
    gen = torch.Generator().manual_seed(7)
    chunk = torch.randn(plan.out_device, generator=gen) * 0.1
    sola.buffer.copy_(chunk[300:300 + sola.overlap])
    sola.prev_offset.fill_(123)
    assert sola.hold_eps == 0.0 and sola.find_offset(chunk).item() == 300


def test_hold_keeps_the_previous_offset_within_eps_and_jumps_beyond_it():
    """A 200 Hz sine: the previous offset one period away from the best scores within a
    hair of it (both are phase-aligned); a quarter period away it scores ~0."""
    plan, sola = make()
    freq = 200.0  # period 240 samples
    sola.buffer.copy_(sine(1000, sola.overlap, freq))
    chunk = sine(1000 - 100, plan.out_device, freq)  # best offsets: 100 and 340
    score = sola.scores(chunk)
    assert score.shape[0] == sola.search + 1 and score.max() <= 1.0 + 1e-5
    assert score[100] > 0.99 and score[340] > 0.99 and abs(score[160]) < 0.2
    best = int(torch.argmax(score))
    other = 340 if best == 100 else 100
    sola.set_hold_eps(0.05)
    sola.prev_offset.fill_(other)
    assert sola.find_offset(chunk).item() == other  # held: within eps of the max
    sola.prev_offset.fill_(160)
    assert sola.find_offset(chunk).item() == best  # a quarter period off: jump
    sola.set_hold_eps(0.0)
    sola.prev_offset.fill_(160)
    assert sola.find_offset(chunk).item() == best  # eps 0: argmax
    with pytest.raises(ValueError):
        sola.set_hold_eps(-1.0)


def test_hold_reduces_offset_jumps_on_a_steady_tone_and_the_histograms_count_them():
    """Coherent renders of a 50 Hz tone (period 960 > the search) with a little noise: the
    correlation peak is broad, so the plain argmax wanders a few samples hop to hop on the
    noise; with damping the offset is held while the wander is within eps."""
    freq = 50.0
    outcomes = {}
    for eps in (0.0, 0.05):
        plan, sola = make()
        sola.set_hold_eps(eps)
        gen = torch.Generator().manual_seed(1)
        position = 0
        for _ in range(60):
            render = sine(position, plan.out_device, freq) + 0.1 * torch.randn(
                plan.out_device, generator=gen)
            sola.stitch(render)
            position += plan.block
        assert sola.offset_hist.sum().item() == 60 and sola.jump_hist.sum().item() == 60
        jumps = sola.jumps_ms(SR)[2:]  # after the fade-in from silence
        outcomes[eps] = float(np.mean(jumps > 0))
    assert outcomes[0.0] > 0.5  # the plain argmax re-times most hops on the noise
    assert outcomes[0.05] < 0.1


def test_hold_state_resets_and_is_graph_state():
    plan, sola = make()
    sola.set_hold_eps(0.02)
    sola.stitch(torch.randn(plan.out_device))
    sola.stitch(torch.randn(plan.out_device))
    assert sola.jump_hist.sum().item() == 2
    state = sola.state_tensors()
    assert any(t is sola.prev_offset for t in state) and any(t is sola.jump_hist for t in state)
    sola.reset()
    assert sola.prev_offset.item() == 0 and sola.jump_hist.sum().item() == 0
    assert sola.hold_eps == 0.02  # the parameter survives a reset; only state is cleared


@pytest.mark.gpu
@pytest.mark.parametrize("eps", [0.0, 0.05])
def test_hold_never_synchronises_with_the_host(eps):
    plan, sola = make(device="cuda")
    sola.set_hold_eps(eps)
    chunks = [torch.randn(plan.out_device, device="cuda") for _ in range(4)]
    sola.stitch(chunks[0])
    torch.cuda.set_sync_debug_mode("error")
    try:
        for chunk in chunks[1:]:
            sola.stitch(chunk)
    finally:
        torch.cuda.set_sync_debug_mode("default")
