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
