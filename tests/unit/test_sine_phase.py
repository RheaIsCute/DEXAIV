"""NSF sine source: the experimental phase carry across hops (``sine_phase_carry``)."""

import pytest
import torch

from vox.synthesis.nets.generator import SineGen

SR, UPP = 48000, 480  # a 48 kHz model: one 10 ms frame is 480 samples


def glide(frames: int) -> torch.Tensor:
    """A sung note with vibrato and an unvoiced gap: [1, frames] Hz."""
    t = torch.arange(frames, dtype=torch.float32) / 100.0
    f0 = 220.0 * (1.0 + 0.02 * torch.sin(2 * torch.pi * 5.5 * t))
    f0[30:34] = 0.0
    return f0.unsqueeze(0)


def render(gen: SineGen, f0: torch.Tensor, phase=None, frames: int = 0) -> torch.Tensor:
    noise = torch.zeros(gen.noise_shape(f0.shape[1]))  # no noise: compare the sine itself
    return gen(f0, noise, phase, frames)[0, :, 0]


@pytest.mark.parametrize("window, shift", [(9, 2), (12, 4), (14, 6)])
def test_carried_phase_makes_hops_one_continuous_sine(window, shift):
    """Hops render overlapping windows shifted by ``shift`` frames, like the streaming
    decoder. With the carry each hop's first ``shift`` frames are the one long render's;
    without it every hop restarts at phase zero."""
    gen = SineGen(SR, UPP)
    f0 = glide(80)
    whole = render(gen, f0)
    phase = torch.zeros(1, 1, 1)
    worst_carry = worst_reset = 0.0
    for start in range(0, 80 - window, shift):
        piece = f0[:, start:start + window]
        span = slice(start * UPP, (start + shift) * UPP)
        carried = render(gen, piece, phase, shift)[:shift * UPP]
        reset = render(gen, piece)[:shift * UPP]
        worst_carry = max(worst_carry, (carried - whole[span]).abs().max().item())
        worst_reset = max(worst_reset, (reset - whole[span]).abs().max().item())
    assert worst_carry < 2e-3  # float32 phase accumulation over 80 frames
    assert worst_reset > 0.05  # sine amplitude is 0.1: the reset is a real phase jump


def test_without_a_phase_buffer_the_source_is_unchanged():
    gen = SineGen(SR, UPP)
    f0 = glide(20)
    assert torch.equal(render(gen, f0), render(gen, f0, None, 0))
    phase = torch.zeros(1, 1, 1)
    assert torch.equal(render(gen, f0, phase, 5), render(gen, f0))  # zero start phase
    assert phase.abs().item() > 0  # and the carry moved on
