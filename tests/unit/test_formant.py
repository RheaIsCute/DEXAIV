"""Formant shift bookkeeping (synthesis/formant.py, docs/DEVIATIONS.md §12): the factor,
the decode length, and that stretch + resample is an identity on timing and pitch. CPU."""

import math

import numpy as np
import pytest
import torch

from vox.config.schemas import RuntimeParams
from vox.engine import chunker
from vox.synthesis.formant import (
    FORMANT_MAX_ST,
    FormantWarp,
    effective_factor,
    resample_units,
    snap_formant,
    stretched_frames,
)


def test_snap_clamps_and_rounds_to_half_semitones():
    assert snap_formant(0.26) == 0.5 and snap_formant(-0.74) == -0.5
    assert snap_formant(9) == FORMANT_MAX_ST and snap_formant(-9) == -4.0
    assert RuntimeParams().formant_st == 0.0


@pytest.mark.parametrize("upp", [320, 400, 480])  # 32k, 40k, 48k models
@pytest.mark.parametrize("st", [-4.0, -2.0, -0.5, 0.5, 1.0, 2.0, 4.0])
def test_effective_factor_is_within_a_thirtieth_of_a_semitone(upp, st):
    f = effective_factor(st, upp)
    assert abs(12 * math.log2(f) - st) < 1 / 30
    assert f == resample_units(st, upp) / upp
    assert (f > 1) == (st > 0)  # positive = formants up = lighter


@pytest.mark.parametrize("st", [-4.0, -1.5, 2.0, 4.0])
def test_decode_length_covers_the_stretched_frames(st):
    plan = chunker.plan(20, 20, 1000, 10, 48000, 32000, decoder_context_ms=80)
    d = plan.decode_frames
    assert d == plan.return_length + plan.dec_context == 13
    d2 = stretched_frames(d, st, 320)
    assert d2 == math.ceil(d * effective_factor(st, 320))
    assert d2 * 320 >= d * resample_units(st, 320)  # the kept span fits in what is decoded


@pytest.mark.parametrize("st", [-2.0, 2.0])
def test_stretch_then_compress_restores_length_and_pitch(st):
    """A sine at the frame rate's sample grid, stretched by f and resampled by 1/f, comes
    back at its own frequency and length: the warp moves the spectrum, not the timing."""
    upp, frames, sr = 320, 40, 32000
    warp = FormantWarp(st, frames, upp, "cpu")
    t = torch.arange(frames * upp) / sr
    tone = torch.sin(2 * math.pi * 220.0 * t)[None, None]
    stretched = warp.stretch_source(tone)
    assert stretched.shape[-1] == warp.frames2 * upp
    # stretched in time: its frequency dropped by the factor
    spec = np.abs(np.fft.rfft(stretched[0, 0, :warp.keep].numpy() * np.hanning(warp.keep)))
    peak = np.argmax(spec) * sr / warp.keep
    assert peak == pytest.approx(220.0 / warp.factor, rel=0.02)
    back = warp.compress(stretched)
    assert back.shape[-1] == frames * upp
    core = slice(upp * 4, upp * (frames - 4))  # away from the resampler's edges
    np.testing.assert_allclose(back[0, 0, core].numpy(), tone[0, 0, core].numpy(), atol=0.02)


def test_latent_stretch_maps_frame_centres_like_interpolate():
    """Same sample positions as F.interpolate(align_corners=False) for an exact factor."""
    warp = FormantWarp(2.0, 13, 320, "cpu")
    x = torch.arange(13, dtype=torch.float32).reshape(1, 1, 13)
    y = warp.stretch_frames(x)[0, 0]
    expected = ((torch.arange(warp.frames2) + 0.5) / warp.factor - 0.5).clamp(0, 12)
    torch.testing.assert_close(y, expected.float())
