"""Formant shift by decode-stretch-resample (the rvc-webui realtime method, ``infer/rtrvc.py``).

The decoder renders ``D`` frames of latent and excitation stretched in time by ``f``; the
``f``-times-longer audio is then resampled back to ``D`` frames. Resampling scales the whole
spectrum by ``f``: the resonances the decoder painted move up by ``f`` (formant +st is
"lighter"), and the sine the decoder was driven with, slowed by the stretch to ``pitch / f``,
comes back to exactly ``pitch``. The coarse pitch the text encoder sees is set to
``pitch / f`` as in rtrvc (the content is rendered "as if" at the lower pitch).

Where this departs from rtrvc (docs/DEVIATIONS.md §12):

* rtrvc stretches ``return_length`` frames to ``ceil(return_length * f)`` and resamples by
  ``floor(f * upp) / upp``: the two ratios differ, so pitch and timing are off by their
  mismatch. That is invisible at rvc-webui's 250 ms block and up to a quarter semitone at
  our 20 ms hop (5 frames). Here one rational ratio ``f_e = upp_res / upp`` (``upp_res`` the
  nearest integer to ``f * upp``) drives both the stretch and the resample, so the pitch
  and the length come back exact; ``f_e`` is within 1/(2 upp) of ``f`` (< 0.03 st).
* The stretch is linear interpolation with precomputed indices and weights (static shapes,
  graph-safe), at the same sample positions ``F.interpolate(align_corners=False)`` uses.
* The decoder context frames (``dec_context``) are stretched and dropped with the tail.

``FormantWarp`` is built once per (frames, step) on the control path; the hot path only
gathers, lerps and convolves.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

FORMANT_MIN_ST = -4.0
FORMANT_MAX_ST = 4.0
FORMANT_STEP_ST = 0.5


def snap_formant(st: float) -> float:
    """Clamp to +-4 st and snap to the 0.5 st grid (the graph cache's unit)."""
    st = max(FORMANT_MIN_ST, min(FORMANT_MAX_ST, float(st)))
    return FORMANT_STEP_ST * round(st / FORMANT_STEP_ST)


def resample_units(st: float, upp: int) -> int:
    """``upp_res``: decoded samples per 10 ms frame before the resample back to ``upp``."""
    return max(1, round(upp * 2.0 ** (st / 12.0)))


def effective_factor(st: float, upp: int) -> float:
    """The exact factor used: formants scale by it, the stretch and the resample share it."""
    return resample_units(st, upp) / upp


def stretched_frames(frames: int, st: float, upp: int) -> int:
    """Frames the decoder renders for ``frames`` output frames (>= frames * f_e)."""
    return math.ceil(frames * resample_units(st, upp) / upp)


def _lerp_plan(n_in: int, n_out: int, factor: float, device) -> tuple[Tensor, Tensor, Tensor]:
    """Indices and weights stretching ``n_in`` points by ``factor`` onto ``n_out`` points:
    output ``m`` reads input position ``(m + 0.5) / factor - 0.5``, clamped to the input."""
    pos = (torch.arange(n_out, dtype=torch.float64) + 0.5) / factor - 0.5
    pos = pos.clamp(0.0, n_in - 1)
    lo = pos.floor().long().clamp(max=max(n_in - 2, 0))
    hi = (lo + 1).clamp(max=n_in - 1)
    w = (pos - lo).float()
    return lo.to(device), hi.to(device), w.to(device)


class FormantWarp:
    """The stretch and the resample for ``frames`` decoded frames at one step."""

    def __init__(self, st: float, frames: int, upp: int, device: torch.device | str) -> None:
        import torchaudio

        self.st = float(st)
        self.frames = frames
        self.upp = upp
        self.upp_res = resample_units(st, upp)
        self.factor = self.upp_res / upp
        self.frames2 = stretched_frames(frames, st, upp)
        self._f = _lerp_plan(frames, self.frames2, self.factor, device)
        self._s = _lerp_plan(frames * upp, self.frames2 * upp, self.factor, device)
        self.keep = frames * self.upp_res  # stretched samples covering the D frames
        self.resampler = torchaudio.transforms.Resample(
            self.upp_res, upp, dtype=torch.float32).to(device)

    @staticmethod
    def _apply(x: Tensor, plan: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        lo, hi, w = plan
        a, b = x.index_select(-1, lo), x.index_select(-1, hi)
        return torch.lerp(a, b, w.to(x.dtype))

    def stretch_frames(self, x: Tensor) -> Tensor:
        """``[b, c, frames]`` -> ``[b, c, frames2]``."""
        return self._apply(x, self._f)

    def stretch_source(self, source: Tensor) -> Tensor:
        """``[b, 1, frames * upp]`` excitation -> ``[b, 1, frames2 * upp]``."""
        return self._apply(source, self._s)

    def compress(self, audio: Tensor) -> Tensor:
        """``[b, 1, frames2 * upp]`` decoded audio -> ``[b, 1, frames * upp]``."""
        y = self.resampler(audio[..., :self.keep].float())
        return y[..., :self.frames * self.upp].to(audio.dtype)
