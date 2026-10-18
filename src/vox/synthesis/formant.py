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
* **The stretch is anchored to absolute time, not to the hop's window.** The window moves
  ``shift`` frames per hop, i.e. ``shift * f_e`` frames of the stretched latent: not a whole
  number, so a window-anchored stretch puts the decoder's frame grid somewhere else every
  hop. Consecutive renders of the same audio then stop agreeing and SOLA's alignment drifts
  one way (measured: +28 cents at -2 st on a steady tone at hop 20, docs/DEVIATIONS.md §12).
  Here the decoded stretched span always starts on an absolute frame boundary: one extra
  frame is decoded and the hop's window starts ``r_n = (n * shift * upp_res) mod upp``
  stretched samples into it. ``r_n`` cycles with period ``upp / gcd(shift * upp_res, upp)``
  hops; the interpolation plans and the crop for every phase are precomputed, and the hop's
  are copied into static buffers before the synth stage (outside any graph, like the noise).
* The stretch is linear interpolation with precomputed indices and weights (graph-safe).
  Positions outside the decoded window read zero, as the convolutions' padding does at an
  unstretched edge; holding the last frame instead skewed every hop's right edge, which is
  what the next hop's SOLA aligns to (-6.6 samples of hop-to-hop lag at -2 st, +28 cents
  of drift; zero-padded: -0.35 samples, within 9 cents of formant 0).
* The decoder context frames (``dec_context``) are stretched and dropped with the tail.
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


def stretched_frames(frames: int, st: float, upp: int, streaming: bool = False) -> int:
    """Frames the decoder renders for ``frames`` output frames: enough to hold the
    ``frames * upp_res`` stretched samples, plus up to one frame of phase offset when
    streaming (the absolute-time anchor)."""
    upp_res = resample_units(st, upp)
    return math.ceil((frames * upp_res + (upp - 1 if streaming else 0)) / upp)


def phase_period(shift_frames: int, st: float, upp: int) -> int:
    """Hops after which the stretched grid's offset ``r_n`` repeats."""
    return upp // math.gcd(shift_frames * resample_units(st, upp), upp)


def _plan(n_in: int, n_out: int, factor: float, offset: float) -> tuple[Tensor, Tensor, Tensor]:
    """Indices and weights: output ``m`` reads input position
    ``(m - offset + 0.5) / factor - 0.5`` (``F.interpolate(align_corners=False)`` at an
    exact factor when ``offset == 0``), clamped to the input."""
    pos = (torch.arange(n_out, dtype=torch.float64) - offset + 0.5) / factor - 0.5
    return _lerp(pos, n_in)


def _lerp(pos: Tensor, n_in: int) -> tuple[Tensor, Tensor, Tensor]:
    """``(lo, hi, w)`` reading ``pos`` (float64) from ``n_in`` points; positions more than
    half a step outside the input read zero (both weights 0), like the convolutions' zero
    padding at an unstretched edge. Holding the edge value there instead biased the right
    edge of every hop and with it SOLA's alignment (docs/DEVIATIONS.md §12)."""
    inside = (pos >= -0.5) & (pos <= n_in - 0.5)
    clamped = pos.clamp(0.0, n_in - 1)
    lo = clamped.floor().long().clamp(max=max(n_in - 2, 0))
    hi = (lo + 1).clamp(max=n_in - 1)
    # second row: 1 inside, 0 outside (a gain on the interpolated value)
    return lo, hi, torch.stack(((clamped - lo).float(), inside.float()))


class FormantWarp:
    """The stretch and the resample for ``frames`` decoded frames at one step.

    ``shift_frames`` > 0 (the streaming engine): the stretched grid is anchored to absolute
    time; call ``select(hop)`` before each hop's decode (outside a graph). 0 (a whole file,
    typed speech): one fixed plan anchored at the start.
    """

    def __init__(self, st: float, frames: int, upp: int, device: torch.device | str,
                 shift_frames: int = 0) -> None:
        import torchaudio

        self.st = float(st)
        self.frames = frames
        self.upp = upp
        self.upp_res = resample_units(st, upp)
        self.factor = self.upp_res / upp
        self.shift_frames = shift_frames
        streaming = shift_frames > 0
        self.frames2 = stretched_frames(frames, st, upp, streaming)
        self.keep = frames * self.upp_res  # stretched samples covering the D frames
        self.period = phase_period(shift_frames, st, upp) if streaming else 1
        n_src, n_out = frames * upp, self.frames2 * upp
        frame_lo, frame_hi, frame_w, src_lo, src_hi, src_w, crop = ([] for _ in range(7))
        for k in range(self.period):
            r = (k * shift_frames * self.upp_res) % upp
            lo, hi, w = _plan(n_src, n_out, self.factor, r)
            src_lo.append(lo)
            src_hi.append(hi)
            src_w.append(w)
            # latent frame j: the source position of its centre sample, in frames
            centre = torch.arange(self.frames2, dtype=torch.float64) * upp + (upp - 1) / 2
            pos = ((centre - r + 0.5) / self.factor - 0.5 - (upp - 1) / 2) / upp
            flo, fhi, fw = _lerp(pos, frames)
            frame_lo.append(flo)
            frame_hi.append(fhi)
            frame_w.append(fw)
            crop.append(torch.arange(self.keep) + r)
        dev = torch.device(device)
        self._table = [torch.stack(t).to(dev) for t in
                       (frame_lo, frame_hi, frame_w, src_lo, src_hi, src_w, crop)]
        # the static buffers the decoder reads (and a captured graph records)
        self._live = [t[0].clone() for t in self._table]
        self.resampler = torchaudio.transforms.Resample(
            self.upp_res, upp, dtype=torch.float32).to(dev)

    def phase(self, hop: int) -> int:
        """Index of hop ``hop``'s plan (``r_n``'s position in the cycle)."""
        return hop % self.period

    def select(self, hop: int) -> None:
        """Before hop ``hop``'s decode, outside any graph: load its plan into the static
        buffers (seven small device copies, no sync)."""
        k = self.phase(hop)
        for live, table in zip(self._live, self._table):
            live.copy_(table[k])

    @staticmethod
    def _apply(x: Tensor, lo: Tensor, hi: Tensor, wg: Tensor) -> Tensor:
        w, gain = wg[0].to(x.dtype), wg[1].to(x.dtype)
        return torch.lerp(x.index_select(-1, lo), x.index_select(-1, hi), w) * gain

    def stretch_frames(self, x: Tensor) -> Tensor:
        """``[b, c, frames]`` -> ``[b, c, frames2]``."""
        return self._apply(x, *self._live[0:3])

    def stretch_source(self, source: Tensor) -> Tensor:
        """``[b, 1, frames * upp]`` excitation -> ``[b, 1, frames2 * upp]``."""
        return self._apply(source, *self._live[3:6])

    def compress(self, audio: Tensor) -> Tensor:
        """``[b, 1, frames2 * upp]`` decoded audio -> ``[b, 1, frames * upp]``."""
        y = self.resampler(audio.index_select(-1, self._live[6]).float())
        return y[..., :self.frames * self.upp].to(audio.dtype)
