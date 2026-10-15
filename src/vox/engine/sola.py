"""SOLA stitching of consecutive hops (docs/research/03 §4).

Each hop the pipeline delivers ``block + overlap + search`` samples. SOLA picks the offset
in ``[0, search]`` whose segment best matches the tail saved from the previous hop:

    NCC(k) = sum_i y[k+i] p[i] / sqrt(sum_i y[k+i]^2 + eps)

then crossfades the first ``overlap`` samples against that tail with sin^2 / cos^2 windows,
emits ``block`` samples and saves the next ``overlap`` as the new tail.

The offset never leaves the device: it is applied with ``index_select`` on a tensor index,
so the hop has no ``.item()`` sync and the stage is CUDA-graph capturable. ``stitch`` runs
``_impl``, which graphs.py may replace with a graph replay of ``_stitch_eager``.

Gate protocol hooks (docs/research/03 §5, Applio): ``begin_drain`` / ``drain_step`` fade
the saved tail out over exactly one overlap when the gate closes; ``zero`` forgets it. A
zero tail makes the next hop fade in from silence through the ordinary crossfade, which is
the onset fade.

The overlap does not exceed the hop (the chunker clamps it unless ``overlap_beyond_hop``
asks otherwise, an A/B setting): every emitted sample is then a sin^2 / cos^2 mix of at
most two renders.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from vox.engine.chunker import ChunkPlan

MODES = ("plain", "pv")


class Sola:
    def __init__(self, plan: ChunkPlan, device: torch.device | str,
                 mode: str = "plain") -> None:
        if mode not in MODES:
            raise ValueError(f"sola mode must be one of {MODES}")
        self.mode = mode
        self.block = plan.block
        self.overlap = plan.sola_buffer
        self.search = plan.sola_search
        self.expected = plan.out_device
        n = self.overlap
        ramp = torch.linspace(0.0, 1.0, n, device=device)
        self.fade_in = torch.sin(0.5 * math.pi * ramp) ** 2
        self.fade_out = 1.0 - self.fade_in
        self.buffer = torch.zeros(n, device=device)  # previous hop's tail
        self._ones = torch.ones(1, 1, n, device=device)
        self._base = torch.arange(self.block + n, device=device)
        self._index = torch.zeros(self.block + n, dtype=torch.long, device=device)
        self._aligned = torch.zeros(self.block + n, device=device)
        self.out = self._aligned[:self.block]  # stable view: the hop's emitted samples
        self.offset_hist = torch.zeros(self.search + 1, dtype=torch.long, device=device)
        self._one = torch.ones(1, dtype=torch.long, device=device)
        self._drain_out = torch.zeros(self.block, device=device)
        self._drain_pos = n  # == overlap: nothing left to drain
        if mode == "pv":
            self._pv_window = torch.sqrt(self.fade_out * self.fade_in)
            self._pv_bins = 2 * math.pi * torch.arange(n // 2 + 1, device=device)
            self._pv_time = torch.arange(n, device=device).unsqueeze(-1) / n
        self._impl = self._stitch_eager

    def find_offset(self, infer: Tensor) -> Tensor:
        """0-dim device tensor: the best alignment offset in ``[0, search]``."""
        head = infer[None, None, :self.overlap + self.search]
        numerator = F.conv1d(head, self.buffer[None, None, :])
        denominator = torch.sqrt(F.conv1d(head * head, self._ones) + 1e-8)
        return torch.argmax(numerator[0, 0] / denominator[0, 0])

    @torch.inference_mode()
    def stitch(self, infer: Tensor) -> Tensor:
        """``[block + overlap + search]`` -> ``[block]`` (a view valid until the next call)."""
        return self._impl(infer)

    def _stitch_eager(self, infer: Tensor) -> Tensor:
        offset = self.find_offset(infer)
        torch.add(self._base, offset, out=self._index)
        torch.index_select(infer, 0, self._index, out=self._aligned)
        head = self._aligned[:self.overlap]
        if self.mode == "pv":
            head.copy_(self._phase_vocoder(self.buffer, head))
        else:
            head.mul_(self.fade_in).addcmul_(self.buffer, self.fade_out)
        # The tail is saved *after* the crossfade. With overlap <= block (always, unless
        # ``overlap_beyond_hop``) the two ranges are disjoint and the order is immaterial.
        # With overlap > block it is what keeps the seam continuous: the saved tail starts
        # on the mixed sample the emitted block ended on (a three-render blend whose weights
        # still sum to one). Saving it before the crossfade would restart the next hop on
        # the raw render: a step at every seam (tests/unit/test_sola.py).
        self.buffer.copy_(self._aligned[self.block:self.block + self.overlap])
        self.offset_hist.index_add_(0, offset.view(1), self._one)
        return self.out

    def _phase_vocoder(self, prev: Tensor, new: Tensor) -> Tensor:
        """Crossfade that phase-aligns the two segments per frequency bin.

        A plain fade sums two signals whose partials may be out of phase and partly cancel.
        Here the faded time-domain mix is topped up with a sinusoidal resynthesis whose
        phase glides from the old segment's to the new one's (docs/research/03 §4.2).
        """
        n = prev.shape[0]
        fa = torch.fft.rfft(prev * self._pv_window)
        fb = torch.fft.rfft(new * self._pv_window)
        magnitude = fa.abs() + fb.abs()
        # Double every bin that has a mirrored negative frequency (all but DC and Nyquist).
        magnitude[1:(-1 if n % 2 == 0 else None)] *= 2
        phase_a = fa.angle()
        delta = fb.angle() - phase_a
        delta = delta - 2 * math.pi * torch.floor(delta / (2 * math.pi) + 0.5)
        glide = torch.cos((self._pv_bins + delta) * self._pv_time + phase_a)
        resynth = (magnitude * glide).sum(-1) * self._pv_window / n
        return prev * self.fade_out**2 + new * self.fade_in**2 + resynth

    # ------------------------------------------------------------------ gate protocol

    def begin_drain(self) -> None:
        """Gate just closed: the saved tail will be faded out over the coming hops."""
        self._drain_pos = 0

    @property
    def draining(self) -> bool:
        return self._drain_pos < self.overlap

    @torch.inference_mode()
    def drain_step(self) -> Tensor:
        """One hop of the fade-out of the saved tail; silence once it is spent.

        The saved tail is the natural continuation of the last emitted sample, so fading it
        to zero over one overlap is a click-free ending. Eager (a handful of kernels, only
        on the hops right after a gate close).
        """
        out = self._drain_out
        out.zero_()
        pos = self._drain_pos
        n = min(self.block, self.overlap - pos)
        if n > 0:
            torch.mul(self.buffer[pos:pos + n], self.fade_out[pos:pos + n], out=out[:n])
            self._drain_pos = pos + n
            if self._drain_pos >= self.overlap:
                self.buffer.zero_()
        return out

    def zero(self) -> None:
        """Forget the saved tail: the next hop fades in from silence."""
        self.buffer.zero_()
        self._drain_pos = self.overlap

    # ----------------------------------------------------------------- control path

    def state_tensors(self) -> list[Tensor]:
        return [self.buffer, self._aligned, self._index, self.offset_hist]

    def offsets_ms(self, device_sr: int):
        """Histogram of chosen offsets, expanded to per-hop values in ms. Control path."""
        import numpy as np

        counts = self.offset_hist.cpu().numpy()
        return np.repeat(np.arange(len(counts)) * 1000.0 / device_sr, counts)

    def reset(self) -> None:
        self.buffer.zero_()
        self.offset_hist.zero_()
        self._drain_pos = self.overlap
