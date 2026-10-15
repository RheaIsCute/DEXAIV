"""Rolling pitch cache (docs/research/05 §3.1; proven in rvc-webui and Applio).

F0 is recomputed only over a ~300 ms tail each hop. Older frames are reused verbatim: the
caches shift left by one hop and the fresh estimate overwrites the tail, minus its three
leading and one trailing frames, which are polluted by window padding.

Buffers are allocated once and never replaced, so their pointers stay valid for CUDA-graph
capture (principle P15).
"""

from __future__ import annotations

import torch
from torch import Tensor

TRIM_HEAD = 3
TRIM_TAIL = 1  # upstream; ``RuntimeParams.f0_trim_tail`` = 2 is the A/B right-edge guard


class PitchCache:
    def __init__(self, convert_frames: int, shift_frames: int, p_len: int,
                 device: torch.device | str, trim_tail: int = TRIM_TAIL) -> None:
        self.length = convert_frames + 1  # +1: extractors emit one more frame than n // 160
        self.shift = shift_frames
        # Fresh frames dropped at the right edge, where the tracker (RMVPE's BiGRU) has the
        # least right context. Like upstream, the rest is written flush with the end of the
        # cache, so each extra frame trimmed also delays the pitch by 10 ms against the
        # features. Part of the f0 stage's op sequence: changing it means a recapture.
        self.trim_tail = trim_tail
        if not 0 < shift_frames < self.length:
            raise ValueError("hop must be shorter than the conversion window")
        self.pitch = torch.ones(self.length, dtype=torch.long, device=device)
        self.pitchf = torch.zeros(self.length, dtype=torch.float32, device=device)
        keep = self.length - shift_frames
        self._tmp_pitch = torch.empty(keep, dtype=torch.long, device=device)
        self._tmp_pitchf = torch.empty(keep, dtype=torch.float32, device=device)
        # Stable [1, p_len] views handed to the synthesizer every hop.
        self.pitch_view = self.pitch[-p_len:].unsqueeze(0)
        self.pitchf_view = self.pitchf[-p_len:].unsqueeze(0)

    def roll_and_write(self, coarse: Tensor, pitchf: Tensor) -> None:
        """Shift one hop left, then write the interior of the fresh tail estimate."""
        self._tmp_pitch.copy_(self.pitch[self.shift:])
        self.pitch[:-self.shift].copy_(self._tmp_pitch)
        self._tmp_pitchf.copy_(self.pitchf[self.shift:])
        self.pitchf[:-self.shift].copy_(self._tmp_pitchf)
        interior = min(coarse.shape[0] - TRIM_HEAD - self.trim_tail, self.length)
        end = coarse.shape[0] - self.trim_tail
        self.pitch[-interior:].copy_(coarse[end - interior:end])
        self.pitchf[-interior:].copy_(pitchf[end - interior:end])

    def advance_silent(self) -> None:
        """A hop the gate kept closed: shift one hop left like ``roll_and_write`` and mark
        only the new hop unvoiced, so the history stays aligned with the 16 kHz ring (which
        advances every closed hop too). Wiping it instead left most of the window unvoiced
        against voiced features after every reopen."""
        self._tmp_pitch.copy_(self.pitch[self.shift:])
        self.pitch[:-self.shift].copy_(self._tmp_pitch)
        self.pitch[-self.shift:].fill_(1)
        self._tmp_pitchf.copy_(self.pitchf[self.shift:])
        self.pitchf[:-self.shift].copy_(self._tmp_pitchf)
        self.pitchf[-self.shift:].zero_()

    def zero(self) -> None:
        """Forget history (start of a stream): every frame unvoiced."""
        self.pitch.fill_(1)
        self.pitchf.zero_()
