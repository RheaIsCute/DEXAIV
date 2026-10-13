"""Post-processing chain after synthesis (principle P7). V1 ships one stage: RMS mix.

``volume_envelope`` (rms_mix_rate) 1.0 keeps the model's own loudness; lower values pull
the output envelope toward the input's. Envelopes use 40 ms frames at a 10 ms hop, like
the rvc-webui realtime path, but stay on the device.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from vox.engine.chunker import ChunkPlan


class RmsMix:
    graph_safe = True

    def __init__(self, plan: ChunkPlan, device: torch.device | str) -> None:
        self.hop = plan.zc
        self.frame = 4 * plan.zc
        self.size = plan.out_device
        self._exponent = torch.zeros((), device=device)

    def set_rate(self, rate: float) -> None:
        self._exponent.fill_(1.0 - rate)

    def _envelope(self, x: Tensor) -> Tensor:
        power = F.pad((x * x)[None, None], (self.frame // 2, self.frame // 2))
        rms = torch.sqrt(F.avg_pool1d(power, self.frame, self.hop))
        return F.interpolate(rms, size=self.size + 1, mode="linear",
                             align_corners=True)[0, 0, :-1]

    def run(self, audio: Tensor, reference: Tensor) -> None:
        """Scale ``audio`` in place by ``(rms_ref / rms_out) ** (1 - rate)``."""
        target = self._envelope(reference)
        current = self._envelope(audio).clamp(min=1e-3)
        audio.mul_(torch.pow(target / current, self._exponent))


class PostChain:
    """Ordered, individually skippable stages. A disabled stage costs nothing."""

    def __init__(self, plan: ChunkPlan, device: torch.device | str) -> None:
        self.rms_mix = RmsMix(plan, device)
        self.rate = 1.0

    def set_volume_envelope(self, rate: float) -> None:
        self.rate = float(rate)
        self.rms_mix.set_rate(self.rate)

    @property
    def needs_reference(self) -> bool:
        return self.rate < 1.0

    def run(self, audio: Tensor, reference: Tensor) -> None:
        if self.rate < 1.0:
            self.rms_mix.run(audio, reference)
