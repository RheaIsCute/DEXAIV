"""The content ``Embedder`` interface (docs/research/12 §3.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

# HuBERT-Base conv frontend: total stride 320, receptive field 400 samples at 16 kHz.
FRONTEND_STRIDE = 320
FRONTEND_FIELD = 400


def frames_for(n_samples: int) -> int:
    """Number of 50 Hz feature frames the conv frontend emits for ``n_samples`` of audio."""
    return (n_samples - FRONTEND_FIELD) // FRONTEND_STRIDE + 1


class Embedder(Protocol):
    name: str
    dim: int

    def load(self, device: torch.device, assets_dir: Path) -> None:
        """Control path only."""
        ...

    def embed(self, audio_16k: Tensor) -> Tensor:
        """``[1, n]`` 16 kHz audio -> ``[1, frames_for(n), dim]`` features at 50 Hz, FP32."""
        ...
