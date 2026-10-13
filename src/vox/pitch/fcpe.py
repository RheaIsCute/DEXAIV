"""FCPE pitch extractor: the V1 alternate (docs/research/05 §2.2).

Wraps the bundled ``torchfcpe`` model. It is always fed at least 300 ms of audio: the hop
length controls latency, not pitch quality (the Applio floor).

``graph_safe`` is False for now: the stock ``torchfcpe`` inference path has not been made
sync-free (Applio replaces its mel frontend and decoder to get there). That work belongs
with CUDA-graph capture in Phase 6.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

FLOOR_SAMPLES = 4800
GUARD_SAMPLES = 800
HOP = 160


class FcpeExtractor:
    name = "fcpe"
    graph_safe = False

    def __init__(self, threshold: float = 0.006) -> None:
        self.threshold = threshold
        self.model = None

    def load(self, device: torch.device, assets_dir: Path) -> None:
        from torchfcpe import spawn_bundled_infer_model

        self.model = spawn_bundled_infer_model(device=device)
        self.device = torch.device(device)

    def tail_samples(self, block_16k: int) -> int:
        return max(block_16k + GUARD_SAMPLES, FLOOR_SAMPLES)

    @torch.inference_mode()
    def extract(self, audio_16k: Tensor) -> Tensor:
        frames = audio_16k.shape[0] // HOP + 1
        f0 = self.model.infer(
            audio_16k[None, :, None], sr=16000, decoder_mode="local_argmax",
            threshold=self.threshold, f0_min=32.7, f0_max=1975.5, interp_uv=False,
            output_interp_target_length=frames)
        return f0[0, :, 0]
