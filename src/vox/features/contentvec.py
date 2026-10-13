"""ContentVec embedder via HF Transformers (docs/research/02 §3.3, 07 §A).

v2 models take the final (12th) encoder layer, 768-d. v1 models take hidden layer 9 through
``final_proj``, 256-d. The embedder always runs over the full conversion window: HuBERT's
attention is bidirectional, so there is nothing valid to cache between hops.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


def _model_class():
    from transformers import HubertModel

    class HubertModelWithFinalProj(HubertModel):
        """HuBERT plus the 768->256 projection that the ContentVec checkpoint carries."""

        def __init__(self, config):
            super().__init__(config)
            self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

    return HubertModelWithFinalProj


class ContentVec:
    name = "contentvec"

    def __init__(self, version: str = "v2", fp16: bool = False) -> None:
        if version not in ("v1", "v2"):
            raise ValueError(f"unsupported RVC feature version {version!r}")
        self.version = version
        self.dim = 768 if version == "v2" else 256
        self.fp16 = fp16
        self.model = None

    def load(self, device: torch.device, assets_dir: Path) -> None:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        path = Path(assets_dir) / "_embedders" / "contentvec"
        model = _model_class().from_pretrained(str(path), local_files_only=True)
        self.model = model.float().eval().requires_grad_(False).to(device)
        self.device = torch.device(device)
        self.fp16 = self.fp16 and self.device.type == "cuda"

    def _forward(self, audio_16k: Tensor) -> Tensor:
        if self.version == "v2":
            return self.model(input_values=audio_16k, output_hidden_states=False,
                              return_dict=True).last_hidden_state
        hidden = self.model(input_values=audio_16k, output_hidden_states=True,
                            return_dict=True).hidden_states[9]
        return self.model.final_proj(hidden)

    @torch.inference_mode()
    def embed(self, audio_16k: Tensor) -> Tensor:
        if not self.fp16:
            return self._forward(audio_16k)
        # FP16 around the embedder only; retrieval and synthesis keep FP32 features.
        with torch.autocast("cuda", dtype=torch.float16):
            feats = self._forward(audio_16k)
        return feats.float()
