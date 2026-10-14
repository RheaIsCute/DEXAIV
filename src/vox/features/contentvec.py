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


AUTOCAST_DTYPES = {"off": None, "fp16": torch.float16, "bf16": torch.bfloat16}


class ContentVec:
    name = "contentvec"

    def __init__(self, version: str = "v2", fp16: bool = False, autocast: str = "off") -> None:
        if version not in ("v1", "v2"):
            raise ValueError(f"unsupported RVC feature version {version!r}")
        self.version = version
        self.dim = 768 if version == "v2" else 256
        self.model = None
        self.device = torch.device("cpu")
        self.autocast = "fp16" if fp16 else autocast

    @property
    def autocast(self) -> str:
        """``"off"`` | ``"fp16"`` | ``"bf16"``: the autocast dtype around the HuBERT forward.
        Settable between hops only by the control path (warmup's precision ladder); it
        changes the op sequence, so captured graphs are keyed on it."""
        return self._autocast

    @autocast.setter
    def autocast(self, value: str) -> None:
        if value not in AUTOCAST_DTYPES:
            raise ValueError(f"autocast must be one of {tuple(AUTOCAST_DTYPES)}, not {value!r}")
        self._autocast = value

    @property
    def fp16(self) -> bool:  # kept for callers written against the old flag
        return self.active_dtype is torch.float16

    @property
    def active_dtype(self) -> torch.dtype | None:
        """The dtype autocast will actually use: None on the CPU or when off."""
        if self.device.type != "cuda":
            return None
        return AUTOCAST_DTYPES[self._autocast]

    def load(self, device: torch.device, assets_dir: Path) -> None:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        path = Path(assets_dir) / "_embedders" / "contentvec"
        model = _model_class().from_pretrained(str(path), local_files_only=True)
        self.model = model.float().eval().requires_grad_(False).to(device)
        self.device = torch.device(device)

    def _forward(self, audio_16k: Tensor) -> Tensor:
        if self.version == "v2":
            return self.model(input_values=audio_16k, output_hidden_states=False,
                              return_dict=True).last_hidden_state
        hidden = self.model(input_values=audio_16k, output_hidden_states=True,
                            return_dict=True).hidden_states[9]
        return self.model.final_proj(hidden)

    @torch.inference_mode()
    def embed(self, audio_16k: Tensor) -> Tensor:
        dtype = self.active_dtype
        if dtype is None:
            return self._forward(audio_16k)
        # Autocast around the embedder only; retrieval and synthesis keep FP32 features.
        with torch.autocast("cuda", dtype=dtype):
            feats = self._forward(audio_16k)
        return feats.float()
