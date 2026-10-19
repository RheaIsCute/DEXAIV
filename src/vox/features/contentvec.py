"""ContentVec embedder via HF Transformers (docs/research/02 §3.3, 07 §A).

v2 models take the final (12th) encoder layer, 768-d. v1 models take hidden layer 9 through
``final_proj``, 256-d. The embedder always runs over the full conversion window: HuBERT's
attention is bidirectional, so there is nothing valid to cache between hops.

Conv layer 0's normalisation (``norm_mode``, docs/DEVIATIONS.md §13, report 18 §C7): the
shipped checkpoint uses ``feat_extract_norm = "group"``, a GroupNorm with one group per
channel, i.e. each channel is normalised over the *time axis of the whole input*. In a
sliding window that makes every frame's features depend on what else is in the window
(and inflates pauses, R4). ``"frozen"`` replaces it with a per-channel affine using
statistics measured once (``measure_norm_stats``) so the conv stack becomes shift-invariant;
a train/test mismatch of unknown size, kept as an A/B. The switch changes the op sequence:
captured graphs are keyed on it.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

NORM_MODES = ("window", "frozen")
FROZEN_NORM_FILE = "frozen_norm.pt"  # under models/_embedders/contentvec/


def _model_class():
    from transformers import HubertModel

    class HubertModelWithFinalProj(HubertModel):
        """HuBERT plus the 768->256 projection that the ContentVec checkpoint carries."""

        def __init__(self, config):
            super().__init__(config)
            self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

    return HubertModelWithFinalProj


AUTOCAST_DTYPES = {"off": None, "fp16": torch.float16, "bf16": torch.bfloat16}


class FrozenGroupNorm(nn.Module):
    """Per-channel affine standing in for a GroupNorm(C, C): ``(x - mean) / sqrt(var + eps)
    * weight + bias`` with ``mean`` / ``var`` fixed buffers instead of the input's own
    over-time statistics."""

    def __init__(self, norm: nn.GroupNorm, mean: Tensor, var: Tensor) -> None:
        super().__init__()
        c = norm.num_channels
        if mean.shape != (c,) or var.shape != (c,):
            raise ValueError(f"frozen norm statistics must be [{c}] per channel")
        self.eps = norm.eps
        self.weight = norm.weight
        self.bias = norm.bias
        self.register_buffer("mean", mean.detach().clone().float())
        self.register_buffer("scale", torch.rsqrt(var.detach().clone().float() + norm.eps))

    def forward(self, x: Tensor) -> Tensor:
        shift = (self.bias - self.mean * self.scale * self.weight).view(1, -1, 1)
        return x * (self.scale * self.weight).view(1, -1, 1) + shift


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
        self._window_norm: nn.GroupNorm | None = None  # conv layer 0's own GroupNorm
        self.norm_mode = "window"

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
        layer0 = self.model.feature_extractor.conv_layers[0]
        norm = getattr(layer0, "layer_norm", None)
        self._window_norm = norm if isinstance(norm, nn.GroupNorm) else None
        self.norm_mode = "window"

    # --------------------------------------------------------- conv layer 0 normalisation

    @property
    def has_group_norm(self) -> bool:
        """Whether this checkpoint normalises conv layer 0 over time (the ``"frozen"``
        option applies only then)."""
        return self._window_norm is not None

    def _layer0(self):
        return self.model.feature_extractor.conv_layers[0]

    @torch.inference_mode()
    def measure_norm_stats(self, audio_16k: Tensor, chunk_s: float = 20.0
                           ) -> dict[str, Tensor]:
        """Per-channel mean and variance of conv layer 0's output over the time axis of
        ``audio_16k`` (``[1, n]`` or ``[n]``), in FP32: the statistics ``"frozen"`` applies.
        Long audio is processed in chunks and the moments pooled."""
        if not self.has_group_norm:
            raise RuntimeError("this checkpoint has no over-time GroupNorm to freeze")
        x = audio_16k.to(self.device, torch.float32).reshape(1, -1)
        conv = self._layer0().conv
        step = int(chunk_s * 16000)
        total = torch.zeros(conv.out_channels, dtype=torch.float64, device=self.device)
        total_sq = torch.zeros_like(total)
        count = 0
        for start in range(0, x.shape[1], step):
            y = conv(x[:, start:start + step].unsqueeze(1))[0].double()  # [C, T]
            total += y.sum(dim=1)
            total_sq += (y * y).sum(dim=1)
            count += y.shape[1]
        mean = total / count
        var = total_sq / count - mean * mean
        return {"mean": mean.float().cpu(), "var": var.clamp(min=0).float().cpu(),
                "frames": torch.tensor(count)}

    def set_norm(self, mode: str, stats: dict[str, Tensor] | None = None) -> None:
        """Control path: ``"window"`` (the checkpoint's GroupNorm) or ``"frozen"`` with
        ``stats`` from ``measure_norm_stats`` (or ``load_norm_stats``)."""
        if mode not in NORM_MODES:
            raise ValueError(f"norm mode must be one of {NORM_MODES}, not {mode!r}")
        if self.model is None:
            raise RuntimeError("load the embedder first")
        if not self.has_group_norm:
            if mode == "frozen":
                raise RuntimeError("this checkpoint has no over-time GroupNorm to freeze")
            return
        layer0 = self._layer0()
        if mode == "window":
            layer0.layer_norm = self._window_norm
        else:
            if stats is None:
                raise ValueError("frozen norm needs statistics (measure_norm_stats)")
            layer0.layer_norm = FrozenGroupNorm(self._window_norm, stats["mean"],
                                                stats["var"]).to(self.device)
        self.norm_mode = mode

    @staticmethod
    def norm_stats_path(assets_dir: Path) -> Path:
        return Path(assets_dir) / "_embedders" / "contentvec" / FROZEN_NORM_FILE

    @classmethod
    def load_norm_stats(cls, assets_dir: Path) -> dict[str, Tensor] | None:
        path = cls.norm_stats_path(assets_dir)
        if not path.is_file():
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    @classmethod
    def save_norm_stats(cls, assets_dir: Path, stats: dict[str, Tensor]) -> Path:
        path = cls.norm_stats_path(assets_dir)
        torch.save({k: v.detach().cpu() for k, v in stats.items()}, path)
        return path

    # ------------------------------------------------------------------------- hot path

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
