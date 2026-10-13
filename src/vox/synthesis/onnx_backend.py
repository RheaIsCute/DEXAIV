"""ONNX decoder backend: stub that pins the interface (V1.x, docs/research/06 §7 and §12).

Template when this gets built: Applio ``rvc/realtime/onnx_decoder.py`` (decoder-only export,
IO binding, ORT CUDA graphs, SHA-256-keyed export cache, excitation kept in torch). Only
worth doing if the decoder dominates the post-graph profile (open question Q5).
"""

from __future__ import annotations

from torch import Tensor


class OnnxDecoder:
    def decode(self, latent: Tensor, source: Tensor, speaker: Tensor) -> Tensor:
        raise NotImplementedError("ONNX decoder backend is deferred (docs/research/14 Q5)")
