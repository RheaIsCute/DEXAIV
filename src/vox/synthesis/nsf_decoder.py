"""Decoder backend seam (principle P5, docs/research/06 §7).

The synthesizer splits into ``infer_latent`` (encoder + flow, always torch) and a decoder
that turns ``(latent, source, speaker)`` into audio. V1 has one backend, the torch decoder
inside the loaded model; the Protocol exists so an ONNX decoder can slot in later without
touching the pipeline.
"""

from __future__ import annotations

from typing import Protocol

from torch import Tensor

from vox.synthesis.nets.synthesizer import Synthesizer


class DecoderBackend(Protocol):
    def decode(self, latent: Tensor, source: Tensor, speaker: Tensor) -> Tensor:
        """``latent [1, C, T]``, ``source [1, 1, T * upp]``, ``speaker [1, G, 1]`` -> audio."""
        ...


class TorchDecoder:
    """The NSF-HiFiGAN stage of an already-loaded synthesizer."""

    def __init__(self, net: Synthesizer) -> None:
        if not net.has_f0:
            raise ValueError("TorchDecoder wraps the NSF decoder of a pitch-guided model")
        self._dec = net.dec

    def decode(self, latent: Tensor, source: Tensor, speaker: Tensor) -> Tensor:
        return self._dec(latent, None, g=speaker, source=source)
