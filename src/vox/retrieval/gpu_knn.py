"""Brute-force k-NN retrieval on the device (docs/research/07 §B.2/B.4).

Exact squared-L2 search with one ``addmm`` + ``topk`` over the reconstructed index
vectors, then the canonical blend: ``w_j = 1 / d_j^2`` over k = 8 neighbours. No CPU
round-trip, static shapes, so it can live inside the captured hop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

K_NEIGHBOURS = 8


@dataclass
class GpuIndex:
    vectors: Tensor  # [N, dim] float32, contiguous
    norms: Tensor  # [N] squared L2 norm of each vector

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    @property
    def ntotal(self) -> int:
        return self.vectors.shape[0]


def retrieve(feats: Tensor, index: GpuIndex, k: int = K_NEIGHBOURS) -> Tensor:
    """``[F, dim]`` query frames -> ``[F, dim]`` inverse-square-distance weighted neighbours."""
    query_norm = (feats * feats).sum(dim=1, keepdim=True)
    dist = torch.addmm(query_norm + index.norms, feats, index.vectors.T, alpha=-2.0)
    dist, ix = torch.topk(dist, k, dim=1, largest=False)
    weight = dist.clamp(min=1e-8).reciprocal().square()
    weight = weight / weight.sum(dim=1, keepdim=True)
    return (index.vectors[ix] * weight.unsqueeze(-1)).sum(dim=1)


def blend(feats: Tensor, index: GpuIndex, index_rate: Tensor | float,
          skip_offset: int = 0) -> None:
    """Blend ``feats [1, F, dim]`` toward the index in place, past ``skip_offset`` frames.

    Context frames (``skip_offset = skip_head // 2`` at 50 Hz) keep their raw embeddings:
    their output is discarded, so searching them would be wasted work.
    """
    tail = feats[0, skip_offset:]
    tail.lerp_(retrieve(tail, index), index_rate)
