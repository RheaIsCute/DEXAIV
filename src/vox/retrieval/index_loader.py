"""Read a FAISS ``.index`` and upload its vectors (docs/research/07 §B, 12 §3.4).

faiss-cpu is used only here, to read the file and reconstruct the raw vectors; searching
happens in ``gpu_knn``. Nothing in this module runs unless ``index_rate > 0`` (principle P6).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vox.retrieval.gpu_knn import GpuIndex

MAX_VECTORS = 200_000  # brute-force cost and VRAM stay bounded past this


class IndexFileError(ValueError):
    """The index file cannot be used with this model."""


def reconstruct(path: Path, emb_dim: int) -> np.ndarray:
    import faiss

    path = Path(path)
    if path.name.startswith("trained_"):
        raise IndexFileError(f"{path.name}: a trained_* index holds no vectors; use added_*")
    index = faiss.read_index(str(path))
    if index.d != emb_dim:
        raise IndexFileError(f"{path.name}: index dim {index.d} != model embedding dim {emb_dim}")
    if index.ntotal == 0:
        raise IndexFileError(f"{path.name}: index is empty (trained but nothing added)")
    return index.reconstruct_n(0, index.ntotal)


def load(path: Path, emb_dim: int, device: torch.device | str,
         max_vectors: int = MAX_VECTORS) -> GpuIndex:
    vectors = reconstruct(path, emb_dim)
    if len(vectors) > max_vectors:
        keep = np.random.default_rng(0).choice(len(vectors), max_vectors, replace=False)
        vectors = vectors[np.sort(keep)]
    tensor = torch.from_numpy(np.ascontiguousarray(vectors, dtype=np.float32)).to(device)
    return GpuIndex(tensor, (tensor * tensor).sum(dim=1))
