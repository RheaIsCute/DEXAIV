"""Shared fixtures. Tests needing downloaded weights or a GPU skip cleanly without them."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

# The pure-CPU engine path exists only for unit tests (TARGET.md rule 1).
os.environ.setdefault("VOX_ALLOW_CPU", "1")

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
CORPUS = ROOT / "benchmarks" / "corpus"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TEST_MODEL = "base_f0_48k_v2"
TEST_CLIP = CORPUS / "libri_1272.wav"


def _assets_ready() -> bool:
    needed = (MODELS / "_embedders" / "contentvec" / "pytorch_model.bin",
              MODELS / "_predictors" / "rmvpe.pt", MODELS / TEST_MODEL / "model.pth", TEST_CLIP)
    return all(p.is_file() for p in needed)


def pytest_collection_modifyitems(config, items):
    skip_assets = pytest.mark.skip(reason="run scripts/fetch_assets.py --all first")
    skip_gpu = pytest.mark.skip(reason="needs CUDA")
    for item in items:
        if "assets" in item.keywords and not _assets_ready():
            item.add_marker(skip_assets)
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def models_dir() -> Path:
    return MODELS


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def clip_48k() -> np.ndarray:
    """The first 8 s of the test clip, mono float32 at 48 kHz."""
    from vox.audio.resample import load_audio

    audio, _ = load_audio(TEST_CLIP, 48000)
    return audio[: 8 * 48000]
