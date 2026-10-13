"""Scan and import models into the ``models/<Name>/`` layout (docs/research/09 §6).

    models/<Name>/{model.pth, model.index, config.json}
    models/_embedders/...   models/_predictors/...      (shared, underscore-prefixed)
"""

from __future__ import annotations

import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from vox.config.schemas import IndexInfo, ModelConfig
from vox.config.store import save_as
from vox.models.metadata import ModelMeta, find_index, inspect_pth, sha256_file

MODEL_FILE = "model.pth"
INDEX_FILE = "model.index"
CONFIG_FILE = "config.json"


def scan(models_dir: Path) -> list[str]:
    """Names of every importable model directory (those holding a ``model.pth``)."""
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    return sorted(p.name for p in models_dir.iterdir()
                  if p.is_dir() and not p.name.startswith("_") and (p / MODEL_FILE).is_file())


def build_config(name: str, meta: ModelMeta, pth: Path, source_file: str,
                 index: IndexInfo, warnings: list[str]) -> ModelConfig:
    return ModelConfig(
        name=name, source_file=source_file, sha256=sha256_file(pth), version=meta.version,
        sr=meta.sr, f0=meta.f0, n_spk=meta.n_spk, sid=0, embedder=meta.embedder,
        emb_dim=meta.emb_dim, emb_output_layer=meta.emb_output_layer,
        use_final_proj=meta.use_final_proj, vocoder=meta.vocoder, index=index,
        defaults={}, warnings=[*meta.warnings, *warnings],
        imported_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def ensure_config(model_dir: Path) -> Path:
    """Write ``config.json`` for a model directory that lacks one. Returns its path."""
    model_dir = Path(model_dir)
    config_path = model_dir / CONFIG_FILE
    if config_path.is_file():
        return config_path
    pth = model_dir / MODEL_FILE
    meta = inspect_pth(pth)
    candidate, warnings = find_index(model_dir, meta.emb_dim)
    index = IndexInfo()
    if candidate is not None:
        index = IndexInfo(candidate.path.name, candidate.dim, candidate.ntotal)
    save_as(build_config(model_dir.name, meta, pth, pth.name, index, warnings), config_path)
    return config_path


def import_model(pth: Path, models_dir: Path, name: str | None = None,
                 index: Path | None = None) -> Path:
    """Copy a dropped ``.pth`` (and a paired ``.index``) into the layout. Returns the dir."""
    pth = Path(pth)
    meta = inspect_pth(pth)  # raises MetadataError before anything is copied
    name = name or re.sub(r"[^A-Za-z0-9_.-]+", "_", pth.stem)
    target = Path(models_dir) / name
    if (target / MODEL_FILE).exists():
        raise FileExistsError(f"model {name!r} already exists in {models_dir}")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pth, target / MODEL_FILE)

    warnings: list[str] = []
    index_info = IndexInfo()
    if index is None:
        candidate, warnings = find_index(pth.parent, meta.emb_dim)
    else:
        candidate, warnings = find_index(Path(index).parent, meta.emb_dim)
        if candidate is not None and candidate.path != Path(index):
            candidate = None
            warnings.append(f"{Path(index).name}: not usable with this model")
    if candidate is not None:
        shutil.copy2(candidate.path, target / INDEX_FILE)
        index_info = IndexInfo(INDEX_FILE, candidate.dim, candidate.ntotal)
    save_as(build_config(name, meta, target / MODEL_FILE, pth.name, index_info, warnings),
            target / CONFIG_FILE)
    return target
