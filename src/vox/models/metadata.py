"""Checkpoint metadata auto-detection (docs/research/09 §5). Control path, CPU only.

Detection sniffs the weights rather than trusting dict keys: community checkpoints omit or
misreport ``version`` / ``f0``. ``detect_checkpoint`` is pure (dict in, dataclass out) so it
is testable with synthetic fixtures.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

OFFICIAL_CONFIG_LEN = 18
SUPPORTED_SR = (32000, 40000, 48000)
_SR_STRINGS = {"32k": 32000, "40k": 40000, "48k": 48000}
_EMB_KEY = "enc_p.emb_phone.weight"
_SPK_KEY = "emb_g.weight"
_F0_KEY = "dec.m_source.l_linear.weight"


class MetadataError(ValueError):
    """The file cannot be used as an RVC inference model; the message says why."""


@dataclass
class ModelMeta:
    format: str = "pth"
    sr: int = 48000
    f0: bool = True
    version: str = "v2"
    emb_dim: int = 768
    n_spk: int = 1
    embedder: str = "contentvec"
    emb_output_layer: int = 12
    use_final_proj: bool = False
    vocoder: str = "HiFi-GAN"
    config: list[Any] = field(default_factory=list)
    info: str = ""
    fork: bool = False
    warnings: list[str] = field(default_factory=list)


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(tensor.shape)


def _parse_sr(value: Any) -> int:
    if isinstance(value, str):
        if value in _SR_STRINGS:
            return _SR_STRINGS[value]
        if value.isdigit():
            return int(value)
        raise MetadataError(f"unrecognised sample rate {value!r} in config[-1]")
    return int(value)


def detect_checkpoint(cpt: Any) -> ModelMeta:
    """Derive model metadata from a loaded checkpoint dict. Raises ``MetadataError``."""
    if not isinstance(cpt, dict):
        raise MetadataError("checkpoint is not a dict; not an RVC model")
    if "weight" not in cpt or "config" not in cpt:
        if "model" in cpt or "optimizer" in cpt:
            raise MetadataError(
                "this is a training checkpoint (has 'model'/'optimizer'); export the small "
                "inference model from your training tool and import that instead")
        raise MetadataError("missing 'weight'/'config': not an RVC inference checkpoint")
    weight = cpt["weight"]
    config = list(cpt["config"])
    warnings: list[str] = []
    for key in (_EMB_KEY, _SPK_KEY):
        if key not in weight:
            raise MetadataError(f"weights lack '{key}': not an RVC synthesizer")
    if len(config) < OFFICIAL_CONFIG_LEN:
        raise MetadataError(f"config has {len(config)} fields, expected {OFFICIAL_CONFIG_LEN}")

    fork = len(config) != OFFICIAL_CONFIG_LEN
    if fork:
        # ddPn08-style forks append fields after the official 18 (docs/research/09 §2).
        warnings.append(f"fork model: config has {len(config)} fields; using the first 18")
        config = config[:OFFICIAL_CONFIG_LEN]

    # 1. sample rate: config[-1], cross-checked against the decoder's total upsampling.
    sr = _parse_sr(config[-1])
    upp = math.prod(config[12])
    if upp * 100 != sr:
        warnings.append(f"config sr {sr} disagrees with upsample rates ({upp * 100}); "
                        "trusting the upsample rates")
        sr = upp * 100
    if sr not in SUPPORTED_SR:
        warnings.append(f"unusual sample rate {sr}")
    config[-1] = sr

    # 2. f0: the NSF source module exists only in pitch-guided models (authoritative).
    f0 = _F0_KEY in weight
    declared_f0 = cpt.get("f0")
    if declared_f0 is not None and bool(declared_f0) != f0:
        warnings.append(f"'f0' key says {declared_f0} but weights say {int(f0)}; "
                        "trusting weights")

    # 3. version: the text-encoder input width is the 256/768 discriminator.
    emb_dim = _shape(weight[_EMB_KEY])[1]
    if emb_dim not in (256, 768):
        raise MetadataError(f"unsupported embedding width {emb_dim} (expected 256 or 768)")
    version = "v2" if emb_dim == 768 else "v1"
    declared = cpt.get("version")
    if declared is not None and declared != version:
        warnings.append(f"'version' key says {declared!r} but weights are {version}; "
                        "trusting weights")

    # 4. speakers: the stored count is unreliable, the embedding table is not.
    n_spk = _shape(weight[_SPK_KEY])[0]
    config[-3] = n_spk

    # 5. embedder.
    embedder = str(cpt.get("embedder_name", "contentvec"))
    embedder = re.sub(r"768$", "", embedder) or "contentvec"
    if embedder not in ("contentvec", "hubert_base"):
        warnings.append(f"trained on embedder {embedder!r}; only contentvec is bundled")
    out_layer = int(cpt.get("embedder_output_layer", 9 if emb_dim == 256 else 12))

    # 6. vocoder (Applio fork extension).
    vocoder = str(cpt.get("vocoder", "HiFi-GAN"))
    if vocoder != "HiFi-GAN" or not any(k.startswith("dec.ups.0.") for k in weight):
        raise MetadataError(f"unsupported vocoder {vocoder!r}; only NSF-HiFi-GAN is implemented")

    return ModelMeta(
        format="pth", sr=sr, f0=f0, version=version, emb_dim=emb_dim, n_spk=n_spk,
        embedder="contentvec" if embedder == "hubert_base" else embedder,
        emb_output_layer=out_layer, use_final_proj=emb_dim == 256, vocoder=vocoder,
        config=config, info=str(cpt.get("info", "")), fork=fork, warnings=warnings,
    )


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a ``.pth`` with ``weights_only=True``. Never falls back to full unpickling."""
    import pickle

    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError) as err:
        raise MetadataError(
            f"{Path(path).name} could not be loaded safely (weights_only=True): {err}"
        ) from err


def inspect_pth(path: Path) -> ModelMeta:
    return detect_checkpoint(load_checkpoint(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class IndexCandidate:
    path: Path
    dim: int
    ntotal: int


def find_index(directory: Path, emb_dim: int) -> tuple[IndexCandidate | None, list[str]]:
    """Pair a usable ``.index`` from ``directory``: prefer ``added_*``, verify the dimension."""
    import faiss

    warnings: list[str] = []
    candidates = sorted(Path(directory).glob("*.index"),
                        key=lambda p: (not p.name.startswith("added_"), p.name))
    for path in candidates:
        if path.name.startswith("trained_"):
            warnings.append(f"{path.name}: a trained_* index has no vectors; use added_*")
            continue
        try:
            index = faiss.read_index(str(path))
        except RuntimeError as err:
            warnings.append(f"{path.name}: unreadable index ({err})")
            continue
        if index.d != emb_dim:
            warnings.append(f"{path.name}: index dim {index.d} != model dim {emb_dim}")
            continue
        if index.ntotal == 0:
            warnings.append(f"{path.name}: index is empty (trained but nothing added)")
            continue
        return IndexCandidate(path, index.d, index.ntotal), warnings
    return None, warnings
