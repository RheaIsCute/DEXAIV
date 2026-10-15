"""JSON load/save for profiles, with atomic writes (docs/research/11 §5, 12 §3.10)."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from vox.config import defaults
from vox.config.schemas import (
    RUNTIME_PARAM_FIELDS,
    SCHEMA_VERSION,
    AudioDeviceProfile,
    CalibrationProfile,
    ModelConfig,
    ModelOverrides,
    RuntimeParams,
    RuntimeProfile,
    VoiceProfile,
    from_dict,
    to_dict,
)

T = TypeVar("T")

PROFILE_FILES: dict[str, type] = {
    "voice_profile": VoiceProfile,
    "calibration": CalibrationProfile,
    "audio_device": AudioDeviceProfile,
    "runtime": RuntimeProfile,
    "model_overrides": ModelOverrides,
}


class SchemaError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: expected a JSON object")
    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Write via a temp file + rename so a crash never leaves a half-written profile."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def load_as(cls: type[T], path: Path) -> T:
    data = load_json(path)
    schema = data.get("schema", SCHEMA_VERSION)
    if schema > SCHEMA_VERSION:
        raise SchemaError(f"{path}: schema {schema} is newer than supported {SCHEMA_VERSION}")
    return from_dict(cls, data)


def _drop_unset(data: Any) -> Any:
    """Remove ``None`` values from (nested) dicts: an unset field is not written, so it
    keeps falling through to the engine default when the file is read back."""
    if isinstance(data, dict):
        return {k: _drop_unset(v) for k, v in data.items() if v is not None}
    return data


def save_as(obj: Any, path: Path) -> None:
    save_json(path, _drop_unset(to_dict(obj)))


@dataclass
class Profile:
    """The five per-user files of one ``profiles/<name>/`` directory."""

    voice_profile: VoiceProfile = field(default_factory=VoiceProfile)
    calibration: CalibrationProfile = field(default_factory=CalibrationProfile)
    audio_device: AudioDeviceProfile = field(default_factory=AudioDeviceProfile)
    runtime: RuntimeProfile = field(default_factory=RuntimeProfile)
    model_overrides: ModelOverrides = field(default_factory=ModelOverrides)


def load_profile(directory: Path) -> Profile:
    """Load whichever of the five files exist; missing files fall back to defaults."""
    directory = Path(directory)
    profile = Profile()
    for stem, cls in PROFILE_FILES.items():
        path = directory / f"{stem}.json"
        if path.is_file():
            setattr(profile, stem, load_as(cls, path))
    return profile


def save_profile(profile: Profile, directory: Path) -> None:
    for stem in PROFILE_FILES:
        save_as(getattr(profile, stem), Path(directory) / f"{stem}.json")


def load_model_config(path: Path) -> ModelConfig:
    return load_as(ModelConfig, path)


# ------------------------------------------------------------ persisted warmup decisions


def load_tuned(profile_dir: Path | None, key: str) -> dict[str, Any] | None:
    """The warmup record stored under ``key`` in ``runtime.json``, or None."""
    if profile_dir is None:
        return None
    path = Path(profile_dir) / "runtime.json"
    if not path.is_file():
        return None
    return load_as(RuntimeProfile, path).tuned.get(key)


def save_tuned(profile_dir: Path | None, key: str, record: dict[str, Any]) -> None:
    """Store ``record`` under ``key`` in ``runtime.json`` (created from defaults if absent)."""
    if profile_dir is None:
        return
    path = Path(profile_dir) / "runtime.json"
    runtime = load_as(RuntimeProfile, path) if path.is_file() else RuntimeProfile()
    runtime.tuned[key] = record
    save_as(runtime, path)


def resolve_runtime_params(
    profile: Profile | None = None,
    model_config: ModelConfig | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> RuntimeParams:
    """Resolution chain of docs/research/08 §7, with explicit CLI values winning last."""
    values = dataclasses.asdict(defaults.DEFAULT_RUNTIME)
    layers: list[dict[str, Any]] = []
    if model_config is not None:
        layers.append({**model_config.defaults, "sid": model_config.sid})
    if profile is not None:
        f0 = profile.voice_profile.f0_extractor
        calibrated = {"f0_method": f0.method, "f0_min_hz": f0.f0_min_hz,
                      "f0_max_hz": f0.f0_max_hz}
        layers.append({k: v for k, v in calibrated.items() if v is not None})
        if model_config is not None:
            layers.append(profile.model_overrides.models.get(model_config.name, {}))
    if cli_overrides:
        layers.append({k: v for k, v in cli_overrides.items() if v is not None})
    for layer in layers:
        values.update({k: v for k, v in layer.items() if k in RUNTIME_PARAM_FIELDS})
    return RuntimeParams(**values)
