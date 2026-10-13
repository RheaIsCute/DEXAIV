import json

import pytest

from vox.config import defaults
from vox.config.schemas import (
    ModelConfig,
    ModelOverrides,
    RuntimeParams,
    RuntimeProfile,
    VoiceProfile,
    from_dict,
    to_dict,
)
from vox.config.store import (
    Profile,
    SchemaError,
    load_as,
    load_profile,
    resolve_runtime_params,
    save_as,
    save_profile,
)


def test_nested_roundtrip():
    profile = VoiceProfile(created="2026-10-13T00:00:00Z")
    profile.source_voice.f0_median_hz = 118.4
    profile.f0_extractor.octave_flip_rate = {"rmvpe": 0.004}
    again = from_dict(VoiceProfile, json.loads(json.dumps(to_dict(profile))))
    assert again == profile


def test_unknown_keys_are_ignored():
    assert from_dict(RuntimeParams, {"f0_up_key": 7, "from_the_future": 1}).f0_up_key == 7


def test_profile_directory_roundtrip(tmp_path):
    profile = Profile()
    profile.runtime.block_ms = 30
    profile.model_overrides.models["aria"] = {"f0_up_key": 7}
    save_profile(profile, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "audio_device.json", "calibration.json", "model_overrides.json", "runtime.json",
        "voice_profile.json"]
    assert load_profile(tmp_path) == profile


def test_missing_profile_files_fall_back_to_defaults(tmp_path):
    assert load_profile(tmp_path) == Profile()


def test_newer_schema_is_rejected(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({"schema": 99}))
    with pytest.raises(SchemaError):
        load_as(RuntimeProfile, path)


def test_resolution_chain_order():
    """engine defaults <- model config <- profile <- per-model override <- CLI."""
    config = ModelConfig(name="aria", sid=3, defaults={"f0_up_key": 12, "protect": 0.4})
    profile = Profile(model_overrides=ModelOverrides(models={"aria": {"f0_up_key": 7}}))
    profile.voice_profile.f0_extractor.f0_max_hz = 300.0

    assert resolve_runtime_params() == defaults.DEFAULT_RUNTIME
    resolved = resolve_runtime_params(profile, config)
    assert (resolved.f0_up_key, resolved.protect, resolved.sid) == (7, 0.4, 3)
    assert resolved.f0_max_hz == 300.0
    cli = resolve_runtime_params(profile, config, {"f0_up_key": 5, "protect": None})
    assert (cli.f0_up_key, cli.protect) == (5, 0.4)


def test_runtime_profile_maps_to_shape_and_precision():
    runtime = RuntimeProfile(block_ms=30, context_extra_ms=300, hubert_fp16=True)
    assert runtime.shape().block_ms == 30 and runtime.shape().extra_ms == 300
    assert runtime.precision().hubert_fp16


def test_model_config_roundtrip(tmp_path):
    config = ModelConfig(name="m", sr=40000)
    config.index.file = "model.index"
    save_as(config, tmp_path / "config.json")
    assert load_as(ModelConfig, tmp_path / "config.json") == config
