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


def test_profiles_store_only_what_was_chosen_and_new_defaults_fall_through(tmp_path,
                                                                           monkeypatch):
    """runtime.json / voice_profile.json never hold copies of engine defaults: an engine
    field added or retuned later is not shadowed by a stale 0.0 / false from the file."""
    save_profile(Profile(), tmp_path)
    runtime_file = json.loads((tmp_path / "runtime.json").read_text())
    assert not {"block_ms", "decoder_context_ms", "tf32", "graphs"} & set(runtime_file)
    f0_file = json.loads((tmp_path / "voice_profile.json").read_text())["f0_extractor"]
    assert not {"method", "f0_min_hz", "f0_max_hz"} & set(f0_file)

    import dataclasses

    monkeypatch.setattr(defaults, "DEFAULT_SHAPE", dataclasses.replace(
        defaults.DEFAULT_SHAPE, decoder_context_ms=80.0, embed_pad_ms=20.0))
    monkeypatch.setattr(defaults, "DEFAULT_RUNTIME", dataclasses.replace(
        defaults.DEFAULT_RUNTIME, volume_envelope=0.4, f0_max_hz=900.0))
    monkeypatch.setattr(defaults, "DEFAULT_PRECISION", dataclasses.replace(
        defaults.DEFAULT_PRECISION, graphs="on"))
    profile = load_profile(tmp_path)
    shape = profile.runtime.shape()
    assert (shape.decoder_context_ms, shape.embed_pad_ms) == (80.0, 20.0)
    assert profile.runtime.precision().graphs == "on"
    params = resolve_runtime_params(profile, ModelConfig(name="m"))
    assert (params.volume_envelope, params.f0_max_hz) == (0.4, 900.0)
    # a value somebody did choose still wins over the default
    profile.runtime.decoder_context_ms = 0.0
    assert profile.runtime.shape().decoder_context_ms == 0.0
    save_profile(profile, tmp_path)
    assert json.loads((tmp_path / "runtime.json").read_text())["decoder_context_ms"] == 0.0


def test_model_config_roundtrip(tmp_path):
    config = ModelConfig(name="m", sr=40000)
    config.index.file = "model.index"
    save_as(config, tmp_path / "config.json")
    assert load_as(ModelConfig, tmp_path / "config.json") == config
