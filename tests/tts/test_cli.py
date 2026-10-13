"""``vox say`` argument plumbing, with the backend and the conversion mocked."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from vox import cli
from vox.tts import backends, speak
from vox.tts.base import TtsError
from vox.tts.cli import DEFAULT_OUT, TTS_DIR


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A models directory with one model config, plus recorders for backend/speak/play."""
    models = tmp_path / "models"
    (models / "Voice").mkdir(parents=True)
    (models / "Voice" / "model.pth").write_bytes(b"")
    (models / "Voice" / "config.json").write_text(json.dumps(
        {"name": "Voice", "sid": 3, "defaults": {"index_rate": 0.5, "f0_up_key": 9}}))
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "model_overrides.json").write_text(json.dumps(
        {"models": {"Voice": {"target_f0_median_hz": 150.0}}}))
    seen = SimpleNamespace(picked=None, speak=None, played=None, models=models,
                           base=["--models-dir", str(models), "--profile", str(profile)])

    def pick_backend(name, voice, cache_dir):
        seen.picked = (name, voice, cache_dir)
        return SimpleNamespace(name=name or "kokoro", voice=voice or "af_heart",
                               voices=lambda: ["af_heart", "am_adam"])

    def fake_speak(text, backend, models_dir, model, params, pitch, target_f0, precision,
                   device, seed, out):
        seen.speak = SimpleNamespace(text=text, model=model, params=params, pitch=pitch,
                                     target_f0=target_f0, device=device, seed=seed, out=out)
        return speak.SayResult(np.zeros(48000, np.float32), 48000, 5.0 if pitch is None else pitch,
                               pitch is None, 150.0, target_f0, 1, 1.0, out)

    monkeypatch.setattr(backends, "pick_backend", pick_backend)
    monkeypatch.setattr(speak, "speak", fake_speak)
    monkeypatch.setattr(speak, "play", lambda audio, sr, device: setattr(
        seen, "played", (len(audio), sr, device)))
    return seen


def test_defaults(env, capsys):
    assert cli.main(["say", "Hello.", "--model", "Voice", *env.base]) == 0
    assert env.picked == (None, None, env.models / TTS_DIR)
    s = env.speak
    assert (s.text, s.model, s.pitch, s.seed, s.device) == ("Hello.", "Voice", None, 0, None)
    assert s.target_f0 == 150.0  # from the profile's model overrides
    assert (s.params.index_rate, s.params.sid, s.params.f0_method) == (0.5, 3, "rmvpe")
    assert s.out == DEFAULT_OUT and env.played is None
    out = capsys.readouterr().out
    assert "auto pitch: +5 semitones" in out and "150.0 Hz -> target 150.0 Hz" in out


def test_every_flag_reaches_its_destination(env, capsys):
    argv = ["say", "Hola.", "--model", "Voice", "--backend", "edge", "--voice",
            "es-MX-DaliaNeural", "--pitch", "-2", "--index-rate", "0.75", "--protect", "0.1",
            "--f0-method", "fcpe", "--seed", "7", "--out", "x.wav", "--play", "--device",
            "Speakers", "--torch-device", "cpu", *env.base]
    assert cli.main(argv) == 0
    assert env.picked[:2] == ("edge", "es-MX-DaliaNeural")
    s = env.speak
    assert (s.pitch, s.seed, s.device, str(s.out)) == (-2.0, 7, "cpu", "x.wav")
    assert (s.params.index_rate, s.params.protect, s.params.f0_method) == (0.75, 0.1, "fcpe")
    assert env.played == (48000, 48000, "Speakers")
    out = capsys.readouterr().out
    assert "pitch: -2 semitones" in out and "auto pitch" not in out


def test_play_without_out_writes_no_file(env):
    assert cli.main(["say", "Hi.", "--model", "Voice", "--play", *env.base]) == 0
    assert env.speak.out is None and env.played == (48000, 48000, None)


def test_text_from_file(env, tmp_path):
    script = tmp_path / "script.txt"
    script.write_text("Línea uno. Línea dos.", encoding="utf-8")
    assert cli.main(["say", "--file", str(script), "--model", "Voice", *env.base]) == 0
    assert env.speak.text == "Línea uno. Línea dos."


def test_list_voices_needs_no_model(env, capsys):
    assert cli.main(["say", "--list-voices", "--backend", "kokoro", *env.base]) == 0
    assert capsys.readouterr().out.split() == ["af_heart", "am_adam"]
    assert env.picked[0] == "kokoro" and env.speak is None


@pytest.mark.parametrize("argv, message", [
    (["say", "Hi."], "--model is required"),
    (["say", "--model", "Voice"], "either as an argument or with --file"),
    (["say", "Hi.", "--file", "x.txt", "--model", "Voice"], "either as an argument"),
    (["say", "Hi.", "--model", "Missing"], "no model named 'Missing'"),
    (["say", "--file", "does_not_exist.txt", "--model", "Voice"], "does_not_exist.txt"),
])
def test_usage_errors(env, capsys, argv, message):
    assert cli.main([*argv, *env.base]) == 1
    assert message in capsys.readouterr().err and env.speak is None


def test_backend_errors_are_reported(env, capsys, monkeypatch):
    def unavailable(name, voice, cache_dir):
        raise TtsError("no TTS backend available: kokoro: not installed")

    monkeypatch.setattr(backends, "pick_backend", unavailable)
    assert cli.main(["say", "Hi.", "--model", "Voice", *env.base]) == 1
    assert "no TTS backend available" in capsys.readouterr().err


def test_help_carries_the_pitch_tip(capsys):
    with pytest.raises(SystemExit):
        cli.main(["say", "--help"])
    out = capsys.readouterr().out
    assert "female TTS voice for a female model" in out and "--list-voices" in out
