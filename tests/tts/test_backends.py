"""The ``TtsBackend`` contract and backend selection. Nothing here touches the network."""

from pathlib import Path

import pytest

from vox.tts import backends
from vox.tts.backends import BACKENDS, EdgeBackend, KokoroBackend, pick_backend
from vox.tts.base import TtsBackend, TtsError


class FakeBackend(TtsBackend):
    name = "fake"
    default_voice = "one"
    reason: str | None = None

    def unavailable_reason(self):
        return self.reason

    def voices(self):
        return ["one", "two"]

    def synthesize(self, text, out_wav_path):
        return 24000


def test_interface_is_abstract():
    with pytest.raises(TypeError):
        TtsBackend()


def test_voice_defaults_and_availability_follows_the_reason(tmp_path):
    backend = FakeBackend(cache_dir=tmp_path)
    assert backend.voice == "one" and backend.cache_dir == tmp_path
    assert FakeBackend("two").voice == "two"
    assert backend.available()
    backend.reason = "not installed"
    assert not backend.available()


@pytest.mark.parametrize("cls", sorted(BACKENDS.values(), key=lambda c: c.name))
def test_real_backends_honour_the_contract(cls, tmp_path, monkeypatch):
    assert issubclass(cls, TtsBackend) and BACKENDS[cls.name] is cls
    backend = cls(cache_dir=tmp_path)
    assert backend.voice == cls.default_voice
    assert isinstance(backend.available(), bool)

    monkeypatch.setattr(backends, "_installed", lambda module: False)
    reason = backend.unavailable_reason()
    assert not backend.available()
    assert reason and "\n" not in reason and "requirements/tts.txt" in reason
    for call in (backend.voices, lambda: backend.synthesize("hi", tmp_path / "x.wav")):
        with pytest.raises(TtsError):
            call()


def test_kokoro_needs_a_cache_directory(monkeypatch):
    monkeypatch.setattr(backends, "_installed", lambda module: True)
    assert "cache" in KokoroBackend().unavailable_reason()
    assert KokoroBackend(cache_dir=Path("x")).available()


def test_kokoro_language_follows_the_voice_prefix():
    assert backends.KOKORO_LANGS["a"] == "en-us" and backends.KOKORO_LANGS["e"] == "es"


def test_pick_named_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "_installed", lambda module: True)
    backend = pick_backend("edge", "es-MX-DaliaNeural", tmp_path)
    assert isinstance(backend, EdgeBackend) and backend.voice == "es-MX-DaliaNeural"
    with pytest.raises(TtsError, match="unknown TTS backend"):
        pick_backend("festival")
    monkeypatch.setattr(backends, "_installed", lambda module: False)
    with pytest.raises(TtsError, match="edge: edge-tts is not installed"):
        pick_backend("edge")


def test_default_is_kokoro_with_edge_as_the_fallback(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(backends, "_installed", lambda module: True)
    assert isinstance(pick_backend(None, "am_adam", tmp_path), KokoroBackend)

    monkeypatch.setattr(backends, "_installed", lambda module: module == "edge_tts")
    backend = pick_backend(None, None, tmp_path)
    assert isinstance(backend, EdgeBackend) and backend.voice == EdgeBackend.default_voice
    assert "kokoro-onnx is not installed" in capsys.readouterr().err
    with pytest.raises(TtsError, match="--backend edge"):  # a kokoro voice means nothing to edge
        pick_backend(None, "am_adam", tmp_path)

    monkeypatch.setattr(backends, "_installed", lambda module: False)
    with pytest.raises(TtsError, match="no TTS backend available"):
        pick_backend(None, None, tmp_path)
