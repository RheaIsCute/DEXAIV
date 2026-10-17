"""External TTS backends: the worker protocol end to end with a fake engine (no model)."""

import sys
import textwrap
import wave
from pathlib import Path

import pytest

from vox.tts import external
from vox.tts.base import TtsError

FAKE = textwrap.dedent('''
    import sys
    sys.path.insert(0, {workers!r})
    from _protocol import serve
    import numpy as np

    class Fake:
        def load(self):
            print("noise a library prints on load")  # must not reach the reply stream
            return 24000
        def voices(self):
            return ["a", "b"]
        def synth(self, text, voice, speed, controls):
            if text == "boom":
                raise ValueError("cannot say that")
            n = int(24000 * 0.1 * len(text) * float(controls.get("stretch", 1.0)))
            return 0.1 * np.ones(n, np.float32), 24000

    serve(Fake())
''')


@pytest.fixture
def backend(tmp_path):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE.format(workers=str(external.WORKERS)))

    class FakeBackend(external.ExternalBackend):
        name = "fake"
        default_voice = "a"
        env_name = "unused"
        default_controls = {"stretch": 1.0}

    FakeBackend.script = str(script)  # absolute: WORKERS / script is the script itself
    FakeBackend.python = property(lambda self: Path(sys.executable))
    b = FakeBackend(None, tmp_path / "cache")
    yield b
    external.ExternalBackend.shutdown_all()


def test_a_worker_renders_lists_voices_and_survives_an_error(backend, tmp_path):
    assert backend.unavailable_reason() is None
    assert backend.voices() == ["a", "b"]
    out = tmp_path / "x.wav"
    assert backend.synthesize("hello", out, controls={"stretch": 2.0}) == 24000
    with wave.open(str(out)) as fh:
        assert fh.getframerate() == 24000 and fh.getnframes() == 24000
    assert backend.last_render["seconds"] == pytest.approx(1.0)
    with pytest.raises(TtsError, match="cannot say that"):
        backend.synthesize("boom", out)
    assert backend.synthesize("hi", out) == 24000  # still up, one process all along
    assert len(external.ExternalBackend._workers) == 1
    assert "noise a library prints" in (tmp_path / "cache" / "fake_worker.log").read_text()


def test_a_missing_environment_is_reported_not_raised(tmp_path):
    b = external.ChatterboxBackend(None, tmp_path)
    if b.python.is_file():
        pytest.skip("the chatterbox environment is installed here")
    assert "not installed" in b.unavailable_reason()
    with pytest.raises(TtsError, match="not installed"):
        b.synthesize("hi", tmp_path / "x.wav")


def test_orpheus_needs_its_server_and_weights(tmp_path):
    b = external.OrpheusBackend(None, tmp_path)
    reason = b.unavailable_reason()
    assert reason is not None and ("not installed" in reason or "missing" in reason)
