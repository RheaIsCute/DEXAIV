"""TTS backends: Kokoro-82M (local, ONNX) and edge-tts (Microsoft Edge voices, online).

Both packages are optional (requirements/tts.txt) and imported lazily, so the rest of the
project never depends on them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import urllib.request
from pathlib import Path

from vox.tts.base import TtsBackend, TtsError

_KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
# file -> sha256, checked once after download
KOKORO_FILES = {
    "kokoro-v1.0.onnx": "beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}
# Kokoro voice names start with a language letter; espeak-ng needs the matching code.
KOKORO_LANGS = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi", "i": "it",
                "j": "ja", "p": "pt-br", "z": "cmn"}


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _download(url: str, dst: Path, sha256: str) -> None:
    """Fetch ``url`` to ``dst`` through a ``.part`` file, verifying the digest."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    print(f"downloading {url} -> {dst}", file=sys.stderr)
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=30) as response, open(tmp, "wb") as fh:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
                digest.update(chunk)
    except OSError as err:
        tmp.unlink(missing_ok=True)
        raise TtsError(f"could not download {url}: {err}") from err
    if sha256 and digest.hexdigest() != sha256:
        tmp.unlink()
        raise TtsError(f"{url}: sha256 {digest.hexdigest()} does not match the pinned {sha256}")
    os.replace(tmp, dst)


class KokoroBackend(TtsBackend):
    """Kokoro-82M through ``kokoro-onnx`` (onnxruntime on the CPU, 24 kHz)."""

    name = "kokoro"
    default_voice = "af_heart"

    def __init__(self, voice: str | None = None, cache_dir: Path | None = None) -> None:
        super().__init__(voice, cache_dir)
        self._kokoro = None

    def unavailable_reason(self) -> str | None:
        if not _installed("kokoro_onnx"):
            return "kokoro-onnx is not installed (pip install -r requirements/tts.txt)"
        if self.cache_dir is None:
            return "no cache directory for the Kokoro weights"
        return None

    def _file(self, filename: str) -> Path:
        """Path of one weight file under ``<cache>/kokoro/``, downloaded on first use."""
        path = self.cache_dir / "kokoro" / filename
        if not path.is_file():
            _download(f"{_KOKORO_RELEASE}/{filename}", path, KOKORO_FILES[filename])
        return path

    def _require(self) -> None:
        if reason := self.unavailable_reason():
            raise TtsError(reason)

    def voices(self) -> list[str]:
        import numpy as np

        self._require()
        with np.load(self._file("voices-v1.0.bin")) as styles:  # only the small file
            return sorted(styles.files)

    def _model(self):
        if self._kokoro is None:
            from kokoro_onnx import Kokoro

            self._kokoro = Kokoro(str(self._file("kokoro-v1.0.onnx")),
                                  str(self._file("voices-v1.0.bin")))
        return self._kokoro

    def load(self) -> None:
        """Build the ONNX session and run one short sentence (espeak and the graph both
        initialise lazily). Skipped while the weights are not downloaded yet."""
        directory = (self.cache_dir or Path()) / "kokoro"
        if self.unavailable_reason() or not all((directory / f).is_file() for f in KOKORO_FILES):
            return
        kokoro = self._model()
        if self.voice in kokoro.get_voices():
            kokoro.create("Hi.", voice=self.voice, speed=1.0,
                          lang=KOKORO_LANGS.get(self.voice[0], "en-us"))

    def synthesize(self, text: str, out_wav_path: Path, speed: float = 1.0) -> int:
        from vox.audio.resample import save_audio

        self._require()
        kokoro = self._model()
        if self.voice not in kokoro.get_voices():
            raise TtsError(f"kokoro has no voice {self.voice!r} (see 'vox say --list-voices')")
        lang = KOKORO_LANGS.get(self.voice[0], "en-us")
        samples, sr = kokoro.create(text, voice=self.voice, speed=float(speed), lang=lang)
        save_audio(out_wav_path, samples, sr)
        return sr


def edge_rate(speed: float) -> str:
    """edge-tts prosody rate for a relative speed: 0.92 -> "-8%", 1.05 -> "+5%"."""
    return f"{round((speed - 1.0) * 100.0):+d}%"


class EdgeBackend(TtsBackend):
    """Microsoft Edge neural voices through ``edge-tts``. Needs internet, no weights."""

    name = "edge"
    default_voice = "en-US-AriaNeural"

    def unavailable_reason(self) -> str | None:
        if not _installed("edge_tts"):
            return "edge-tts is not installed (pip install -r requirements/tts.txt)"
        return None

    def _edge(self):
        if reason := self.unavailable_reason():
            raise TtsError(reason)
        import edge_tts

        return edge_tts

    def _run(self, coroutine):
        import asyncio

        try:
            return asyncio.run(coroutine)
        except Exception as err:  # aiohttp / edge-tts errors: offline, bad voice, no audio
            raise TtsError(f"edge-tts failed ({type(err).__name__}: {err}); "
                           f"it needs an internet connection") from err

    def voices(self) -> list[str]:
        edge_tts = self._edge()
        return sorted(v["ShortName"] for v in self._run(edge_tts.list_voices()))

    def synthesize(self, text: str, out_wav_path: Path, speed: float = 1.0) -> int:
        from vox.audio.resample import load_audio, save_audio

        edge_tts = self._edge()
        out_wav_path = Path(out_wav_path)
        mp3 = out_wav_path.with_suffix(".mp3")  # the service streams MP3
        # The rate is only sent when it differs from normal, so speed 1.0 is the old request.
        prosody = {"rate": edge_rate(speed)} if speed != 1.0 else {}
        try:
            self._run(edge_tts.Communicate(text, self.voice, **prosody).save(str(mp3)))
            audio, sr = load_audio(mp3)
        finally:
            mp3.unlink(missing_ok=True)
        save_audio(out_wav_path, audio, sr)
        return sr


BACKENDS: dict[str, type[TtsBackend]] = {"kokoro": KokoroBackend, "edge": EdgeBackend}


def pick_backend(name: str | None = None, voice: str | None = None,
                 cache_dir: Path | None = None) -> TtsBackend:
    """The named backend, or the first available one (kokoro, then edge) when ``name`` is None.

    A voice only makes sense for the backend it was written for, so falling back while a
    voice was requested is an error rather than a silent switch to another speaker.
    """
    if name is not None:
        if name not in BACKENDS:
            raise TtsError(f"unknown TTS backend {name!r}; choose from {', '.join(BACKENDS)}")
        backend = BACKENDS[name](voice, cache_dir)
        if reason := backend.unavailable_reason():
            raise TtsError(f"{name}: {reason}")
        return backend
    reasons = []
    for candidate, cls in BACKENDS.items():
        backend = cls(None if reasons else voice, cache_dir)
        reason = backend.unavailable_reason()
        if reason is None:
            if reasons and voice:
                raise TtsError(f"{'; '.join(reasons)}; pass --backend {candidate} with one of "
                               f"its own voices instead of {voice!r}")
            for skipped in reasons:
                print(f"note: {skipped}; using {candidate}", file=sys.stderr)
            return backend
        reasons.append(f"{candidate}: {reason}")
    raise TtsError("no TTS backend available: " + "; ".join(reasons))
