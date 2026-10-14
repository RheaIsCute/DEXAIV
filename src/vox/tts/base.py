"""The ``TtsBackend`` interface: text in, a WAV file out.

Control path only. A backend renders plain speech in one of its own voices; turning that
into the target voice is the converter's job (``vox.tts.speak``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TtsError(RuntimeError):
    """A backend cannot be used or failed to render (missing package, no network, ...)."""


class TtsBackend(ABC):
    name: str
    default_voice: str

    def __init__(self, voice: str | None = None, cache_dir: Path | None = None) -> None:
        self.voice = voice or self.default_voice
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def available(self) -> bool:
        return self.unavailable_reason() is None

    @abstractmethod
    def unavailable_reason(self) -> str | None:
        """One line saying why the backend cannot run here, or None when it can.

        Must stay cheap and offline: it only checks what is installed.
        """

    @abstractmethod
    def voices(self) -> list[str]:
        """Names accepted as ``voice``."""

    @abstractmethod
    def synthesize(self, text: str, out_wav_path: Path, speed: float = 1.0) -> int:
        """Render ``text`` in ``self.voice`` to a mono WAV. Returns its sample rate.

        ``speed`` is the render rate relative to the voice's normal pace (1.0); the
        expressive layer passes values in roughly 0.78 .. 1.05. A backend must render
        ``speed=1.0`` exactly as it did before the parameter existed.
        """
