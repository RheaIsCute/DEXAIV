"""Text -> TTS wav -> ``FullFileConverter`` -> audio in the model's voice.

Long text is split on sentence boundaries into pieces of at most ~30 s, each piece is
rendered and converted on its own, and the results are joined with a 10 ms crossfade. One
transposition is used for the whole utterance so the voice does not jump between pieces.
"""

from __future__ import annotations

import dataclasses
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vox.audio.resample import load_audio, save_audio
from vox.config.schemas import ModelConfig, PrecisionParams, RuntimeParams
from vox.config.store import Profile
from vox.offline.convert import SR, FullFileConverter
from vox.offline.simulate import load_engine
from vox.tts.base import TtsBackend, TtsError

DEFAULT_TARGET_F0_HZ = 200.0
TARGET_F0_KEY = "target_f0_median_hz"  # optional, in model_overrides.json or config.json
MAX_PIECE_S = 30.0
CHARS_PER_SECOND = 14.0  # conservative speaking rate used to size pieces before rendering
CROSSFADE_MS = 10.0
PIECE_GAP_S = 0.2  # backends trim silence, so restore a sentence pause between pieces
MAX_PITCH = 24

_SENTENCE_END = re.compile(  # after the terminator, or after one closing quote / bracket
    r"(?<=[.!?…。！？][\"'”’)\]])\s+|(?<=[.!?…。！？])\s+|\n+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _pack(parts: list[str], limit: int) -> list[str]:
    """Greedily join consecutive parts with spaces while they fit in ``limit`` characters."""
    packed: list[str] = []
    for part in parts:
        if packed and len(packed[-1]) + 1 + len(part) <= limit:
            packed[-1] += " " + part
        else:
            packed.append(part)
    return packed


def split_text(text: str, max_seconds: float = MAX_PIECE_S,
               chars_per_second: float = CHARS_PER_SECOND) -> list[str]:
    """Pieces of whole sentences, each estimated to speak in at most ``max_seconds``.

    A single sentence longer than that is cut at clause punctuation, then between words.
    """
    limit = max(1, int(max_seconds * chars_per_second))
    parts: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence) <= limit:
            parts.append(sentence)
            continue
        for clause in _pack(_CLAUSE_END.split(sentence), limit):
            parts.extend(_pack(clause.split(), limit) if len(clause) > limit else [clause])
    return _pack(parts, limit)


def crossfade_concat(pieces: list[np.ndarray], sr: int, ms: float = CROSSFADE_MS) -> np.ndarray:
    """Join pieces, overlapping each seam by ``ms`` with a linear crossfade."""
    out = pieces[0].astype(np.float32)
    for piece in pieces[1:]:
        n = min(int(sr * ms / 1000.0), len(out), len(piece))
        if n == 0:
            out = np.concatenate([out, piece])
            continue
        fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
        seam = out[-n:] * (1.0 - fade) + piece[:n] * fade
        out = np.concatenate([out[:-n], seam, piece[n:]])
    return out.astype(np.float32)


@torch.inference_mode()
def median_voiced_f0(extractor, pieces_16k: list[np.ndarray], device: torch.device) -> float:
    """Median F0 in Hz over the voiced frames of all pieces; 0.0 when nothing is voiced."""
    voiced = []
    for audio in pieces_16k:
        f0 = extractor.extract(torch.from_numpy(audio).to(device)).flatten()
        voiced.append(f0[f0 > 0].float().cpu())
    f0 = torch.cat(voiced)
    return float(f0.median()) if f0.numel() else 0.0


def auto_pitch(tts_median_f0: float, target_f0: float) -> int:
    """Semitones that move the TTS voice's median F0 onto the target's (docs/research/08)."""
    if tts_median_f0 <= 0 or target_f0 <= 0:
        return 0
    pitch = round(12.0 * math.log2(target_f0 / tts_median_f0))
    return max(-MAX_PITCH, min(MAX_PITCH, pitch))


def target_f0_hint(profile: Profile | None, config: ModelConfig) -> float:
    """The model's median F0 from the profile or the model config, else 200 Hz."""
    layers = [config.defaults]
    if profile is not None:
        layers.insert(0, profile.model_overrides.models.get(config.name, {}))
    for layer in layers:
        value = layer.get(TARGET_F0_KEY)
        if value and float(value) > 0:
            return float(value)
    return DEFAULT_TARGET_F0_HZ


@dataclass
class SayResult:
    audio: np.ndarray  # mono float32 at ``sr`` (the model's sample rate)
    sr: int
    pitch: float
    auto_pitch: bool
    tts_median_f0: float
    target_f0: float
    pieces: int
    tts_seconds: float
    path: Path | None = None

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.sr


def render_tts(backend: TtsBackend, pieces: list[str]) -> list[np.ndarray]:
    """Render each text piece and load it as the mono 16 kHz the converter expects."""
    rendered = []
    with tempfile.TemporaryDirectory(prefix="vox_tts_") as tmp:
        for n, piece in enumerate(pieces):
            wav = Path(tmp) / f"piece_{n:03d}.wav"
            backend.synthesize(piece, wav)
            audio, _ = load_audio(wav, SR)
            if not len(audio):
                raise TtsError(f"{backend.name} produced no audio for {piece[:40]!r}")
            if n < len(pieces) - 1:
                audio = np.concatenate([audio, np.zeros(int(SR * PIECE_GAP_S), np.float32)])
            rendered.append(audio)
    return rendered


def speak(text: str, backend: TtsBackend, models_dir: Path, model_name: str,
          params: RuntimeParams | None = None, pitch: float | None = None,
          target_f0: float = DEFAULT_TARGET_F0_HZ, precision: PrecisionParams | None = None,
          device: str | None = None, seed: int | None = 0, out: Path | None = None,
          max_piece_s: float = MAX_PIECE_S) -> SayResult:
    """Speak ``text`` in the voice of ``model_name``.

    ``pitch=None`` measures the rendered TTS audio and transposes its median F0 onto
    ``target_f0``; ``params.f0_up_key`` is ignored either way (it is calibrated for the
    user's microphone voice, not for a TTS voice).
    """
    pieces = split_text(text, max_piece_s)
    if not pieces:
        raise TtsError("nothing to say: the text is empty")
    rendered = render_tts(backend, pieces)

    engine = load_engine(models_dir, model_name, None, params, precision, device, warmup_hops=0)
    converter = FullFileConverter.from_engine(engine)
    measured = 0.0
    auto = pitch is None
    if auto:
        if converter.extractor is None:  # a model without F0 conditioning cannot transpose
            pitch = 0
        else:
            measured = median_voiced_f0(converter.extractor, rendered, engine.device)
            pitch = auto_pitch(measured, target_f0)
    params = dataclasses.replace(engine.params, f0_up_key=float(pitch))

    converted = []
    for audio in rendered:
        if seed is not None:
            torch.manual_seed(seed)
        converted.append(converter.convert(audio, params))
    sr = engine.loaded.model.tgt_sr
    result = SayResult(crossfade_concat(converted, sr), sr, float(pitch), auto, measured,
                       target_f0, len(pieces), sum(len(a) for a in rendered) / SR)
    if out is not None:
        save_audio(out, result.audio, sr)
        result.path = Path(out)
    return result


def play(audio: np.ndarray, sr: int, device: str | None = None) -> None:
    """Blocking playback on the default output device, or one given by index or name."""
    import sounddevice as sd

    target = int(device) if device is not None and device.isdigit() else device
    sd.play(audio, sr, device=target)
    sd.wait()
