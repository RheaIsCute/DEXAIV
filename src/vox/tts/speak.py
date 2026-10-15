"""Text -> TTS wav -> ``FullFileConverter`` -> audio in the model's voice.

Long text is split on sentence boundaries into pieces of at most ~30 s, each piece is
rendered and converted on its own, and the results are joined with a 10 ms crossfade. One
transposition is used for the whole utterance so the voice does not jump between pieces.

Informal text gets its render hints from ``vox.tts.expressive``: per-piece TTS speed, a
per-boundary pause (dots and dashes) and a per-piece gain applied to the *converted* audio
so it survives the RVC stage. A text without any mark takes exactly the old path.
"""

from __future__ import annotations

import dataclasses
import math
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from vox.audio.resample import load_audio, save_audio
from vox.config.schemas import ModelConfig, PrecisionParams, RuntimeParams
from vox.config.store import Profile
from vox.offline.convert import SR, FullFileConverter
from vox.offline.simulate import load_engine
from vox.tts.base import TtsBackend, TtsError
from vox.tts.expressive import SENTENCE_GAP_S, ExpressivePiece, analyze, split_sentences

DEFAULT_TARGET_F0_HZ = 200.0
TARGET_F0_KEY = "target_f0_median_hz"  # optional, in model_overrides.json or config.json
MAX_PIECE_S = 30.0
CHARS_PER_SECOND = 14.0  # conservative speaking rate used to size pieces before rendering
CROSSFADE_MS = 10.0
PIECE_GAP_S = SENTENCE_GAP_S  # backends trim silence, so restore a sentence pause between pieces
PEAK = 0.99  # ceiling after per-piece gain (the converter itself normalises to this)
MAX_PITCH = 24

_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")

__all__ = ["ExpressivePiece", "TypedPolish", "analyze", "split_pieces", "split_sentences",
           "split_text"]


@dataclass(frozen=True)
class TypedPolish:
    """Optional prosody treatment of typed speech (docs/DEVIATIONS.md §7, Step 3). The
    defaults are the plain path, byte for byte; nothing here touches the live mic.

    ``envelope``: ``volume_envelope`` for the typed clip only (1.0 = the model's own level;
    lower keeps more of the TTS loudness contour). ``rms_frame_ms``: the loudness frames of
    that mix (None = rvc-webui's 1 s; 40 carries syllable stress). ``contour_scale``: voiced
    F0 widened around the utterance median in semitones (1.0 = as rendered).
    """

    envelope: float = 1.0
    rms_frame_ms: float | None = None
    contour_scale: float = 1.0

    @property
    def plain(self) -> bool:
        return self == TypedPolish()


def _pack(parts: list[str], limit: int) -> list[str]:
    """Greedily join consecutive parts with spaces while they fit in ``limit`` characters."""
    packed: list[str] = []
    for part in parts:
        if packed and len(packed[-1]) + 1 + len(part) <= limit:
            packed[-1] += " " + part
        else:
            packed.append(part)
    return packed


def _pack_pieces(parts: list[ExpressivePiece], limit: int) -> list[ExpressivePiece]:
    """``_pack`` for pieces: only neighbours with the same hints join, never across an
    explicit pause; the joined piece keeps the pause of its last member."""
    packed: list[ExpressivePiece] = []
    for part in parts:
        last = packed[-1] if packed else None
        if (last is not None and last.same_hints(part) and not last.explicit
                and len(last.text) + 1 + len(part.text) <= limit):
            packed[-1] = replace(part, text=last.text + " " + part.text)
        else:
            packed.append(part)
    return packed


def split_pieces(text: str, max_seconds: float = MAX_PIECE_S,
                 chars_per_second: float = CHARS_PER_SECOND) -> list[ExpressivePiece]:
    """Pieces of whole sentences with render hints, each estimated to speak in at most
    ``max_seconds``. A single piece longer than that is cut at clause punctuation, then
    between words; the cuts inherit the hints and the explicit pause stays on the last cut.
    """
    limit = max(1, int(max_seconds * chars_per_second))
    parts: list[ExpressivePiece] = []
    for piece in analyze(text):
        if len(piece.text) <= limit:
            parts.append(piece)
            continue
        cuts: list[str] = []
        for clause in _pack(_CLAUSE_END.split(piece.text), limit):
            cuts.extend(_pack(clause.split(), limit) if len(clause) > limit else [clause])
        for n, cut in enumerate(cuts):
            last = n == len(cuts) - 1
            parts.append(replace(piece, text=cut,
                                 pause_after_s=piece.pause_after_s if last else PIECE_GAP_S,
                                 explicit=piece.explicit and last))
    return _pack_pieces(parts, limit)


def split_text(text: str, max_seconds: float = MAX_PIECE_S,
               chars_per_second: float = CHARS_PER_SECOND) -> list[str]:
    """The texts of ``split_pieces``."""
    return [p.text for p in split_pieces(text, max_seconds, chars_per_second)]


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
    hints: list[ExpressivePiece] = dataclasses.field(default_factory=list)

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.sr

    @property
    def expressive(self) -> bool:
        """Did any piece carry a render hint (speed, gain or an explicit pause)?"""
        return not all(p.plain for p in self.hints)


def render_tts(backend: TtsBackend, pieces: list[ExpressivePiece]) -> list[np.ndarray]:
    """Render each piece at its speed and load it as the mono 16 kHz the converter expects.

    Each piece but the last gets its boundary pause appended as silence (both backends trim
    their own trailing silence).
    """
    rendered = []
    with tempfile.TemporaryDirectory(prefix="vox_tts_") as tmp:
        for n, piece in enumerate(pieces):
            wav = Path(tmp) / f"piece_{n:03d}.wav"
            backend.synthesize(piece.text, wav, speed=piece.speed)
            audio, _ = load_audio(wav, SR)
            if not len(audio):
                raise TtsError(f"{backend.name} produced no audio for {piece.text[:40]!r}")
            if n < len(pieces) - 1:
                gap = np.zeros(round(SR * piece.pause_after_s), np.float32)
                audio = np.concatenate([audio, gap])
            rendered.append(audio)
    return rendered


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Per-piece gain on converted audio; 0 dB returns the input object untouched."""
    if gain_db == 0.0:
        return audio
    return (audio * 10.0 ** (gain_db / 20.0)).astype(np.float32)


def limit_peak(audio: np.ndarray, peak: float = PEAK) -> np.ndarray:
    """Scale the whole utterance down if a gained piece pushed it past ``peak``."""
    top = float(np.abs(audio).max()) if len(audio) else 0.0
    return (audio * (peak / top)).astype(np.float32) if top > peak else audio


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
    pieces = split_pieces(text, max_piece_s)
    if not pieces:
        raise TtsError("nothing to say: the text is empty")
    rendered = render_tts(backend, pieces)

    engine = load_engine(models_dir, model_name, None, params, precision, device, warmup_hops=0)
    result = convert_rendered(FullFileConverter.from_engine(engine), engine.device,
                              engine.loaded.model.tgt_sr, rendered, engine.params, pitch,
                              target_f0, seed, pieces)
    if out is not None:
        save_audio(out, result.audio, result.sr)
        result.path = Path(out)
    return result


def convert_rendered(converter: FullFileConverter, device: torch.device, sr: int,
                     rendered: list[np.ndarray], params: RuntimeParams,
                     pitch: float | None = None, target_f0: float = DEFAULT_TARGET_F0_HZ,
                     seed: int | None = 0,
                     pieces: list[ExpressivePiece] | None = None,
                     polish: TypedPolish | None = None) -> SayResult:
    """Convert rendered 16 kHz TTS pieces with an already loaded converter.

    The half of ``speak`` that needs the models; the control API calls it with the live
    engine's converter so a typed clip uses the voice that is already loaded. ``pieces``
    are the render hints the audio was made from; their per-piece gain is applied here.
    ``polish`` (default: none) applies the typed-speech prosody options.
    """
    polish = polish or TypedPolish()
    measured = 0.0
    auto = pitch is None
    if auto:
        if converter.extractor is None:  # a model without F0 conditioning cannot transpose
            pitch = 0
        else:
            measured = median_voiced_f0(converter.extractor, rendered, device)
            pitch = auto_pitch(measured, target_f0)
    params = dataclasses.replace(params, f0_up_key=float(pitch))
    if not polish.plain:
        params = dataclasses.replace(params, volume_envelope=polish.envelope)

    gains = [p.gain_db for p in pieces] if pieces else [0.0] * len(rendered)
    converted = []
    for gain_db, audio in zip(gains, rendered):
        if seed is not None:
            torch.manual_seed(seed)
        # Gain goes on the converted audio: applied before RVC it would be normalised away.
        out = (converter.convert(audio, params) if polish.plain else
               converter.convert(audio, params, polish.rms_frame_ms, polish.contour_scale))
        converted.append(apply_gain(out, gain_db))
    joined = crossfade_concat(converted, sr)
    if any(gains):
        joined = limit_peak(joined)
    return SayResult(joined, sr, float(pitch), auto, measured, target_f0, len(rendered),
                     sum(len(a) for a in rendered) / SR, hints=list(pieces or []))


def play(audio: np.ndarray, sr: int, device: str | None = None) -> None:
    """Blocking playback on the default output device, or one given by index or name."""
    import sounddevice as sd

    target = int(device) if device is not None and device.isdigit() else device
    sd.play(audio, sr, device=target)
    sd.wait()
