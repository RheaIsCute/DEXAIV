"""Quality metrics for ``vox bench`` (docs/research/15, Phase 0.1): a decision tool for the
things the existing metrics cannot see.

    WER           faster-whisper (small.en, CPU, int8) on a fixed script against its
                  transcript; reported with the un-converted microphone's WER beside it
    speaker cos   ECAPA-TDNN (speechbrain) or WavLM-SV cosine of the output against
                  held-out recordings of the target (takes/target/<Model>/*.wav), or,
                  when none exist, the model's own ``vox say`` renders, labelled "proxy";
                  always with the seed floor (two renders of the same thing) beside it
    UTMOS         SpeechMOS utmos22_strong, a naturalness proxy; never rank on it alone
    F0 stats      median, IQR in semitones, octave flips per voiced minute, voiced ratio
                  (the engine's own RMVPE track, 100 Hz)

The existing R3 log-mel, click ratio and HF/LF stay where they are (``offline/metrics.py``,
``bench.py``). Not here, on purpose: MCD, F0 RMSE, PESQ, NISQA-PSTN (SVCC 2023/2025:
they do not track MOS for voice conversion). Objective-subjective agreement tops out
near 0.8 Spearman, so every table is paired with blind A/B wavs (``write_pairs``): the
user's ear decides, these numbers say where to listen.

Pure functions first (unit-tested on the CPU); the model-backed scorers below them load
lazily and cache their weights under ``models/`` (gitignored). Every scorer runs on the
CPU so a GPU engine job is never disturbed.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TARGETS_DIR = ROOT / "takes" / "target"  # held-out recordings of each target voice
METRIC_SR = 16000
WHISPER_MODEL = "small.en"
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
WAVLM_SOURCE = "microsoft/wavlm-base-plus-sv"
UTMOS_REPO = "tarepan/SpeechMOS:v1.2.0"
UTMOS_CHUNK_S = 15.0  # UTMOS on the CPU: score in chunks, average by duration
FLIP_SEMITONES = 8.0  # a 10 ms step this large between voiced frames is an octave error
MIN_SEGMENT_S = 1.0  # speaker embeddings need at least this much speech
MATRIX_PAIRS_SEED = 2026  # X/Y assignment of every blind pair this session writes

# ----------------------------------------------------------------------------- words / WER

_WORD = re.compile(r"[a-z0-9']+")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})


def words(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens; punctuation and case never count as errors."""
    return _WORD.findall(text.translate(_APOSTROPHES).lower())


def word_errors(reference: Sequence[str], hypothesis: Sequence[str]) -> dict[str, float]:
    """Levenshtein alignment at the word level: substitutions, insertions, deletions and
    WER = (S + D + I) / N over ``N`` reference words (``inf`` for an empty reference)."""
    n, m = len(reference), len(hypothesis)
    # cost[i][j] = (edits, S, D, I) for reference[:i] vs hypothesis[:j]
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cur.append(prev[j - 1])
                continue
            sub = prev[j - 1]
            dele = prev[j]
            ins = cur[j - 1]
            best = min((sub[0] + 1, 0, sub), (dele[0] + 1, 1, dele), (ins[0] + 1, 2, ins))
            _, kind, src = best
            cur.append((src[0] + 1, src[1] + (kind == 0), src[2] + (kind == 1),
                        src[3] + (kind == 2)))
        prev = cur
    edits, s, d, i = prev[m]
    return {"wer": edits / n if n else float("inf"), "substitutions": s, "deletions": d,
            "insertions": i, "reference_words": n, "hypothesis_words": m}


def wer(reference_text: str, hypothesis_text: str) -> float:
    return word_errors(words(reference_text), words(hypothesis_text))["wer"]


# ------------------------------------------------------------------------------ F0 stats


def f0_stats(f0_100hz: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """Statistics of a 100 Hz pitch track (0 = unvoiced) over ``mask`` (10 ms frames;
    default: every frame).

    ``median_hz`` / ``p10_hz`` / ``p90_hz`` of the voiced frames; ``iqr_st`` the
    interquartile range in semitones (the intonation spread); ``flips_per_min`` the
    steps of at least ``FLIP_SEMITONES`` between consecutive voiced frames, per minute of
    voiced audio (an octave error lands at 12 st, a glide never moves 8 st in 10 ms);
    ``voiced_ratio`` the share of masked frames with a pitch.
    """
    f0 = np.asarray(f0_100hz, dtype=np.float64)
    if mask is None:
        mask = np.ones(len(f0), dtype=bool)
    n = min(len(f0), len(mask))
    f0, mask = f0[:n], np.asarray(mask[:n], dtype=bool)
    voiced = mask & (f0 > 0)
    out = {"median_hz": 0.0, "p10_hz": 0.0, "p90_hz": 0.0, "iqr_st": 0.0,
           "flips_per_min": 0.0, "voiced_ratio": float(voiced.sum() / mask.sum())
           if mask.any() else float("nan"), "voiced_s": float(voiced.sum()) / 100.0}
    values = f0[voiced]
    if values.size == 0:
        return out
    p10, p25, p50, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90])
    out.update(median_hz=float(p50), p10_hz=float(p10), p90_hz=float(p90),
               iqr_st=float(12.0 * np.log2(p75 / p25)) if p25 > 0 else 0.0)
    both = voiced[1:] & voiced[:-1]
    if both.any():
        steps = np.abs(12.0 * np.log2(f0[1:][both] / f0[:-1][both]))
        flips = int((steps >= FLIP_SEMITONES).sum())
        out["flips_per_min"] = 60.0 * flips / max(out["voiced_s"], 1e-9)
    return out


# ------------------------------------------------------------------------- speaker cosine


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def speaker_summary(embedding: np.ndarray, targets: Sequence[np.ndarray]) -> dict[str, float]:
    """Cosine of one embedding against a set of target embeddings: against their centroid
    (``cos``), the mean and min over the clips, and the targets' own mean pairwise cosine
    (``target_self``: how alike two real recordings of the target are, the ceiling)."""
    if not len(targets):
        return {"cos": float("nan"), "cos_mean": float("nan"), "cos_min": float("nan"),
                "target_self": float("nan")}
    units = [unit(t) for t in targets]
    per_clip = [cosine(embedding, t) for t in units]
    centroid = unit(np.mean(units, axis=0))
    pairs = [cosine(units[i], units[j]) for i in range(len(units))
             for j in range(i + 1, len(units))]
    return {"cos": cosine(embedding, centroid), "cos_mean": float(np.mean(per_clip)),
            "cos_min": float(np.min(per_clip)),
            "target_self": float(np.mean(pairs)) if pairs else float("nan")}


# ---------------------------------------------------------------------------- blind pairs


@dataclass(frozen=True)
class Pair:
    """One blind comparison: ``a`` and ``b`` are variant names, ``what`` says what to
    listen for. The listener sees X and Y; the key says which is which."""

    a: str
    b: str
    what: str = ""


def write_pairs(pairs: Sequence[Pair], files: dict[str, Path], out_dir: Path,
                seed: int = 0, title: str = "") -> dict:
    """Copy each pair as ``pairNN_X.wav`` / ``pairNN_Y.wav`` with X/Y assigned at random,
    write ``pairs.md`` (what to listen for, a blank per pair) for the listener and
    ``answer_key.json`` (do not open until the sheet is filled). Returns the key."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    key: dict = {"title": title, "seed": seed, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "pairs": {}}
    sheet = [f"# Blind pairs{': ' + title if title else ''}", "",
             ("Listen to X and Y of each pair (same input, one setting changed). Write which "
              "you prefer, where in the clip and what you heard. Open answer_key.json only "
              "after. One line per pair, e.g. `pair01: Y, warble on the second sentence`."), ""]
    for n, pair in enumerate(pairs, 1):
        name = f"pair{n:02d}"
        flip = bool(rng.random() < 0.5)
        x, y = (pair.b, pair.a) if flip else (pair.a, pair.b)
        for label, variant in (("X", x), ("Y", y)):
            src = Path(files[variant])
            shutil.copyfile(src, out_dir / f"{name}_{label}{src.suffix}")
        key["pairs"][name] = {"X": x, "Y": y, "what": pair.what,
                              "sources": {"X": str(files[x]), "Y": str(files[y])}}
        sheet.append(f"- **{name}** ({pair.what or 'which sounds better?'}): "
                     f"`{name}_X.wav` vs `{name}_Y.wav` -> ")
    (out_dir / "pairs.md").write_text("\n".join(sheet) + "\n", encoding="utf-8")
    (out_dir / "answer_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    return key


# ----------------------------------------------------------------------- audio utilities


def to_metric_rate(audio: np.ndarray, sr: int) -> np.ndarray:
    from vox.audio.resample import resample_array

    return np.ascontiguousarray(resample_array(np.asarray(audio, dtype=np.float32), sr,
                                               METRIC_SR))


def speech_segments(audio_16k: np.ndarray, min_s: float = MIN_SEGMENT_S,
                    floor_db: float = 35.0) -> list[np.ndarray]:
    """Spans of signal (frames within ``floor_db`` of the loud ones, gaps < 300 ms bridged)
    at least ``min_s`` long, for speaker embeddings: silence and gate zeros would pull the
    embedding toward nothing in particular. Falls back to the whole clip."""
    from vox.offline.metrics import phrases

    spans = [(s, e) for s, e in phrases(audio_16k, METRIC_SR, below_p95_db=floor_db,
                                        merge_ms=300.0, min_ms=min_s * 1000.0)]
    if not spans:
        return [audio_16k]
    return [audio_16k[s:e] for s, e in spans]


# --------------------------------------------------------------------- model-backed scorers


class Transcriber:
    """faster-whisper on the CPU (int8). ``text(audio, sr)`` -> the transcript."""

    def __init__(self, model_size: str = WHISPER_MODEL, cache_dir: Path | None = None) -> None:
        self.model_size = model_size
        self.cache_dir = Path(cache_dir) if cache_dir else ROOT / "models" / "_tools" / "whisper"
        self._model = None

    def load(self) -> None:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8",
                                       download_root=str(self.cache_dir))

    def text(self, audio: np.ndarray, sr: int) -> str:
        self.load()
        segments, _ = self._model.transcribe(to_metric_rate(audio, sr), language="en",
                                             beam_size=5, vad_filter=False,
                                             condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()


class SpeakerEmbedder:
    """``ecapa`` (speechbrain ECAPA-TDNN, 192-d) or ``wavlm`` (WavLM-base-plus-sv, 512-d),
    CPU. ``embed(audio, sr)`` averages the unit embeddings of the clip's speech segments."""

    def __init__(self, backend: str = "ecapa", cache_dir: Path | None = None) -> None:
        if backend not in ("ecapa", "wavlm"):
            raise ValueError("speaker backend must be 'ecapa' or 'wavlm'")
        self.backend = backend
        self.cache_dir = Path(cache_dir) if cache_dir else ROOT / "models" / "_embedders"
        self._model = None
        self._extractor = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch

        if self.backend == "ecapa":
            from speechbrain.inference.speaker import EncoderClassifier

            self._model = EncoderClassifier.from_hparams(
                source=ECAPA_SOURCE, savedir=str(self.cache_dir / "ecapa"),
                run_opts={"device": "cpu"})
        else:
            from transformers import AutoFeatureExtractor, WavLMForXVector

            cache = str(self.cache_dir / "hf")
            self._extractor = AutoFeatureExtractor.from_pretrained(WAVLM_SOURCE, cache_dir=cache)
            self._model = WavLMForXVector.from_pretrained(WAVLM_SOURCE, cache_dir=cache).eval()
        torch.set_num_threads(max(1, torch.get_num_threads()))

    def _one(self, segment_16k: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(segment_16k, dtype=np.float32))
            if self.backend == "ecapa":
                emb = self._model.encode_batch(x[None]).squeeze()
            else:
                inputs = self._extractor(segment_16k, sampling_rate=METRIC_SR, return_tensors="pt")
                emb = self._model(**inputs).embeddings.squeeze()
        return unit(emb.cpu().numpy())

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        self.load()
        x = to_metric_rate(audio, sr)
        segments = speech_segments(x)
        weights = np.array([len(s) for s in segments], dtype=np.float64)
        embs = np.stack([self._one(s) for s in segments])
        return unit((embs * weights[:, None]).sum(axis=0))


class Naturalness:
    """UTMOS22 strong (SpeechMOS) on the CPU: a 1-5 naturalness proxy. ``score`` averages
    over ``UTMOS_CHUNK_S`` chunks, weighted by duration."""

    def __init__(self, hub_dir: Path | None = None) -> None:
        self.hub_dir = Path(hub_dir) if hub_dir else ROOT / "models" / "_predictors" / "hub"
        self._model = None

    def load(self) -> None:
        if self._model is None:
            import torch

            torch.hub.set_dir(str(self.hub_dir))
            self._model = torch.hub.load(UTMOS_REPO, "utmos22_strong", trust_repo=True,
                                         verbose=False).eval()

    def score(self, audio: np.ndarray, sr: int) -> float:
        import torch

        self.load()
        x = to_metric_rate(audio, sr)
        chunk = int(UTMOS_CHUNK_S * METRIC_SR)
        pieces = [x[i:i + chunk] for i in range(0, len(x), chunk)]
        if len(pieces) > 1 and len(pieces[-1]) < METRIC_SR:  # a sub-second tail: fold it in
            pieces[-2] = np.concatenate([pieces[-2], pieces[-1]])
            pieces.pop()
        scores, weights = [], []
        with torch.inference_mode():
            for piece in pieces:
                if not np.any(piece):
                    continue
                scores.append(float(self._model(torch.from_numpy(piece)[None], METRIC_SR)))
                weights.append(len(piece))
        return float(np.average(scores, weights=weights)) if scores else float("nan")


# ----------------------------------------------------------------------- the quality kit


@dataclass
class TargetSet:
    """Speaker embeddings of the target voice: held-out recordings, or a labelled proxy."""

    model: str
    source: str  # "held-out" | "proxy: vox say" | "none"
    embeddings: list[np.ndarray] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.embeddings)


@dataclass
class Scores:
    """One variant's quality numbers. ``nan`` where a measure did not apply."""

    wer: float = float("nan")
    wer_mic: float = float("nan")  # the un-converted microphone against the same script
    transcript: str = ""
    speaker_cos: float = float("nan")
    speaker_cos_min: float = float("nan")
    target_self: float = float("nan")
    seed_floor: float = float("nan")  # cosine between two renders differing only in seed
    target_source: str = "none"
    utmos: float = float("nan")
    f0: dict[str, float] = field(default_factory=dict)
    extra: dict[str, float] = field(default_factory=dict)  # click ratio, R3, HF/LF, ...

    @property
    def wer_delta(self) -> float:
        return self.wer - self.wer_mic

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["wer_delta"] = self.wer_delta
        return d


class QualityKit:
    """The scorers behind one interface. ``f0_track(audio, sr) -> f0_100hz`` comes from
    the caller (the engine's RMVPE, so the F0 numbers match the rest of the bench)."""

    def __init__(self, models_dir: Path | None = None, speaker_backend: str = "ecapa",
                 whisper: str = WHISPER_MODEL,
                 f0_track: Callable[[np.ndarray, int], np.ndarray] | None = None,
                 enable: Sequence[str] = ("wer", "speaker", "utmos", "f0")) -> None:
        models_dir = Path(models_dir) if models_dir else ROOT / "models"
        self.enable = set(enable)
        self.transcriber = Transcriber(whisper, models_dir / "_tools" / "whisper")
        self.speaker = SpeakerEmbedder(speaker_backend, models_dir / "_embedders")
        self.speaker_backend = speaker_backend  # names the embedding cache files
        self.naturalness = Naturalness(models_dir / "_predictors" / "hub")
        self.f0_track = f0_track
        self._cache_dir = ROOT / "out" / "quality"

    # ------------------------------------------------------------------- targets

    def targets(self, model: str, proxy: Callable[[], list[tuple[np.ndarray, int]]] | None = None,
                targets_dir: Path | None = None) -> TargetSet:
        """Held-out recordings from ``takes/target/<model>/`` (embeddings cached by file
        size and mtime), else ``proxy()`` renders (``vox say`` through the model), labelled
        so. ``TargetSet.available`` is False when neither exists."""
        folder = Path(targets_dir) if targets_dir else TARGETS_DIR / model
        files = sorted(p for p in folder.glob("*.wav")) if folder.is_dir() else []
        if files:
            return TargetSet(model, "held-out", [self._cached_embedding(p) for p in files],
                             [str(p) for p in files])
        if proxy is None:
            return TargetSet(model, "none")
        clips = proxy()
        embs = [self.speaker.embed(audio, sr) for audio, sr in clips]
        return TargetSet(model, "proxy: vox say", embs, [f"proxy render {i}" for i in
                                                         range(len(embs))])

    def _cached_embedding(self, path: Path) -> np.ndarray:
        from vox.audio.resample import load_audio

        stat = path.stat()
        cache = self._cache_dir / "targets" / f"{path.stem}_{self.speaker_backend}.npz"
        if cache.is_file():
            with np.load(cache) as z:
                if int(z["size"]) == stat.st_size and float(z["mtime"]) == stat.st_mtime:
                    return z["embedding"]
        audio, sr = load_audio(path)
        emb = self.speaker.embed(audio, sr)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, embedding=emb, size=stat.st_size, mtime=stat.st_mtime)
        return emb

    # -------------------------------------------------------------------- scoring

    def transcript_of(self, audio: np.ndarray, sr: int) -> str:
        return self.transcriber.text(audio, sr)

    def score(self, audio: np.ndarray, sr: int, *, reference_text: str | None = None,
              mic_wer: float = float("nan"), targets: TargetSet | None = None,
              seed_pair: tuple[np.ndarray, np.ndarray] | None = None,
              mask: np.ndarray | None = None) -> Scores:
        """Every enabled measure of one rendered clip. ``reference_text`` is the script;
        ``mic_wer`` the microphone's own WER against it (computed once by the caller);
        ``seed_pair`` two renders of the same variant with different seeds (their cosine
        is the seed floor); ``mask`` the 10 ms active frames for the F0 statistics."""
        s = Scores(wer_mic=mic_wer)
        if "wer" in self.enable and reference_text is not None:
            s.transcript = self.transcript_of(audio, sr)
            s.wer = wer(reference_text, s.transcript)
        if "speaker" in self.enable and targets is not None and targets.available:
            emb = self.speaker.embed(audio, sr)
            summary = speaker_summary(emb, targets.embeddings)
            s.speaker_cos, s.speaker_cos_min = summary["cos"], summary["cos_min"]
            s.target_self, s.target_source = summary["target_self"], targets.source
            if seed_pair is not None:
                s.seed_floor = cosine(self.speaker.embed(seed_pair[0], sr),
                                      self.speaker.embed(seed_pair[1], sr))
        elif targets is not None:
            s.target_source = targets.source
        if "utmos" in self.enable:
            s.utmos = self.naturalness.score(audio, sr)
        if "f0" in self.enable and self.f0_track is not None:
            s.f0 = f0_stats(self.f0_track(audio, sr), mask)
        return s


# ------------------------------------------------------------------------------- tables

COLUMNS = (
    ("WER", lambda s: s.wer, "{:.1%}"),
    ("dWER", lambda s: s.wer_delta, "{:+.1%}"),
    ("spk cos", lambda s: s.speaker_cos, "{:.3f}"),
    ("seed flr", lambda s: s.seed_floor, "{:.3f}"),
    ("UTMOS", lambda s: s.utmos, "{:.2f}"),
    ("F0 med", lambda s: s.f0.get("median_hz", float("nan")), "{:.0f}"),
    ("IQR st", lambda s: s.f0.get("iqr_st", float("nan")), "{:.1f}"),
    ("flips/m", lambda s: s.f0.get("flips_per_min", float("nan")), "{:.1f}"),
    ("voiced", lambda s: s.f0.get("voiced_ratio", float("nan")), "{:.0%}"),
)


def _cell(value: float, fmt: str) -> str:
    try:
        return "-" if value is None or (isinstance(value, float) and math.isnan(value)) \
            else fmt.format(value)
    except (TypeError, ValueError):
        return str(value)


def table(rows: dict[str, Scores], extra: Sequence[tuple[str, str, str]] = ()) -> str:
    """One Markdown table, one row per variant. ``extra``: ``(header, key, fmt)`` columns
    read from ``Scores.extra`` (click ratio, R3, ...)."""
    columns = list(COLUMNS) + [(h, (lambda s, k=k: s.extra.get(k, float("nan"))), f)
                               for h, k, f in extra]
    head = "| variant | " + " | ".join(h for h, _, _ in columns) + " |"
    sep = "|---|" + "|".join("---:" for _ in columns) + "|"
    lines = [head, sep]
    for name, s in rows.items():
        lines.append(f"| {name} | " + " | ".join(_cell(get(s), fmt) for _, get, fmt in columns)
                     + " |")
    sources = {s.target_source for s in rows.values()}
    mic = {s.wer_mic for s in rows.values() if not math.isnan(s.wer_mic)}
    notes = []
    if mic:
        notes.append(f"microphone WER against the same script: {_cell(mic.pop(), '{:.1%}')}")
    if sources:
        notes.append("speaker cosine against " + ", ".join(sorted(sources))
                     + ("; the target recordings agree with each other at "
                        + _cell(next(iter(s.target_self for s in rows.values())), "{:.3f}")
                        if any(not math.isnan(s.target_self) for s in rows.values()) else ""))
    notes.append("UTMOS is a naturalness proxy: report, never rank on it alone; "
                 "listen to the blind pairs")
    return "\n".join(lines) + "\n\n" + "\n".join(f"- {n}" for n in notes)


def load_script(path: Path) -> str:
    """The fixed speech script (a UTF-8 text file; lines starting with # are comments)."""
    text = Path(path).read_text(encoding="utf-8")
    return " ".join(line.strip() for line in text.splitlines()
                    if line.strip() and not line.lstrip().startswith("#"))
