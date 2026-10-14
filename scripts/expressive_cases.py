"""Render the expressive acceptance cases with ``vox say`` and measure what the ear should
hear: the pause length at the dots, the level of the emphasised piece, and the duration of
the elongated words against their normalised spelling.

    .venv\\Scripts\\python scripts\\expressive_cases.py --model base_f0_48k_v2

Writes ``out/expressive/<case>.wav`` plus a ``<case>_plain.wav`` reference rendered from
the normalised text with the expressive hints switched off, and prints a table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CASES = {
    "dots_emphasis": ("hey... hey!! are you fr right now", "hey. hey! are you fr right now"),
    "elongation": ("nooooo waaaay", "noo waay"),
    "caps": ("that is SO not what i said", "that is so not what i said!"),
    "formal": ("Hello, this is a test of the text to speech path.",
               "Hello, this is a test of the text to speech path."),
}


def rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def speech_segments(audio: np.ndarray, sr: int, floor_db: float = -40.0,
                    min_gap_ms: float = 60.0) -> list[tuple[int, int]]:
    """(start, end) sample spans above ``floor_db`` (10 ms RMS frames), gaps shorter than
    ``min_gap_ms`` bridged."""
    hop = sr // 100
    frames = len(audio) // hop
    level = np.array([rms_db(audio[i * hop:(i + 1) * hop]) for i in range(frames)])
    on = level > floor_db
    spans = []
    start = None
    for i, flag in enumerate(on):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            spans.append([start, i])
            start = None
    if start is not None:
        spans.append([start, frames])
    merged: list[list[int]] = []
    for s in spans:
        if merged and (s[0] - merged[-1][1]) * 10 < min_gap_ms:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(a * hop, b * hop) for a, b in merged]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base_f0_48k_v2")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "expressive")
    parser.add_argument("--backend", default="kokoro")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    args = parser.parse_args()

    from vox.audio.resample import load_audio
    from vox.tts import expressive, speak
    from vox.tts.backends import pick_backend
    from vox.tts.cli import TTS_DIR

    backend = pick_backend(args.backend, args.voice, args.models_dir / TTS_DIR)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in args.cases:
        text, plain_text = CASES[case]
        paths = {}
        def plain_analyze(t: str) -> list:  # the pre-change path: sentences, no hints
            return [expressive.ExpressivePiece(s) for s in expressive.split_sentences(t)]

        for tag, say_text, hints in (("", text, True), ("_plain", plain_text, False)):
            path = args.out / f"{case}{tag}.wav"
            speak.analyze = expressive.analyze if hints else plain_analyze
            try:
                result = speak.speak(say_text, backend, args.models_dir, args.model,
                                     pitch=args.pitch, seed=0, out=path)
            finally:
                speak.analyze = expressive.analyze
            paths[tag] = path
            print(f"{case}{tag}: {say_text!r} -> {path.name} ({result.seconds:.2f} s, "
                  f"{result.pieces} piece(s))")
            for n, piece in enumerate(result.hints):
                print(f"    piece {n + 1}: {piece.text!r} speed {piece.speed:.2f} gain "
                      f"{piece.gain_db:+.1f} dB pause {piece.pause_after_s:.2f} s"
                      f"{' (explicit)' if piece.explicit else ''}")
        audio, sr = load_audio(paths[""], None)
        plain, _ = load_audio(paths["_plain"], None)
        segs = speech_segments(audio, sr)
        segs_plain = speech_segments(plain, sr)
        gap = (segs[1][0] - segs[0][1]) / sr * 1000 if len(segs) > 1 else float("nan")
        gap_plain = ((segs_plain[1][0] - segs_plain[0][1]) / sr * 1000
                     if len(segs_plain) > 1 else float("nan"))
        first = rms_db(audio[segs[0][0]:segs[0][1]]) if segs else float("nan")
        second = rms_db(audio[segs[1][0]:segs[1][1]]) if len(segs) > 1 else float("nan")
        second_plain = (rms_db(plain[segs_plain[1][0]:segs_plain[1][1]])
                        if len(segs_plain) > 1 else float("nan"))
        rows.append((case, len(audio) / sr, len(plain) / sr, gap, gap_plain, second - first,
                     second - second_plain, len(segs)))
    print(f"\n{'case':<14} {'len s':>6} {'plain s':>7} {'gap1 ms':>8} {'plain':>6} "
          f"{'seg2-seg1 dB':>12} {'seg2 vs plain dB':>16} segs")
    for r in rows:
        print(f"{r[0]:<14} {r[1]:>6.2f} {r[2]:>7.2f} {r[3]:>8.0f} {r[4]:>6.0f} {r[5]:>12.1f} "
              f"{r[6]:>16.1f} {r[7]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
