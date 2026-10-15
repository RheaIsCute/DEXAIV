"""Attribute every digital-zero cut in a live output recording (``vox run --record-output``).

    python scripts/analyze_tap.py out/live_tap.wav [--input benchmarks/corpus/mic_script.wav]

The tap holds what the output callback handed the device and a mask of the samples that
came out of the ring. A zero run with any zero-filled (mask 0) sample is an **underrun**;
one that came entirely out of the ring was written by the worker for a gated hop: the
**gate**. With ``--input`` (the file the session looped), runs are also placed against the
input's phrases: the lag between the tap and the input is found by cross-correlating their
envelopes, and a run counts as mid-speech when it overlaps an input phrase by >= 10 ms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vox.audio.resample import load_audio
from vox.offline.metrics import frame_rms_dbfs, phrases, zero_runs


def envelope_lag(tap: np.ndarray, source: np.ndarray, sr: int, max_s: float = 1.0) -> int:
    """Samples the tap trails ``source`` (envelopes at 10 ms, cross-correlated)."""
    hop = sr // 100
    a = np.maximum(frame_rms_dbfs(tap, sr), -80.0) + 80.0
    b = np.maximum(frame_rms_dbfs(source, sr), -80.0) + 80.0
    n = min(len(a), len(b) * 3)
    best, lag = -np.inf, 0
    for k in range(int(max_s * 100)):
        m = min(n - k, len(b))
        if m <= 0:
            break
        score = float(np.dot(a[k:k + m] - a[k:k + m].mean(), b[:m] - b[:m].mean()))
        if score > best:
            best, lag = score, k
    return lag * hop


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("tap", type=Path)
    p.add_argument("--input", type=Path, default=None, help="the file the session looped")
    p.add_argument("--min-ms", type=float, default=10.0)
    args = p.parse_args()

    tap, sr = load_audio(args.tap)
    mask = np.load(args.tap.with_suffix(".npz"))["from_ring"][:len(tap)]
    log = json.loads(args.tap.with_suffix(".events.json").read_text())
    runs = zero_runs(tap, int(sr * args.min_ms / 1000))
    in_phrase = None
    if args.input is not None:
        source, _ = load_audio(args.input, sr)
        lag = envelope_lag(tap, source, sr)
        loops = -(-max(0, len(tap) - lag) // len(source))
        looped = np.tile(source, loops + 1)
        in_phrase = np.zeros(len(tap), dtype=bool)
        for s, e in phrases(source, sr):
            for k in range(loops + 1):
                lo, hi = s + k * len(source) + lag, e + k * len(source) + lag
                in_phrase[max(lo, 0):max(min(hi, len(tap)), 0)] = True
        print(f"tap trails the input by {1000 * lag / sr:.0f} ms (envelope match); "
              f"{len(looped) / sr:.0f} s of looped input covered")
    counts = {"gate": 0, "underrun": 0}
    mid = {"gate": 0, "underrun": 0}
    lost = {"gate": 0.0, "underrun": 0.0}
    for a, b in runs:
        cause = "underrun" if mask[a:b].min() == 0 else "gate"
        counts[cause] += 1
        if in_phrase is not None:
            speech = int(in_phrase[a:b].sum())
            if speech >= sr // 100:
                mid[cause] += 1
                lost[cause] += 1000.0 * speech / sr
    print(f"{len(tap) / sr:.1f} s recorded; zero runs >= {args.min_ms:g} ms: {len(runs)} "
          f"(gate {counts['gate']}, underrun {counts['underrun']}); underrun-filled samples "
          f"{int((mask == 0).sum())}")
    if in_phrase is not None:
        print(f"mid-speech: gate {mid['gate']} ({lost['gate']:.0f} ms of speech), underrun "
              f"{mid['underrun']} ({lost['underrun']:.0f} ms)")
    print(f"event log: {log['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
