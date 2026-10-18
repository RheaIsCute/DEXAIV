"""Robust formant / pitch check of rendered formant steps (docs/DEVIATIONS.md §12).

LPC formant tracking fails on high voices (at ~200 Hz F0 its "F1" is the fundamental), so
two measures that do not depend on resolving formants frame by frame:

* **pitch**: the median, over frames voiced in both renders, of the per-frame cent
  difference between a step and step 0 (same input timeline), from RMVPE at 100 Hz.
* **envelope shift**: the long-term average spectrum (LTAS) of each render's active
  frames, cepstrally smoothed to an envelope; the factor ``a`` that best aligns
  ``env_step(f)`` with ``env_0(f / a)`` over 300-5000 Hz (log-frequency cross-correlation,
  1-cent grid). A formant shift of ``f`` shows as ``a ≈ f``.

    python benchmarks/formant_check.py out/ab/formant [--prefix live_]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def envelope_db(audio: np.ndarray, sr: int, n_fft: int = 4096, lifter: int = 40) -> tuple:
    """LTAS over active frames, cepstrally liftered: ``(freqs, envelope dB)``."""
    from vox.offline.metrics import active_frames

    hop = sr // 100
    x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    spec = torch.stft(x, n_fft, hop_length=hop, window=torch.hann_window(n_fft), center=True,
                      return_complex=True).abs().pow(2).numpy()
    mask = active_frames(audio, sr)[:spec.shape[1]]
    ltas = spec[:, :len(mask)][:, mask].mean(axis=1) + 1e-12
    log = np.log(ltas)
    cep = np.fft.irfft(log)
    cep[lifter:len(cep) - lifter] = 0.0
    env = np.fft.rfft(cep).real[:len(log)]
    return np.fft.rfftfreq(n_fft, 1.0 / sr), 10 * env / np.log(10)


def envelope_shift(ref: np.ndarray, test: np.ndarray, sr: int, lo: float = 300.0,
                   hi: float = 5000.0, span_st: float = 6.0) -> float:
    """Factor ``a`` maximising the correlation of env_test(f) with env_ref(f / a)."""
    f, e_ref = envelope_db(ref, sr)
    _, e_test = envelope_db(test, sr)
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 2000))
    t = np.interp(grid, f, e_test)
    t = t - t.mean()
    best = (-np.inf, 1.0)
    for cents in np.arange(-100 * span_st, 100 * span_st + 1, 1.0):
        a = 2 ** (cents / 1200)
        r = np.interp(grid / a, f, e_ref)
        r = r - r.mean()
        c = float(np.dot(r, t) / (np.linalg.norm(r) * np.linalg.norm(t) + 1e-12))
        best = max(best, (c, a))
    return best[1]


def framewise_cents(ref_f0: np.ndarray, test_f0: np.ndarray) -> tuple[float, float]:
    n = min(len(ref_f0), len(test_f0))
    both = (ref_f0[:n] > 0) & (test_f0[:n] > 0)
    d = 1200 * np.log2(test_f0[:n][both] / ref_f0[:n][both])
    d = d[np.abs(d) < 600]  # octave errors are not a pitch shift
    return float(np.median(d)), float(np.mean(np.abs(d) < 50))


def main() -> int:
    from vox.audio.resample import load_audio, resample_array
    from vox.pitch import make_extractor

    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--prefixes", default="live_,typed_")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rmvpe = make_extractor("rmvpe")
    rmvpe.load(device, ROOT / "models")

    def f0(audio, sr):
        y = resample_array(audio, sr, 16000)
        with torch.inference_mode():
            return rmvpe.extract(torch.from_numpy(y).to(device)).float().cpu().numpy()

    for prefix in args.prefixes.split(","):
        files = sorted(args.folder.glob(f"{prefix}*.wav"))
        steps = {float(p.stem[len(prefix):]): p for p in files}
        if 0.0 not in steps:
            continue
        ref, sr = load_audio(steps[0.0])
        ref_f0 = f0(ref, sr)
        print(f"\n{prefix.rstrip('_')}: | step | expected factor | envelope shift | "
              "pitch vs step 0 (median cents, frames within 50 cents) |")
        for st, path in sorted(steps.items()):
            audio, _ = load_audio(path, sr)
            shift = envelope_shift(ref, audio, sr) if st != 0.0 else 1.0
            cents, within = framewise_cents(ref_f0, f0(audio, sr))
            print(f"  {st:+.1f} | {2 ** (st / 12):.3f} | {shift:.3f} "
                  f"({1200 * math.log2(shift) / 100:+.2f} st) | {cents:+.1f} c, {within:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
