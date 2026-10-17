"""Per-model target register (docs/research/15, Phase 0.2; open question Q13).

``target_f0_median_hz`` is what auto pitch and ``vox say`` aim at. No model ships it, so
until now every model fell back to 200 Hz. Three estimators, from the model itself:

1. **rendered** (the one written, as the prompt defines it): the fixed speech script
   through Kokoro at pitch 0 (no auto pitch) and through the model; the median voiced F0
   of the OUTPUT. Caveat, stated plainly in the table: RVC copies the source contour, so at
   pitch 0 this largely echoes the TTS voice's register; it is the decoder's answer, not its
   training distribution.
2. **sweep**: the same render at several transpositions; at each, how faithfully the output
   carries the requested pitch (median cents error, voicing kept, octave flips). The
   shift where fidelity is best is where the decoder is most at home; its output median is
   the sweep estimate. Flat curves (all within tolerance) carry no information.
3. **embedding**: the fine-tuned model's coarse-pitch embedding table against its
   pretrained base (``models/_pretrained/f0G<sr>.pth``): the buckets that moved during
   fine-tuning are the buckets the training data occupied. Weighted median in Hz. Only when
   a base of the model's sample rate is present and the model derives from it (unknown for
   community models; a flat movement profile says it does not).

All three are stored under ``defaults.target_f0_calibration`` of the model's config.json;
``target_f0_median_hz`` takes the rendered value. Disagreement is printed for the user.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TTS_VOICE = "af_heart"
SWEEP_ST = (-8.0, -4.0, 0.0, 4.0, 8.0)
# Coarse-pitch buckets (pitch/base.py f0_to_coarse): 1 = unvoiced, 2..255 cover 50-1100 Hz.
_MEL_MIN = 1127.0 * math.log(1.0 + 50.0 / 700.0)
_MEL_MAX = 1127.0 * math.log(1.0 + 1100.0 / 700.0)


def bucket_to_hz(bucket: np.ndarray) -> np.ndarray:
    """Centre frequency of each coarse-pitch bucket (the inverse of ``f0_to_coarse``)."""
    mel = (np.asarray(bucket, dtype=np.float64) - 1.0) * (_MEL_MAX - _MEL_MIN) / 254.0 + _MEL_MIN
    return 700.0 * (np.exp(mel / 1127.0) - 1.0)


def embedding_register(fine: np.ndarray, base: np.ndarray) -> dict[str, float] | None:
    """Where the fine-tuned pitch embedding moved away from its base: per bucket, the L2
    of the difference, as a weight over 50-1100 Hz. Returns the weighted median / p10 / p90
    in Hz and ``peak_ratio`` (largest weight over the median weight: a flat profile, under
    ~3, means the model did not derive from this base or trained every pitch alike)."""
    fine, base = np.asarray(fine, dtype=np.float64), np.asarray(base, dtype=np.float64)
    if fine.shape != base.shape or fine.shape[0] < 256:
        return None
    moved = np.linalg.norm(fine - base, axis=1)[2:256]  # bucket 1 is "unvoiced"
    if not np.any(moved > 0):
        return None
    hz = bucket_to_hz(np.arange(2, 256))
    weights = moved / moved.sum()
    cdf = np.cumsum(weights)
    pick = lambda q: float(np.interp(q, cdf, hz))
    return {"median_hz": pick(0.5), "p10_hz": pick(0.1), "p90_hz": pick(0.9),
            "peak_ratio": float(moved.max() / max(np.median(moved), 1e-12))}


def load_pitch_table(path: Path) -> np.ndarray | None:
    import torch

    cpt = torch.load(path, map_location="cpu", weights_only=True)
    weights = cpt.get("weight") or cpt.get("model") or {}
    table = weights.get("enc_p.emb_pitch.weight")
    return None if table is None else table.float().numpy()


def fidelity(f0_in_100hz: np.ndarray, ratio: float, f0_out_100hz: np.ndarray) -> dict[str, float]:
    """How the output carried the requested pitch, without frame alignment: the median of
    the output's voiced F0 against the requested median (cents), the voicing the output kept
    (its voiced share over the input's) and its octave flips per voiced minute."""
    import sys

    sys.path.insert(0, str(ROOT / "benchmarks"))
    from quality import f0_stats

    s_in, s_out = f0_stats(f0_in_100hz), f0_stats(f0_out_100hz)
    requested = s_in["median_hz"] * ratio
    cents = 1200.0 * math.log2(s_out["median_hz"] / requested) if requested > 0 and \
        s_out["median_hz"] > 0 else float("nan")
    kept = s_out["voiced_ratio"] / s_in["voiced_ratio"] if s_in["voiced_ratio"] > 0 else \
        float("nan")
    return {"requested_hz": requested, "median_hz": s_out["median_hz"], "cents_error": cents,
            "voicing_kept": kept, "flips_per_min": s_out["flips_per_min"],
            "iqr_st": s_out["iqr_st"]}


def fidelity_cost(row: dict[str, float]) -> float:
    """One number to rank sweep points: cents off target (per 10), voicing lost (per 1%),
    flips per minute. Lower is better; nan sorts last."""
    parts = (abs(row["cents_error"]) / 10.0, 100.0 * max(0.0, 1.0 - row["voicing_kept"]),
             row["flips_per_min"])
    return float("inf") if any(math.isnan(p) for p in parts) else float(sum(parts))


@dataclass
class Calibration:
    model: str
    tts_voice: str
    script: str
    rendered_hz: float = 0.0  # the written value: output median at pitch 0
    tts_hz: float = 0.0  # the TTS voice's own median (what pitch 0 echoes)
    sweep: list[dict[str, float]] = field(default_factory=list)
    sweep_best_st: float = float("nan")
    sweep_hz: float = float("nan")
    embedding: dict[str, float] | None = None
    utmos: dict[str, float] = field(default_factory=dict)  # per shift, when a kit is given
    when: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def as_defaults(self) -> dict:
        """What goes into the model's ``config.json`` ``defaults``."""
        return {"target_f0_median_hz": round(self.rendered_hz, 1),
                "target_f0_calibration": {
                    "method": "rendered: fixed script, Kokoro " + self.tts_voice
                              + ", pitch 0, median voiced F0 of the output (RMVPE)",
                    "when": self.when, "rendered_hz": round(self.rendered_hz, 1),
                    "tts_hz": round(self.tts_hz, 1),
                    "sweep_best_st": self.sweep_best_st, "sweep_hz": round(self.sweep_hz, 1)
                    if not math.isnan(self.sweep_hz) else None,
                    "embedding": self.embedding, "sweep": self.sweep, "utmos": self.utmos}}

    def disagreement_st(self) -> dict[str, float]:
        """Semitones between the rendered value and the other two estimators."""
        out = {}
        for name, hz in (("sweep", self.sweep_hz),
                         ("embedding", (self.embedding or {}).get("median_hz", float("nan")))):
            out[name] = 12.0 * math.log2(hz / self.rendered_hz) if hz and hz > 0 and \
                self.rendered_hz > 0 and not math.isnan(hz) else float("nan")
        return out


def table(cals: list[Calibration]) -> str:
    lines = [("| model | sr | rendered (pitch 0) | TTS voice | sweep best | sweep Hz | "
              "embedding Hz (p10-p90, peak) | written |"),
             "|---|---:|---:|---:|---:|---:|---|---:|"]
    for c in cals:
        emb = "-" if not c.embedding else (f"{c.embedding['median_hz']:.0f} "
                                          f"({c.embedding['p10_hz']:.0f}-{c.embedding['p90_hz']:.0f}, "
                                          f"x{c.embedding['peak_ratio']:.1f})")
        best = "-" if math.isnan(c.sweep_best_st) else f"{c.sweep_best_st:+.0f} st"
        swp = "-" if math.isnan(c.sweep_hz) else f"{c.sweep_hz:.0f}"
        lines.append(f"| {c.model} | | {c.rendered_hz:.0f} Hz | {c.tts_hz:.0f} Hz | {best} | "
                     f"{swp} | {emb} | {c.as_defaults()['target_f0_median_hz']:.0f} Hz |")
    return "\n".join(lines)


def sweep_table(cal: Calibration) -> str:
    lines = [(f"{cal.model}: shift | requested | out median | cents | voicing kept | "
              "flips/min | IQR st | UTMOS")]
    for row in cal.sweep:
        u = cal.utmos.get(f"{row['shift_st']:g}", float("nan"))
        lines.append(f"  {row['shift_st']:+5.0f} | {row['requested_hz']:9.0f} | "
                     f"{row['median_hz']:10.0f} | {row['cents_error']:+6.0f} | "
                     f"{100 * row['voicing_kept']:11.0f}% | {row['flips_per_min']:9.1f} | "
                     f"{row['iqr_st']:6.1f} | {'-' if math.isnan(u) else f'{u:.2f}'}")
    return "\n".join(lines)
