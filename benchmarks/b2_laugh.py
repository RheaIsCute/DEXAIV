"""Laughs on the live path (report 19 §3.3, docs/DEVIATIONS.md §14.2 / §15).

    .venv\\Scripts\\python.exe benchmarks\\b2_laugh.py --take <mic_raw.wav>
        [--segments "3.2-5.0,12.1-13.4"] [--label laugh] [--out DIR]

Renders the take through the user's chain (benchmarks/paraling.py: hop 80, gate -28 /
hangover 500, RNNoise, auto level, Mom +12, ceiling 330) in Speak and Natural, each without
and with the laugh guard, and writes ``<label>_<variant>.wav``. Then, on the conditioned
input's F0 (RMVPE, what the tracker sees), it replays the engine's own register tracker,
laugh guard and soft ceiling hop by hop (fresh = the hop's frames, as in the f0 stage) and
reports, per laugh segment (given, or found: >= 3 voiced pulses of 40-300 ms within 1.2 s):

    register_move_st     how far the auto-pitch register moved across the segment
    register_peak_st     its largest excursion inside the segment and the 3 s after
    recovery_s           time after the segment until the register is back within 1 st of
                         where it was before (None: not within the take)
    ceiling_frames       voiced frames whose transposed F0 is over the ceiling's knee
                         (ceiling - 4 st: the soft ceiling is bending them), and their share
    out_f0_p90_hz        p90 of the transposed F0 vs the knee
    flips_per_s          voicing flips per second inside the segment (speech: see
                         ``speech_flips_per_s``, the 1 s windows outside every segment)
    gate_closes_inside   from the Speak render
    unvoiced_share       RMVPE frames the laugh gets as unvoiced (breath, not voice)

Every number is printed as one ``B2 {json}`` line and saved to ``<label>_report.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import paraling as pl  # noqa: E402

KNEE_ST = 4.0
RECOVERY_ST = 1.0


def voiced_runs(voiced: np.ndarray) -> list[tuple[int, int]]:
    runs, i = [], 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def find_segments(runs: list[tuple[int, int]]) -> list[tuple[float, float]]:
    pulses = [(a, b) for a, b in runs if 4 <= b - a <= 30]
    segs: list[tuple[float, float]] = []
    for k in range(len(pulses)):
        group = [pulses[k]]
        for a, b in pulses[k + 1:]:
            if a - group[0][0] <= 120:
                group.append((a, b))
        if len(group) >= 3 and (not segs or group[0][0] / 100 > segs[-1][1]):
            segs.append((group[0][0] / 100, group[-1][1] / 100))
    return segs


def flips_per_s(voiced: np.ndarray) -> float:
    if len(voiced) < 2:
        return 0.0
    return float(np.abs(np.diff(voiced.astype(int))).sum() * 100.0 / len(voiced))


def replay(f0: np.ndarray, fresh: int, params, guard_on: bool) -> dict[str, np.ndarray]:
    """The f0 stage's tracker / guard / ceiling, hop by hop on the CPU (no note detector:
    a laugh is not a held note)."""
    from vox.pitch.base import LaughGuard, RegisterTracker, semitones_to_ratio

    reg = RegisterTracker("cpu")
    reg.fresh = fresh
    reg.set(params.pitch_mode == "auto", params.target_f0_hz, params.pitch_follow_s,
            params.pitch_range)
    guard = LaughGuard("cpu")
    guard.fresh = fresh
    guard.set(guard_on, params.laugh_jump_st, params.laugh_flip_rate, params.laugh_ceiling_hz)
    fixed = torch.tensor(semitones_to_ratio(params.f0_up_key))
    base_ceiling = torch.tensor(params.f0_ceiling_hz if params.f0_ceiling_hz > 0 else math.inf)
    n = len(f0) // fresh * fresh
    ft = torch.from_numpy(np.concatenate([np.zeros(32, np.float32), f0[:n]]))
    level, ratio, ceiling, weight, lift, rate = (np.zeros(n) for _ in range(6))
    for h in range(n // fresh):
        end = 32 + (h + 1) * fresh
        win = ft[end - 32:end]
        w = guard.update(win, reg.level, reg.heard)
        r = reg.ratio(win, fixed, weight=w)[-fresh:]
        c = guard.ceiling(base_ceiling)
        sl = slice(h * fresh, (h + 1) * fresh)
        level[sl] = float(reg.level)
        ratio[sl] = r.numpy()
        ceiling[sl] = float(c)
        weight[sl], lift[sl], rate[sl] = float(w), float(guard.lift), float(guard.rate)
    return {"level": level, "ratio": ratio, "ceiling": ceiling, "weight": weight,
            "lift": lift, "rate": rate, "n": n}


def segment_rows(f0, voiced, runs, segs, trace, closes) -> list[dict]:
    rows = []
    n = trace["n"]
    level = trace["level"]
    out = f0[:n] * trace["ratio"]
    knee = trace["ceiling"] * 2.0 ** (-KNEE_ST / 12.0)
    for a, b in segs:
        fa, fb = int(a * 100), min(int(b * 100), n - 1)
        if fb <= fa:
            continue
        v = voiced[fa:fb]
        pre = level[max(0, fa - 1)]
        after = level[fb:min(n, fb + 300)]
        move = 12.0 * (level[fb] - pre)
        peak = 12.0 * float(np.max(np.abs(np.concatenate([level[fa:fb], after]) - pre)))
        back = np.flatnonzero(np.abs(level[fb:] - pre) * 12.0 <= RECOVERY_ST)
        engaged = (out[fa:fb] > knee[fa:fb]) & v
        seg_out = out[fa:fb][v]
        rows.append({
            "t0": a, "t1": b, "pulses": sum(1 for x, y in runs if x >= fa and y <= fb),
            "unvoiced_share": float(1 - v.mean()),
            "f0_in_p10": float(np.percentile(f0[fa:fb][v], 10)) if v.any() else None,
            "f0_in_p90": float(np.percentile(f0[fa:fb][v], 90)) if v.any() else None,
            "register_move_st": round(float(move), 2), "register_peak_st": round(peak, 2),
            "recovery_s": (round(float(back[0]) / 100.0, 2) if len(back) else None),
            "ceiling_frames": int(engaged.sum()),
            "ceiling_share": float(engaged.sum() / max(1, v.sum())),
            "out_f0_p90_hz": float(np.percentile(seg_out, 90)) if len(seg_out) else None,
            "knee_hz": float(np.median(knee[fa:fb])),
            "flips_per_s": round(flips_per_s(voiced[fa:fb]), 1),
            "guard_weight_min": round(float(trace["weight"][fa:fb].min()), 3),
            "ceiling_lift_max": round(float(trace["lift"][fa:fb].max()), 3),
            "gate_closes_inside": sum(1 for t in closes if a <= t <= b),
        })
    return rows


def speech_flip_rates(voiced: np.ndarray, segs) -> dict:
    """Voicing flips/s over 1 s windows outside every segment, where something is voiced:
    the speech baseline the flicker threshold must stay above."""
    mask = np.ones(len(voiced), bool)
    for a, b in segs:
        mask[max(0, int(a * 100) - 100):int(b * 100) + 100] = False
    rates = []
    for s in range(0, len(voiced) - 100, 50):
        if mask[s:s + 100].all() and voiced[s:s + 100].mean() > 0.2:
            rates.append(flips_per_s(voiced[s:s + 100]))
    if not rates:
        return {"windows": 0}
    return {"windows": len(rates), "p50": float(np.percentile(rates, 50)),
            "p90": float(np.percentile(rates, 90)), "max": float(np.max(rates))}


def main() -> int:
    from vox.audio.conditioning import ConditioningConfig, InputConditioner
    from vox.audio.resample import load_audio, resample_array, save_audio

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--take", required=True)
    ap.add_argument("--segments", default="", help="laugh spans in s, '3.2-5.0,12.1-13.4'")
    ap.add_argument("--out", default=str(ROOT / "out" / "ab" / "paraling" / "laugh"))
    ap.add_argument("--label", default="laugh")
    ap.add_argument("--variants", default="speak,speak_guard,natural,natural_guard")
    ap.add_argument("--no-render", action="store_true", help="metrics only (no GPU renders)")
    args = ap.parse_args()
    sr = pl.SR
    audio, _ = load_audio(args.take, sr)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"take": args.take, "seconds": len(audio) / sr}
    closes: list[float] = []
    engine = None
    for name in args.variants.split(","):
        preset = "natural" if name.startswith("natural") else "speak"
        variant = "laugh" if name.endswith("_guard") else "base"
        engine = pl.load_variant(variant, ROOT / "models", ROOT / "profiles" / "default",
                                 preset=preset)
        if args.no_render:
            break
        res = pl.render(engine, audio)
        save_audio(out_dir / f"{args.label}_{name}.wav", res.audio, sr)
        if name == "speak":
            closes = [e["hop"] * engine.plan.block / sr for e in res.events
                      if e["kind"] == "gate_close"]
        engine.graphs.release()
        engine = None if name != args.variants.split(",")[-1] else engine
    if engine is None:
        engine = pl.load_variant("base", ROOT / "models", ROOT / "profiles" / "default")
    plan = engine.plan
    cond = ConditioningConfig(**pl.USER_COND)
    c = InputConditioner(sr, plan.block, cond)
    n_hops = math.ceil(len(audio) / plan.block)
    padded = np.zeros(n_hops * plan.block, np.float32)
    padded[:len(audio)] = audio
    conditioned = np.concatenate([c.process(padded[i * plan.block:(i + 1) * plan.block]).copy()
                                  for i in range(n_hops)])
    x16 = resample_array(conditioned, sr, 16000).astype(np.float32)
    with torch.inference_mode():
        f0 = engine.pipeline.extractor.extract(
            torch.from_numpy(x16).to(engine.device)).float().cpu().numpy()
    engine.graphs.release()
    voiced = f0 > 0
    runs = voiced_runs(voiced)
    if args.segments:
        segs = [tuple(float(v) for v in s.split("-")) for s in args.segments.split(",")]
    else:
        segs = find_segments(runs)
    from vox.config.schemas import RuntimeParams

    for name in ("speak", "speak_guard", "natural", "natural_guard"):
        base = pl.NATURAL_PARAMS if name.startswith("natural") else pl.USER_PARAMS
        params = RuntimeParams(**base)
        trace = replay(f0, plan.shift_frames, params, name.endswith("_guard"))
        report[name] = segment_rows(f0, voiced, runs, segs, trace, closes)
    report["segments"] = segs
    report["speech_flips_per_s"] = speech_flip_rates(voiced, segs)
    report["voiced_share"] = float(voiced.mean())
    report["gate_closes_total"] = len(closes)
    pl.write_report(out_dir / f"{args.label}_report.json", report)
    print("B2 " + json.dumps(report, default=float), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
