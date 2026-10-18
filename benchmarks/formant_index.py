"""``vox bench formant`` and ``vox bench index`` (docs/DEVIATIONS.md §12).

formant: one raw-mic take and one typed line rendered at each formant step (live = the
stream simulator with the input chain, typed = Kokoro + the full-file converter), saved to
``out/ab/formant/``; per render the output median F0 (must not move), LPC F1/F2 (must move
by the factor), the hop-seam click ratio, and the synth stage's cost at 0 vs the largest
step.

index: the same raw-mic take at index_rate {0, 0.3, 0.6, 1.0} with one seed, plus a second
seed at 0 for the noise floor, all from one engine (the index loaded throughout, so only the
rate differs). Per variant: how far the blend moved the features the decoder saw (mean
cosine distance on the decoded frames), the output's log-mel L1 against index 0 over speech
frames, speaker cosine, WER against the script, sibilant centroid and loudness. Then the
verdict rule of the feature prompt, and a 16-trial ABX set when the output does change.
"""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TAKE = (ROOT / "recordings" / "matrix_20261017_192237" /
                "shure_gateoff_agcoff_hop20_mic_raw.wav")
INDEX_RATES = (0.0, 0.3, 0.6, 1.0)
INDEX_PRIME = 1e-6  # load the index while the rate is 0 (as the GUI does)


# ------------------------------------------------------------------------- measures


def lpc_formants(audio: np.ndarray, sr: int, order: int = 12) -> tuple[float, float]:
    """Median F1 / F2 (LPC at 10 kHz, 25 ms frames, loudest 30%)."""
    from scipy.linalg import solve_toeplitz

    from vox.audio.resample import resample_array

    y = resample_array(audio, sr, 10000).astype(np.float64)
    win, hop = 250, 100
    starts = list(range(0, len(y) - win, hop))
    energy = np.array([np.sum(y[i:i + win] ** 2) for i in starts])
    if not len(energy):
        return float("nan"), float("nan")
    thr = np.percentile(energy, 70)
    f1, f2 = [], []
    for e, i in zip(energy, starts):
        if e < thr:
            continue
        seg = y[i:i + win] * np.hamming(win)
        seg = np.append(seg[0], seg[1:] - 0.63 * seg[:-1])
        r = np.correlate(seg, seg, "full")[win - 1:win + order + 1]
        if r[0] <= 0:
            continue
        a = solve_toeplitz(r[:order], -r[1:order + 1])
        roots = np.roots(np.concatenate(([1.0], a)))
        roots = roots[np.imag(roots) > 0]
        freq = np.angle(roots) * 10000 / (2 * np.pi)
        bw = -10000 / np.pi * np.log(np.abs(roots)) / 2
        f = np.sort(freq[(freq > 200) & (freq < 4500) & (bw < 400)])
        if len(f) >= 2:
            f1.append(f[0])
            f2.append(f[1])
    return (float(np.median(f1)), float(np.median(f2))) if f1 else (float("nan"), float("nan"))


def sibilant_centroid(audio: np.ndarray, sr: int) -> float:
    """Mean spectral centroid (Hz) of the frames whose energy is mostly above 4 kHz (the
    's' and 'sh' frames), among active frames. NaN without any."""
    from vox.offline.metrics import active_frames

    hop, win = sr // 100, sr // 25
    n_fft = 1 << (win - 1).bit_length()
    x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    spec = torch.stft(x, n_fft, hop_length=hop, win_length=win, window=torch.hann_window(win),
                      center=True, return_complex=True).abs().pow(2).numpy()
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= 100) & (freqs < min(16000, sr / 2))
    spec = spec[band]
    freqs = freqs[band]
    total = spec.sum(axis=0) + 1e-12
    high = spec[freqs >= 4000].sum(axis=0) / total
    active = active_frames(audio, sr)
    n = min(len(active), spec.shape[1])
    pick = active[:n] & (high[:n] > 0.5)
    if not pick.any():
        return float("nan")
    centroid = (freqs[:, None] * spec[:, :n]).sum(axis=0) / total[:n]
    return float(centroid[pick].mean())


def active_dbfs(audio: np.ndarray, sr: int) -> float:
    from vox.offline.metrics import active_frames, frame_rms_dbfs

    level = frame_rms_dbfs(audio, sr)
    mask = active_frames(audio, sr)[:len(level)]
    if not mask.any():
        return float("nan")
    return float(10 * np.log10(np.mean(10 ** (level[mask] / 10)) + 1e-12))


# ------------------------------------------------------------------- stream with probe


def stream_with_feature_probe(engine, audio: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    """Open stream through the conditioned engine path, hop by hop (like ``_sing_trace``),
    measuring per hop the mean cosine distance between the features before the index
    blend (``feats0``) and what the decoder received (``feats``) over the decoded frames.
    Returns (aligned output, mean distance over voiced-or-not active hops)."""
    plan, pipe, sola, cond = engine.plan, engine.pipeline, engine.sola, engine.conditioner
    torch.manual_seed(seed)
    pipe.reset()
    sola.reset()
    cond.reset()
    block, hops = plan.block, len(audio) // plan.block
    out = np.zeros(hops * block, dtype=np.float32)
    frames = plan.decode_frames
    dist = []
    for n in range(hops):
        pipe.upload(cond.process(audio[n * block:(n + 1) * block]))
        stitched = sola.stitch(pipe.process_hop())
        out[n * block:(n + 1) * block] = stitched.cpu().numpy()
        a = pipe.feats0[0, -frames:]
        b = pipe.feats[0, -frames:]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        energy = a.norm(dim=-1)
        dist.append(float((1.0 - cos)[energy > 0].mean()) if bool((energy > 0).any()) else 0.0)
    settle = math.ceil((plan.context_ms / 1000 + 0.5) * 1000 / plan.block_ms)
    aligned = out[plan.stream_delay:][:len(audio)]
    return aligned, float(np.mean(dist[settle:])) if len(dist) > settle else float("nan")


# --------------------------------------------------------------------------- ABX


def write_abx(a: Path, b: Path, out_dir: Path, trials: int = 16, seed: int = 2026,
              title: str = "") -> dict:
    """``A.wav``, ``B.wav`` and ``trialNN_X.wav`` (X is A or B at random, balanced 8/8),
    ``abx.md`` to fill in and ``answer_key.json``. 13 of 16 correct is p < 0.011."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copyfile(a, out_dir / "A.wav")
    shutil.copyfile(b, out_dir / "B.wav")
    rng = np.random.default_rng(seed)
    answers = rng.permutation(["A"] * (trials // 2) + ["B"] * (trials - trials // 2)).tolist()
    for n, ans in enumerate(answers, 1):
        shutil.copyfile(a if ans == "A" else b, out_dir / f"trial{n:02d}_X.wav")
    sheet = [f"# ABX{': ' + title if title else ''}", "",
             ("Listen to A.wav and B.wav, then to each trialNN_X.wav, and write whether X is "
              "A or B. Guess if you are unsure; every trial needs an answer. 13 or more of "
              "16 right means you hear the difference (fewer than 1.1% would get there by "
              "chance). Open answer_key.json only after."), ""]
    sheet += [f"- trial{n:02d}: " for n in range(1, trials + 1)]
    (out_dir / "abx.md").write_text("\n".join(sheet) + "\n", encoding="utf-8")
    key = {"title": title, "seed": seed, "A": str(a), "B": str(b),
           "answers": {f"trial{n:02d}": ans for n, ans in enumerate(answers, 1)},
           "heard_if_correct_at_least": 13}
    (out_dir / "answer_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    return key


def abx_threshold(trials: int = 16, alpha: float = 0.02) -> int:
    """Smallest number correct with a one-sided binomial p below ``alpha``. The feature
    prompt's bar is 13 of 16 (p = 0.011); 12 of 16 would already pass at 5% (p = 0.038)."""
    for k in range(trials + 1):
        p = sum(math.comb(trials, j) for j in range(k, trials + 1)) / 2 ** trials
        if p < alpha:
            return k
    return trials + 1


def verdict(change_06: float, seed_floor: float, heard: bool | None = None) -> str:
    """The feature prompt's rule. ``change_06``: log-mel L1 of index 0.6 vs index 0 (same
    seed); ``seed_floor``: index 0 seed 1 vs seed 0."""
    if change_06 < seed_floor:
        return "inert"  # default 0, skip the stage, grey the slider
    if heard is None:
        return "changes output: needs the ABX"
    return "heard" if heard else "not heard"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def report_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")


def replace(params, **kw):
    return dataclasses.replace(params, **kw)


INDEX_EFFECT_KEY = "index_effect"  # model config defaults / profile model_overrides


def save_index_verdict(models_dir: Path, profile_dir: Path, model: str, record: dict) -> None:
    """Store the verdict where the controller reads it: the model's config.json defaults and
    the profile's model_overrides.json (the committed copy). ``record`` holds ``verdict``,
    ``index`` (the file measured), ``change_06``, ``seed_floor`` and ``when``."""
    from vox.config.store import load_json, save_json

    path = Path(models_dir) / model / "config.json"
    data = load_json(path)
    data.setdefault("defaults", {})[INDEX_EFFECT_KEY] = record
    save_json(path, data)
    over = Path(profile_dir) / "model_overrides.json"
    overrides = load_json(over) if over.is_file() else {"schema": 1}
    overrides.setdefault("models", {}).setdefault(model, {})[INDEX_EFFECT_KEY] = record
    save_json(over, overrides)
