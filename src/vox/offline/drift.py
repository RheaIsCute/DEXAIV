"""Feature and F0 drift of the sliding window (docs/DEVIATIONS.md §13; report 18 §C7-C8).

Every hop re-embeds the whole window, so the ContentVec feature (and the cached F0) of one
instant changes as the window slides: the frame is first seen at the window's edge (age 0,
no right context) and then again with 20, 40, 80 ... ms of audio to its right. Nothing
published measures that drift; this probe does, against the whole-file features and F0 of
the same frames, which is what the commit-once design (report 18 §F9) has to approximate
at its chosen lookahead.

``measure_drift`` runs the stream simulator with a per-hop probe (``on_hop``) that copies
the frames of the requested ages out of the pipeline's buffers; the whole-file reference
is the embedder and the extractor over the entire take, on the same 50 Hz grid parity as
the window (the window's grid is offset by one 10 ms frame when its start frame is odd).
Control path: it syncs every hop and is not a timing run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from vox.audio.resample import resample_array
from vox.engine.pipeline import upsample_features
from vox.offline.metrics import active_frames
from vox.offline.simulate import Engine, StreamSimulator
from vox.pitch.base import postprocess_f0, semitones_to_ratio

SR16 = 16000
FRAME = 160
FLIP_CENTS = 600.0  # half an octave: an octave error, not a tracking wobble


@dataclass
class DriftResult:
    ages_ms: tuple[int, ...]
    feature: dict[int, dict[str, float]] = field(default_factory=dict)  # by age
    f0: dict[int, dict[str, float]] = field(default_factory=dict)  # nominal frame pairing
    f0_lagged: dict[int, dict[str, float]] = field(default_factory=dict)  # trim-corrected
    frames: int = 0
    trim_tail: int = 0

    def table(self) -> str:
        head = (f"{'age ms':>6} {'cos dist mean':>14} {'cos p95':>8} {'n':>6} | "
                f"{'cents rmse':>10} {'p95':>6} {'flips':>7} {'voicing':>8} | "
                f"{'lagged rmse':>11} {'p95':>6}")
        rows = [head]
        for age in self.ages_ms:
            f, p, q = self.feature[age], self.f0[age], self.f0_lagged[age]
            rows.append(f"{age:>6d} {f['cos_mean']:>14.4f} {f['cos_p95']:>8.4f} {f['n']:>6d} | "
                        f"{p['cents_rmse']:>10.1f} {p['cents_p95']:>6.1f} {p['flip_rate']:>7.4f} "
                        f"{p['voicing_agreement']:>8.3f} | {q['cents_rmse']:>11.1f} "
                        f"{q['cents_p95']:>6.1f}")
        return "\n".join(rows)


def _reference_features(engine: Engine, audio_16k: np.ndarray, n_frames: int
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """100 Hz whole-file features on the even and the odd 50 Hz grid: ``[2, n_frames, C]``
    where ``[p][a]`` is frame ``a`` as a window whose start frame has parity ``p`` sees it."""
    embedder = engine.pipeline.embedder
    device = engine.device
    out = torch.zeros(2, n_frames, embedder.dim, device=device)
    wave = torch.from_numpy(audio_16k).to(device)
    with torch.inference_mode():
        for parity in (0, 1):
            feats50 = embedder.embed(wave[parity * FRAME:].unsqueeze(0))
            buf = torch.zeros(1, n_frames, embedder.dim, device=device)
            upsample_features(feats50, buf)
            if parity == 0:
                out[0] = buf[0]
            else:
                out[1, 1:] = buf[0, :-1]
                out[1, 0] = buf[0, 0]
    return out


def _reference_f0(engine: Engine, audio_16k: np.ndarray, n_frames: int) -> torch.Tensor:
    """Whole-file continuous F0 at 100 Hz after the engine's own post-processing
    (transposition, bounds, unvoiced contract, ceiling): ``[n_frames]``."""
    pipeline, params = engine.pipeline, engine.params
    wave = torch.from_numpy(audio_16k).to(engine.device)
    with torch.inference_mode():
        f0 = pipeline.extractor.extract(wave)
        _, cont = postprocess_f0(f0, semitones_to_ratio(params.f0_up_key), params.f0_min_hz,
                                 params.f0_max_hz, params.f0_unvoiced == "interp",
                                 round(params.f0_bridge_ms / 10.0), params.f0_ceiling_hz)
    out = torch.zeros(n_frames, device=engine.device)
    n = min(n_frames, cont.shape[0])
    out[:n] = cont[:n]
    return out


def measure_drift(engine: Engine, audio: np.ndarray, ages_ms: tuple[int, ...] = (0, 20, 40, 80),
                  seed: int | None = 0, start_s: float = 1.5) -> DriftResult:
    """Drift by age of the frames the stream computes for ``audio`` (device rate), against
    the whole-file features and F0 of the same frames, over active input frames from
    ``start_s`` on. Ages are measured from the window's end (age 0 = the newest frame); with
    ``extra_right_ms`` = R the kept region's newest frame has age R.

    Feature drift: cosine distance per frame (mean, p95). F0: cents between the cached
    value (what the decoder sees) and the whole-file value of the nominally paired frame
    (RMSE and p95 over frames voiced in both, octave-flip rate at >= 600 cents, voicing
    agreement), and the same with the cache's known ``f0_trim_tail`` lag taken out
    (``f0_lagged``: the cache's frame ``a`` holds the tracker's frame ``a + 1 - trim``)."""
    plan, pipeline = engine.plan, engine.pipeline
    if pipeline.extractor is None:
        raise ValueError("drift needs a pitch-guided model")
    for age in ages_ms:
        if age % 10 or age // 10 >= plan.p_len:
            raise ValueError(f"age {age} ms must be a 10 ms multiple inside the window")
    audio_16k = resample_array(audio, plan.device_sr, SR16).astype(np.float32)
    n_frames = len(audio_16k) // FRAME
    ref_feats = _reference_features(engine, audio_16k, n_frames)
    ref_f0 = _reference_f0(engine, audio_16k, n_frames)
    active = active_frames(audio, plan.device_sr)
    trim = pipeline.pitch_cache.trim_tail
    shift, p_len = plan.shift_frames, plan.p_len
    keep: dict[int, list[tuple[int, int]]] = {age: [] for age in ages_ms}
    feat_rows: dict[int, list[torch.Tensor]] = {age: [] for age in ages_ms}
    f0_rows: dict[int, list[torch.Tensor]] = {age: [] for age in ages_ms}
    first_frame = int(start_s * 100)

    def probe(n: int) -> None:
        end = (n + 1) * shift  # absolute frame after the window's last frame
        start = end - p_len
        parity = start & 1
        for age in ages_ms:
            j = p_len - 1 - age // 10
            a = start + j
            if a < first_frame or a >= n_frames or a >= len(active) or not active[a]:
                continue
            keep[age].append((a, parity))
            feat_rows[age].append(pipeline.feats[0, j].clone())
            f0_rows[age].append(pipeline.pitch_cache.pitchf_view[0, j].clone())

    StreamSimulator(engine).run(audio, seed=seed, on_hop=probe)
    result = DriftResult(ages_ms=tuple(ages_ms), trim_tail=trim)
    for age in ages_ms:
        if not keep[age]:
            result.feature[age] = {"cos_mean": float("nan"), "cos_p95": float("nan"), "n": 0}
            empty = {"cents_rmse": float("nan"), "cents_p95": float("nan"),
                     "flip_rate": float("nan"), "voicing_agreement": float("nan"), "n": 0}
            result.f0[age] = dict(empty)
            result.f0_lagged[age] = dict(empty)
            continue
        idx = torch.tensor([a for a, _ in keep[age]], device=engine.device)
        par = torch.tensor([p for _, p in keep[age]], device=engine.device)
        stream = torch.stack(feat_rows[age])
        reference = ref_feats[par, idx]
        cos = 1.0 - torch.nn.functional.cosine_similarity(stream, reference, dim=1)
        cos_np = cos.cpu().numpy()
        result.feature[age] = {"cos_mean": float(cos_np.mean()),
                               "cos_p95": float(np.percentile(cos_np, 95)), "n": len(cos_np)}
        f0_stream = torch.stack(f0_rows[age]).cpu().numpy()
        for name, lag in (("f0", 0), ("f0_lagged", 1 - trim)):
            ref_idx = (idx + lag).clamp(0, n_frames - 1)
            f0_ref = ref_f0[ref_idx].cpu().numpy()
            both = (f0_stream > 0) & (f0_ref > 0)
            agree = float(np.mean((f0_stream > 0) == (f0_ref > 0)))
            if both.sum() == 0:
                stats = {"cents_rmse": float("nan"), "cents_p95": float("nan"),
                         "flip_rate": float("nan"), "voicing_agreement": agree, "n": 0}
            else:
                cents = 1200.0 * np.log2(f0_stream[both] / f0_ref[both])
                stats = {"cents_rmse": float(np.sqrt(np.mean(cents ** 2))),
                         "cents_p95": float(np.percentile(np.abs(cents), 95)),
                         "flip_rate": float(np.mean(np.abs(cents) >= FLIP_CENTS)),
                         "voicing_agreement": agree, "n": int(both.sum())}
            getattr(result, name)[age] = stats
    result.frames = max(len(v) for v in keep.values())
    return result
