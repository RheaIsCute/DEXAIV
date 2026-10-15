"""Objective audio comparisons for the golden tests and the benchmark (docs/research/10 §6).

Triage, not truth: these catch regressions and block-boundary damage. They cannot rate
voice quality. The synthesizer draws fresh noise on every run, so two renders of the same
settings are never sample-identical; compare against ``the noise floor`` (two full-file
renders with different seeds), not against zero.
"""

from __future__ import annotations

import numpy as np
import torch


def log_mel(audio: np.ndarray, sr: int, n_mels: int = 80, hop_ms: float = 10.0,
            win_ms: float = 40.0) -> torch.Tensor:
    """``[frames, n_mels]`` natural-log mel spectrogram on the CPU."""
    import torchaudio

    hop = int(sr * hop_ms / 1000)
    win = int(sr * win_ms / 1000)
    n_fft = 1 << (win - 1).bit_length()
    x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    spec = torch.stft(x, n_fft, hop_length=hop, win_length=win,
                      window=torch.hann_window(win), center=True, return_complex=True).abs()
    fbank = torchaudio.functional.melscale_fbanks(
        n_fft // 2 + 1, 50.0, min(sr / 2, 12000.0), n_mels, sr, norm="slaney",
        mel_scale="htk")
    return torch.log(torch.clamp(spec.T @ fbank, min=1e-5))


def _shift(audio: np.ndarray, lag: int) -> np.ndarray:
    """Delay ``audio`` by ``lag`` samples (negative advances), keeping the length."""
    if lag == 0:
        return audio
    out = np.zeros_like(audio)
    if lag > 0:
        out[lag:] = audio[:-lag]
    else:
        out[:lag] = audio[-lag:]
    return out


def log_mel_distance(ref: np.ndarray, test: np.ndarray, sr: int, lag_search_ms: float = 0.0,
                     step_ms: float = 2.5) -> tuple[float, int]:
    """Mean absolute log-mel difference, minimised over a small lag search.

    Returns ``(distance, best_lag_samples)``. The lag search absorbs the few milliseconds
    of SOLA alignment jitter in a streamed render.
    """
    n = min(len(ref), len(test))
    ref, test = ref[:n], test[:n]
    ref_mel = log_mel(ref, sr)
    step = max(1, int(sr * step_ms / 1000))
    span = int(sr * lag_search_ms / 1000) // step
    best = (float("inf"), 0)
    for k in range(-span, span + 1):
        lag = k * step
        dist = (log_mel(_shift(test, lag), sr) - ref_mel).abs().mean().item()
        best = min(best, (dist, lag))
    return best


def compare(ref: np.ndarray, test: np.ndarray, sr: int, lag_search_ms: float = 0.0,
            start_s: float = 0.0) -> dict[str, float]:
    """Log-mel comparison split into speech and quiet frames, from ``start_s`` onward.

    ``speech`` frames are those above the reference median energy. Quiet frames sit near
    the log floor, where inaudible differences produce large distances, so ``l1_speech`` is
    the number that tracks what is heard; ``bias`` (mean signed difference, nepers) shows a
    level offset. ``start_s`` skips a stream's start-up, when its history is still zeros.
    """
    n = min(len(ref), len(test))
    begin = int(start_s * sr)
    ref, test = ref[begin:n], test[begin:n]
    l1, lag = log_mel_distance(ref, test, sr, lag_search_ms)
    ref_mel = log_mel(ref, sr)
    diff = log_mel(_shift(test, lag), sr) - ref_mel
    energy = ref_mel.mean(dim=1)
    speech = energy > energy.median()
    return {"l1": l1, "l1_speech": diff[speech].abs().mean().item(),
            "l1_quiet": diff[~speech].abs().mean().item(), "bias": diff.mean().item(),
            "bias_speech": diff[speech].mean().item(), "lag_ms": 1000.0 * lag / sr}


def boundary_click_ratio(audio: np.ndarray, block: int, half_width: int = 8) -> float:
    """Sample-level roughness at hop boundaries relative to mid-hop positions.

    A hard splice shows up as a spike in the second difference at multiples of ``block``.
    ~1.0 means boundaries are indistinguishable from the rest of the signal.
    """
    accel = np.abs(np.diff(audio.astype(np.float64), n=2))
    hops = len(accel) // block
    if hops < 4:
        return float("nan")

    def local(center: np.ndarray) -> np.ndarray:
        idx = center[:, None] + np.arange(-half_width, half_width)[None, :]
        return accel[np.clip(idx, 0, len(accel) - 1)].mean(axis=1)

    boundaries = np.arange(1, hops) * block
    edge = local(boundaries)
    middle = local(boundaries - block // 2)
    active = middle > np.percentile(middle, 25)  # ignore silence, where both are ~0
    if not active.any():
        return float("nan")
    return float(np.median(edge[active]) / np.median(middle[active]))


def transition_click_ratio(audio: np.ndarray, positions: np.ndarray, block: int,
                           half_width: int = 8) -> float:
    """``boundary_click_ratio`` restricted to given sample positions (gate transitions).

    Roughness (|second difference|) in a +-``half_width`` window around each position,
    relative to the roughness mid-hop in the active hops around it. ~1.0 = seamless.
    """
    accel = np.abs(np.diff(audio.astype(np.float64), n=2))
    positions = np.asarray(positions, dtype=np.int64)
    positions = positions[(positions > block) & (positions < len(accel) - block)]
    if positions.size == 0:
        return float("nan")

    def local(center: np.ndarray) -> np.ndarray:
        idx = center[:, None] + np.arange(-half_width, half_width)[None, :]
        return accel[np.clip(idx, 0, len(accel) - 1)].mean(axis=1)

    edge = local(positions)
    # reference: mid-hop roughness of the neighbouring hops (one before, one after)
    middle = np.maximum(local(positions - block // 2), local(positions + block // 2))
    middle = np.maximum(middle, 1e-7)
    return float(np.median(edge / middle))


def frame_rms_dbfs(audio: np.ndarray, sr: int, hop_ms: float = 10.0,
                   win_ms: float = 40.0) -> np.ndarray:
    """Per-frame RMS level in dBFS (centred windows), for pause / speech region masks."""
    hop, win = int(sr * hop_ms / 1000), int(sr * win_ms / 1000)
    padded = np.pad(audio.astype(np.float64), (win // 2, win // 2))
    csum = np.concatenate(([0.0], np.cumsum(padded * padded)))
    starts = np.arange(0, len(audio), hop)
    energy = (csum[np.minimum(starts + win, len(padded))] - csum[starts]) / win
    return 10.0 * np.log10(energy + 1e-12)


def rms_dbfs(audio: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(audio, dtype=np.float64))) + 1e-12))


# ------------------------------------------------------------- live-mic diagnostics
# The measures of the external audio report (docs/DEVIATIONS.md §7): voicing, HF/LF tilt,
# gate chatter and digital-zero cuts. Frames are 40 ms windows at a 10 ms hop throughout,
# so a 100 Hz F0 track lines up with them frame for frame.


def active_frames(audio: np.ndarray, sr: int, below_p95_db: float = 30.0,
                  floor_dbfs: float = -60.0) -> np.ndarray:
    """10 ms frames carrying signal: within ``below_p95_db`` of the clip's loud frames
    and above ``floor_dbfs``. Digital silence is never active."""
    level = frame_rms_dbfs(audio, sr)
    return level >= max(floor_dbfs, float(np.percentile(level, 95)) - below_p95_db)


def band_ratio_db(audio: np.ndarray, sr: int, high: tuple[float, float] = (3000.0, 12000.0),
                  low: tuple[float, float] = (100.0, 1000.0)) -> np.ndarray:
    """Per-frame energy in ``high`` over energy in ``low``, dB (hiss / breathiness tilt)."""
    hop, win = sr // 100, sr // 25
    n_fft = 1 << (win - 1).bit_length()
    x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    spec = torch.stft(x, n_fft, hop_length=hop, win_length=win, window=torch.hann_window(win),
                      center=True, return_complex=True).abs().pow(2).numpy()
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    def band(lo: float, hi: float) -> np.ndarray:
        return spec[(freqs >= lo) & (freqs < hi)].sum(axis=0)

    return 10.0 * np.log10((band(*high) + 1e-12) / (band(*low) + 1e-12))


def hf_lf_p90(audio: np.ndarray, sr: int, mask: np.ndarray | None = None) -> float:
    """90th percentile of ``band_ratio_db`` over active frames (the report's HF/LF ratio)."""
    ratio = band_ratio_db(audio, sr)
    mask = active_frames(audio, sr) if mask is None else mask
    n = min(len(ratio), len(mask))
    picked = ratio[:n][mask[:n]]
    return float(np.percentile(picked, 90)) if picked.size else float("nan")


def voiced_ratio(f0_100hz: np.ndarray, mask: np.ndarray) -> float:
    """Share of active 10 ms frames with a pitch (``f0 > 0``)."""
    n = min(len(f0_100hz), len(mask))
    active = mask[:n]
    return float(np.mean(f0_100hz[:n][active] > 0)) if active.any() else float("nan")


def phrases(audio: np.ndarray, sr: int, below_p95_db: float = 18.0, merge_ms: float = 250.0,
            min_ms: float = 100.0) -> list[tuple[int, int]]:
    """Sample spans where the *input* is speaking: frames within ``below_p95_db`` of the
    loud frames, gaps shorter than ``merge_ms`` bridged, spans shorter than ``min_ms``
    dropped. 18 dB keeps the talker's soft syllables and leaves out what a microphone hears
    20 dB and more below them (room tone, other people, speakers bleeding in)."""
    mask = active_frames(audio, sr, below_p95_db=below_p95_db, floor_dbfs=-80.0)
    hop = sr // 100
    spans, start = [], None
    for i, on in enumerate(np.append(mask, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            spans.append([start, i])
            start = None
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and (s - merged[-1][1]) * 10 < merge_ms:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s * hop, e * hop) for s, e in merged if (e - s) * 10 >= min_ms]


def zero_runs(audio: np.ndarray, min_samples: int) -> list[tuple[int, int]]:
    """Spans of exact digital zero at least ``min_samples`` long (gated hops, underruns)."""
    zero = np.append(np.append(False, audio == 0.0), False).astype(np.int8)
    edges = np.flatnonzero(np.diff(zero))
    starts, ends = edges[0::2], edges[1::2]
    keep = ends - starts >= min_samples
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))
