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


def rms_dbfs(audio: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(audio, dtype=np.float64))) + 1e-12))
