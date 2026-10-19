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


# ------------------------------------------------------------ seam-rate modulation prominence
# Report 18 §E11 / brief 16: a streamed render joins a render every hop, so whatever the two
# renders disagree on shows up as an amplitude modulation at the hop rate (50 Hz at a 20 ms
# hop: the ear's roughness band). Log-mel L1 averages it away inside its 25 ms frames; the
# envelope spectrum within critical bands sees it directly.

SEAM_BAND_LOW_HZ = 200.0
SEAM_BANDS = 17  # 1/3-octave centres 200 Hz .. ~8 kHz (fewer when sr / 2 is lower)
SEAM_ENV_SR = 1000  # envelope sample rate


def third_octave_bands(sr: int, low_hz: float = SEAM_BAND_LOW_HZ, count: int = SEAM_BANDS
                       ) -> list[tuple[float, float]]:
    """``(lo, hi)`` edges of 1/3-octave bands from ``low_hz`` up, below 0.45 x sr."""
    edges = []
    for k in range(count):
        centre = low_hz * 2.0 ** (k / 3.0)
        lo, hi = centre / 2.0 ** (1.0 / 6.0), centre * 2.0 ** (1.0 / 6.0)
        if hi >= 0.45 * sr:
            break
        edges.append((lo, hi))
    return edges


def band_envelopes(audio: np.ndarray, sr: int, bands: list[tuple[float, float]] | None = None,
                   env_sr: int = SEAM_ENV_SR) -> np.ndarray:
    """``[bands, samples @ env_sr]`` Hilbert envelopes of the 1/3-octave band signals."""
    from fractions import Fraction

    from scipy import signal

    bands = bands or third_octave_bands(sr)
    x = np.ascontiguousarray(audio, dtype=np.float64)
    ratio = Fraction(env_sr, sr)
    out = []
    for lo, hi in bands:
        sos = signal.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
        y = signal.sosfiltfilt(sos, x)
        env = np.abs(signal.hilbert(y))
        out.append(signal.resample_poly(env, ratio.numerator, ratio.denominator))
    return np.stack(out)


def seam_prominence(audio: np.ndarray, sr: int, hop_ms: float, block_ms: float = 400.0,
                    harmonics: int = 3, env_sr: int = SEAM_ENV_SR,
                    envelopes: np.ndarray | None = None) -> dict[str, float]:
    """Seam-rate modulation prominence, dB: the envelope-spectrum peak at the hop rate
    (and its first harmonics) over the neighbouring modulation floor, per 1/3-octave band,
    per ``block_ms`` block (Hann, half overlap), energy-weighted over bands and averaged
    over speech blocks (block energy above the median).

    ``seam_db`` is the fundamental (50 Hz at hop 20, 25 at 40, 16.7 at 60), ``seam_h2_db``
    / ``seam_h3_db`` its harmonics, ``seam_sum_db`` the three peaks' summed power over the
    summed floors; ``seam_depth_pct`` reads the fundamental as an AM depth (the quantity
    ``impose_am`` imposes: on speech the floor-relative dB compress, a 20 % dip reading
    ~+3 dB, while the depth tracks it ~linearly on top of the talker's own modulation).
    Only bands at least 2.5 x the hop rate wide take part (narrower filters remove the AM
    sidebands). The floor of each harmonic is the median envelope power within
    +-0.4 x hop rate of it, excluding the peak's own bins, so the talker's syllabic
    modulation (2-10 Hz) stays out of the comparison. Compare against the whole-file
    render of the same take: a value near the whole-file's means no seam modulation the
    metric can see. Calibrated on imposed AM (``impose_am``): docs/DEVIATIONS.md §13.
    """
    f_hop = 1000.0 / hop_ms
    env = band_envelopes(audio, sr, env_sr=env_sr) if envelopes is None else envelopes
    # A band narrower than ~2.5 x the hop rate cannot carry that rate's AM sidebands (its
    # filter removes them), so at 50 Hz the bands below ~500 Hz are left out.
    wide = [i for i, (lo, hi) in enumerate(third_octave_bands(sr)) if hi - lo >= 2.5 * f_hop]
    env = env[wide] if len(wide) < env.shape[0] else env
    block = max(16, int(round(env_sr * block_ms / 1000.0)))
    hop = block // 2
    nan = float("nan")
    if env.shape[1] < block:
        return {"seam_db": nan, "seam_h2_db": nan, "seam_h3_db": nan, "seam_sum_db": nan,
                "hop_rate_hz": f_hop, "blocks": 0}
    window = np.hanning(block)
    freqs = np.fft.rfftfreq(block, 1.0 / env_sr)
    df = freqs[1]
    starts = np.arange(0, env.shape[1] - block + 1, hop)
    peaks = np.full((len(starts), harmonics), nan)
    floors = np.full((len(starts), harmonics), nan)
    depth = np.full(len(starts), nan)  # AM depth at the hop rate, 0-1
    energy = np.zeros(len(starts))
    for i, s in enumerate(starts):
        seg = env[:, s:s + block]
        mean = seg.mean(axis=1)
        weight = mean ** 2  # band energy in the block
        energy[i] = weight.sum()
        if energy[i] <= 0:
            continue
        weight = weight / weight.sum()
        spectrum = np.fft.rfft((seg - mean[:, None]) * window, axis=1)
        spec = np.abs(spectrum) ** 2
        for k in range(1, harmonics + 1):
            f = k * f_hop
            near = np.abs(freqs - f) <= 1.5 * df
            around = (np.abs(freqs - f) <= 0.4 * f_hop) & ~near & (freqs > 0)
            if not near.any() or around.sum() < 3:
                continue
            peak = spec[:, near].max(axis=1)
            floor = np.median(spec[:, around], axis=1)
            peaks[i, k - 1] = float((weight * peak).sum())
            floors[i, k - 1] = float((weight * floor).sum())
            if k == 1:
                # a dip of depth d once per period is a cosine of amplitude d/2 x the mean
                # envelope: |X| = (d/2) x mean x sum(window) / 2
                amplitude = np.abs(spectrum[:, near]).max(axis=1)
                d = 4.0 * amplitude / (np.maximum(mean, 1e-12) * window.sum())
                depth[i] = float((weight * d).sum())
    speech = energy > np.median(energy)
    if not speech.any():
        speech = energy > 0
    p, f = peaks[speech], floors[speech]
    with np.errstate(divide="ignore", invalid="ignore"):
        per_block = 10.0 * np.log10(p / f)
        total = 10.0 * np.log10(np.nansum(p, axis=1) / np.nansum(f, axis=1))

    def mean_db(column: int) -> float:
        if column >= harmonics or np.isnan(per_block[:, column]).all():
            return nan
        return float(np.nanmean(per_block[:, column]))

    return {"seam_db": mean_db(0), "seam_h2_db": mean_db(1), "seam_h3_db": mean_db(2),
            "seam_sum_db": float(np.nanmean(total)) if np.isfinite(total).any() else nan,
            # the fundamental's AM depth in %, the same quantity ``impose_am`` imposes
            "seam_depth_pct": float(100.0 * np.nanmean(depth[speech]))
            if np.isfinite(depth[speech]).any() else nan,
            "hop_rate_hz": f_hop, "blocks": int(speech.sum())}


def impose_am(audio: np.ndarray, sr: int, rate_hz: float, depth: float) -> np.ndarray:
    """``audio`` with a raised-cosine amplitude dip of ``depth`` (0-1) once per period of
    ``rate_hz``: the gain runs between ``1 - depth`` and 1. The calibration signal for
    ``seam_prominence``."""
    t = np.arange(len(audio)) / sr
    gain = 1.0 - depth * 0.5 * (1.0 - np.cos(2 * np.pi * rate_hz * t))
    return (audio * gain).astype(np.float32)
