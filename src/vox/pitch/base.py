"""The ``PitchExtractor`` interface and the shared, trained-in F0 post-processing.

Everything here is tensor-native and sync-free (no ``.item()``, no boolean-mask indexing)
so it can run inside the per-hop function and later inside a CUDA graph.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

# The coarse-pitch contract every RVC synthesizer was trained on (docs/research/02 §3.2).
COARSE_F0_MIN = 50.0
COARSE_F0_MAX = 1100.0
_MEL_MIN = 1127.0 * math.log(1.0 + COARSE_F0_MIN / 700.0)
_MEL_MAX = 1127.0 * math.log(1.0 + COARSE_F0_MAX / 700.0)


class PitchExtractor(Protocol):
    name: str
    graph_safe: bool  # may live inside a captured CUDA graph

    def load(self, device: torch.device, assets_dir: Path) -> None:
        """Control path only."""
        ...

    def tail_samples(self, block_16k: int) -> int:
        """How much trailing 16 kHz audio the extractor wants for a hop of ``block_16k``."""
        ...

    def extract(self, audio_16k: Tensor) -> Tensor:
        """``[n]`` samples -> ``[n // 160 + 1]`` F0 values in Hz at 100 Hz; 0 = unvoiced."""
        ...


def f0_to_coarse(f0: Tensor) -> Tensor:
    """Hz -> the 1..255 mel-scaled bucket fed to the pitch embedding (1 = unvoiced)."""
    mel = 1127.0 * torch.log1p(f0 / 700.0)
    scaled = (mel - _MEL_MIN) * (254.0 / (_MEL_MAX - _MEL_MIN)) + 1.0
    mel = torch.where(mel > 0, scaled, mel)
    return torch.round(mel.clamp(1.0, 255.0)).long()


def clamp_voicing(f0: Tensor, f0_min: Tensor | float, f0_max: Tensor | float) -> Tensor:
    """Treat estimates outside the calibrated source range as unvoiced.

    A frame far outside the speaker's own range is almost always an octave error; calling
    it unvoiced beats transposing a wrong pitch (docs/research/08 §2.4).
    """
    return torch.where((f0 >= f0_min) & (f0 <= f0_max), f0, torch.zeros_like(f0))


def interp_unvoiced(f0: Tensor) -> Tensor:
    """Linearly interpolate across ``f0 == 0`` gaps, holding the nearest value at the edges.

    Same result as ``np.interp(uv_idx, voiced_idx, f0[voiced])`` in current rvc-webui, but
    with static shapes and no host round-trip. All-unvoiced input is returned unchanged.
    """
    n = f0.shape[-1]
    idx = torch.arange(n, device=f0.device)
    voiced = f0 > 0
    prev = torch.cummax(torch.where(voiced, idx, -1), dim=-1).values
    nxt = torch.flip(
        torch.cummin(torch.flip(torch.where(voiced, idx, n), [-1]), dim=-1).values, [-1])
    f_prev = f0.gather(-1, prev.clamp(min=0))
    f_next = f0.gather(-1, nxt.clamp(max=n - 1))
    left = torch.where(prev >= 0, f_prev, f_next)
    right = torch.where(nxt < n, f_next, f_prev)
    weight = (idx - prev).to(f0.dtype) / (nxt - prev).clamp(min=1).to(f0.dtype)
    return left + (right - left) * weight.clamp(0.0, 1.0)


def bridge_gaps(f0: Tensor, max_frames: int, max_semitones: float = 4.0) -> Tensor:
    """Fill ``f0 == 0`` runs of at most ``max_frames`` that sit between two voiced frames no
    more than ``max_semitones`` apart, interpolating in log-F0: a tracker dropout inside a
    vowel, not a consonant (those border on a jump or a longer gap). Longer gaps and the
    edges are left unvoiced. Static shapes, no host round-trip (graph-safe)."""
    n = f0.shape[-1]
    idx = torch.arange(n, device=f0.device)
    voiced = f0 > 0
    prev = torch.cummax(torch.where(voiced, idx, -1), dim=-1).values
    nxt = torch.flip(
        torch.cummin(torch.flip(torch.where(voiced, idx, n), [-1]), dim=-1).values, [-1])
    log_prev = torch.log(f0.gather(-1, prev.clamp(min=0)).clamp(min=1.0))
    log_next = torch.log(f0.gather(-1, nxt.clamp(max=n - 1)).clamp(min=1.0))
    span = (nxt - prev).clamp(min=1)
    weight = (idx - prev).to(f0.dtype) / span.to(f0.dtype)
    interpolated = torch.exp(log_prev + (log_next - log_prev) * weight)
    close = (log_next - log_prev).abs() <= max_semitones * math.log(2.0) / 12.0
    fill = ~voiced & (prev >= 0) & (nxt < n) & (span - 1 <= max_frames) & close
    return torch.where(fill, interpolated, f0)


CEILING_KNEE_ST = 4.0  # the soft ceiling starts compressing this far below the ceiling


def soft_ceiling(pitchf: Tensor, ceiling: Tensor | float) -> Tensor:
    """Soft-knee F0 limit in semitone space: unchanged up to ``ceiling`` - 4 st, then
    ``knee * 2 ** (4 tanh(st / 4) / 12)`` (slope 1 at the knee, asymptotic to ``ceiling``).

    For singing through a speech-trained model: belted notes that the transposition pushes
    past what the model saw are bent down instead of cracking. It is a safety, not a fix:
    a note 2.5 st over the knee comes out ~0.3 st flat, 6 st over ~2.3 st flat (off-key).
    ``ceiling = inf`` is an exact no-op with the same op sequence (graph-safe); unvoiced
    frames (0 Hz) are never touched.
    """
    knee = ceiling * 2.0 ** (-CEILING_KNEE_ST / 12.0)
    over = 12.0 * torch.log2(pitchf.clamp(min=1e-3) / knee)
    bent = knee * torch.exp2(CEILING_KNEE_ST * torch.tanh(over.clamp(min=0.0) / CEILING_KNEE_ST)
                             / 12.0)
    return torch.where(pitchf > knee, bent, pitchf)


def postprocess_f0(f0: Tensor, ratio: Tensor | float, f0_min: Tensor | float = COARSE_F0_MIN,
                   f0_max: Tensor | float = COARSE_F0_MAX, interp: bool = False,
                   bridge_frames: int = 0, ceiling: Tensor | float = 0.0
                   ) -> tuple[Tensor, Tensor]:
    """Raw extractor output -> ``(coarse int64, continuous Hz)`` after transposition.

    ``ratio`` is ``2 ** (f0_up_key / 12)``; pass a 0-dim device tensor on the hot path so
    changing the key never changes the op sequence. The same goes for the voicing bounds:
    as tensors they are always applied (use 0 / inf to disable), as floats only when
    tighter than the coarse-pitch contract. ``bridge_frames`` > 0 fills short dropouts
    first (``bridge_gaps``). ``ceiling`` (Hz) bends transposed pitch above it softly
    (``soft_ceiling``); as a tensor it is always applied (inf = off), as a float only > 0.
    """
    tensors = isinstance(f0_min, Tensor) or isinstance(f0_max, Tensor)
    if tensors or f0_min > COARSE_F0_MIN or f0_max < COARSE_F0_MAX:
        f0 = clamp_voicing(f0, f0_min, f0_max)
    if bridge_frames > 0:
        f0 = bridge_gaps(f0, bridge_frames)
    if interp:
        f0 = interp_unvoiced(f0)
    pitchf = f0 * ratio
    if isinstance(ceiling, Tensor) or ceiling > 0:
        pitchf = soft_ceiling(pitchf, ceiling)
    return f0_to_coarse(pitchf), pitchf


def semitones_to_ratio(f0_up_key: float) -> float:
    return 2.0 ** (f0_up_key / 12.0)
