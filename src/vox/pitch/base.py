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


def clamp_voicing(f0: Tensor, f0_min: float, f0_max: float) -> Tensor:
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


def postprocess_f0(f0: Tensor, ratio: Tensor | float, f0_min: float = COARSE_F0_MIN,
                   f0_max: float = COARSE_F0_MAX, interp: bool = False
                   ) -> tuple[Tensor, Tensor]:
    """Raw extractor output -> ``(coarse int64, continuous Hz)`` after transposition.

    ``ratio`` is ``2 ** (f0_up_key / 12)``; pass a 0-dim device tensor on the hot path so
    changing the key never changes the op sequence.
    """
    if f0_min > COARSE_F0_MIN or f0_max < COARSE_F0_MAX:
        f0 = clamp_voicing(f0, f0_min, f0_max)
    if interp:
        f0 = interp_unvoiced(f0)
    pitchf = f0 * ratio
    return f0_to_coarse(pitchf), pitchf


def semitones_to_ratio(f0_up_key: float) -> float:
    return 2.0 ** (f0_up_key / 12.0)
