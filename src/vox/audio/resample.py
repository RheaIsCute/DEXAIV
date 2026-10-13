"""Resampling: two fixed-rate GPU resamplers for the hot path, soxr for files.

Hot path (docs/research/03 §6): ``device_sr -> 16 kHz`` incrementally, resampling only the
new block plus two 10 ms units of history and discarding the first 160 output samples
(the filter's left-edge transient); and ``model_sr -> device_sr`` for the decoded tail.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

ANALYSIS_SR = 16000


class IncrementalResampler16k:
    """Streaming ``device_sr -> 16 kHz`` with the overlap-and-discard trick.

    Each ``push`` returns ``block_16k + 160`` samples: the last 10 ms of the previous hop
    recomputed with proper right-hand context, followed by the new block. The caller
    overwrites the tail of its 16 kHz ring with them.
    """

    def __init__(self, device_sr: int, block: int, device: torch.device | str) -> None:
        import torchaudio

        self.zc = device_sr // 100
        if block % self.zc:
            raise ValueError(f"block {block} is not a multiple of {self.zc} (10 ms)")
        self.block = block
        self.out_len = 160 * (block // self.zc + 1)
        self._resample = torchaudio.transforms.Resample(
            device_sr, ANALYSIS_SR, dtype=torch.float32).to(device)
        self._window = torch.zeros(2 * self.zc + block, device=device)
        self._stash = torch.zeros(2 * self.zc, device=device)

    def push(self, block: Tensor) -> Tensor:
        stash = 2 * self.zc
        self._stash.copy_(self._window[-stash:])
        self._window[:stash].copy_(self._stash)
        self._window[stash:].copy_(block)
        return self._resample(self._window)[160:]

    def reset(self) -> None:
        self._window.zero_()


class OutputResampler:
    """``model_sr -> device_sr`` for the decoded tail. Identity when the rates match."""

    def __init__(self, model_sr: int, device_sr: int, device: torch.device | str) -> None:
        self.identity = model_sr == device_sr
        self._resample = None
        if not self.identity:
            import torchaudio

            self._resample = torchaudio.transforms.Resample(
                model_sr, device_sr, dtype=torch.float32).to(device)

    def run(self, audio: Tensor) -> Tensor:
        return audio if self.identity else self._resample(audio)


# ------------------------------------------------------------------ control path (files)


def resample_array(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32, copy=False)
    import soxr

    return soxr.resample(audio, sr_in, sr_out, quality="VHQ").astype(np.float32)


def load_audio(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Read a file as mono float32, optionally resampled to ``sr``."""
    import soundfile as sf

    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr is not None and sr != file_sr:
        return resample_array(audio, file_sr, sr), sr
    return audio, file_sr


def save_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")
