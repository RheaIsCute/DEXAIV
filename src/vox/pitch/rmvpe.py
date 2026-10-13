"""RMVPE pitch extractor: the V1 default (docs/research/05 §2.1).

Tensor-native end to end: mel frontend, network and the local-average-cents decode all stay
on the device, with no numpy round-trip and no data-dependent control flow. FP32 always
(the cents decode sits near FP16 granularity, docs/research/06 §5).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from vox.pitch._rmvpe_net import E2E, N_CLASS, N_MELS

SR = 16000
HOP = 160
N_FFT = 1024
TILE_FRAMES = 32  # the U-Net halves time five times
TILE_SAMPLES = TILE_FRAMES * HOP  # 5120
GUARD_SAMPLES = 800
CENTS_BASE = 1997.3794084376191
CENTS_PER_BIN = 20.0


class RmvpeExtractor:
    name = "rmvpe"
    graph_safe = True

    def __init__(self, threshold: float = 0.03) -> None:
        self.threshold = threshold
        self.model: E2E | None = None

    def load(self, device: torch.device, assets_dir: Path) -> None:
        import torchaudio

        path = Path(assets_dir) / "_predictors" / "rmvpe.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = E2E()
        model.load_state_dict(state)
        self.model = model.float().eval().requires_grad_(False).to(device)
        self.device = torch.device(device)
        # Same filterbank as librosa.filters.mel(htk=True): slaney-normalised HTK mels.
        fbank = torchaudio.functional.melscale_fbanks(
            n_freqs=N_FFT // 2 + 1, f_min=30.0, f_max=8000.0, n_mels=N_MELS,
            sample_rate=SR, norm="slaney", mel_scale="htk")
        self.mel_basis = fbank.T.contiguous().to(device)
        self.window = torch.hann_window(N_FFT, device=device)
        # Cents of each bin, zero-padded by 4 either side for the 9-bin local average.
        cents = CENTS_PER_BIN * torch.arange(N_CLASS, dtype=torch.float32) + CENTS_BASE
        self.cents = F.pad(cents, (4, 4)).to(device)
        self.offsets = torch.arange(9, device=device)

    def tail_samples(self, block_16k: int) -> int:
        """``block + 800`` rounded up to whole 32-frame tiles, minus one hop.

        With ``center=True`` that yields exactly ``32 n`` mel frames, so the tile padding in
        ``_salience`` is a no-op: 4960 samples (310 ms) for any hop up to ~270 ms.
        """
        return TILE_SAMPLES * ((block_16k + GUARD_SAMPLES - 1) // TILE_SAMPLES + 1) - HOP

    def _mel(self, audio: Tensor) -> Tensor:
        spec = torch.stft(audio, N_FFT, hop_length=HOP, win_length=N_FFT, window=self.window,
                          center=True, return_complex=True)
        return torch.log(torch.clamp(torch.matmul(self.mel_basis, spec.abs()), min=1e-5))

    def _salience(self, mel: Tensor) -> Tensor:
        n_frames = mel.shape[-1]
        n_pad = TILE_FRAMES * ((n_frames - 1) // TILE_FRAMES + 1) - n_frames
        if n_pad > 0:
            mel = F.pad(mel, (0, n_pad), mode="constant")
        return self.model(mel)[:, :n_frames]

    def _decode(self, salience: Tensor) -> Tensor:
        """Salience-weighted mean of the cents in the 9 bins around each frame's peak."""
        peak, center = salience.max(dim=-1)
        index = center.unsqueeze(-1) + self.offsets  # into the 4-padded axis
        local = F.pad(salience, (4, 4)).gather(-1, index)
        cents = (local * self.cents[index]).sum(-1) / local.sum(-1)
        f0 = 10.0 * torch.pow(2.0, cents / 1200.0)
        return torch.where(peak > self.threshold, f0, torch.zeros_like(f0))

    @torch.inference_mode()
    def extract(self, audio_16k: Tensor) -> Tensor:
        return self._decode(self._salience(self._mel(audio_16k.unsqueeze(0))))[0]
