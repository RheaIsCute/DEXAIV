"""NSF-HiFiGAN decoder: sine excitation from continuous F0 injected into HiFi-GAN upsampling.

Reference: rvc-webui ``infer/module/models.py`` (SineGen, SourceModuleHnNSF, GeneratorNSF,
Generator) and ``modules.py`` (ResBlock1/2), MIT. Weight norm is fused at load time.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

LRELU_SLOPE = 0.1


def _padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                      padding=_padding(kernel_size, d))
            for d in dilation
        )
        self.convs2 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, 1, padding=_padding(kernel_size))
            for _ in dilation
        )

    def forward(self, x: Tensor) -> Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(F.leaky_relu(x, LRELU_SLOPE))
            xt = c2(F.leaky_relu(xt, LRELU_SLOPE))
            x = xt + x
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3)) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                      padding=_padding(kernel_size, d))
            for d in dilation
        )

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = conv(F.leaky_relu(x, LRELU_SLOPE)) + x
        return x


class SineGen(nn.Module):
    """Frame-rate F0 (Hz, 0 = unvoiced) -> sample-rate sine + noise excitation.

    Always FP32: the per-frame phase accumulation loses pitch accuracy in half precision
    (docs/research/06 §5). Phase restarts at zero on every call, exactly like the
    references; continuity across hops is SOLA's job.
    """

    def __init__(self, samp_rate: int, upp: int, harmonic_num: int = 0, sine_amp: float = 0.1,
                 noise_std: float = 0.003) -> None:
        super().__init__()
        self.sampling_rate = samp_rate
        self.upp = upp
        self.dim = harmonic_num + 1
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.register_buffer("_ramp", torch.arange(1, upp + 1, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("_harmonics", torch.arange(1, self.dim + 1, dtype=torch.float32)
                             .reshape(1, 1, -1), persistent=False)

    def noise_shape(self, frames: int) -> tuple[int, int, int]:
        """Shape of the ``noise`` tensor ``forward`` consumes for ``frames`` F0 frames."""
        return (1, frames * self.upp, self.dim)

    @torch.no_grad()
    def forward(self, f0: Tensor, noise: Tensor | None = None) -> Tensor:
        """``noise``: pre-drawn standard-normal samples of ``noise_shape(t)``. Passing them
        in keeps every random draw outside a captured CUDA graph (the hot path pre-fills
        the buffer each hop); ``None`` draws fresh, the upstream behaviour."""
        f0 = f0.float().unsqueeze(-1)  # [b, t, 1]
        # Phase advance within each frame, then carry the wrapped end-of-frame phase forward.
        rad = f0 / self.sampling_rate * self._ramp  # [b, t, upp]
        frame_end = torch.fmod(rad[..., -1:] + 0.5, 1.0) - 0.5
        carried = frame_end.cumsum(dim=1).fmod(1.0)
        rad = rad + F.pad(carried, (0, 0, 1, -1))
        rad = rad.reshape(f0.shape[0], -1, 1) * self._harmonics
        if self.dim > 1:
            # Upstream draws a random start phase per harmonic; the fundamental stays at 0.
            initial_phase = torch.rand(1, 1, self.dim, device=f0.device)
            initial_phase[..., 0] = 0
            rad = rad + initial_phase
        sine = torch.sin(2 * math.pi * rad) * self.sine_amp

        uv = (f0 > 0).to(sine.dtype)
        uv = uv.expand(-1, -1, self.upp).reshape(f0.shape[0], -1, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        if noise is None:
            noise = torch.randn_like(sine)
        return sine * uv + noise_amp * noise


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate: int, upp: int, harmonic_num: int = 0) -> None:
        super().__init__()
        self.l_sin_gen = SineGen(sampling_rate, upp, harmonic_num)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def forward(self, f0: Tensor, noise: Tensor | None = None) -> Tensor:
        with torch.autocast(f0.device.type, enabled=False):
            sine = self.l_sin_gen(f0, noise)
            merged = torch.tanh(self.l_linear(sine.to(self.l_linear.weight.dtype)))
        return merged.transpose(1, 2)  # [b, 1, t * upp]


class _GeneratorBase(nn.Module):
    """Shared HiFi-GAN trunk: conv_pre -> (upsample -> resblocks) x N -> conv_post."""

    def __init__(self, initial_channel: int, resblock: str, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates, upsample_initial_channel: int,
                 upsample_kernel_sizes, gin_channels: int) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upp = math.prod(upsample_rates)
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        block = ResBlock1 if str(resblock) == "1" else ResBlock2
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        channels = upsample_initial_channel
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            channels = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(nn.ConvTranspose1d(channels * 2, channels, kernel, rate,
                                               padding=(kernel - rate) // 2))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(block(channels, k, d))
        self.conv_post = nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False)
        if gin_channels:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def _resblocks(self, x: Tensor, stage: int) -> Tensor:
        base = stage * self.num_kernels
        total = self.resblocks[base](x)
        for j in range(1, self.num_kernels):
            total = total + self.resblocks[base + j](x)
        return total / self.num_kernels

    def _finish(self, x: Tensor) -> Tensor:
        # Default slope (0.01) here, unlike the 0.1 used inside the trunk: upstream behaviour.
        return torch.tanh(self.conv_post(F.leaky_relu(x)))


class Generator(_GeneratorBase):
    """Plain HiFi-GAN for non-F0 (``*_nono``) models."""

    def forward(self, x: Tensor, g: Tensor | None = None) -> Tensor:
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = self.ups[i](F.leaky_relu(x, LRELU_SLOPE))
            x = self._resblocks(x, i)
        return self._finish(x)


class GeneratorNSF(_GeneratorBase):
    def __init__(self, initial_channel: int, resblock: str, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates, upsample_initial_channel: int,
                 upsample_kernel_sizes, gin_channels: int, sr: int) -> None:
        super().__init__(initial_channel, resblock, resblock_kernel_sizes,
                         resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
                         upsample_kernel_sizes, gin_channels)
        self.m_source = SourceModuleHnNSF(sr, self.upp)
        self.noise_convs = nn.ModuleList()
        for i in range(len(upsample_rates)):
            channels = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(nn.Conv1d(1, channels, kernel_size=stride * 2,
                                                  stride=stride, padding=stride // 2))
            else:
                self.noise_convs.append(nn.Conv1d(1, channels, kernel_size=1))

    def source(self, f0: Tensor, noise: Tensor | None = None) -> Tensor:
        """The excitation signal; kept separate so a decoder backend can reuse it (P5)."""
        return self.m_source(f0, noise)

    def forward(self, x: Tensor, f0: Tensor, g: Tensor | None = None,
                source: Tensor | None = None, source_noise: Tensor | None = None) -> Tensor:
        if source is None:
            source = self.source(f0, source_noise)
        source = source.to(x.dtype)
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = self.ups[i](F.leaky_relu(x, LRELU_SLOPE))
            x = x + self.noise_convs[i](source)
            x = self._resblocks(x, i)
        return self._finish(x)
