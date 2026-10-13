"""RMVPE network (deep U-Net over the mel plane + BiGRU), inference only.

Vendored because there is no canonical package (docs/research/11 §4). Layer layout and
parameter names follow rvc-webui ``infer/rmvpe.py`` (MIT) so ``rmvpe.pt`` loads unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

N_MELS = 128
N_CLASS = 360


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.shortcut = (nn.Conv2d(in_channels, out_channels, 1)
                         if in_channels != out_channels else None)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x) + (x if self.shortcut is None else self.shortcut(x))


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, n_blocks: int = 1,
                 momentum: float = 0.01) -> None:
        super().__init__()
        self.conv = nn.ModuleList(
            ConvBlockRes(in_channels if i == 0 else out_channels, out_channels, momentum)
            for i in range(n_blocks)
        )
        self.pool = nn.AvgPool2d(kernel_size=kernel_size) if kernel_size is not None else None

    def forward(self, x: Tensor):
        for conv in self.conv:
            x = conv(x)
        return x if self.pool is None else (x, self.pool(x))


class Encoder(nn.Module):
    def __init__(self, in_channels: int, n_encoders: int, kernel_size, n_blocks: int,
                 out_channels: int = 16, momentum: float = 0.01) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks,
                                               momentum))
            in_channels, out_channels = out_channels, out_channels * 2
        self.out_channel = out_channels

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        skips: list[Tensor] = []
        x = self.bn(x)
        for layer in self.layers:
            skip, x = layer(x)
            skips.append(skip)
        return x, skips


class Intermediate(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_inters: int, n_blocks: int,
                 momentum: float = 0.01) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            ResEncoderBlock(in_channels if i == 0 else out_channels, out_channels, None,
                            n_blocks, momentum)
            for i in range(n_inters)
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride, n_blocks: int = 1,
                 momentum: float = 0.01) -> None:
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride, 1,
                               output_padding=out_padding, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList(
            ConvBlockRes(out_channels * 2 if i == 0 else out_channels, out_channels, momentum)
            for i in range(n_blocks)
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = torch.cat((self.conv1(x), skip), dim=1)
        for conv in self.conv2:
            x = conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels: int, n_decoders: int, stride, n_blocks: int,
                 momentum: float = 0.01) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            self.layers.append(ResDecoderBlock(in_channels, in_channels // 2, stride, n_blocks,
                                               momentum))
            in_channels //= 2

    def forward(self, x: Tensor, skips: list[Tensor]) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, skips[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks: int, en_de_layers: int = 5, inter_layers: int = 4,
                 in_channels: int = 1, en_out_channels: int = 16) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, en_de_layers, kernel_size, n_blocks,
                               en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2,
                                         self.encoder.out_channel, inter_layers, n_blocks)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x: Tensor) -> Tensor:
        x, skips = self.encoder(x)
        return self.decoder(self.intermediate(x), skips)


class BiGRU(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, num_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers,
                          batch_first=True, bidirectional=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.gru(x)[0]


class E2E(nn.Module):
    """``mel [b, 128, frames]`` -> pitch salience ``[b, frames, 360]`` (20-cent bins)."""

    def __init__(self, n_blocks: int = 4, n_gru: int = 1, kernel_size=(2, 2)) -> None:
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks)
        self.cnn = nn.Conv2d(16, 3, 3, padding=1)
        self.fc = nn.Sequential(
            BiGRU(3 * N_MELS, 256, n_gru),
            nn.Linear(512, N_CLASS),
            nn.Dropout(0.25),
            nn.Sigmoid(),
        )

    def forward(self, mel: Tensor) -> Tensor:
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        return self.fc(x)
