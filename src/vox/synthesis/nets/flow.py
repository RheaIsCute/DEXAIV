"""Residual coupling flow, inverse direction only (prior latent -> decoder latent).

Reference: rvc-webui ``infer/module/modules.py`` (WN, ResidualCouplingLayer) and
``models.py`` (ResidualCouplingBlock), MIT. Weight norm is fused into plain weights at load
time (see ``synthesizer.fuse_weight_norm``), so these are ordinary convolutions.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class WN(nn.Module):
    """WaveNet-style gated dilated conv stack with global (speaker) conditioning."""

    def __init__(self, hidden_channels: int, kernel_size: int, dilation_rate: int,
                 n_layers: int, gin_channels: int = 0) -> None:
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        if gin_channels:
            self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                                            dilation=dilation, padding=padding))
            out = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(nn.Conv1d(hidden_channels, out, 1))

    def forward(self, x: Tensor, x_mask: Tensor, g: Tensor | None = None) -> Tensor:
        h = self.hidden_channels
        output = torch.zeros_like(x)
        if g is not None:
            g = self.cond_layer(g)
        for i, (in_layer, res_skip_layer) in enumerate(zip(self.in_layers, self.res_skip_layers)):
            acts = in_layer(x)
            if g is not None:
                acts = acts + g[:, i * 2 * h:(i + 1) * 2 * h, :]
            acts = torch.tanh(acts[:, :h, :]) * torch.sigmoid(acts[:, h:, :])
            res_skip = res_skip_layer(acts)
            if i < self.n_layers - 1:
                x = (x + res_skip[:, :h, :]) * x_mask
                output = output + res_skip[:, h:, :]
            else:
                output = output + res_skip
        return output * x_mask


class ResidualCouplingLayer(nn.Module):
    """Mean-only affine coupling; only the inverse is needed at inference."""

    def __init__(self, channels: int, hidden_channels: int, kernel_size: int,
                 dilation_rate: int, n_layers: int, gin_channels: int = 0) -> None:
        super().__init__()
        assert channels % 2 == 0
        self.half_channels = channels // 2
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels, 1)

    def reverse(self, x: Tensor, x_mask: Tensor, g: Tensor | None) -> Tensor:
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.enc(self.pre(x0) * x_mask, x_mask, g=g)
        mean = self.post(h) * x_mask
        return torch.cat([x0, (x1 - mean) * x_mask], 1)


class ResidualCouplingBlock(nn.Module):
    """Four coupling layers, each followed by a channel flip.

    ``flows`` keeps the upstream layout (layers at even indices, parameterless flips at odd
    ones) so checkpoint keys ``flow.flows.{0,2,4,6}.*`` load unchanged.
    """

    def __init__(self, channels: int, hidden_channels: int, kernel_size: int,
                 dilation_rate: int, n_layers: int, n_flows: int = 4,
                 gin_channels: int = 0) -> None:
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size,
                                                    dilation_rate, n_layers, gin_channels))
            self.flows.append(nn.Identity())

    def reverse(self, x: Tensor, x_mask: Tensor, g: Tensor | None = None) -> Tensor:
        for layer in reversed(self.flows[::2]):
            x = layer.reverse(torch.flip(x, [1]), x_mask, g)
        return x
