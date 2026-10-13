"""Text-encoder transformer with windowed relative-position attention (VITS lineage).

Inference-only reimplementation. Parameter names match RVC checkpoints so ``state_dict``s
load unchanged (reference: rvc-webui ``infer/module/attentions.py``, MIT).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """Layer norm over the channel axis of a ``[B, C, T]`` tensor."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        x = F.layer_norm(x.transpose(1, -1), (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class MultiHeadAttention(nn.Module):
    """Self-attention with relative-position keys/values clipped to ``±window_size`` frames."""

    def __init__(self, channels: int, n_heads: int, window_size: int = 10) -> None:
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.window_size = window_size
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)
        # Shared across heads (heads_share=True upstream).
        self.emb_rel_k = nn.Parameter(torch.zeros(1, window_size * 2 + 1, self.k_channels))
        self.emb_rel_v = nn.Parameter(torch.zeros(1, window_size * 2 + 1, self.k_channels))

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        b, d, t = x.shape
        split = (b, self.n_heads, self.k_channels, t)
        query = self.conv_q(x).view(split).transpose(2, 3)  # [b, h, t, d_k]
        key = self.conv_k(x).view(split).transpose(2, 3)
        value = self.conv_v(x).view(split).transpose(2, 3)

        query = query / math.sqrt(self.k_channels)
        scores = torch.matmul(query, key.transpose(-2, -1))
        rel_keys = self._relative_embeddings(self.emb_rel_k, t)
        rel_logits = torch.matmul(query, rel_keys.unsqueeze(0).transpose(-2, -1))
        scores = scores + self._relative_to_absolute(rel_logits)
        scores = scores.masked_fill(attn_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)

        output = torch.matmul(p_attn, value)
        rel_values = self._relative_embeddings(self.emb_rel_v, t)
        rel_weights = self._absolute_to_relative(p_attn)
        output = output + torch.matmul(rel_weights, rel_values.unsqueeze(0))
        output = output.transpose(2, 3).contiguous().view(b, d, t)
        return self.conv_o(output)

    def _relative_embeddings(self, emb: Tensor, length: int) -> Tensor:
        """Slice/pad the ``2w+1`` table to the ``2*length-1`` offsets of this sequence."""
        pad = max(length - (self.window_size + 1), 0)
        start = max((self.window_size + 1) - length, 0)
        if pad > 0:
            emb = F.pad(emb, [0, 0, pad, pad, 0, 0])
        return emb[:, start:start + 2 * length - 1]

    @staticmethod
    def _relative_to_absolute(x: Tensor) -> Tensor:
        """``[b, h, l, 2l-1]`` relative logits -> ``[b, h, l, l]`` absolute positions."""
        b, h, length, _ = x.shape
        x = F.pad(x, [0, 1])
        x = F.pad(x.view(b, h, length * 2 * length), [0, length - 1])
        return x.view(b, h, length + 1, 2 * length - 1)[:, :, :length, length - 1:]

    @staticmethod
    def _absolute_to_relative(x: Tensor) -> Tensor:
        """``[b, h, l, l]`` attention weights -> ``[b, h, l, 2l-1]`` relative offsets."""
        b, h, length, _ = x.shape
        x = F.pad(x, [0, length - 1])
        x = F.pad(x.view(b, h, length * length + length * (length - 1)), [length, 0])
        return x.view(b, h, length, 2 * length)[:, :, :, 1:]


class FFN(nn.Module):
    """Two same-padded convolutions with a ReLU in between."""

    def __init__(self, channels: int, filter_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, channels, kernel_size)
        self._pad = [(kernel_size - 1) // 2, kernel_size // 2] if kernel_size > 1 else None

    def _same(self, x: Tensor) -> Tensor:
        return x if self._pad is None else F.pad(x, self._pad)

    def forward(self, x: Tensor, x_mask: Tensor) -> Tensor:
        x = torch.relu(self.conv_1(self._same(x * x_mask)))
        x = self.conv_2(self._same(x * x_mask))
        return x * x_mask


class Encoder(nn.Module):
    def __init__(self, hidden_channels: int, filter_channels: int, n_heads: int, n_layers: int,
                 kernel_size: int = 1, window_size: int = 10) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(int(n_layers)):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, n_heads, window_size))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, filter_channels, kernel_size))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x: Tensor, x_mask: Tensor) -> Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for attn, norm_1, ffn, norm_2 in zip(
            self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2
        ):
            x = norm_1(x + attn(x, attn_mask))
            x = norm_2(x + ffn(x, x_mask))
        return x * x_mask


class TextEncoder(nn.Module):
    """Content features (+ coarse pitch embedding) -> prior mean / log-std at 100 Hz."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int,
                 filter_channels: int, n_heads: int, n_layers: int, kernel_size: int,
                 f0: bool = True) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.emb_phone = nn.Linear(in_channels, hidden_channels)
        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone: Tensor, pitch: Tensor | None, lengths: Tensor,
                skip_head: int = 0) -> tuple[Tensor, Tensor, Tensor]:
        x = self.emb_phone(phone)
        if pitch is not None:
            x = x + self.emb_pitch(pitch)
        x = F.leaky_relu(x * math.sqrt(self.hidden_channels), 0.1)
        x = x.transpose(1, -1)  # [b, h, t]
        steps = torch.arange(x.size(2), dtype=lengths.dtype, device=lengths.device)
        x_mask = (steps.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        if skip_head:
            x = x[:, :, skip_head:]
            x_mask = x_mask[:, :, skip_head:]
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask
