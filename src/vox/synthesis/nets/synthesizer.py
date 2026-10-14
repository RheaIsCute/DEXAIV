"""RVC v1/v2 synthesizer, inference only: TextEncoder -> noise latent -> flow^-1 -> decoder.

One class covers the four upstream variants (256/768-d input, with/without F0). Reference:
rvc-webui ``infer/module/models.py`` ``SynthesizerTrnMs{256,768}NSFsid[_nono].infer``, MIT.

Randomness: the latent noise and the NSF excitation noise may be passed in pre-drawn
(``noise`` / ``source_noise``). The streaming engine fills those buffers outside its CUDA
graphs each hop, so no random draw happens inside a captured region. Left ``None``, both
are drawn on the spot, exactly as upstream.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from vox.synthesis.nets.attentions import TextEncoder
from vox.synthesis.nets.flow import ResidualCouplingBlock
from vox.synthesis.nets.generator import Generator, GeneratorNSF

# Frames of flow context kept ahead of the decoded tail (the flow's receptive field).
FLOW_CONTEXT_FRAMES = 24
NOISE_SCALE = 0.66666


class Synthesizer(nn.Module):
    def __init__(self, config: Sequence, emb_dim: int, f0: bool) -> None:
        super().__init__()
        (_spec, _segment, inter_channels, hidden_channels, filter_channels, n_heads, n_layers,
         kernel_size, _p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
         upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim,
         gin_channels, sr) = config[:18]
        self.has_f0 = f0
        self.sr = int(sr)
        self.inter_channels = int(inter_channels)
        self.enc_p = TextEncoder(emb_dim, inter_channels, hidden_channels, filter_channels,
                                 n_heads, n_layers, kernel_size, f0=f0)
        decoder_args = (inter_channels, resblock, resblock_kernel_sizes,
                        resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
                        upsample_kernel_sizes, gin_channels)
        self.dec = GeneratorNSF(*decoder_args, sr=self.sr) if f0 else Generator(*decoder_args)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3,
                                          gin_channels=gin_channels)
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)
        self.upp = self.dec.upp

    @staticmethod
    def _flow_head(skip_head: int, return_length: int | None, dec_context: int) -> int:
        if return_length is None:
            return 0
        return max(skip_head - dec_context - FLOW_CONTEXT_FRAMES, 0)

    def noise_shapes(self, p_len: int, skip_head: int = 0, return_length: int | None = None,
                     dec_context: int = 0) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        """Shapes of ``(noise, source_noise)`` for a window of ``p_len`` frames."""
        if return_length is not None:
            dec_context = min(dec_context, skip_head)
        latent = (1, self.inter_channels,
                  p_len - self._flow_head(skip_head, return_length, dec_context))
        if not self.has_f0:
            return latent, None
        frames = p_len if return_length is None else return_length + dec_context
        return latent, self.dec.m_source.l_sin_gen.noise_shape(frames)

    def infer_latent(self, phone: Tensor, lengths: Tensor, pitch: Tensor | None, sid: Tensor,
                     skip_head: int = 0, return_length: int | None = None,
                     dec_context: int = 0, noise: Tensor | None = None
                     ) -> tuple[Tensor, Tensor]:
        """Everything up to the decoder input. Returns ``(z * mask, g)``.

        With ``return_length`` set, the encoder still attends over the whole window, the
        flow runs from 24 frames before the first returned frame, and the frames
        ``[skip_head - dec_context, skip_head + return_length)`` are returned (tail-only
        decode, docs/research/02 §3.7; ``dec_context`` is our addition, see ``infer``).
        """
        g = self.emb_g(sid).unsqueeze(-1)
        first = skip_head - dec_context
        flow_head = self._flow_head(skip_head, return_length, dec_context)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, lengths, flow_head)
        if noise is None:
            noise = torch.randn_like(m_p)
        z_p = (m_p + torch.exp(logs_p) * noise * NOISE_SCALE) * x_mask
        z = self.flow.reverse(z_p, x_mask, g=g)
        if return_length is not None:
            start, end = first - flow_head, skip_head - flow_head + return_length
            z = z[:, :, start:end]
            x_mask = x_mask[:, :, start:end]
        return z * x_mask, g

    def infer(self, phone: Tensor, lengths: Tensor, pitch: Tensor | None,
              pitchf: Tensor | None, sid: Tensor, skip_head: int = 0,
              return_length: int | None = None, dec_context: int = 0,
              noise: Tensor | None = None, source_noise: Tensor | None = None) -> Tensor:
        """Returns audio ``[b, 1, return_length * upp]`` at the model sample rate.

        ``dec_context`` frames ahead of the tail are decoded too and their samples dropped.
        The receptive field of the decoder is roughly +-90 ms; without left context a short
        tail (a 20 ms hop decodes ~60 ms) is rendered entirely against zero padding. This
        costs compute, not latency. ``dec_context=0`` is the upstream behaviour.
        """
        dec_context = min(dec_context, skip_head) if return_length is not None else 0
        z, g = self.infer_latent(phone, lengths, pitch, sid, skip_head, return_length,
                                 dec_context, noise)
        if not self.has_f0:
            audio = self.dec(z, g=g)
        else:
            if return_length is not None:
                pitchf = pitchf[:, skip_head - dec_context:skip_head + return_length]
            audio = self.dec(z, pitchf, g=g, source_noise=source_noise)
        return audio[:, :, dec_context * self.upp:]


def fuse_weight_norm(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Fold weight-norm ``(g, v)`` pairs into plain FP32 weights: ``w = g * v / ||v||``.

    Handles both the legacy ``weight_g``/``weight_v`` keys and the parametrization keys
    (``parametrizations.weight.original0``/``original1``) written by newer torch versions.
    """
    legacy = (".weight_g", ".weight_v")
    param = (".parametrizations.weight.original0", ".parametrizations.weight.original1")
    fused: dict[str, Tensor] = {}
    for key, value in state.items():
        for g_suffix, v_suffix in (legacy, param):
            if key.endswith(g_suffix):
                base = key[: -len(g_suffix)]
                v = state[base + v_suffix].float()
                norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
                fused[base + ".weight"] = v * (value.float() / norm)
                break
            if key.endswith(v_suffix):
                break
        else:
            fused[key] = value.float() if value.is_floating_point() else value
    return fused
