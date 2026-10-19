"""The RVC synthesizer as a loaded, device-resident object (docs/research/12 §3.5).

Loading (control path) applies the shared invariants of docs/research/09 §1:
``weights_only=True``, n_spk patched from ``emb_g``, ``tgt_sr = config[-1]``, ``enc_q``
dropped, weight norm fused away. ``infer`` is the hot-path entry point.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from vox.models.metadata import ModelMeta, detect_checkpoint, load_checkpoint
from vox.synthesis.nets.synthesizer import Synthesizer, fuse_weight_norm


class RvcModel:
    def __init__(self, net: Synthesizer, meta: ModelMeta, device: torch.device) -> None:
        self.net = net
        self.meta = meta
        self.device = device
        self.tgt_sr = meta.sr
        self.upp = net.upp
        self.emb_dim = meta.emb_dim
        self.has_f0 = meta.f0
        self.decoder_fp16 = False

    @classmethod
    def load(cls, path: Path, device: torch.device | str,
             meta: ModelMeta | None = None) -> RvcModel:
        device = torch.device(device)
        cpt = load_checkpoint(Path(path))
        meta = meta or detect_checkpoint(cpt)
        net = Synthesizer(meta.config, meta.emb_dim, meta.f0)
        state = fuse_weight_norm(
            {k: v for k, v in cpt["weight"].items() if not k.startswith("enc_q.")}
        )
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"{Path(path).name}: weights do not match the detected architecture "
                f"(missing {list(missing)[:4]}, unexpected {list(unexpected)[:4]})")
        net.eval().requires_grad_(False).to(device)
        return cls(net, meta, device)

    def noise_shapes(self, p_len: int, skip_head: int = 0, return_length: int | None = None,
                     dec_context: int = 0, dec_right: int = 0):
        """Shapes of the ``(noise, source_noise)`` buffers ``infer`` accepts (control path)."""
        return self.net.noise_shapes(p_len, skip_head, return_length, dec_context, dec_right)

    @torch.inference_mode()
    def infer(self, feats: Tensor, p_len: Tensor, pitch: Tensor | None,
              pitchf: Tensor | None, sid: Tensor, skip_head: int = 0,
              return_length: int | None = None, dec_context: int = 0,
              noise: Tensor | None = None, source_noise: Tensor | None = None,
              phase: Tensor | None = None, phase_frames: int = 0, warp=None,
              dec_right: int = 0) -> Tensor:
        """``feats [1, T, C]`` at 100 Hz -> audio ``[frames * upp]`` at ``tgt_sr``.

        ``skip_head`` / ``return_length`` are host ints from the static chunk plan, so the
        tail slice never needs a GPU ``.item()``. ``noise`` / ``source_noise`` are optional
        pre-drawn standard-normal buffers (see ``nets.synthesizer``). ``phase`` is the
        caller's carried NSF sine phase (``SineGen``); None = restart at zero (upstream).
        ``dec_right`` frames past the tail are decoded and dropped (right context).
        """
        if not self.decoder_fp16:
            return self.net.infer(feats, p_len, pitch, pitchf, sid, skip_head,
                                  return_length, dec_context, noise, source_noise,
                                  phase, phase_frames, warp, dec_right)[0, 0]
        if return_length is None:
            dec_context = dec_right = 0
        else:
            dec_context = min(dec_context, skip_head)
        z, g = self.net.infer_latent(feats, p_len, pitch, sid, skip_head, return_length,
                                     dec_context, noise, dec_right)
        if self.has_f0 and return_length is not None:
            pitchf = pitchf[:, skip_head - dec_context:skip_head + return_length + dec_right]
        with torch.autocast(self.device.type, dtype=torch.float16):
            audio = (self.net.dec(z, pitchf, g=g, source_noise=source_noise, phase=phase,
                                  phase_frames=phase_frames, warp=warp) if self.has_f0
                     else self.net.dec(z, g=g, warp=warp))
        if return_length is None:
            return audio[0, 0].float()
        return audio[0, 0, dec_context * self.upp:(dec_context + return_length) * self.upp].float()
