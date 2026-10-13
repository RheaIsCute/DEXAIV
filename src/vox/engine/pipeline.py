"""The per-hop conversion: one function, ``HopPipeline.process_hop`` (docs/research/12 §3.6).

    upload -> incremental resample to 16 kHz -> F0 on the tail -> pitch cache
           -> ContentVec over the full window -> x2 to 100 Hz -> [index + protect]
           -> tail-only synthesis -> resample to device rate -> post chain

Design constraints, because this function is the future CUDA-graph capture target:
static shapes, every persistent buffer allocated once in ``realloc`` and never replaced,
scalars that may change between hops held in 0-dim device tensors, and no ``.item()``,
``.cpu()`` or boolean-mask indexing anywhere in the hop.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from vox.audio.resample import IncrementalResampler16k, OutputResampler
from vox.config.schemas import RuntimeParams
from vox.engine.chunker import ChunkPlan
from vox.engine.postchain import PostChain
from vox.features.base import Embedder
from vox.pitch.base import PitchExtractor, postprocess_f0, semitones_to_ratio
from vox.pitch.pitch_cache import PitchCache
from vox.profiling.timers import StageTimers
from vox.retrieval import gpu_knn
from vox.retrieval.gpu_knn import GpuIndex
from vox.synthesis.rvc_model import RvcModel

# Stage names shared by the pipeline and whoever drives it (worker or stream simulator).
STAGES = ("upload", "resample_in", "f0", "embed", "index", "synth", "post", "sola",
          "download")


def upsample_features(feats: Tensor, out: Tensor) -> None:
    """50 Hz ``[1, F, C]`` -> 100 Hz ``out [1, P, C]``: each frame twice, last frame held.

    Equals duplicating the final frame, ``F.interpolate(scale_factor=2)`` and truncating to
    ``P`` frames (docs/research/02 §3.5), written straight into a preallocated buffer.
    """
    n_in, n_out = feats.shape[1], out.shape[1]
    for phase in (0, 1):
        count = (n_out + 1 - phase) // 2
        direct = min(count, n_in)
        out[:, phase:2 * direct:2].copy_(feats[:, :direct])
        if count > n_in:
            out[:, 2 * n_in + phase].copy_(feats[:, n_in - 1])


def protect_blend(feats: Tensor, feats0: Tensor, pitchf: Tensor, protect: Tensor | float,
                  out: Tensor | None = None) -> Tensor:
    """Pull unvoiced frames back toward the pre-index features (docs/research/02 §3.6).

    Voiced frames keep the index-blended features; unvoiced frames (``pitchf < 1`` Hz,
    the upstream test) keep only a ``protect`` share of them. Upstream semantics: ``feats0``
    is cloned before the index blend and the gate is the continuous pitch.
    """
    mix = torch.where(pitchf < 1, protect, 1.0).unsqueeze(-1)
    return torch.lerp(feats0, feats, mix, out=out)


class HopPipeline:
    def __init__(self, plan: ChunkPlan, model: RvcModel, embedder: Embedder,
                 extractor: PitchExtractor | None, device: torch.device | str,
                 params: RuntimeParams, timers: StageTimers | None = None,
                 index: GpuIndex | None = None) -> None:
        if model.has_f0 and extractor is None:
            raise ValueError("a pitch-guided model needs a pitch extractor")
        if plan.model_sr != model.tgt_sr:
            raise ValueError(f"plan is for {plan.model_sr} Hz, model is {model.tgt_sr} Hz")
        self.plan = plan
        self.model = model
        self.embedder = embedder
        self.extractor = extractor if model.has_f0 else None
        self.device = torch.device(device)
        self.timers = timers or StageTimers(STAGES, self.device, enabled=False)
        self.index = index
        self.realloc()
        self.set_params(params)

    # ------------------------------------------------------------------ control path

    def realloc(self) -> None:
        """Allocate every persistent buffer from the plan. Never called while running."""
        p, dev = self.plan, self.device
        pin = dev.type == "cuda"
        self._host_in = torch.zeros(p.block, pin_memory=pin)
        self._host_in_np = self._host_in.numpy()
        self.in_block = torch.zeros(p.block, device=dev)
        self.resampler_in = IncrementalResampler16k(p.device_sr, p.block, dev)
        self.resampler_out = OutputResampler(p.model_sr, p.device_sr, dev)
        self.ring_16k = torch.zeros(p.history_16k, device=dev)
        self._ring_tmp = torch.zeros(p.history_16k - p.block_16k, device=dev)
        self._fresh_len = self.resampler_in.out_len
        self.window_16k = self.ring_16k[-p.convert_16k:].unsqueeze(0)
        self.f0_tail = self.ring_16k[-p.f0_tail_16k:] if self.extractor else None
        self.pitch_cache = PitchCache(p.convert_frames, p.shift_frames, p.p_len, dev)
        self.feats = torch.zeros(1, p.p_len, self.model.emb_dim, device=dev)
        self.feats0 = torch.zeros_like(self.feats)
        self.p_len_t = torch.tensor([p.p_len], dtype=torch.long, device=dev)
        self.sid_t = torch.zeros(1, dtype=torch.long, device=dev)
        self.ratio_t = torch.ones((), device=dev)
        self.index_rate_t = torch.zeros((), device=dev)
        self.protect_t = torch.zeros((), device=dev)
        self.out = torch.zeros(p.out_device, device=dev)
        # Device-rate copy of the input under the decoded tail: the RMS-mix reference.
        self.in_tail = torch.zeros(p.out_device, device=dev)
        self._in_tail_tmp = torch.zeros(p.out_device - p.block, device=dev)
        self.post = PostChain(p, dev)

    def set_params(self, params: RuntimeParams) -> None:
        """Apply hot parameters between hops. Scalars go into existing device tensors."""
        self.params = params
        self.ratio_t.fill_(semitones_to_ratio(params.f0_up_key))
        self.index_rate_t.fill_(params.index_rate)
        self.protect_t.fill_(params.protect)
        self.sid_t.fill_(params.sid)
        self.post.set_volume_envelope(params.volume_envelope)
        self._interp = params.f0_unvoiced == "interp"
        self._use_index = self.index is not None and params.index_rate > 0
        self._use_protect = self._use_index and self.extractor is not None \
            and params.protect < 0.5

    def set_index(self, index: GpuIndex | None) -> None:
        self.index = index
        self.set_params(self.params)

    def reset(self) -> None:
        """Zero all streaming state (start of a stream, or after warmup)."""
        self.resampler_in.reset()
        self.ring_16k.zero_()
        self.in_tail.zero_()
        self.pitch_cache.zero()
        self.out.zero_()

    def buffers(self) -> dict[str, Tensor]:
        """Persistent buffers, for pointer-stability checks and future graph cache keys."""
        return {
            "in_block": self.in_block, "ring_16k": self.ring_16k, "feats": self.feats,
            "feats0": self.feats0, "pitch": self.pitch_cache.pitch,
            "pitchf": self.pitch_cache.pitchf, "out": self.out, "in_tail": self.in_tail,
        }

    # ---------------------------------------------------------------------- hot path

    def upload(self, block: np.ndarray) -> None:
        """The one host -> device copy of the hop."""
        self._host_in_np[:] = block
        self.in_block.copy_(self._host_in, non_blocking=True)

    @torch.inference_mode()
    def process_hop(self) -> Tensor:
        """Convert the block in ``in_block``. Returns ``out``: ``block+overlap+search``."""
        p, timers = self.plan, self.timers

        with timers.stage("resample_in"):
            fresh = self.resampler_in.push(self.in_block)
            self._ring_tmp.copy_(self.ring_16k[p.block_16k:])
            self.ring_16k[:-p.block_16k].copy_(self._ring_tmp)
            self.ring_16k[-self._fresh_len:].copy_(fresh)
            if self.post.needs_reference:
                self._in_tail_tmp.copy_(self.in_tail[p.block:])
                self.in_tail[:-p.block].copy_(self._in_tail_tmp)
                self.in_tail[-p.block:].copy_(self.in_block)

        pitch = pitchf = None
        if self.extractor is not None:
            with timers.stage("f0"):
                f0 = self.extractor.extract(self.f0_tail)
                coarse, cont = postprocess_f0(f0, self.ratio_t, self.params.f0_min_hz,
                                              self.params.f0_max_hz, self._interp)
                self.pitch_cache.roll_and_write(coarse, cont)
                pitch, pitchf = self.pitch_cache.pitch_view, self.pitch_cache.pitchf_view

        with timers.stage("embed"):
            feats50 = self.embedder.embed(self.window_16k)

        with timers.stage("index"):
            if self._use_protect:
                upsample_features(feats50, self.feats0)
            if self._use_index:
                gpu_knn.blend(feats50, self.index, self.index_rate_t, p.skip_head // 2)
            upsample_features(feats50, self.feats)
            if self._use_protect:
                protect_blend(self.feats, self.feats0, pitchf[0], self.protect_t,
                              out=self.feats)

        with timers.stage("synth"):
            audio = self.model.infer(self.feats, self.p_len_t, pitch, pitchf, self.sid_t,
                                     p.skip_head, p.return_length, p.dec_context)

        with timers.stage("post"):
            self.out.copy_(self.resampler_out.run(audio))
            self.post.run(self.out, self.in_tail)
        return self.out
