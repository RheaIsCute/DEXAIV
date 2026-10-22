"""The per-hop conversion: one function, ``HopPipeline.process_hop`` (docs/research/12 §3.6).

    upload -> incremental resample to 16 kHz -> F0 on the tail -> pitch cache
           -> ContentVec over the full window -> x2 to 100 Hz -> [index + protect]
           -> tail-only synthesis -> resample to device rate -> post chain

The hop is a fixed sequence of *stages*. Each stage is a method that reads and writes only
persistent, pointer-stable buffers owned by this object, so every stage is independently
capturable as a CUDA graph (``engine/graphs.py``): ``stage_impl[name]`` holds either the
eager method or a graph replay, and ``process_hop`` does not care which.

Design constraints: static shapes, every persistent buffer allocated once in ``realloc`` and
never replaced, scalars that may change between hops held in 0-dim device tensors, no
``.item()``, ``.cpu()`` or boolean-mask indexing anywhere in the hop, and no random draw
inside a stage (the decoder noise is pre-filled per hop, outside any graph). The op
sequence of a stage depends only on the static configuration (index loaded, F0 extractor
present, ``f0_unvoiced``), never on a hot parameter value: ``index_rate = 0`` is a blend
with weight zero, ``protect >= 0.5`` a protect weight of one, ``volume_envelope = 1`` an
RMS exponent of zero. All three are exact no-ops numerically.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from vox.audio.resample import IncrementalResampler16k, OutputResampler
from vox.config.schemas import RuntimeParams
from vox.engine.chunker import ChunkPlan
from vox.engine.postchain import PostChain
from vox.features.base import Embedder
from vox.pitch.base import (
    COARSE_F0_MAX,
    COARSE_F0_MIN,
    PitchExtractor,
    NoteDetector,
    RegisterTracker,
    f0_to_coarse,
    postprocess_f0,
    semitones_to_ratio,
)
from vox.pitch.pitch_cache import PitchCache
from vox.profiling.timers import StageTimers
from vox.retrieval import gpu_knn
from vox.retrieval.gpu_knn import GpuIndex
from vox.synthesis.formant import FormantWarp, snap_formant
from vox.synthesis.rvc_model import RvcModel

FORMANT_WARPS_KEPT = 8  # built FormantWarp objects per plan (cheap: indices + a FIR kernel)

# Stage names shared by the pipeline and whoever drives it (worker or stream simulator).
STAGES = ("upload", "gate", "resample_in", "f0", "embed", "index", "synth", "post", "sola",
          "download")
# The GPU stages of the hop, in execution order. Each is a capture unit for graphs.py.
GPU_STAGES = ("resample_in", "f0", "embed", "index", "synth", "post")


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
        self.model = model
        self.embedder = embedder
        self.extractor = extractor if model.has_f0 else None
        self.device = torch.device(device)
        self.timers = timers or StageTimers(STAGES, self.device, enabled=False)
        self.index = index
        self.params = params
        self._interp = params.f0_unvoiced == "interp"
        self._bridge = round(params.f0_bridge_ms / 10.0)  # F0 frames (10 ms each)
        self._trim_tail = params.f0_trim_tail
        self._phase_carry = params.sine_phase_carry
        self.formant_st = 0.0  # applied step (snapped); 0 = no warp
        self.warp: FormantWarp | None = None
        self._warps: dict[float, FormantWarp] = {}
        self.set_plan(plan)
        self.set_params(params)

    # ------------------------------------------------------------------ control path

    def set_plan(self, plan: ChunkPlan) -> None:
        """Adopt a chunk plan: reallocate every buffer. Never called while running."""
        if plan.model_sr != self.model.tgt_sr:
            raise ValueError(f"plan is for {plan.model_sr} Hz, model is {self.model.tgt_sr} Hz")
        self.plan = plan
        self.realloc()

    def realloc(self) -> None:
        """Allocate every persistent buffer from the plan. Never called while running."""
        p, dev = self.plan, self.device
        pin = dev.type == "cuda"
        self._host_in = torch.zeros(p.block, pin_memory=pin)
        self._host_in_np = self._host_in.numpy()
        self._upload_done = torch.cuda.Event() if pin else None
        self.in_block = torch.zeros(p.block, device=dev)
        self.resampler_in = IncrementalResampler16k(p.device_sr, p.block, dev)
        self.resampler_out = OutputResampler(p.model_sr, p.device_sr, dev, p.decoder_out,
                                             p.decoder_out * p.block // p.out_device)
        self.ring_16k = torch.zeros(p.history_16k, device=dev)
        self._ring_tmp = torch.zeros(p.history_16k - p.block_16k, device=dev)
        self._fresh_len = self.resampler_in.out_len
        self.window_16k = self.ring_16k[-p.convert_16k:].unsqueeze(0)
        # A/B (``embed_pad_ms``): the window plus its last sample replicated, so the last
        # 50 Hz ContentVec frame reaches the end of the window instead of stopping short
        # and being held for the final decoded frames.
        self._window_padded = (torch.zeros(1, p.convert_16k + p.embed_pad_16k, device=dev)
                               if p.embed_pad_16k else None)
        self.f0_tail = self.ring_16k[-p.f0_tail_16k:] if self.extractor else None
        self.pitch_cache = PitchCache(p.convert_frames, p.shift_frames, p.p_len, dev,
                                      self._trim_tail)
        self.feats50 = torch.zeros(1, p.hubert_frames, self.model.emb_dim, device=dev)
        self.feats = torch.zeros(1, p.p_len, self.model.emb_dim, device=dev)
        self.feats0 = torch.zeros_like(self.feats)
        self.p_len_t = torch.tensor([p.p_len], dtype=torch.long, device=dev)
        self.sid_t = torch.zeros(1, dtype=torch.long, device=dev)
        self.ratio_t = torch.ones((), device=dev)
        self.index_rate_t = torch.zeros((), device=dev)
        self.protect_t = torch.ones((), device=dev)
        self.f0_min_t = torch.zeros((), device=dev)
        self.f0_max_t = torch.full((), math.inf, device=dev)
        self.f0_ceiling_t = torch.full((), math.inf, device=dev)  # inf = no soft ceiling
        # Formant: the coarse pitch is built from pitchf * this (1 / factor), as rtrvc does;
        # a hot value inside the f0 stage (1.0 = no shift, same op sequence).
        self.formant_inv_t = torch.ones((), device=dev)
        self._warps = {}  # built for the previous plan's frame count
        self.warp = None
        # Hops since the stream started (open or gated): the formant warp's absolute-time
        # phase (synthesis.formant). Host int, advanced with the 16 kHz ring.
        self.hop_index = 0
        self.register = RegisterTracker(dev)  # auto pitch (set_params)
        self.register.fresh = p.shift_frames
        # Held notes (Natural mode): the ceiling and the pitch range they switch to.
        self.note = NoteDetector(dev)
        self.note.fresh = p.shift_frames
        self.note_ceiling_t = torch.full((), math.inf, device=dev)
        self.note_range_t = torch.ones((), device=dev)
        # Experimental ``sine_phase_carry``: the NSF sine's phase (cycles) where the next
        # hop's decoded slice starts; written by the synth stage, zeroed at gate onsets.
        self.sine_phase = torch.zeros(1, 1, 1, device=dev)
        latent, source = self.model.noise_shapes(p.p_len, p.skip_head, p.return_length,
                                                 p.dec_context, p.right_frames)
        self.noise_z = torch.zeros(latent, device=dev)
        self.noise_src = torch.zeros(source, device=dev) if source is not None else None
        # A/B (``noise_roll``): scratch for shifting the noise with the window.
        self._noise_z_tmp = torch.zeros_like(self.noise_z[..., p.shift_frames:])
        self._noise_src_tmp = (torch.zeros_like(
            self.noise_src[:, p.shift_frames * self.model.upp:])
            if self.noise_src is not None else None)
        self.audio_model = torch.zeros(p.decoder_out, device=dev)
        self.out = torch.zeros(p.out_device, device=dev)
        # Device-rate copy of the input under the decoded tail: the RMS-mix reference. With
        # lookahead the tail ends ``right`` samples before the newest input, so the ring is
        # that much longer and the reference is its first ``out_device`` samples.
        self.in_tail = torch.zeros(p.out_device + p.right, device=dev)
        self.in_tail_ref = self.in_tail[:p.out_device]
        self._in_tail_tmp = torch.zeros(p.out_device + p.right - p.block, device=dev)
        self.post = PostChain(p, dev)
        self.stage_fns: dict[str, Callable[[], None]] = {
            "resample_in": self._stage_resample_in,
            "f0": self._stage_f0,
            "embed": self._stage_embed,
            "index": self._stage_index,
            "synth": self._stage_synth,
            "post": self._stage_post,
        }
        if self.extractor is None:
            self.stage_fns["f0"] = self._stage_noop
        # graphs.py swaps entries here for graph replays; process_hop only reads it.
        self.stage_impl: dict[str, Callable[[], None]] = dict(self.stage_fns)
        if hasattr(self, "params"):
            self.set_params(self.params)

    def set_params(self, params: RuntimeParams) -> bool:
        """Apply hot parameters between hops. Scalars go into existing device tensors.

        Returns True when the change alters a stage's op sequence (``f0_unvoiced``,
        ``f0_bridge_ms``, ``f0_trim_tail``, ``sine_phase_carry``), which means any captured
        graph of that stage is stale and must be recaptured.
        """
        self.params = params
        self.ratio_t.fill_(semitones_to_ratio(params.f0_up_key))
        self.index_rate_t.fill_(params.index_rate if self.index is not None else 0.0)
        # Upstream disables the protect blend at >= 0.5; a weight of 1 is the same thing.
        self.protect_t.fill_(params.protect if params.protect < 0.5 else 1.0)
        # Voicing bounds only act when tighter than the coarse-pitch contract (D9 / doc 08).
        self.f0_min_t.fill_(params.f0_min_hz if params.f0_min_hz > COARSE_F0_MIN else 0.0)
        self.f0_max_t.fill_(params.f0_max_hz if params.f0_max_hz < COARSE_F0_MAX else math.inf)
        self.f0_ceiling_t.fill_(params.f0_ceiling_hz if params.f0_ceiling_hz > 0 else math.inf)
        self.register.set(params.pitch_mode == "auto", params.target_f0_hz,
                          params.pitch_follow_s, params.pitch_range)
        self.note.set(params.note_detect, params.note_min_ms, params.note_tolerance_st,
                      params.note_freeze_ms)
        note_ceiling = params.note_ceiling_hz if params.note_ceiling_hz > 0 else params.f0_ceiling_hz
        self.note_ceiling_t.fill_(note_ceiling if note_ceiling > 0 else math.inf)
        self.note_range_t.fill_(max(params.note_pitch_range, 0.0))
        self.sid_t.fill_(params.sid)
        self.post.set_volume_envelope(params.volume_envelope)
        self.set_formant(params.formant_st)
        interp = params.f0_unvoiced == "interp"
        bridge = round(params.f0_bridge_ms / 10.0)
        trim, carry = params.f0_trim_tail, params.sine_phase_carry
        changed = (interp != self._interp or bridge != self._bridge or trim != self._trim_tail
                   or carry != self._phase_carry)
        self._interp, self._bridge = interp, bridge
        self._trim_tail = self.pitch_cache.trim_tail = trim
        if carry != self._phase_carry:
            self.sine_phase.zero_()
        self._phase_carry = carry
        return changed

    def formant_is_built(self, st: float) -> bool:
        """Whether switching to ``st`` is free on the pipeline side (no warp to build)."""
        st = snap_formant(st)
        return st == 0.0 or st in self._warps

    def set_formant(self, st: float) -> bool:
        """Select the formant step (snapped to 0.5 st, clamped to +-4). Returns True when
        the step changed: the synth stage then has a new shape, which ``StageGraphs``
        handles with its per-step graph cache (``sync_formant``), never a full recapture."""
        st = snap_formant(st)
        changed = st != self.formant_st or (st != 0.0 and self.warp is None)
        self.formant_st = st
        if st == 0.0:
            self.warp = None
            self.formant_inv_t.fill_(1.0)
            return changed
        warp = self._warps.get(st)
        if warp is None:
            if len(self._warps) >= FORMANT_WARPS_KEPT:
                self._warps.pop(next(iter(self._warps)))
            warp = FormantWarp(st, self.plan.decode_frames, self.model.upp, self.device,
                               shift_frames=self.plan.shift_frames)
            self._warps[st] = warp
        self.warp = warp
        self.formant_inv_t.fill_(1.0 / warp.factor)
        return changed

    def set_index(self, index: GpuIndex | None) -> None:
        """Load/unload the retrieval index: changes the index stage's op sequence."""
        self.index = index
        self.set_params(self.params)

    def reset(self) -> None:
        """Zero all streaming state (start of a stream, or after warmup)."""
        self.resampler_in.reset()
        self.resampler_out.reset()
        self.ring_16k.zero_()
        self.in_tail.zero_()
        self.pitch_cache.zero()
        self.out.zero_()
        self.audio_model.zero_()
        self.sine_phase.zero_()
        self.register.reset()
        self.note.reset()
        self.hop_index = 0
        # Fresh decoder noise: with ``noise_roll`` the window would otherwise start on the
        # previous stream's draws (and a seeded run would not be reproducible).
        self.noise_z.normal_()
        if self.noise_src is not None:
            self.noise_src.normal_()

    def on_onset(self) -> None:
        """Gate onset (or new graphs): the carried sine phase restarts at zero, as on the
        first hop of a stream. A no-op without ``sine_phase_carry``."""
        self.sine_phase.zero_()

    def buffers(self) -> dict[str, Tensor]:
        """Persistent buffers, for pointer-stability checks and graph cache keys."""
        return {
            "in_block": self.in_block, "ring_16k": self.ring_16k, "feats50": self.feats50,
            "feats": self.feats, "feats0": self.feats0, "pitch": self.pitch_cache.pitch,
            "pitchf": self.pitch_cache.pitchf, "noise_z": self.noise_z,
            "audio_model": self.audio_model, "out": self.out, "in_tail": self.in_tail,
        }

    def state_tensors(self) -> list[Tensor]:
        """Every tensor a stage mutates: snapshot/restore around graph warmup and capture."""
        tensors = [self.in_block, self.ring_16k, self.in_tail, self.pitch_cache.pitch,
                   self.pitch_cache.pitchf, self.feats50, self.feats, self.feats0,
                   self.audio_model, self.out, self.resampler_in._window,
                   *self.resampler_out.state_tensors()]
        if self.noise_src is not None:
            tensors.append(self.noise_src)
        tensors.append(self.noise_z)
        tensors.append(self.sine_phase)
        tensors.extend(self.register.state_tensors())
        tensors.extend(self.note.state_tensors())
        return tensors

    def static_key(self) -> tuple:
        """Everything that fixes a stage's op sequence and its memory addresses."""
        return (
            tuple(sorted((k, v.data_ptr()) for k, v in self.buffers().items())),
            self.plan, id(self.model), id(self.embedder), id(self.extractor),
            id(self.index), self._interp, self._bridge, self._trim_tail, self._phase_carry,
            self.model.decoder_fp16,
            getattr(self.embedder, "autocast", None), getattr(self.embedder, "norm_mode", None),
            self.post.rms_mix.graph_safe,
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark,
        )

    def stage_graph_safe(self, name: str) -> bool:
        if name == "f0":
            return self.extractor is None or getattr(self.extractor, "graph_safe", False)
        return True

    # ---------------------------------------------------------------------- hot path

    def upload(self, block: np.ndarray) -> None:
        """The one host -> device copy of the hop.

        The pinned staging buffer may only be rewritten once the previous asynchronous copy
        out of it has landed; the event wait makes that explicit instead of relying on the
        hop's own download sync (under graph replay a hop is launched in ~1 ms of host
        time, so without the wait the next block would overwrite the staging buffer while
        the DMA of the previous one is still queued behind the running graphs).
        """
        if self._upload_done is not None:
            self._upload_done.synchronize()
        self._host_in_np[:] = block
        self.in_block.copy_(self._host_in, non_blocking=True)
        if self._upload_done is not None:
            self._upload_done.record()

    def fill_noise(self) -> None:
        """Draw this hop's decoder noise. Outside every graph: RNG never lives in a capture.

        With ``noise_roll`` (A/B) the noise moves with the window instead: shifted left by
        one hop and drawn only for the frames that entered, so the 40 ms two consecutive
        hops both decode see the same latent and excitation noise.
        """
        if not self.params.noise_roll:
            self.noise_z.normal_()
            if self.noise_src is not None:
                self.noise_src.normal_()
            return
        s = self.plan.shift_frames
        self._noise_z_tmp.copy_(self.noise_z[..., s:])
        self.noise_z[..., :-s].copy_(self._noise_z_tmp)
        self.noise_z[..., -s:].normal_()
        if self.noise_src is not None:
            s = s * self.model.upp
            self._noise_src_tmp.copy_(self.noise_src[:, s:])
            self.noise_src[:, :-s].copy_(self._noise_src_tmp)
            self.noise_src[:, -s:].normal_()

    def _stage_noop(self) -> None:
        return None

    def _stage_resample_in(self) -> None:
        p = self.plan
        fresh = self.resampler_in.push(self.in_block)
        self._ring_tmp.copy_(self.ring_16k[p.block_16k:])
        self.ring_16k[:-p.block_16k].copy_(self._ring_tmp)
        self.ring_16k[-self._fresh_len:].copy_(fresh)
        self._in_tail_tmp.copy_(self.in_tail[p.block:])
        self.in_tail[:-p.block].copy_(self._in_tail_tmp)
        self.in_tail[-p.block:].copy_(self.in_block)

    def _stage_f0(self) -> None:
        f0 = self.extractor.extract(self.f0_tail)
        # Held note? Then the note ceiling / range apply and the register holds still.
        note, frozen = self.note.update(f0)
        ratio = self.register.ratio(f0, self.ratio_t,  # per frame; the fixed ratio unless auto
                                    torch.where(note, self.note_range_t, self.register.range),
                                    frozen)
        ceiling = torch.where(note, self.note_ceiling_t, self.f0_ceiling_t)
        _, cont = postprocess_f0(f0, ratio, self.f0_min_t, self.f0_max_t,
                                 self._interp, self._bridge, ceiling)
        # The excitation keeps the requested pitch; the coarse pitch is that over the
        # formant factor (1 without a shift: the same buckets postprocess_f0 makes).
        coarse = f0_to_coarse(cont * self.formant_inv_t)
        self.pitch_cache.roll_and_write(coarse, cont)

    def _stage_embed(self) -> None:
        window = self.window_16k
        if self._window_padded is not None:
            n = self.plan.convert_16k
            self._window_padded[:, :n].copy_(window)
            self._window_padded[:, n:].copy_(window[:, -1:].expand(-1, self.plan.embed_pad_16k))
            window = self._window_padded
        self.feats50.copy_(self.embedder.embed(window))

    def _stage_index(self) -> None:
        protect = self.index is not None and self.extractor is not None
        if protect:
            upsample_features(self.feats50, self.feats0)
        if self.index is not None:
            gpu_knn.blend(self.feats50, self.index, self.index_rate_t, self.plan.skip_head // 2)
        upsample_features(self.feats50, self.feats)
        if protect:
            protect_blend(self.feats, self.feats0, self.pitch_cache.pitchf_view[0],
                          self.protect_t, out=self.feats)

    def _stage_synth(self) -> None:
        p = self.plan
        pitch = pitchf = None
        if self.extractor is not None:
            pitch, pitchf = self.pitch_cache.pitch_view, self.pitch_cache.pitchf_view
        phase = self.sine_phase if self._phase_carry and pitch is not None else None
        audio = self.model.infer(self.feats, self.p_len_t, pitch, pitchf, self.sid_t,
                                 p.skip_head, p.return_length, p.dec_context,
                                 self.noise_z, self.noise_src, phase, p.shift_frames,
                                 self.warp, p.right_frames)
        self.audio_model.copy_(audio)

    def _stage_post(self) -> None:
        self.out.copy_(self.resampler_out.run(self.audio_model))
        self.post.run(self.out, self.in_tail_ref)

    @torch.inference_mode()
    def process_hop(self) -> Tensor:
        """Convert the block in ``in_block``. Returns ``out``: ``block+overlap+search``."""
        timers, impl = self.timers, self.stage_impl
        with timers.stage("resample_in"):
            impl["resample_in"]()
        with timers.stage("f0"):
            impl["f0"]()
        with timers.stage("embed"):
            impl["embed"]()
        with timers.stage("index"):
            impl["index"]()
        with timers.stage("synth"):
            self.fill_noise()
            if self.warp is not None:
                self.warp.select(self.hop_index)  # outside the graph, like the noise
            impl["synth"]()
        with timers.stage("post"):
            impl["post"]()
        self.hop_index += 1
        return self.out

    @torch.inference_mode()
    def advance_context(self) -> None:
        """Gate closed: keep the 16 kHz ring, the RMS reference and the pitch cache current.

        The ring is then the true recent history when the gate reopens, with no stale
        speech to replay and no refill stall (docs/research/03 §5, Applio protocol). The
        pitch cache shifts with it (the closed hop's frames unvoiced), so at the reopen the
        F0 history lines up with the audio ContentVec sees.
        """
        with self.timers.stage("resample_in"):
            self.stage_impl["resample_in"]()
            self.pitch_cache.advance_silent()
        self.hop_index += 1

    def forget_speech(self) -> None:
        """Gate closed: drop the rendered output of the audio that just ended.

        Features and pitch are left alone: the features are recomputed from the ring at
        every open hop, and the pitch cache advances with the ring (``advance_context``).
        """
        self.audio_model.zero_()
        self.out.zero_()
        self.resampler_out.reset()  # its history would splice old speech into the onset
