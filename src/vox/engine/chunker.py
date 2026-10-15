"""The chunking law: the single owner of every derived buffer size (principle P17).

    convert = extra + crossfade + sola_search + block      (padded to 160 @ 16 kHz)

Only the trailing ``block + crossfade + sola_search`` of the model output is kept; the
``extra`` region exists to give the encoder and F0 tracker context. ``skip_head`` and
``return_length`` tell the synthesizer which frames to decode (docs/research/03 §1).

Pure integer math: no torch, no I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

ANALYSIS_SR = 16000
FRAME_16K = 160  # one 10 ms feature frame
MAX_SOLA_UNITS = 4  # overlap is pure algorithmic latency: capped at 40 ms
HUBERT_STRIDE = 320
HUBERT_FIELD = 400


class ChunkPlanError(ValueError):
    pass


@dataclass(frozen=True)
class ChunkPlan:
    device_sr: int
    model_sr: int
    zc: int  # device samples per 10 ms unit
    # device-rate sizes (all multiples of zc)
    block: int
    crossfade: int  # requested overlap, after 10 ms rounding
    sola_buffer: int  # overlap actually used: min(crossfade, 40 ms, block)
    sola_search: int
    extra: int
    # 16 kHz sizes
    block_16k: int
    convert_16k: int
    history_16k: int  # 16 kHz ring length: max(convert, F0 tail)
    f0_tail_16k: int
    # frame counts at 100 Hz
    convert_frames: int
    skip_head: int
    return_length: int
    dec_context: int  # frames decoded ahead of the tail and discarded (left context)
    shift_frames: int  # pitch-cache shift per hop
    hubert_frames: int  # 50 Hz frames the embedder emits for the window
    p_len: int
    # output sizes
    decoder_out: int  # samples at model_sr
    out_device: int  # samples at device_sr: block + sola_buffer + sola_search
    embed_pad_16k: int = 0  # replicated right edge appended before ContentVec (A/B)

    def ms(self, device_samples: int) -> float:
        return 1000.0 * device_samples / self.device_sr

    @property
    def block_ms(self) -> float:
        return self.ms(self.block)

    @property
    def overlap_ms(self) -> float:
        return self.ms(self.sola_buffer)

    @property
    def search_ms(self) -> float:
        return self.ms(self.sola_search)

    @property
    def extra_ms(self) -> float:
        return self.ms(self.extra)

    @property
    def context_ms(self) -> float:
        return 1000.0 * self.convert_16k / ANALYSIS_SR

    @property
    def stream_delay(self) -> int:
        """Device samples the stitched stream trails the newest input (before SOLA offset)."""
        return self.sola_buffer + self.sola_search

    @property
    def algorithmic_latency_ms(self) -> float:
        """Hop accumulation plus the SOLA overlap and search the output is emitted behind."""
        return self.block_ms + self.overlap_ms + self.search_ms

    @property
    def overlap_note(self) -> str:
        """Why the overlap in use is shorter than the one asked for ("" when it is not)."""
        if self.sola_buffer == self.crossfade:
            return ""
        why = ("the overlap cannot exceed the hop" if self.sola_buffer == self.block
               else f"the overlap is capped at {MAX_SOLA_UNITS * 10} ms")
        return (f"crossfade {self.ms(self.crossfade):.0f} ms -> {self.overlap_ms:.0f} ms: {why} "
                f"(hop {self.block_ms:.0f} ms)")

    @property
    def hubert_phase_stable(self) -> bool:
        """True when the hop is a whole number of 20 ms HuBERT frames.

        Otherwise the window start alternates by half a HuBERT frame between hops, so the
        same audio lands on a different 50 Hz grid from one hop to the next.
        """
        return self.block_16k % HUBERT_STRIDE == 0

    def latency_estimate_ms(self, ring_fill_ms: float = 0.0, device_ms: float = 0.0) -> float:
        return self.algorithmic_latency_ms + ring_fill_ms + device_ms


def _units(ms: float, minimum: int) -> int:
    """Milliseconds -> whole 10 ms units, rounding half up."""
    return max(minimum, math.floor(ms / 10.0 + 0.5))


def from_shape(shape, model_sr: int, f0_tail_16k: int = 0) -> ChunkPlan:
    """``plan`` for a ``ShapeParams`` (read by attribute: the chunker imports nothing)."""
    return plan(shape.block_ms, shape.crossfade_ms, shape.extra_ms, shape.sola_search_ms,
                shape.device_sr, model_sr, f0_tail_16k, shape.decoder_context_ms,
                shape.overlap_beyond_hop, round(ANALYSIS_SR * shape.embed_pad_ms / 1000.0))


def plan(block_ms: float, crossfade_ms: float, extra_ms: float, sola_search_ms: float,
         device_sr: int, model_sr: int, f0_tail_16k: int = 0,
         decoder_context_ms: float = 0.0, overlap_beyond_hop: bool = False,
         embed_pad_16k: int = 0) -> ChunkPlan:
    """Derive every size from the four user-facing durations.

    ``f0_tail_16k`` is the pitch extractor's tail requirement; when it exceeds the
    conversion window the 16 kHz ring is simply made that long. ``overlap_beyond_hop``
    lifts the overlap <= hop clamp (the pre-fix geometry, for A/B listening only).
    ``embed_pad_16k`` samples of replicated right edge are appended to the window before
    ContentVec (A/B: so its last 50 Hz frame covers the end of the window).
    """
    if device_sr % 100 or model_sr % 100:
        raise ChunkPlanError("sample rates must be multiples of 100 Hz")
    if min(block_ms, crossfade_ms, sola_search_ms) <= 0 or extra_ms < 0:
        raise ChunkPlanError("block, crossfade and sola_search must be > 0, extra >= 0")
    zc = device_sr // 100
    block_u = _units(block_ms, 1)
    crossfade_u = _units(crossfade_ms, 1)
    search_u = _units(sola_search_ms, 1)
    extra_u = _units(extra_ms, 0)
    # The overlap does not exceed the hop: then every emitted sample is a two-render mix and
    # the overlap costs no more latency than it buys smoothing. A longer overlap blends
    # three renders per sample (SOLA's tail then overlaps its own crossfade); kept only as
    # an A/B setting. See docs/DEVIATIONS.md §7, fix 1.
    sola_u = min(crossfade_u, MAX_SOLA_UNITS)
    if not overlap_beyond_hop:
        sola_u = min(sola_u, block_u)

    # One 10 ms device unit is exactly 160 samples at 16 kHz.
    block_16k = FRAME_16K * block_u
    convert_16k = FRAME_16K * (block_u + sola_u + search_u + extra_u)
    if convert_16k % FRAME_16K:
        convert_16k += FRAME_16K - convert_16k % FRAME_16K
    if convert_16k < HUBERT_FIELD:
        raise ChunkPlanError("conversion window shorter than one HuBERT frame")
    convert_frames = convert_16k // FRAME_16K
    return_length = block_u + sola_u + search_u
    skip_head = convert_frames - return_length
    if embed_pad_16k < 0:
        raise ChunkPlanError("embed_pad_16k must be >= 0")
    hubert_frames = (convert_16k + embed_pad_16k - HUBERT_FIELD) // HUBERT_STRIDE + 1
    p_len = min(convert_frames, 2 * (hubert_frames + 1))
    if p_len != convert_frames:
        raise ChunkPlanError("embedder cannot cover the conversion window")

    return ChunkPlan(
        device_sr=device_sr, model_sr=model_sr, zc=zc,
        block=block_u * zc, crossfade=crossfade_u * zc, sola_buffer=sola_u * zc,
        sola_search=search_u * zc, extra=extra_u * zc,
        block_16k=block_16k, convert_16k=convert_16k,
        history_16k=max(convert_16k, f0_tail_16k, block_16k + FRAME_16K),
        f0_tail_16k=f0_tail_16k,
        convert_frames=convert_frames, skip_head=skip_head, return_length=return_length,
        dec_context=min(_units(decoder_context_ms, 0), skip_head),
        shift_frames=block_u, hubert_frames=hubert_frames, p_len=p_len,
        decoder_out=return_length * (model_sr // 100), out_device=return_length * zc,
        embed_pad_16k=embed_pad_16k,
    )
