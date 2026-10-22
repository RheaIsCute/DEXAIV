"""The chunking law: the single owner of every derived buffer size (principle P17).

    convert = extra + right + crossfade + sola_search + block      (padded to 160 @ 16 kHz)

Only ``block + crossfade + sola_search`` of the model output is kept; the ``extra`` region
exists to give the encoder and F0 tracker context. ``right`` (``extra_right_ms``, 0 by
default) is lookahead: the kept region ends that far before the window's end, so every
kept sample has that much audio to its right for the encoder, the flow, the F0 tracker and
the decoder (which decodes those frames too and drops them). The stream is emitted
``right`` later (docs/DEVIATIONS.md §13, report 18 §F2). ``skip_head`` and ``return_length``
tell the synthesizer which frames to decode (docs/research/03 §1).

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
# The hop the user picks (GUI, ``vox run``) is a whole number of 20 ms HuBERT frames, so
# ``hubert_phase_stable`` always holds. The chunker itself still plans any 10 ms multiple
# (benchmarks measure 30 ms hops on purpose).
HOP_STEP_MS = 20.0


class ChunkPlanError(ValueError):
    pass


def snap_block_ms(block_ms: float) -> float:
    """Nearest multiple of ``HOP_STEP_MS`` (halves round up), at least one step. Values
    <= 0 pass through unchanged so the planner can reject them."""
    if block_ms <= 0:
        return block_ms
    return HOP_STEP_MS * max(1, math.floor(block_ms / HOP_STEP_MS + 0.5))


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
    right: int  # lookahead: the kept region ends this many device samples before the window's end
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
    right_frames: int  # frames decoded after the tail and discarded (right context)
    shift_frames: int  # pitch-cache shift per hop
    hubert_frames: int  # 50 Hz frames the embedder emits for the window
    p_len: int
    # output sizes
    decoder_out: int  # samples at model_sr
    out_device: int  # samples at device_sr: block + sola_buffer + sola_search
    embed_pad_16k: int = 0  # replicated right edge appended before ContentVec (A/B)
    # Device samples of the fade-in on a gate-onset hop; 0 or >= sola_buffer = the plain
    # overlap fade (engine/sola.py ``mark_onset``).
    onset_ramp: int = 0
    onset_ramp_energy: int = 0  # the fade-in of an energy-only gate onset (0 = onset_ramp)

    def ms(self, device_samples: int) -> float:
        return 1000.0 * device_samples / self.device_sr

    @property
    def decode_frames(self) -> int:
        """Frames the decoder renders per hop: the kept tail plus its left and right
        context. A formant shift stretches exactly these (``synthesis.formant``)."""
        return self.return_length + self.dec_context + self.right_frames

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
    def right_ms(self) -> float:
        """Lookahead: how much later than today's geometry the stream is emitted."""
        return self.ms(self.right)

    @property
    def context_ms(self) -> float:
        return 1000.0 * self.convert_16k / ANALYSIS_SR

    @property
    def stream_delay(self) -> int:
        """Device samples the stitched stream trails the newest input (before SOLA offset):
        the SOLA overlap and search, plus the lookahead."""
        return self.sola_buffer + self.sola_search + self.right

    @property
    def algorithmic_latency_ms(self) -> float:
        """Hop accumulation plus the SOLA overlap, search and lookahead the output is
        emitted behind."""
        return self.block_ms + self.overlap_ms + self.search_ms + self.right_ms

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


# Output-ring fill target (docs/DEVIATIONS.md §13, report 18 §0.1). ``legacy`` is the rule
# the live session has always used, sized for the 20 ms hop where compute is nearly a whole
# hop: two hops plus one callback block. ``hop_plus`` is one hop plus a fixed margin, the
# same cushion at every hop (at hop >= 40 compute is a fraction of the hop, and 2 x hop
# triples the live cost of a bigger hop).
FILL_RULES = ("legacy", "hop_plus")


def fill_target_ms(block_ms: float, callback_ms: float, rule: str = "legacy",
                   margin_ms: float = 20.0) -> float:
    """Milliseconds the output ring is kept filled ahead of the device."""
    if rule == "legacy":
        return 2 * block_ms + callback_ms
    if rule == "hop_plus":
        return block_ms + margin_ms
    raise ChunkPlanError(f"fill rule must be one of {FILL_RULES}, not {rule!r}")


def from_shape(shape, model_sr: int, f0_tail_16k: int = 0) -> ChunkPlan:
    """``plan`` for a ``ShapeParams`` (read by attribute: the chunker imports nothing)."""
    return plan(shape.block_ms, shape.crossfade_ms, shape.extra_ms, shape.sola_search_ms,
                shape.device_sr, model_sr, f0_tail_16k, shape.decoder_context_ms,
                shape.overlap_beyond_hop, round(ANALYSIS_SR * shape.embed_pad_ms / 1000.0),
                shape.onset_ramp_ms, shape.extra_right_ms, shape.onset_ramp_energy_ms)


def plan(block_ms: float, crossfade_ms: float, extra_ms: float, sola_search_ms: float,
         device_sr: int, model_sr: int, f0_tail_16k: int = 0,
         decoder_context_ms: float = 0.0, overlap_beyond_hop: bool = False,
         embed_pad_16k: int = 0, onset_ramp_ms: float = 0.0,
         extra_right_ms: float = 0.0,
         onset_ramp_energy_ms: float = 0.0) -> ChunkPlan:
    """Derive every size from the four user-facing durations.

    ``f0_tail_16k`` is the pitch extractor's tail requirement; when it exceeds the
    conversion window the 16 kHz ring is simply made that long. ``overlap_beyond_hop``
    lifts the overlap <= hop clamp (the pre-fix geometry, for A/B listening only).
    ``embed_pad_16k`` samples of replicated right edge are appended to the window before
    ContentVec (A/B: so its last 50 Hz frame covers the end of the window).
    ``onset_ramp_ms`` is the fade-in of a gate-onset hop (any length, not 10 ms units).
    ``extra_right_ms`` is the lookahead: the window extends that far past the kept region.
    """
    if device_sr % 100 or model_sr % 100:
        raise ChunkPlanError("sample rates must be multiples of 100 Hz")
    if min(block_ms, crossfade_ms, sola_search_ms) <= 0 or extra_ms < 0:
        raise ChunkPlanError("block, crossfade and sola_search must be > 0, extra >= 0")
    if extra_right_ms < 0:
        raise ChunkPlanError("extra_right_ms must be >= 0")
    zc = device_sr // 100
    block_u = _units(block_ms, 1)
    crossfade_u = _units(crossfade_ms, 1)
    search_u = _units(sola_search_ms, 1)
    extra_u = _units(extra_ms, 0)
    right_u = _units(extra_right_ms, 0)
    # The overlap does not exceed the hop: then every emitted sample is a two-render mix and
    # the overlap costs no more latency than it buys smoothing. A longer overlap blends
    # three renders per sample (SOLA's tail then overlaps its own crossfade); kept only as
    # an A/B setting. See docs/DEVIATIONS.md §7, fix 1.
    sola_u = min(crossfade_u, MAX_SOLA_UNITS)
    if not overlap_beyond_hop:
        sola_u = min(sola_u, block_u)

    # One 10 ms device unit is exactly 160 samples at 16 kHz.
    block_16k = FRAME_16K * block_u
    convert_16k = FRAME_16K * (block_u + sola_u + search_u + extra_u + right_u)
    if convert_16k % FRAME_16K:
        convert_16k += FRAME_16K - convert_16k % FRAME_16K
    if convert_16k < HUBERT_FIELD:
        raise ChunkPlanError("conversion window shorter than one HuBERT frame")
    convert_frames = convert_16k // FRAME_16K
    return_length = block_u + sola_u + search_u
    # The kept region sits ``right_u`` frames before the window's end.
    skip_head = convert_frames - return_length - right_u
    if embed_pad_16k < 0:
        raise ChunkPlanError("embed_pad_16k must be >= 0")
    hubert_frames = (convert_16k + embed_pad_16k - HUBERT_FIELD) // HUBERT_STRIDE + 1
    p_len = min(convert_frames, 2 * (hubert_frames + 1))
    if p_len != convert_frames:
        raise ChunkPlanError("embedder cannot cover the conversion window")

    return ChunkPlan(
        device_sr=device_sr, model_sr=model_sr, zc=zc,
        block=block_u * zc, crossfade=crossfade_u * zc, sola_buffer=sola_u * zc,
        sola_search=search_u * zc, extra=extra_u * zc, right=right_u * zc,
        block_16k=block_16k, convert_16k=convert_16k,
        history_16k=max(convert_16k, f0_tail_16k, block_16k + FRAME_16K),
        f0_tail_16k=f0_tail_16k,
        convert_frames=convert_frames, skip_head=skip_head, return_length=return_length,
        dec_context=min(_units(decoder_context_ms, 0), skip_head),
        right_frames=right_u, shift_frames=block_u, hubert_frames=hubert_frames, p_len=p_len,
        decoder_out=return_length * (model_sr // 100), out_device=return_length * zc,
        embed_pad_16k=embed_pad_16k,
        onset_ramp=max(0, round(device_sr * onset_ramp_ms / 1000.0)),
        onset_ramp_energy=max(0, round(device_sr * onset_ramp_energy_ms / 1000.0)),
    )
