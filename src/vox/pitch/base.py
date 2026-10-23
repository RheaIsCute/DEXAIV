"""The ``PitchExtractor`` interface and the shared, trained-in F0 post-processing.

Everything here is tensor-native and sync-free (no ``.item()``, no boolean-mask indexing)
so it can run inside the per-hop function and later inside a CUDA graph.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

# The coarse-pitch contract every RVC synthesizer was trained on (docs/research/02 §3.2).
COARSE_F0_MIN = 50.0
COARSE_F0_MAX = 1100.0
_MEL_MIN = 1127.0 * math.log(1.0 + COARSE_F0_MIN / 700.0)
_MEL_MAX = 1127.0 * math.log(1.0 + COARSE_F0_MAX / 700.0)


class PitchExtractor(Protocol):
    name: str
    graph_safe: bool  # may live inside a captured CUDA graph

    def load(self, device: torch.device, assets_dir: Path) -> None:
        """Control path only."""
        ...

    def tail_samples(self, block_16k: int) -> int:
        """How much trailing 16 kHz audio the extractor wants for a hop of ``block_16k``."""
        ...

    def extract(self, audio_16k: Tensor) -> Tensor:
        """``[n]`` samples -> ``[n // 160 + 1]`` F0 values in Hz at 100 Hz; 0 = unvoiced."""
        ...


def f0_to_coarse(f0: Tensor) -> Tensor:
    """Hz -> the 1..255 mel-scaled bucket fed to the pitch embedding (1 = unvoiced)."""
    mel = 1127.0 * torch.log1p(f0 / 700.0)
    scaled = (mel - _MEL_MIN) * (254.0 / (_MEL_MAX - _MEL_MIN)) + 1.0
    mel = torch.where(mel > 0, scaled, mel)
    return torch.round(mel.clamp(1.0, 255.0)).long()


def clamp_voicing(f0: Tensor, f0_min: Tensor | float, f0_max: Tensor | float) -> Tensor:
    """Treat estimates outside the calibrated source range as unvoiced.

    A frame far outside the speaker's own range is almost always an octave error; calling
    it unvoiced beats transposing a wrong pitch (docs/research/08 §2.4).
    """
    return torch.where((f0 >= f0_min) & (f0 <= f0_max), f0, torch.zeros_like(f0))


def interp_unvoiced(f0: Tensor) -> Tensor:
    """Linearly interpolate across ``f0 == 0`` gaps, holding the nearest value at the edges.

    Same result as ``np.interp(uv_idx, voiced_idx, f0[voiced])`` in current rvc-webui, but
    with static shapes and no host round-trip. All-unvoiced input is returned unchanged.
    """
    n = f0.shape[-1]
    idx = torch.arange(n, device=f0.device)
    voiced = f0 > 0
    prev = torch.cummax(torch.where(voiced, idx, -1), dim=-1).values
    nxt = torch.flip(
        torch.cummin(torch.flip(torch.where(voiced, idx, n), [-1]), dim=-1).values, [-1])
    f_prev = f0.gather(-1, prev.clamp(min=0))
    f_next = f0.gather(-1, nxt.clamp(max=n - 1))
    left = torch.where(prev >= 0, f_prev, f_next)
    right = torch.where(nxt < n, f_next, f_prev)
    weight = (idx - prev).to(f0.dtype) / (nxt - prev).clamp(min=1).to(f0.dtype)
    return left + (right - left) * weight.clamp(0.0, 1.0)


def bridge_gaps(f0: Tensor, max_frames: int, max_semitones: float = 4.0) -> Tensor:
    """Fill ``f0 == 0`` runs of at most ``max_frames`` that sit between two voiced frames no
    more than ``max_semitones`` apart, interpolating in log-F0: a tracker dropout inside a
    vowel, not a consonant (those border on a jump or a longer gap). Longer gaps and the
    edges are left unvoiced. Static shapes, no host round-trip (graph-safe)."""
    n = f0.shape[-1]
    idx = torch.arange(n, device=f0.device)
    voiced = f0 > 0
    prev = torch.cummax(torch.where(voiced, idx, -1), dim=-1).values
    nxt = torch.flip(
        torch.cummin(torch.flip(torch.where(voiced, idx, n), [-1]), dim=-1).values, [-1])
    log_prev = torch.log(f0.gather(-1, prev.clamp(min=0)).clamp(min=1.0))
    log_next = torch.log(f0.gather(-1, nxt.clamp(max=n - 1)).clamp(min=1.0))
    span = (nxt - prev).clamp(min=1)
    weight = (idx - prev).to(f0.dtype) / span.to(f0.dtype)
    interpolated = torch.exp(log_prev + (log_next - log_prev) * weight)
    close = (log_next - log_prev).abs() <= max_semitones * math.log(2.0) / 12.0
    fill = ~voiced & (prev >= 0) & (nxt < n) & (span - 1 <= max_frames) & close
    return torch.where(fill, interpolated, f0)


CEILING_KNEE_ST = 4.0  # the soft ceiling starts compressing this far below the ceiling


def soft_ceiling(pitchf: Tensor, ceiling: Tensor | float) -> Tensor:
    """Soft-knee F0 limit in semitone space: unchanged up to ``ceiling`` - 4 st, then
    ``knee * 2 ** (4 tanh(st / 4) / 12)`` (slope 1 at the knee, asymptotic to ``ceiling``).

    For singing through a speech-trained model: belted notes that the transposition pushes
    past what the model saw are bent down instead of cracking. It is a safety, not a fix:
    a note 2.5 st over the knee comes out ~0.3 st flat, 6 st over ~2.3 st flat (off-key).
    ``ceiling = inf`` is an exact no-op with the same op sequence (graph-safe); unvoiced
    frames (0 Hz) are never touched.
    """
    knee = ceiling * 2.0 ** (-CEILING_KNEE_ST / 12.0)
    over = 12.0 * torch.log2(pitchf.clamp(min=1e-3) / knee)
    bent = knee * torch.exp2(CEILING_KNEE_ST * torch.tanh(over.clamp(min=0.0) / CEILING_KNEE_ST)
                             / 12.0)
    return torch.where(pitchf > knee, bent, pitchf)


def postprocess_f0(f0: Tensor, ratio: Tensor | float, f0_min: Tensor | float = COARSE_F0_MIN,
                   f0_max: Tensor | float = COARSE_F0_MAX, interp: bool = False,
                   bridge_frames: int = 0, ceiling: Tensor | float = 0.0
                   ) -> tuple[Tensor, Tensor]:
    """Raw extractor output -> ``(coarse int64, continuous Hz)`` after transposition.

    ``ratio`` is ``2 ** (f0_up_key / 12)``; pass a 0-dim device tensor on the hot path so
    changing the key never changes the op sequence. The same goes for the voicing bounds:
    as tensors they are always applied (use 0 / inf to disable), as floats only when
    tighter than the coarse-pitch contract. ``bridge_frames`` > 0 fills short dropouts
    first (``bridge_gaps``). ``ceiling`` (Hz) bends transposed pitch above it softly
    (``soft_ceiling``); as a tensor it is always applied (inf = off), as a float only > 0.
    """
    tensors = isinstance(f0_min, Tensor) or isinstance(f0_max, Tensor)
    if tensors or f0_min > COARSE_F0_MIN or f0_max < COARSE_F0_MAX:
        f0 = clamp_voicing(f0, f0_min, f0_max)
    if bridge_frames > 0:
        f0 = bridge_gaps(f0, bridge_frames)
    if interp:
        f0 = interp_unvoiced(f0)
    pitchf = f0 * ratio
    if isinstance(ceiling, Tensor) or ceiling > 0:
        pitchf = soft_ceiling(pitchf, ceiling)
    return f0_to_coarse(pitchf), pitchf


def semitones_to_ratio(f0_up_key: float) -> float:
    return 2.0 ** (f0_up_key / 12.0)


REGISTER_WARM_FRAMES = 50  # voiced frames (0.5 s) heard before the auto ratio takes over
REGISTER_MIN_ST, REGISTER_MAX_ST = -24.0, 24.0  # the auto transposition's range


class RegisterTracker:
    """Auto pitch: a running mean of the talker's log2 F0 over the fresh frames of every hop,
    turned into a per-frame transposition that puts it on ``target`` (docs/DEVIATIONS.md
    §9). Graph-safe: static shapes, 0-dim device state, no host sync; the state is updated
    in place by ``ratio``, so it lives inside the captured F0 stage like the pitch cache.

    ``level``: the register estimate in log2 Hz, moved toward each hop's mean voiced log2
    F0 by ``1 - exp(-voiced_frames / tau_frames)`` (a hop of silence moves nothing).
    ``heard``: voiced frames so far, saturating; below ``REGISTER_WARM_FRAMES`` the fixed
    ratio applies. ``pitch_range`` r: log2 out = log2 target + r (log2 f0 - level).
    """

    def __init__(self, device: torch.device | str) -> None:
        self.level = torch.zeros((), device=device)
        self.heard = torch.zeros((), device=device)
        self.auto = torch.zeros((), device=device)  # 1 = auto mode
        self.target_log = torch.full((), math.log2(200.0), device=device)
        self.tau_frames = torch.full((), 200.0, device=device)
        self._target_log_host = math.log2(200.0)
        self.range = torch.ones((), device=device)
        self.fresh = 1  # frames at the end of ``f0`` that are new this hop
        # Host mirror of (level, heard), written by ``publish`` on the hop thread after the
        # hop has synced; ``offset_st`` reads only this, so no other thread ever syncs the
        # device (a sync during a graph capture poisons the CUDA context).
        self._pair = torch.zeros(2, device=device)
        self._host = torch.zeros(2)

    def set(self, auto: bool, target_hz: float, follow_s: float, pitch_range: float) -> None:
        self.auto.fill_(1.0 if auto else 0.0)
        self.target_log.fill_(math.log2(max(target_hz, 1.0)))
        self._target_log_host = math.log2(max(target_hz, 1.0))
        self.tau_frames.fill_(max(follow_s, 0.05) * 100.0)
        self.range.fill_(max(pitch_range, 0.0))

    def reset(self) -> None:
        self.level.zero_()
        self.heard.zero_()
        self._host.zero_()

    def state_tensors(self) -> list[Tensor]:
        return [self.level, self.heard]

    def publish(self) -> None:
        """Hop thread, after the hop's download (which syncs): mirror the state to the
        host for ``offset_st``. Two scalars; the copy is immediate on a synced stream."""
        self._pair[0].copy_(self.level)
        self._pair[1].copy_(self.heard)
        self._host.copy_(self._pair)

    @property
    def offset_st(self) -> float:
        """Any thread: the auto transposition in semitones as of the last ``publish``
        (nan before warm-up). Reads the host mirror only."""
        level, heard = self._host.tolist()
        if heard < REGISTER_WARM_FRAMES:
            return float("nan")
        st = 12.0 * (self._target_log_host - level)
        return min(REGISTER_MAX_ST, max(REGISTER_MIN_ST, st))

    def ratio(self, f0: Tensor, fixed: Tensor, pitch_range: Tensor | None = None,
              freeze: Tensor | None = None, lift_oct: Tensor | None = None) -> Tensor:
        """Per-frame transposition ratio for ``f0`` (Hz, 0 = unvoiced) after updating the
        register from its last ``fresh`` frames; ``fixed`` is the 0-dim manual ratio.
        ``pitch_range`` (0-dim) overrides the set range for this hop; ``freeze`` (0-dim
        bool) keeps the register where it is (a held note, ``NoteDetector``)."""
        tail = f0[-self.fresh:]
        voiced = tail > 0
        count = voiced.sum().to(f0.dtype)
        log_f0 = torch.log2(f0.clamp(min=1.0))
        mean = torch.where(voiced, log_f0[-self.fresh:], 0.0).sum() / count.clamp(min=1.0)
        a = 1.0 - torch.exp(-count / self.tau_frames)
        first = (self.heard == 0) & (count > 0)
        level = torch.where(first, mean, self.level + a * (mean - self.level))
        moving = count > 0
        if freeze is not None:
            moving = moving & ~freeze
        self.level.copy_(torch.where(moving, level, self.level))
        self.heard.copy_((self.heard + count).clamp(max=1e6))
        shift = (self.target_log - self.level).clamp(REGISTER_MIN_ST / 12, REGISTER_MAX_ST / 12)
        # per frame: target * (f0 / level) ** range, as a ratio against f0
        r = self.range if pitch_range is None else pitch_range
        log_ratio = shift + (r - 1.0) * (log_f0 - self.level)
        if lift_oct is not None:  # the sung-register lift, in octaves (NoteDetector)
            log_ratio = log_ratio + lift_oct
        warmed = (self.auto > 0) & (self.heard >= REGISTER_WARM_FRAMES)
        return torch.where(warmed, torch.exp2(log_ratio), fixed.expand_as(f0))


NOTE_REF_ALPHA = 0.1  # per frame: the run's reference pitch follows a slow vibrato


class NoteDetector:
    """Held-note detector on the talker's own F0 (docs/DEVIATIONS.md §14, Natural mode).

    A run of consecutive voiced frames whose log2 F0 stays within ``tolerance`` of the
    run's reference (an EMA of the run, so vibrato and a slow drift stay inside) is a
    note once it is ``min_frames`` long. Frame by frame over the hop's ``fresh`` frames
    (a static count, so the op sequence never changes): unvoiced resets the run, a frame
    outside the tolerance starts a new run on its own pitch. Graph-safe: 0-dim device
    state, no host sync. ``update`` returns ``(note, frozen)``: the run is at least
    ``min_frames`` / ``freeze_frames`` old. ``enabled`` masks both (0 = never a note).
    """

    def __init__(self, device: torch.device | str) -> None:
        self.ref = torch.zeros((), device=device)  # log2 Hz of the current run
        self.run = torch.zeros((), device=device)  # frames in the current run
        self.enabled = torch.zeros((), device=device)
        self.min_frames = torch.full((), 40.0, device=device)
        self.freeze_frames = torch.full((), 10.0, device=device)
        self.tolerance = torch.full((), 1.0 / 12.0, device=device)  # octaves
        self.fresh = 1
        self.notes = torch.zeros((), device=device)  # hops spent inside a note (stats)
        # Sung-register lift (docs/DEVIATIONS.md 14.8): a held note marks the phrase as
        # sung; ``lift`` (0..1) glides to 1 while it is, back to 0 after ``release``
        # frames without voice. The pipeline adds lift * note_target_offset_st.
        self.lift = torch.zeros((), device=device)
        self.quiet = torch.zeros((), device=device)  # unvoiced frames since the last voice
        self.sung = torch.zeros((), device=device)
        self.lift_alpha = 0.05  # per frame: ~200 ms to 63 %
        self.release = torch.full((), 150.0, device=device)  # 1.5 s of silence ends a sung phrase

    def set(self, enabled: bool, min_ms: float, tolerance_st: float, freeze_ms: float) -> None:
        self.enabled.fill_(1.0 if enabled else 0.0)
        self.min_frames.fill_(max(1.0, round(min_ms / 10.0)))
        self.freeze_frames.fill_(max(1.0, round(freeze_ms / 10.0)))
        self.tolerance.fill_(max(tolerance_st, 0.0) / 12.0)

    def reset(self) -> None:
        self.ref.zero_()
        self.run.zero_()
        self.notes.zero_()
        self.lift.zero_()
        self.quiet.zero_()
        self.sung.zero_()

    def state_tensors(self) -> list[Tensor]:
        return [self.ref, self.run, self.notes, self.lift, self.quiet, self.sung]

    def update(self, f0: Tensor) -> tuple[Tensor, Tensor]:
        """Advance over the last ``fresh`` frames of ``f0`` (Hz, 0 = unvoiced); returns
        0-dim bools ``(note, frozen)`` for this hop."""
        tail = f0[-self.fresh:]
        log_f0 = torch.log2(tail.clamp(min=1.0))
        ref, run = self.ref, self.run
        for k in range(self.fresh):  # static unroll: fresh is a plan constant
            voiced = tail[k] > 0
            inside = voiced & ((log_f0[k] - ref).abs() <= self.tolerance) & (run > 0)
            ref = torch.where(inside, ref + NOTE_REF_ALPHA * (log_f0[k] - ref),
                              torch.where(voiced, log_f0[k], ref))
            run = torch.where(inside, run + 1.0, torch.where(voiced, torch.ones_like(run),
                                                             torch.zeros_like(run)))
        self.ref.copy_(ref)
        self.run.copy_(run)
        on = self.enabled > 0
        note = on & (run >= self.min_frames)
        frozen = on & (run >= self.freeze_frames)
        self.notes.copy_(self.notes + note.to(self.notes.dtype))
        # sung phrase latch and the lift's glide (over the hop's fresh frames)
        voiced_any = (tail > 0).any()
        quiet = torch.where(voiced_any, torch.zeros_like(self.quiet), self.quiet + self.fresh)
        sung = torch.where(note, torch.ones_like(self.sung),
                           torch.where(quiet >= self.release, torch.zeros_like(self.sung), self.sung))
        sung = sung * on.to(sung.dtype)
        a = 1.0 - (1.0 - self.lift_alpha) ** self.fresh
        self.quiet.copy_(quiet)
        self.sung.copy_(sung)
        self.lift.copy_(self.lift + a * (sung - self.lift))
        return note, frozen
