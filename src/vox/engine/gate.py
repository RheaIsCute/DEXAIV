"""Silence gate: energy pre-gate + Silero VAD with hysteresis and hangover, and the
four-buffer protocol.

The decision runs on the host from the hop's device-rate block (docs/research/03 §5,
12 §3.6), after input conditioning (the level the IN meter shows). Silero VAD runs on the
16 kHz decimation of *every* block (CPU, ~0.3 ms per 32 ms chunk), so the decimator and the
VAD's recurrent state never skip audio. The gate opens on a block at least
``threshold_dbfs`` loud that the VAD calls speech, or, before the VAD has caught up, on a
block ``onset_margin_db`` above the threshold (held only ``ONSET_PROBATION_MS`` unless the
VAD confirms it; with ``onset_persist_chunks`` > 1 the rise must also have lasted that many
32 ms VAD chunks, docs/DEVIATIONS.md §14; with ``onset_confirm_ms`` > 0 a 10 ms sub-frame
flag is confirmed over the next 30 ms instead, §15). Once open it stays open down to
``threshold_dbfs - hysteresis_db`` and a VAD probability of ``vad_threshold -
vad_hysteresis`` (Silero's own neg_threshold rule), then for ``hangover_ms`` more, never
less than the stream delay (so the speech still inside the lookahead is rendered before the
drain). No GPU work and no device sync happens here.

What the worker (or the stream simulator) does with a ``GateDecision``:

    speech        -> run the hop normally
    drain         -> gate just closed: ``sola.begin_drain()``, ``pipeline.forget_speech()``
    closed        -> ``pipeline.advance_context()`` (16 kHz ring and pitch cache), emit
                     ``sola.drain_step()`` while ``sola.draining`` else zeros
    onset         -> gate just opened: ``sola.mark_onset()`` (a short fade-in from the
                     silent tail keeps the decoded pre-onset audio) and
                     ``pipeline.on_onset()``; the context ring is already current
                     (advanced while closed)

So the four coupled states stay coherent: the 16 kHz context ring is always the true recent
input (never stale speech, never zeros at onset), the pitch cache advances with it (closed
hops unvoiced), the SOLA tail is faded out exactly once, and the output is silence until
the next onset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vox.engine.chunker import ANALYSIS_SR, ChunkPlan

VAD_CHUNK = 512  # Silero v5 takes exactly 512 samples at 16 kHz (32 ms)
# An energy-only open (the VAD has not seen the onset yet) lasts this long unless the VAD
# confirms speech: long enough for Silero to answer (a chunk or two), short enough that a
# desk thump does not hold the gate open for a whole hangover.
ONSET_PROBATION_MS = 100.0
# ...and it needs a real onset: the block must also stand this far above the quietest
# block of the last ONSET_BACKGROUND_MS. With auto level the room tone of a quiet mic is
# lifted with the voice and can sit above threshold + margin on its own; without this rule
# it reopened the gate on probation every ~120 ms through a pause.
ONSET_RISE_DB = 12.0
ONSET_BACKGROUND_MS = 200.0
# Context-based probation (``onset_confirm_ms`` > 0, report 19 §1.3): the onset flag and its
# confirmation run on 10 ms sub-frames; the decay test looks this far past the flag's peak.
SUB_MS = 10.0
CONFIRM_DECAY_MS = 20.0
ONSET_LOG_MAX = 4000  # flagged onsets kept for the bench (crest factor, HF ratio, verdict)
HF_SPLIT_HZ = 2000.0  # the HF ratio is the energy share above this


@dataclass(frozen=True)
class GateConfig:
    enabled: bool = True
    threshold_dbfs: float = -48.0  # energy pre-gate (calibration.silence_gate_dbfs)
    vad_threshold: float = 0.5  # Silero speech probability to open
    hangover_ms: float = 500.0  # keep open this long after the last speech (200 -> 350 -> 500, 14.7)
    use_vad: bool = True  # False: energy gate only
    hysteresis_db: float = 6.0  # once open, stay open down to threshold - this
    vad_hysteresis: float = 0.15  # once open, stay open down to vad_threshold - this
    onset_margin_db: float = 6.0  # this far above the threshold opens before the VAD agrees
    # A hop of 40 ms or more completes several 512-sample VAD chunks: "max" decides on the
    # most speech-like of them (an onset late in the hop's first chunk still counts),
    # "last" on the newest only (the pre-fix behaviour, docs/DEVIATIONS.md §8).
    vad_reduce: str = "max"
    # Energy-only onset probation (docs/DEVIATIONS.md §14, B4): the rise must have lasted
    # this many completed 32 ms VAD chunks before the gate opens without the VAD. 1 = the
    # block's own level only (today); 2 = one extra chunk, so a 20-50 ms creak or desk tap
    # does not open the gate.
    onset_persist_chunks: int = 1
    # Context-based probation (docs/DEVIATIONS.md §15, report 19 §1.3). 0 = off (the rule
    # above). > 0 replaces it: a 10 ms sub-frame ``onset_margin_db`` over the threshold and
    # ONSET_RISE_DB over the quietest sub-frame of the last 200 ms flags an onset; the gate
    # stays closed while the next ``onset_confirm_ms`` arrive (they sit in the lookahead),
    # then opens if the energy has not fallen ``onset_decay_db`` from its peak within 20 ms
    # (a voice holds, a click does not) or Silero's probability rose ``onset_vad_rise``.
    # A rejected flag mutes new flags for ``onset_pair_ms`` (a key's release impulse).
    # The thresholds are placeholders until the user's typing take.
    onset_confirm_ms: float = 0.0
    onset_decay_db: float = 8.0
    onset_vad_rise: float = 0.25
    onset_pair_ms: float = 120.0


@dataclass(frozen=True)
class GateDecision:
    speech: bool
    onset: bool = False  # first speech hop after a closed period
    drain: bool = False  # first closed hop after speech: fade the SOLA tail out
    rms_dbfs: float = -120.0
    vad_prob: float = 0.0
    energy_only: bool = False  # the onset opened on the energy rule, not the VAD


class SileroVad:
    """Silero VAD on the CPU, fed 512-sample 16 kHz chunks. Control-path ``load``."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.model = None
        self._chunk = torch.zeros(VAD_CHUNK)

    def load(self, assets_dir: Path | None = None) -> None:
        from silero_vad import load_silero_vad

        self.model = load_silero_vad()  # bundled with the package, no download
        self.model.eval()
        # TorchScript's profiling executor re-optimises during the first calls (tens of
        # ms each): take those hits here, never on a live hop.
        with torch.inference_mode():
            for _ in range(8):
                self.model(self._chunk, ANALYSIS_SR)
        self.reset()

    def reset(self) -> None:
        if self.model is not None:
            self.model.reset_states()

    @torch.inference_mode()
    def prob(self, chunk_16k: np.ndarray) -> float:
        """Speech probability of one 512-sample chunk (one host-side float, no CUDA)."""
        self._chunk.numpy()[:] = chunk_16k
        return float(self.model(self._chunk, ANALYSIS_SR).item())


class _Decimator:
    """Stateful FIR decimation ``device_sr -> 16 kHz`` for the VAD only (host, numpy).

    Integer factors (48 k -> 3, 32 k -> 2) use a 63-tap windowed-sinc low-pass carried
    across blocks; anything else falls back to a per-block polyphase resample (VAD-grade).
    """

    def __init__(self, device_sr: int, block: int) -> None:
        self.factor = device_sr // ANALYSIS_SR if device_sr % ANALYSIS_SR == 0 else 0
        self.device_sr = device_sr
        self.out_len = block * ANALYSIS_SR // device_sr
        if self.factor > 1:
            taps = 63
            n = np.arange(taps) - (taps - 1) / 2
            cutoff = 0.45 / self.factor  # cycles/sample, below the 16 kHz Nyquist
            self.taps = (2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(taps)).astype(
                np.float32)
            self.taps /= self.taps.sum()
            self.state = np.zeros(taps - 1, dtype=np.float32)
        else:
            self.taps = None

    def reset(self) -> None:
        if self.taps is not None:
            self.state[:] = 0

    def __call__(self, block: np.ndarray) -> np.ndarray:
        if self.factor == 1:
            return block
        if self.taps is None:
            from scipy.signal import resample_poly

            g = math.gcd(self.device_sr, ANALYSIS_SR)
            return resample_poly(block, ANALYSIS_SR // g, self.device_sr // g).astype(np.float32)
        x = np.concatenate((self.state, block))
        self.state[:] = x[-len(self.state):]
        return np.convolve(x, self.taps, mode="valid")[::self.factor]


def rms_dbfs(block: np.ndarray) -> float:
    return float(10.0 * math.log10(float(np.mean(np.square(block, dtype=np.float64))) + 1e-12))


class SilenceGate:
    """The gate state machine. One ``decide`` per hop, on the (conditioned) host block."""

    def __init__(self, plan: ChunkPlan, config: GateConfig = GateConfig(),
                 vad: SileroVad | None = None) -> None:
        self.plan = plan
        self.config = config
        self.vad = vad if config.use_vad else None
        if self.vad is not None:
            self.vad.threshold = config.vad_threshold
        self._decimate = _Decimator(plan.device_sr, plan.block)
        self._set_hops(config)
        self._recent = np.full(max(1, math.ceil(ONSET_BACKGROUND_MS / plan.block_ms)), -120.0)
        self._acc = np.zeros(VAD_CHUNK + self._decimate.out_len, dtype=np.float32)
        self._acc_len = 0
        self._chunk_levels = np.full(4, -120.0)  # dBFS of the last completed VAD chunks
        self._sub = max(1, plan.device_sr * int(SUB_MS) // 1000)
        self._sub_bg = np.full(max(1, int(ONSET_BACKGROUND_MS / SUB_MS)), -120.0)
        # Every flagged onset, for calibration (report 19 §1.3: log, do not threshold):
        # dicts of time, level, rise, crest factor, HF ratio and the verdict.
        from collections import deque

        self.onset_log: deque[dict] = deque(maxlen=ONSET_LOG_MAX)
        self.transitions = 0
        self.reset()

    def _set_hops(self, config: GateConfig) -> None:
        # Never shorter than the stream delay: the output trails the input by it, so a
        # shorter hangover would drain before the speech inside the lookahead is rendered.
        self.hangover_hops = max(1, math.ceil(config.hangover_ms / self.plan.block_ms),
                                 math.ceil(self.plan.stream_delay / self.plan.block))
        self.probation_hops = min(self.hangover_hops,
                                  max(1, math.ceil(ONSET_PROBATION_MS / self.plan.block_ms)))

    def reset(self) -> None:
        self.open = False
        self._remaining = 0  # hangover hops left
        self._vad_prob = 0.0
        self._acc_len = 0
        self._recent[:] = -120.0  # block levels of the last ONSET_BACKGROUND_MS
        self._recent_pos = 0
        self._chunk_levels[:] = -120.0
        self._sub_bg[:] = -120.0
        self._sub_pos = 0  # absolute sub-frame index of the next block's first sub-frame
        self._cand: dict | None = None  # the flagged onset awaiting confirmation
        self._mute_until = 0  # sub-frame index before which no new flag is raised
        self._vad_before = 0.0  # the VAD probability before this block's chunks
        self._decimate.reset()
        if self.vad is not None:
            self.vad.reset()

    def set_config(self, config: GateConfig) -> None:
        """Hot update (threshold, hangover, enable). Disabling opens the gate at once."""
        self.config = config
        self._set_hops(config)
        if self.vad is not None:
            self.vad.threshold = config.vad_threshold

    def _vad_update(self, block: np.ndarray) -> None:
        """Feed the block's 16 kHz decimation to Silero, one call per completed 512-sample
        chunk. Runs on every block, quiet or not, so neither the FIR state nor the VAD's
        recurrent state ever skips audio (the VAD also sees the way into silence)."""
        if self.vad is None:
            return
        chunk16 = self._decimate(block)
        n = len(chunk16)
        self._acc[self._acc_len:self._acc_len + n] = chunk16
        self._acc_len += n
        peak = -1.0  # no chunk completed this block: the previous probability stands
        while self._acc_len >= VAD_CHUNK:
            prob = self.vad.prob(self._acc[:VAD_CHUNK])
            peak = max(peak, prob) if self.config.vad_reduce == "max" else prob
            # the chunk's own level, for the onset persistence rule
            self._chunk_levels[:-1] = self._chunk_levels[1:]
            self._chunk_levels[-1] = rms_dbfs(self._acc[:VAD_CHUNK])
            self._acc[:self._acc_len - VAD_CHUNK] = self._acc[VAD_CHUNK:self._acc_len]
            self._acc_len -= VAD_CHUNK
        if peak >= 0.0:
            self._vad_prob = peak

    def _rise_persisted(self) -> bool:
        """The energy-only onset's extra rule: the last ``onset_persist_chunks`` completed
        VAD chunks all stood ``onset_margin_db`` over the threshold (the VAD-less gate has
        no chunks: the block's level alone decides)."""
        need = self.config.onset_persist_chunks
        if need <= 1 or self.vad is None:
            return True
        cfg = self.config
        return bool(np.all(self._chunk_levels[-need:] >= cfg.threshold_dbfs + cfg.onset_margin_db))

    def _features(self, x: np.ndarray, start: int) -> dict:
        """Crest factor (dB) and HF energy share of one flagged sub-frame, logged only."""
        rms = math.sqrt(float(np.mean(np.square(x, dtype=np.float64)))) + 1e-12
        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), 1.0 / self.plan.device_sr)
        return {"t": start * self._sub / self.plan.device_sr,
                "crest_db": 20.0 * math.log10(float(np.max(np.abs(x))) / rms + 1e-12),
                "hf_ratio": float(spec[freqs >= HF_SPLIT_HZ].sum() / (spec.sum() + 1e-20))}

    def _confirm(self, block: np.ndarray, armed: bool) -> bool:
        """Context-based probation over this block's sub-frames: True when a flagged onset
        is confirmed in it (the gate opens energy-only). Flags are raised only while
        ``armed`` (the gate closed); the background is tracked either way."""
        cfg = self.config
        n = len(block) // self._sub
        levels = [rms_dbfs(block[k * self._sub:(k + 1) * self._sub]) for k in range(n)]
        need = max(1, math.ceil(cfg.onset_confirm_ms / SUB_MS))
        decay_n = max(1, round(CONFIRM_DECAY_MS / SUB_MS))
        confirmed = False
        for k, lv in enumerate(levels):
            j = self._sub_pos + k
            cand = self._cand
            if cand is None and armed and not confirmed and j >= self._mute_until:
                bg = float(self._sub_bg.min())
                if lv >= cfg.threshold_dbfs + cfg.onset_margin_db and lv >= bg + ONSET_RISE_DB:
                    x = block[k * self._sub:(k + 1) * self._sub]
                    cand = self._cand = {"j": j, "levels": [lv], **self._features(x, j),
                                         "level_db": lv, "rise_db": lv - bg,
                                         "vad_before": self._vad_before}
            elif cand is not None:
                cand["levels"].append(lv)
            self._sub_bg[j % len(self._sub_bg)] = lv
            if cand is None or len(cand["levels"]) <= need:
                continue
            # decide: the peak is in the flagged sub-frame or the next; did it decay?
            lvs = cand["levels"]
            p = int(np.argmax(lvs[:2]))
            after = lvs[p + 1:p + 1 + decay_n]
            held = bool(after) and min(after) > lvs[p] - cfg.onset_decay_db
            vad = self._vad_prob - cand["vad_before"] >= cfg.onset_vad_rise
            verdict = "held" if held else "vad" if vad else "rejected"
            entry = {k2: v for k2, v in cand.items() if k2 != "levels"}
            entry.update(verdict=verdict, levels=[round(v, 1) for v in lvs],
                         vad_after=self._vad_prob)
            self.onset_log.append(entry)
            self._cand = None
            if verdict == "rejected":
                self._mute_until = cand["j"] + max(1, round(cfg.onset_pair_ms / SUB_MS))
            else:
                confirmed = True
        self._sub_pos += n
        return confirmed

    def decide(self, block: np.ndarray) -> GateDecision:
        level = rms_dbfs(block)
        self._vad_before = self._vad_prob
        self._vad_update(block)
        cfg = self.config
        if not cfg.enabled:
            onset = not self.open
            self.open = True
            return GateDecision(True, onset=onset, rms_dbfs=level, vad_prob=self._vad_prob)
        # Hysteresis: opening takes the full threshold and VAD probability, staying open
        # only the lower ones, so a soft syllable inside a phrase does not close the gate.
        if self.open:
            loud = level >= cfg.threshold_dbfs - cfg.hysteresis_db
            voiced = self._vad_prob >= cfg.vad_threshold - cfg.vad_hysteresis
        else:
            loud = level >= cfg.threshold_dbfs
            voiced = self._vad_prob >= cfg.vad_threshold
        if self.vad is None:
            voiced = True
        # The VAD lags the block by up to one chunk (32 ms): a clear onset opens at once
        # rather than losing its first hop, on probation until the VAD has heard it.
        background = float(self._recent.min())
        self._recent[self._recent_pos] = level
        self._recent_pos = (self._recent_pos + 1) % len(self._recent)
        if cfg.onset_confirm_ms > 0:
            if self.open:
                self._cand = None  # the gate is open anyway: nothing to confirm
            strong = self._confirm(block, armed=not self.open) and not self.open
        else:
            strong = (not self.open and level >= cfg.threshold_dbfs + cfg.onset_margin_db
                      and level >= background + ONSET_RISE_DB and self._rise_persisted())
            if strong:  # log the legacy rule's onsets too (loudest sub-frame)
                n = len(block) // self._sub
                k = int(np.argmax([np.mean(np.square(block[i * self._sub:(i + 1) * self._sub]))
                                   for i in range(n)]))
                self.onset_log.append({**self._features(block[k * self._sub:(k + 1) * self._sub],
                                                        self._sub_pos + k),
                                       "level_db": level, "rise_db": level - background,
                                       "verdict": "legacy"})
            self._sub_pos += len(block) // self._sub
        energy_only = False
        if loud and voiced:
            self._remaining = self.hangover_hops
            speech = True
        elif strong:
            self._remaining = self.probation_hops - 1
            # this level is now the background: a second energy-only open needs a new rise
            self._recent[:] = level
            speech = True
            energy_only = True
        elif self._remaining > 0:
            self._remaining -= 1  # this hop is inside the hangover: still open
            speech = True
        else:
            speech = False
        onset = speech and not self.open
        drain = self.open and not speech
        if onset or drain:
            self.transitions += 1
        self.open = speech
        return GateDecision(speech, onset=onset, drain=drain, rms_dbfs=level,
                            vad_prob=self._vad_prob, energy_only=onset and energy_only)
