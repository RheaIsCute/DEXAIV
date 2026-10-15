"""Silence gate: energy pre-gate + Silero VAD with hysteresis and hangover, and the
four-buffer protocol.

The decision runs on the host from the hop's device-rate block (docs/research/03 §5,
12 §3.6), after input conditioning (the level the IN meter shows). Silero VAD runs on the
16 kHz decimation of *every* block (CPU, ~0.3 ms per 32 ms chunk), so the decimator and the
VAD's recurrent state never skip audio. The gate opens on a block at least
``threshold_dbfs`` loud that the VAD calls speech, or, before the VAD has caught up, on a
block ``onset_margin_db`` above the threshold (held only ``ONSET_PROBATION_MS`` unless the
VAD confirms it). Once open it stays open down to ``threshold_dbfs - hysteresis_db`` and a
VAD probability of ``vad_threshold - vad_hysteresis`` (Silero's own neg_threshold rule),
then for ``hangover_ms`` more. No GPU work and no device sync happens here.

What the worker (or the stream simulator) does with a ``GateDecision``:

    speech        -> run the hop normally
    drain         -> gate just closed: ``sola.begin_drain()``, ``pipeline.forget_speech()``
    closed        -> ``pipeline.advance_context()`` (16 kHz ring and pitch cache), emit
                     ``sola.drain_step()`` while ``sola.draining`` else zeros
    onset         -> gate just opened: nothing to do, the zero SOLA tail fades the first
                     hop in; the context ring is already current (advanced while closed)

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


@dataclass(frozen=True)
class GateConfig:
    enabled: bool = True
    threshold_dbfs: float = -48.0  # energy pre-gate (calibration.silence_gate_dbfs)
    vad_threshold: float = 0.5  # Silero speech probability to open
    hangover_ms: float = 350.0  # keep open this long after the last speech (was 200)
    use_vad: bool = True  # False: energy gate only
    hysteresis_db: float = 6.0  # once open, stay open down to threshold - this
    vad_hysteresis: float = 0.15  # once open, stay open down to vad_threshold - this
    onset_margin_db: float = 6.0  # this far above the threshold opens before the VAD agrees


@dataclass(frozen=True)
class GateDecision:
    speech: bool
    onset: bool = False  # first speech hop after a closed period
    drain: bool = False  # first closed hop after speech: fade the SOLA tail out
    rms_dbfs: float = -120.0
    vad_prob: float = 0.0


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
        self.transitions = 0
        self.reset()

    def _set_hops(self, config: GateConfig) -> None:
        self.hangover_hops = max(1, math.ceil(config.hangover_ms / self.plan.block_ms))
        self.probation_hops = min(self.hangover_hops,
                                  max(1, math.ceil(ONSET_PROBATION_MS / self.plan.block_ms)))

    def reset(self) -> None:
        self.open = False
        self._remaining = 0  # hangover hops left
        self._vad_prob = 0.0
        self._acc_len = 0
        self._recent[:] = -120.0  # block levels of the last ONSET_BACKGROUND_MS
        self._recent_pos = 0
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
        while self._acc_len >= VAD_CHUNK:
            self._vad_prob = self.vad.prob(self._acc[:VAD_CHUNK])
            self._acc[:self._acc_len - VAD_CHUNK] = self._acc[VAD_CHUNK:self._acc_len]
            self._acc_len -= VAD_CHUNK

    def decide(self, block: np.ndarray) -> GateDecision:
        level = rms_dbfs(block)
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
        strong = (not self.open and level >= cfg.threshold_dbfs + cfg.onset_margin_db
                  and level >= background + ONSET_RISE_DB)
        if loud and voiced:
            self._remaining = self.hangover_hops
            speech = True
        elif strong:
            self._remaining = self.probation_hops - 1
            # this level is now the background: a second energy-only open needs a new rise
            self._recent[:] = level
            speech = True
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
                            vad_prob=self._vad_prob)
