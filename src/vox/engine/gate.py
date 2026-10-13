"""Silence gate: energy pre-gate + Silero VAD with hangover, and the four-buffer protocol.

The decision runs on the host from the hop's device-rate block (docs/research/03 §5,
12 §3.6): an RMS threshold first (free), Silero VAD on the hop's 16 kHz decimation only
when the energy passes (CPU, ~0.3 ms per 32 ms chunk), then a hangover that keeps the gate
open ``hangover_ms`` after the last speech. No GPU work and no device sync happens here.

What the worker (or the stream simulator) does with a ``GateDecision``:

    speech        -> run the hop normally
    drain         -> gate just closed: ``sola.begin_drain()``, ``pipeline.forget_speech()``
    closed        -> ``pipeline.advance_context()`` (16 kHz ring only), emit
                     ``sola.drain_step()`` while ``sola.draining`` else zeros
    onset         -> gate just opened: nothing to do, the zero SOLA tail fades the first
                     hop in; the context ring is already current (advanced while closed)

So the four coupled states stay coherent: the 16 kHz context ring is always the true recent
input (never stale speech, never zeros at onset), the pitch cache and feature buffers are
zeroed on close, the SOLA tail is faded out exactly once, and the output is silence until
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


@dataclass(frozen=True)
class GateConfig:
    enabled: bool = True
    threshold_dbfs: float = -48.0  # energy pre-gate (calibration.silence_gate_dbfs)
    vad_threshold: float = 0.5  # Silero speech probability
    hangover_ms: float = 200.0  # keep open this long after the last speech
    use_vad: bool = True  # False: energy gate only


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
    """The gate state machine. One ``decide`` per hop, on the host block."""

    def __init__(self, plan: ChunkPlan, config: GateConfig = GateConfig(),
                 vad: SileroVad | None = None) -> None:
        self.plan = plan
        self.config = config
        self.vad = vad if config.use_vad else None
        if self.vad is not None:
            self.vad.threshold = config.vad_threshold
        self._decimate = _Decimator(plan.device_sr, plan.block)
        self.hangover_hops = max(1, math.ceil(config.hangover_ms / plan.block_ms))
        self._acc = np.zeros(VAD_CHUNK + self._decimate.out_len, dtype=np.float32)
        self._acc_len = 0
        self.transitions = 0
        self.reset()

    def reset(self) -> None:
        self.open = False
        self._remaining = 0  # hangover hops left
        self._vad_prob = 0.0
        self._acc_len = 0
        self._decimate.reset()
        if self.vad is not None:
            self.vad.reset()

    def set_config(self, config: GateConfig) -> None:
        """Hot update (threshold, hangover, enable). Disabling opens the gate at once."""
        self.config = config
        self.hangover_hops = max(1, math.ceil(config.hangover_ms / self.plan.block_ms))
        if self.vad is not None:
            self.vad.threshold = config.vad_threshold

    def _vad_speech(self, block: np.ndarray) -> bool:
        """Run Silero on every completed 512-sample chunk of the 16 kHz decimation."""
        if self.vad is None:
            return True
        chunk16 = self._decimate(block)
        n = len(chunk16)
        self._acc[self._acc_len:self._acc_len + n] = chunk16
        self._acc_len += n
        while self._acc_len >= VAD_CHUNK:
            self._vad_prob = self.vad.prob(self._acc[:VAD_CHUNK])
            self._acc[:self._acc_len - VAD_CHUNK] = self._acc[VAD_CHUNK:self._acc_len]
            self._acc_len -= VAD_CHUNK
        return self._vad_prob >= self.config.vad_threshold

    def decide(self, block: np.ndarray) -> GateDecision:
        level = rms_dbfs(block)
        if not self.config.enabled:
            onset = not self.open
            self.open = True
            return GateDecision(True, onset=onset, rms_dbfs=level, vad_prob=self._vad_prob)
        loud = level >= self.config.threshold_dbfs
        # The VAD sees every block while it is loud enough to matter, so its recurrent
        # state tracks the signal; quiet blocks only decay the hangover.
        speech_now = loud and self._vad_speech(block)
        if speech_now:
            self._remaining = self.hangover_hops
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
