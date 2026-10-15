"""Live microphone conditioning, on the host, before the hop's upload (and before the gate).

    raw block -> 2nd-order high-pass (stateful) -> gain (auto level + trim) -> soft limit

The full-file converter high-passes and normalises its input; the live engine used to take
the raw microphone, which on a quiet mic means ContentVec and RMVPE see speech 15-20 dB
below what they were trained on, plus any desk thump below 70 Hz. Here:

* **High-pass**: 2nd-order Butterworth at ``highpass_hz``, state carried across blocks.
* **Auto level**: an AGC toward ``target_dbfs`` of *active* speech RMS. The speech level is
  a moving average of block power over the blocks above the calibrated noise floor
  (+ ``freeze_margin_db``); below it level and gain are both held, so pauses are not pumped
  up. Gain: slow up, fast down, ramped linearly across each block (no zipper noise).
  Bounded by ``min_gain_db`` / ``max_gain_db``.
* **Trim**: the GUI's input-gain slider (plus calibration's ``input_gain_db``), on top of the
  AGC, or alone when auto level is off.
* **Soft limit**: transparent below ``limit_dbfs`` - 3 dB, a tanh knee up to ``limit_dbfs``.

The gate and the IN meter see the conditioned block. Typed clips (TTS, already normalised)
never pass through here: they go straight to the output ring (engine/override.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Time constants (ms). The level is the mean power of active speech over about half a
# second (a symmetric average, so it is the active RMS and not a peak follower); the gain
# rises over ~half a second and falls fast, so a talker who gets louder does not run into
# the limiter for long.
LEVEL_MS = 500.0
GAIN_RISE_MS = 400.0
GAIN_FALL_MS = 60.0


@dataclass(frozen=True)
class ConditioningConfig:
    enabled: bool = True  # False: blocks pass untouched
    auto_level: bool = True  # the GUI's "Auto level" toggle
    trim_db: float = 0.0  # input-gain slider + calibration.input_gain_db (on top of the AGC)
    highpass_hz: float = 70.0  # 0 disables; low voices reach ~70 Hz (the F0 p10 seen here)
    target_dbfs: float = -20.0  # active speech RMS the AGC aims for
    max_gain_db: float = 18.0
    min_gain_db: float = -12.0
    noise_floor_dbfs: float = -60.0  # calibration.noise_floor_dbfs
    freeze_margin_db: float = 6.0  # gain held on blocks below floor + this
    limit_dbfs: float = -3.0  # soft-limiter ceiling


def _alpha(block_ms: float, tau_ms: float) -> float:
    return 1.0 - math.exp(-block_ms / tau_ms)


class InputConditioner:
    """One ``process`` per hop on the worker thread (or the simulator); numpy only."""

    def __init__(self, sr: int, block: int, config: ConditioningConfig | None = None) -> None:
        self.sr = sr
        self.config = config or ConditioningConfig()
        self._sos = None
        self._zi = None
        self._design()
        self.resize(block)
        self.reset()

    # ------------------------------------------------------------------ control path

    def _design(self) -> None:
        from scipy.signal import butter

        hz = self.config.highpass_hz
        self._sos = butter(2, hz, btype="highpass", fs=self.sr, output="sos") if hz > 0 else None
        self._zi = np.zeros((1, 2)) if self._sos is not None else None

    def resize(self, block: int) -> None:
        """New hop size (live rebuild): filter state and gain carry over."""
        self.block = block
        ms = 1000.0 * block / self.sr
        self._a_level = _alpha(ms, LEVEL_MS)
        self._a_gain_up, self._a_gain_down = _alpha(ms, GAIN_RISE_MS), _alpha(ms, GAIN_FALL_MS)
        self._ramp = (np.arange(1, block + 1, dtype=np.float64) / block)
        self._out = np.zeros(block, dtype=np.float32)

    def set_config(self, config: ConditioningConfig) -> None:
        """Hot update between hops (toggle, trim). A new corner frequency redesigns the
        filter (its state restarts: one block of settling)."""
        redesign = config.highpass_hz != self.config.highpass_hz
        self.config = config
        if redesign:
            self._design()

    def reset(self) -> None:
        if self._zi is not None:
            self._zi[:] = 0.0
        self._power: float | None = None  # mean power of active speech, unknown until speech
        self.agc_db = 0.0  # the AGC's own gain
        self._applied_db = self._total_db()

    def _total_db(self) -> float:
        cfg = self.config
        return (self.agc_db if cfg.auto_level else 0.0) + cfg.trim_db

    @property
    def gain_db(self) -> float:
        """The gain applied to the last block (AGC + trim), for the meters."""
        return self._applied_db if self.config.enabled else 0.0

    @property
    def level_db(self) -> float | None:
        """Active speech RMS before the gain (None until the first active block)."""
        return None if self._power is None else 10.0 * math.log10(self._power + 1e-12)

    # ---------------------------------------------------------------------- hot path

    def process(self, block: np.ndarray) -> np.ndarray:
        """Condition one block. Returns an internal buffer, valid until the next call."""
        cfg = self.config
        if not cfg.enabled:
            return block
        x = np.asarray(block, dtype=np.float64)
        if self._sos is not None:
            from scipy.signal import sosfilt

            x, self._zi = sosfilt(self._sos, x, zi=self._zi)
        power = float(np.mean(x * x))
        if 10.0 * math.log10(power + 1e-12) >= cfg.noise_floor_dbfs + cfg.freeze_margin_db:
            # Speech (or at least signal): track its level and steer the gain. In pauses
            # neither moves, so the room tone keeps the gain the last speech set.
            self._power = power if self._power is None else (
                self._power + self._a_level * (power - self._power))
            want = min(cfg.max_gain_db, max(cfg.min_gain_db, cfg.target_dbfs - self.level_db))
            a = self._a_gain_up if want > self.agc_db else self._a_gain_down
            self.agc_db += a * (want - self.agc_db)
        start = 10.0 ** (self._applied_db / 20.0)
        self._applied_db = self._total_db()
        end = 10.0 ** (self._applied_db / 20.0)
        y = x * (start + (end - start) * self._ramp)
        self._out[:] = soft_limit(y, cfg.limit_dbfs)
        return self._out


def soft_limit(x: np.ndarray, limit_dbfs: float) -> np.ndarray:
    """Identity below ``limit - 3 dB``; above it a tanh knee that approaches ``limit``
    (continuous, slope 1 at the knee)."""
    ceiling = 10.0 ** (limit_dbfs / 20.0)
    knee = 10.0 ** ((limit_dbfs - 3.0) / 20.0)
    mag = np.abs(x)
    over = mag > knee
    if not over.any():
        return x
    span = ceiling - knee
    out = x.copy()
    out[over] = np.sign(x[over]) * (knee + span * np.tanh((mag[over] - knee) / span))
    return out
