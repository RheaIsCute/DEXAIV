"""Live microphone conditioning, on the host, before the hop's upload (and before the gate).

    raw block -> 2nd-order high-pass (stateful) -> [de-click] -> [suppression]
              -> gain (auto level + trim) -> soft limit

The full-file converter high-passes and normalises its input; the live engine used to take
the raw microphone, which on a quiet mic means ContentVec and RMVPE see speech 15-20 dB
below what they were trained on, plus any desk thump below 70 Hz. Here:

* **High-pass**: 2nd-order Butterworth at ``highpass_hz``, state carried across blocks.
* **De-click** (``declick``, off by default; docs/DEVIATIONS.md §15): ``Declicker`` clamps
  short transients (keystrokes, taps) inside speech, deciding 20 ms after each candidate
  with the engine's lookahead; the part of a clamp that lands on an earlier block comes back
  as ``retro`` for ``HopPipeline.apply_retro``.
* **Suppression**: RNNoise (``audio/suppression.py``) when ``suppression`` is on, mixed at
  ``suppression_mix``; switching crossfades over 20 ms. Adds 20 ms of delay while on. At
  48 kHz only; elsewhere (or without the library) the stage is absent and the flag ignored.
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
    suppression: bool = False  # RNNoise after the high-pass (LiveSession resolves it)
    suppression_mix: float = 1.0  # 1 = fully denoised; lower keeps some dry (sung notes)
    # "normal" = RNNoise as is; "max" also gates its residual on its own frame VAD
    # (audio/suppression.py; docs/DEVIATIONS.md §14 B4 A/B)
    suppression_strength: str = "normal"
    # In-speech de-clicker (docs/DEVIATIONS.md §15, report 19 §1.4), after the high-pass and
    # before RNNoise and the gain. A sample where the fast envelope jumps ``declick_ratio``
    # over the slow one is a candidate; 20 ms later (inside the engine's 40 ms lookahead)
    # it is a click if the energy fell back within 6 dB of the pre-click level, and is then
    # clamped toward that level + 6 dB with 2 ms ramps. Off by default.
    declick: bool = False
    declick_ratio: float = 4.0  # fast/slow amplitude, +12 dB (placeholder)


def _alpha(block_ms: float, tau_ms: float) -> float:
    return 1.0 - math.exp(-block_ms / tau_ms)


# De-clicker (report 19 §1.4). Every number here is a placeholder until the user's typing
# take gives a distribution (docs/DEVIATIONS.md §15).
DECLICK_FAST_MS = 1.5  # fast envelope time constant
DECLICK_SLOW_MS = 25.0  # slow envelope (the running level the click stands out of)
DECLICK_DECIDE_MS = 20.0  # the decision waits this long after the candidate (lookahead)
DECLICK_SETTLE_MS = 12.0  # a click may stay above the pre-click level this long...
DECLICK_BACK_DB = 6.0  # ...and must then be back within this of it; also the clamp target
DECLICK_RAMP_MS = 2.0
DECLICK_LAG_MS = 2.0  # the fast envelope is compared with the slow one this much earlier
DECLICK_PRE_MS = 6.0  # the level before a candidate: RMS of [-6, -lag] ms
DECLICK_FLOOR_DBFS = -80.0  # candidates quieter than this are ignored
DECLICK_LOOKBACK_MS = 30.0  # the look-back the engine rewrites (inside the 40 ms lookahead)


class Declicker:
    """In-speech de-clicker: look-ahead decision, retroactive clamp (numpy, host).

    Runs on the high-passed block. Two one-pole power followers give a fast and a slow
    amplitude envelope; a candidate is a rising edge of ``fast >= ratio * slow`` (and above
    ``DECLICK_FLOOR_DBFS``), the slow envelope taken ``DECLICK_LAG_MS`` earlier so its own
    rise under the click does not hide it. It is decided once ``DECLICK_DECIDE_MS`` of
    samples after it exist: a click when the signal's RMS over [settle, decide] after the
    candidate (not the fast envelope, whose own decay would still carry the click there in a
    quiet room) is within ``DECLICK_BACK_DB`` of the level just before it (the RMS of the
    ``DECLICK_PRE_MS`` before the candidate: the vowel a clack lands in, the room tone of a
    pause, the silence of a stop closure). A plosive burst passes the first test too, but
    its aspiration and voicing rise above the closure it follows, so it fails the second
    and is left alone. A click's span (candidate to where it
    fell back) is scaled so its RMS sits ``DECLICK_BACK_DB`` over that level (a soft clamp,
    not a zero), with cosine ramps.

    The decision lags the candidate by 20 ms, so part of a clamp can land on samples of an
    earlier block, already passed on. ``process`` returns ``(gain, lookback)``: the gain
    for the current block, and the gain for the last ``lookback_len`` samples before it
    (None when nothing there changed), which the engine applies to its input buffers
    before the next hop (``HopPipeline.apply_retro``). Those samples are still inside the
    40 ms right context, so nothing has been rendered from them yet.
    """

    def __init__(self, sr: int, ratio: float = 4.0) -> None:
        self.sr = sr
        self.ratio = float(ratio)
        n = lambda ms: max(1, round(sr * ms / 1000.0))  # noqa: E731
        self.decide, self.settle, self.ramp = n(DECLICK_DECIDE_MS), n(DECLICK_SETTLE_MS), \
            n(DECLICK_RAMP_MS)
        self.lag = n(DECLICK_LAG_MS)
        self.pre = n(DECLICK_PRE_MS)
        self.lookback_len = n(DECLICK_LOOKBACK_MS)
        self.keep = self.lookback_len + self.decide + self.ramp + self.pre
        self._af = 1.0 - math.exp(-1000.0 / (sr * DECLICK_FAST_MS))
        self._as = 1.0 - math.exp(-1000.0 / (sr * DECLICK_SLOW_MS))
        self.floor = 10.0 ** (DECLICK_FLOOR_DBFS / 20.0)
        self.back = 10.0 ** (DECLICK_BACK_DB / 20.0)
        up = 0.5 - 0.5 * np.cos(np.pi * (np.arange(self.ramp) + 0.5) / self.ramp)
        self._ramp_up = up  # 0 -> 1
        self.reset()

    def reset(self) -> None:
        self._zf = np.zeros(1)
        self._zs = np.zeros(1)
        self._x = np.zeros(0)
        self._ef = np.zeros(0)
        self._es = np.zeros(0)
        self._pos = 0  # absolute index of the next block's first sample
        self._scan = 0  # absolute index from which candidates are still looked for
        self.clicks: list[tuple[float, float]] = []  # (time s, clamp dB), for the bench

    def _envelopes(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from scipy.signal import lfilter

        p = x * x
        pf, self._zf = lfilter([self._af], [1.0, self._af - 1.0], p, zi=self._zf)
        ps, self._zs = lfilter([self._as], [1.0, self._as - 1.0], p, zi=self._zs)
        return np.sqrt(np.maximum(pf, 0.0)), np.sqrt(np.maximum(ps, 0.0))

    def process(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        ef, es = self._envelopes(x)
        hist = len(self._x)
        X = np.concatenate((self._x, x))
        EF = np.concatenate((self._ef, ef))
        ES = np.concatenate((self._es, es))
        base = self._pos - hist  # absolute index of X[0]
        gain = np.ones(len(X))
        lag = min(self.lag, len(ES))
        ES = np.concatenate((np.full(lag, ES[0]), ES[:len(ES) - lag]))  # the earlier level
        mask = (EF >= self.ratio * ES) & (EF > self.floor)
        i = max(self._scan - base, 1)
        while i < len(X):
            edges = np.flatnonzero(mask[i:] & ~mask[i - 1:-1])
            if not len(edges):
                i = len(X)
                break
            ts = i + int(edges[0])
            if ts + self.decide > len(X):
                i = ts  # not decidable yet: look at it again next block
                break
            before = X[max(0, ts - self.pre):max(1, ts - self.lag)]
            ref = max(math.sqrt(float(np.mean(before * before))), self.floor)
            limit = self.back * ref
            after = X[ts + self.settle:ts + self.decide]
            if math.sqrt(float(np.mean(after * after))) <= limit:
                # the click lasts until the energy is back for good (a keystroke's push is
                # two sub-impulses, touch and hit: the dip between them is not the end)
                above = np.flatnonzero(EF[ts:ts + self.settle + 1] > limit)
                end = ts + (int(above[-1]) + 1 if len(above) else 1)
                seg = X[ts:max(end, ts + 1)]
                g = min(1.0, limit / (math.sqrt(float(np.mean(seg * seg))) + 1e-12))
                if g < 1.0:
                    self._clamp(gain, ts, end, g)
                    self.clicks.append(((base + ts) / self.sr, 20.0 * math.log10(g)))
                i = end + 1
            else:
                i = ts + 1
        self._scan = base + i
        # history for the next block (unclamped: the envelopes are the input's own)
        self._x, self._ef = X[-self.keep:], EF[-self.keep:]
        self._es = np.concatenate((self._es, es))[-self.keep:]  # unlagged
        self._pos += len(x)
        lookback = None
        before = gain[max(0, hist - self.lookback_len):hist]
        if (before < 1.0).any():
            lookback = np.ones(self.lookback_len)
            lookback[self.lookback_len - len(before):] = before
        return gain[hist:], lookback

    def _clamp(self, gain: np.ndarray, start: int, end: int, g: float) -> None:
        """Scale [start, end) by g, with a ramp in before and a ramp out after."""
        r = self.ramp
        shape = np.concatenate((1.0 - (1.0 - g) * self._ramp_up, np.full(max(1, end - start), g),
                                g + (1.0 - g) * self._ramp_up))
        lo = start - r
        a, b = max(0, lo), min(len(gain), lo + len(shape))
        gain[a:b] *= shape[a - lo:b - lo]


class InputConditioner:
    """One ``process`` per hop on the worker thread (or the simulator); numpy only."""

    def __init__(self, sr: int, block: int, config: ConditioningConfig | None = None) -> None:
        from vox.audio import suppression

        self.sr = sr
        self.config = config or ConditioningConfig()
        self._sos = None
        self._zi = None
        self._design()
        # Built whenever RNNoise can run here, so suppression can be switched on live.
        self.suppressor = None
        if suppression.unavailable_reason(sr) is None and block % suppression.FRAME == 0:
            self.suppressor = suppression.Suppressor(block, self.config.suppression,
                                                     self.config.suppression_mix,
                                                     self.config.suppression_strength)
        self.declicker: Declicker | None = None
        self._set_declicker()
        # The de-clicker's look-back for the engine (device-rate gains over the last
        # ``declicker.lookback_len`` samples already returned), or None. Valid until the
        # next ``process``.
        self.retro: np.ndarray | None = None
        self.resize(block)
        self.reset()

    # ------------------------------------------------------------------ control path

    def _set_declicker(self) -> None:
        cfg = self.config
        if not cfg.declick:
            self.declicker = None
        elif self.declicker is None:
            self.declicker = Declicker(self.sr, cfg.declick_ratio)
        else:
            self.declicker.ratio = float(cfg.declick_ratio)

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
        if self.suppressor is not None:
            self.suppressor.resize(block)

    def set_config(self, config: ConditioningConfig) -> None:
        """Hot update between hops (toggle, trim, suppression). A new corner frequency
        redesigns the filter (its state restarts: one block of settling)."""
        redesign = config.highpass_hz != self.config.highpass_hz
        self.config = config
        if redesign:
            self._design()
        self._set_declicker()
        if self.suppressor is not None:
            self.suppressor.set(config.suppression, config.suppression_mix,
                                config.suppression_strength)

    def reset(self) -> None:
        if self._zi is not None:
            self._zi[:] = 0.0
        self._power: float | None = None  # mean power of active speech, unknown until speech
        self.agc_db = 0.0  # the AGC's own gain
        self._applied_db = self._total_db()
        self.retro = None
        if self.declicker is not None:
            self.declicker.reset()
        if self.suppressor is not None:
            self.suppressor.reset()

    @property
    def delay_ms(self) -> float:
        """Delay the chain adds to the input right now (RNNoise's 20 ms while it runs)."""
        if self.suppressor is None or not self.config.enabled:
            return 0.0
        return 1000.0 * self.suppressor.delay_samples / self.sr

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
        self.retro = None
        if not cfg.enabled:
            return block
        x = np.asarray(block, dtype=np.float64)
        if self._sos is not None:
            from scipy.signal import sosfilt

            x, self._zi = sosfilt(self._sos, x, zi=self._zi)
        lookback = None
        if self.declicker is not None:
            gain, lookback = self.declicker.process(x)
            x = x * gain
        delay = 0
        if self.suppressor is not None:
            delay = self.suppressor.delay_samples
            x = self.suppressor.process(x)
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
        if lookback is not None:
            self._place_lookback(lookback, delay)
        return self._out

    def _place_lookback(self, lookback: np.ndarray, delay: int) -> None:
        """The de-clicker's look-back gain is on the high-passed timeline; RNNoise (when on)
        delays everything after it by ``delay`` samples. The part that still lands in this
        block's output is applied here, the rest becomes ``retro`` for the engine."""
        n = len(lookback)
        delay = min(delay, n, len(self._out))
        if delay:
            self._out[:delay] *= lookback[n - delay:]
            lookback = np.concatenate((np.ones(delay), lookback[:n - delay]))
        if (lookback < 1.0).any():
            self.retro = lookback


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
