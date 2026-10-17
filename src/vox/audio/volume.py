"""Volume controls: output volume, typed-speech volume and matching typed clips to the
live voice.

``RampedGain`` is the in-place gain the output callback (``streams.DuplexAudio``) and the
clip player (``engine/override.py``) apply: a new value ramps in linearly over ``RAMP_MS``
whatever the block size, so moving a slider never clicks, and it allocates nothing.
``SpeechLevel`` follows the level of the converted voice while it speaks (the worker feeds
it one hop at a time); ``active_level_dbfs`` measures a typed clip the same way, so the
controller can bring a clip to the live voice's level before the typed-speech volume.
"""

from __future__ import annotations

import math

import numpy as np

OUTPUT_DB_RANGE = (-24.0, 6.0)  # output volume slider: everything sent to the output device
TYPED_DB_RANGE = (-12.0, 12.0)  # typed-speech volume slider: typed clips only
DEFAULT_MATCH_DBFS = -20.0  # typed clips are matched to this before any live level is known
MATCH_LIMIT_DB = (-24.0, 18.0)  # auto-match never moves a clip further than this
SPEECH_FLOOR_DBFS = -60.0  # quieter hops / frames are pauses or tails, not speech
LEVEL_TAU_S = 3.0  # the live level follows the last few seconds of speech
RAMP_MS = 10.0
_FLOOR_MS = 10.0 ** (SPEECH_FLOOR_DBFS / 10.0)  # the floor as a mean square


def db_to_gain(db: float) -> float:
    return 10.0 ** (float(db) / 20.0)


def clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(bounds[1], max(bounds[0], float(value)))


class RampedGain:
    """Gain applied in place to consecutive blocks. Set ``target`` (linear) from any
    thread; ``apply`` ramps from the current gain to it over ``ramp`` samples (spanning
    blocks if needed), then holds it. Unity is a no-op: the signal is untouched."""

    def __init__(self, ramp: int, chunk: int = 4096) -> None:
        self.ramp = max(1, int(ramp))
        self.target = 1.0
        self.now = 1.0
        self._goal = 1.0
        self._step = 0.0
        self._left = 0
        self._steps = np.arange(1, chunk + 1, dtype=np.float32)
        self._scratch = np.zeros(chunk, dtype=np.float32)

    @classmethod
    def for_rate(cls, sr: int, ramp_ms: float = RAMP_MS) -> RampedGain:
        return cls(round(sr * ramp_ms / 1000.0))

    def jump(self, gain: float) -> None:
        """Set the gain at once, without a ramp (before any audio flows)."""
        self.target = self.now = self._goal = float(gain)
        self._left = 0

    @property
    def unity(self) -> bool:
        return self.target == 1.0 and self.now == 1.0 and not self._left

    def apply(self, buf: np.ndarray) -> None:
        target = self.target  # read once: the control thread may write it meanwhile
        if target != self._goal:
            self._goal = target
            self._left = self.ramp
            self._step = (target - self.now) / self.ramp
        pos, n = 0, len(buf)
        while self._left and pos < n:
            k = min(n - pos, self._left, len(self._steps))
            ramp = self._scratch[:k]
            np.multiply(self._steps[:k], self._step, out=ramp)
            ramp += self.now
            buf[pos:pos + k] *= ramp
            self._left -= k
            self.now = self._goal if not self._left else self.now + self._step * k
            pos += k
        if pos < n and self.now != 1.0:
            buf[pos:] *= self.now


class SpeechLevel:
    """Running level (dBFS) of a signal over its speech blocks only: an energy average
    with a ``tau_s`` time constant; blocks under ``SPEECH_FLOOR_DBFS`` are skipped. Fed by
    one thread; ``dbfs`` may be read from another (None until something was heard)."""

    def __init__(self, tau_s: float = LEVEL_TAU_S) -> None:
        self.tau_s = tau_s
        self._ms = 0.0
        self.dbfs: float | None = None

    def add(self, block: np.ndarray, seconds: float) -> None:
        if not len(block):
            return
        ms = float(np.dot(block, block)) / len(block)
        if ms < _FLOOR_MS:
            return
        if self.dbfs is None:
            self._ms = ms
        else:
            alpha = min(1.0, seconds / self.tau_s)
            self._ms += alpha * (ms - self._ms)
        self.dbfs = 10.0 * math.log10(self._ms)


def active_level_dbfs(audio: np.ndarray, sr: int, frame_ms: float = 20.0,
                      window_db: float = 30.0) -> float | None:
    """RMS level of the speech in ``audio``: over the frames within ``window_db`` of the
    loudest frame and above ``SPEECH_FLOOR_DBFS`` (pauses and fades do not count). None for
    a clip with no such frame."""
    n = max(1, int(sr * frame_ms / 1000.0))
    frames = len(audio) // n
    if frames == 0:
        return None
    blocks = np.asarray(audio[:frames * n], dtype=np.float64).reshape(frames, n)
    ms = np.mean(blocks * blocks, axis=1)
    top = float(ms.max())
    keep = ms[(ms >= _FLOOR_MS) & (ms >= top * 10.0 ** (-window_db / 10.0))]
    if not len(keep):
        return None
    return 10.0 * math.log10(float(keep.mean()))


def match_gain_db(clip_dbfs: float | None, target_dbfs: float) -> float:
    """dB that bring a clip measured at ``clip_dbfs`` to ``target_dbfs``, within
    ``MATCH_LIMIT_DB``; 0 for a clip without speech."""
    if clip_dbfs is None:
        return 0.0
    return clamp(target_dbfs - clip_dbfs, MATCH_LIMIT_DB)
