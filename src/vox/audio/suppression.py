"""Noise suppression on the microphone: RNNoise, 48 kHz, 10 ms frames (docs/DEVIATIONS.md §8).

The library is xiph's RNNoise as shipped in the ``pyrnnoise`` wheel (``rnnoise.dll``). Only
its DLL is used, through ctypes on preallocated buffers: the package's own Python layer
allocates per frame and imports audio-file tooling the engine does not need, so it is
installed ``--no-deps`` and never imported.

Placement: inside ``InputConditioner``, after the high-pass and before auto level, so the
AGC and the gate measure the denoised signal. Runs on the worker thread, one block per hop
(blocks are whole 10 ms frames at 48 kHz, so there is no frame buffering).

RNNoise's output trails its input by ``DELAY`` samples (20 ms, measured). With
``mix`` < 1 the dry part is taken through the same delay so the two stay aligned (no comb
filter). ``strength = "max"`` (docs/DEVIATIONS.md §14, B4) also uses the speech probability
RNNoise returns per 10 ms frame: frames it calls non-speech (below ``MAX_VAD``) are taken
down a further ``MAX_FLOOR`` (one-pole smoothed over frames, so no zipper), which removes
the residual it leaves on creaks and bumps between sentences. Switching on or off crossfades over ``FADE_MS`` between the undelayed dry signal
and that delayed mix; after an idle period the network first runs ``WARMUP_MS`` unheard so
its recurrent state is settled before it is faded in.
"""

from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path

import numpy as np

SR = 48000
FRAME = 480  # RNNoise's fixed frame: 10 ms at 48 kHz
DELAY = 960  # output trails input by 20 ms (measured on speech, corr 0.99)
SCALE = 32768.0  # RNNoise works on int16-range floats
FADE_MS = 20.0
WARMUP_MS = 100.0
STRENGTHS = ("normal", "max")
MAX_VAD = 0.5  # "max": frames under this speech probability are attenuated further...
MAX_FLOOR = 0.1  # ...down to this gain (-20 dB), the gain gliding over MAX_RELEASE_FRAMES
MAX_RELEASE_FRAMES = 5  # 50 ms one-pole on the extra gain (attack is the same constant)

_LIB: ctypes.CDLL | None = None
_LIB_ERROR: str | None = None


def _library() -> ctypes.CDLL | None:
    """Load ``rnnoise.dll`` from the installed ``pyrnnoise`` wheel once (no package import)."""
    global _LIB, _LIB_ERROR
    if _LIB is not None or _LIB_ERROR is not None:
        return _LIB
    spec = importlib.util.find_spec("pyrnnoise")
    if spec is None or not spec.submodule_search_locations:
        _LIB_ERROR = "pyrnnoise is not installed (pip install --no-deps pyrnnoise==0.4.5)"
        return None
    folder = Path(next(iter(spec.submodule_search_locations)))
    names = ("rnnoise.dll", "librnnoise.so", "librnnoise.dylib")
    path = next((folder / n for n in names if (folder / n).is_file()), None)
    if path is None:
        _LIB_ERROR = f"no RNNoise library in {folder}"
        return None
    try:
        lib = ctypes.CDLL(str(path))
        lib.rnnoise_create.restype = ctypes.c_void_p
        lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        lib.rnnoise_process_frame.restype = ctypes.c_float
        lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.rnnoise_get_frame_size.restype = ctypes.c_int
        if lib.rnnoise_get_frame_size() != FRAME:
            raise OSError(f"frame size {lib.rnnoise_get_frame_size()} != {FRAME}")
    except (OSError, AttributeError) as err:
        _LIB_ERROR = f"could not load {path.name}: {err}"
        return None
    _LIB = lib
    return lib


def unavailable_reason(sr: int = SR) -> str | None:
    """None when suppression can run at ``sr``, else why not (for the UI)."""
    if sr != SR:
        return f"RNNoise needs {SR} Hz (the device runs at {sr})"
    _library()
    return _LIB_ERROR


class Suppressor:
    """One RNNoise state, a dry delay line and the on/off crossfade. Allocation-free per
    block: every buffer is made in ``resize``; ``process`` only writes into them."""

    def __init__(self, block: int, enabled: bool = False, mix: float = 1.0,
                 strength: str = "normal") -> None:
        lib = _library()
        if lib is None:
            raise RuntimeError(_LIB_ERROR or "RNNoise unavailable")
        self._lib = lib
        self._state = lib.rnnoise_create(None)
        self.enabled = enabled
        self.mix = float(mix)
        self._mix_now = self.mix
        self.strength = strength
        self._vad_gain = 1.0  # "max": the extra per-frame gain, smoothed
        self.last_vad = 0.0  # RNNoise's speech probability of the last frame (stats)
        self._step = 1.0 / (SR * FADE_MS / 1000.0)  # wet weight change per sample
        self._warmup = int(SR * WARMUP_MS / 1000.0)
        self.resize(block)
        self.reset()

    def __del__(self) -> None:
        state, self._state = getattr(self, "_state", None), None
        if state:
            self._lib.rnnoise_destroy(state)

    # ------------------------------------------------------------------ control path

    def resize(self, block: int) -> None:
        if block % FRAME:
            raise ValueError(f"block {block} is not a whole number of {FRAME}-sample frames")
        self.block = block
        self._frames = np.zeros(block, dtype=np.float32)
        base = self._frames.ctypes.data
        self._ptrs = [base + 4 * i for i in range(0, block, FRAME)]
        line = np.zeros(DELAY + block, dtype=np.float64)
        if hasattr(self, "_line"):  # keep the recent input across a hop change
            keep = min(DELAY, len(self._line))
            line[DELAY - keep:DELAY] = self._line[len(self._line) - keep:]
        self._line = line
        self._wet = np.zeros(block, dtype=np.float64)
        self._weights = np.zeros(block, dtype=np.float64)
        self._frame_ramp = np.arange(1, FRAME + 1, dtype=np.float64) / FRAME
        self._ramp = np.arange(1, block + 1, dtype=np.float64)
        self._out = np.zeros(block, dtype=np.float64)

    def reset(self) -> None:
        """Start of a stream: fresh network state, empty delay line, no crossfade."""
        self._lib.rnnoise_destroy(self._state)
        self._state = self._lib.rnnoise_create(None)
        self._line[:] = 0.0
        self._w = 1.0 if self.enabled else 0.0
        self._ran = self._warmup if self.enabled else 0
        self._mix_now = self.mix
        self._vad_gain = 1.0

    def set(self, enabled: bool | None = None, mix: float | None = None,
            strength: str | None = None) -> None:
        """Hot: takes effect over the next blocks (crossfade, mix glide)."""
        if enabled is not None:
            self.enabled = bool(enabled)
        if mix is not None:
            self.mix = min(1.0, max(0.0, float(mix)))
        if strength is not None:
            if strength not in STRENGTHS:
                raise ValueError(f"suppression strength must be one of {STRENGTHS}")
            self.strength = strength

    @property
    def active(self) -> bool:
        """RNNoise is running: on, or still fading out."""
        return self.enabled or self._w > 0.0

    @property
    def delay_samples(self) -> int:
        """What the suppressor adds to the input path right now (0 when idle)."""
        return DELAY if self.active else 0

    # ---------------------------------------------------------------------- hot path

    def process(self, x: np.ndarray) -> np.ndarray:
        """One block (float64, ``block`` samples). Returns ``x`` itself while idle, else an
        internal buffer valid until the next call."""
        n = self.block
        line = self._line
        line[DELAY:] = x  # line[:n] is now the input DELAY samples ago, aligned with RNNoise
        if not self.active:
            line[:DELAY] = line[n:n + DELAY]
            self._ran = 0
            return x
        # Casts by assignment, scaling in place: a mixed-dtype ufunc would allocate its
        # cast buffers on every call.
        frames, wet = self._frames, self._wet
        frames[:] = x
        frames *= SCALE
        process = self._lib.rnnoise_process_frame
        state = self._state
        strong = self.strength == "max"
        gain = self._vad_gain
        for k, ptr in enumerate(self._ptrs):
            self.last_vad = vad = process(state, ptr, ptr)
            if strong:
                # extra attenuation on frames RNNoise calls non-speech, glided per frame
                want = 1.0 if vad >= MAX_VAD else MAX_FLOOR
                nxt = gain + (want - gain) / MAX_RELEASE_FRAMES
                seg = frames[k * FRAME:(k + 1) * FRAME]
                seg *= gain + (nxt - gain) * self._frame_ramp
                gain = nxt
        self._vad_gain = gain
        wet[:] = frames
        wet *= 1.0 / SCALE
        # "on" signal: the denoised voice, with (1 - mix) of the equally delayed dry input.
        mix0, mix1 = self._mix_now, self.mix
        weights = self._weights
        np.multiply(self._ramp, (mix1 - mix0) / n, out=weights)
        weights += mix0  # per-sample mix: a glide, no zipper noise on a slider move
        wet -= line[:n]
        wet *= weights
        wet += line[:n]
        self._mix_now = mix1
        warmed = self._ran >= self._warmup
        self._ran += n
        # Toggle crossfade between the undelayed dry signal and the "on" signal.
        w0 = self._w
        target = 1.0 if self.enabled and warmed else (self._w if self.enabled else 0.0)
        if target == w0 and w0 in (0.0, 1.0):
            line[:DELAY] = line[n:n + DELAY]
            if w0 == 1.0:
                self._out[:] = wet
                return self._out
            return x
        sign = 1.0 if target > w0 else -1.0
        np.multiply(self._ramp, sign * self._step, out=weights)
        weights += w0
        np.clip(weights, 0.0, 1.0, out=weights)
        self._w = float(weights[-1])
        out = self._out
        np.subtract(wet, x, out=out)
        out *= weights
        out += x
        line[:DELAY] = line[n:n + DELAY]
        return out
