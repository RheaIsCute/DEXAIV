"""Priority clip override on the worker's output: a rendered clip replaces the live voice.

The worker keeps converting the microphone; just before a hop goes into the output ring,
``ClipOverride.mix`` fades the live block out over the clip's first hop, plays the clip
alone while it lasts, then fades the live block back in over the hop after it ends. The
engine never stops and there is still exactly one writer of the output ring, so a typed
clip and the live voice share one output path (docs/gui/GUI_SPEC.md, type-to-speak).

Host side, numpy only, no allocation per hop. ``submit`` and ``cancel`` are single-slot
mailboxes written by the control path and read by the worker between hops. ``volume``
is the typed-speech volume (``volume.target``, linear, from the control path), ramped in
while a clip plays so the slider works mid-clip.
"""

from __future__ import annotations

import numpy as np

from vox.audio.volume import RampedGain


class ClipOverride:
    def __init__(self, block: int, sr: int = 48000) -> None:
        self._pending: np.ndarray | None = None
        self._cancel = False
        self._clip: np.ndarray | None = None
        self._pos = 0
        self._fade_in_next = False
        self.played = 0  # clips started
        self.volume = RampedGain.for_rate(sr)
        self.resize(block)

    def resize(self, block: int) -> None:
        """New hop size (worker rebuild). A clip in flight keeps playing."""
        ramp = np.linspace(0.0, 1.0, block, endpoint=False, dtype=np.float32)
        self._fade_in = ramp
        self._fade_out = (1.0 - ramp).astype(np.float32)
        self._scratch = np.zeros(block, dtype=np.float32)

    # ----------------------------------------------------------------- control path

    def submit(self, clip: np.ndarray) -> None:
        """Play ``clip`` (mono float32 at the device rate) from the next hop on."""
        self._pending = np.ascontiguousarray(clip, dtype=np.float32)

    def cancel(self) -> None:
        self._cancel = True

    @property
    def active(self) -> bool:
        return self._clip is not None or self._pending is not None

    # ------------------------------------------------------------------- worker hop

    def mix(self, out: np.ndarray) -> bool:
        """Apply the override to one live hop in place. True while a clip owns the output."""
        if self._cancel:
            self._cancel = False
            self._pending = None
            if self._clip is not None:
                self._clip = None
                self._fade_in_next = True
        pending = self._pending
        if pending is not None:
            self._pending = None
            fresh = self._clip is None and not self._fade_in_next
            self._clip, self._pos = pending, 0
            self.played += 1
            if not fresh:  # replacing a clip: live is already muted
                self._fade_in_next = False
                out[:] = 0.0
            else:
                out *= self._fade_out
        elif self._clip is not None:
            out[:] = 0.0
        clip = self._clip
        if clip is None:
            if self._fade_in_next:
                self._fade_in_next = False
                out *= self._fade_in
            return False
        n = min(out.shape[0], clip.shape[0] - self._pos)
        if self.volume.unity:
            out[:n] += clip[self._pos:self._pos + n]
        else:
            part = self._scratch[:n]
            part[:] = clip[self._pos:self._pos + n]
            self.volume.apply(part)
            out[:n] += part
        self._pos += n
        if self._pos >= clip.shape[0]:
            self._clip = None
            self._fade_in_next = True
        return True
