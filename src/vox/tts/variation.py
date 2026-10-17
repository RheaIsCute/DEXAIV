"""Per-utterance variation: the same text typed twice should sound like a person saying it
twice -- the same voice, a slightly different delivery.

``draw(amount, seed, voice, voices)`` makes one ``Variation`` per utterance; every piece of
that utterance renders with it, so nothing jumps inside one utterance. All offsets scale
with ``amount`` (0..1, one knob) and are bounded:

* TTS speed: x(1 +- ``SPEED_JITTER`` * amount)
* TTS style (Kokoro only): the voice's style vector blended with 1-2 other voices of the
  same language and gender, together at most ``BLEND_MAX`` * amount
* transposition: +- ``PITCH_JITTER_ST`` * amount semitones on top of the auto/manual pitch
  (``f0_up_key`` is a float all the way down, so fractional semitones need no other path)
* explicit pauses (dots, dashes): x(1 +- ``PAUSE_JITTER`` * amount)
* the RVC conversion seed

``amount == 0`` draws nothing and returns ``Variation(rvc_seed=seed)``: exactly the
pre-variation path, with the caller's seed (``vox say`` uses 0, the GUI None).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

DEFAULT_AMOUNT = 0.35
SPEED_JITTER = 0.04
BLEND_MAX = 0.15
PITCH_JITTER_ST = 0.4
PAUSE_JITTER = 0.25
CONTOUR_TRY = 1.15  # f0 contour widening worth an A/B; the default stays 1.0 (off)
CONTOUR_MAX = 1.5


@dataclass(frozen=True)
class Variation:
    amount: float = 0.0
    seed: int | None = None  # what reproduces this draw (vox say --seed)
    rvc_seed: int | None = 0
    speed: float = 1.0
    pitch_offset: float = 0.0
    pause_scale: float = 1.0
    blend: tuple[tuple[str, float], ...] = ()  # (other voice, weight)

    @property
    def neutral(self) -> bool:
        return (self.speed == 1.0 and self.pitch_offset == 0.0 and self.pause_scale == 1.0
                and not self.blend)

    def describe(self) -> str:
        if self.neutral:
            return f"variation off (seed {self.rvc_seed})"
        mix = ", ".join(f"{w * 100:.1f}% {v}" for v, w in self.blend) or "none"
        return (f"variation {self.amount:.2f} (seed {self.seed}): speed x{self.speed:.3f}, "
                f"pitch {self.pitch_offset:+.2f} st, pauses x{self.pause_scale:.2f}, "
                f"style blend {mix}")

    def as_dict(self) -> dict:
        return {"amount": self.amount, "seed": self.seed, "speed": self.speed,
                "pitch_offset": self.pitch_offset, "pause_scale": self.pause_scale,
                "blend": [list(b) for b in self.blend]}


def blend_candidates(voice: str | None, voices: list[str] | tuple[str, ...],
                     allow: list[str] | tuple[str, ...] | None = None) -> list[str]:
    """Voices that may be blended into ``voice``: ``allow`` when given (minus ``voice`` and
    unknown names), else every voice with the same Kokoro prefix ("af_" = American female)."""
    if not voice or not voices:
        return []
    known = set(voices)
    if allow:
        return [v for v in allow if v in known and v != voice]
    prefix = voice.split("_", 1)[0] + "_" if "_" in voice else None
    return sorted(v for v in voices if prefix and v.startswith(prefix) and v != voice)


def draw(amount: float, seed: int | None = None, voice: str | None = None,
         voices: list[str] | tuple[str, ...] = (),
         allow: list[str] | tuple[str, ...] | None = None) -> Variation:
    """One utterance's variation. ``seed=None`` (with ``amount > 0``) draws from the clock.

    The random draws happen in a fixed order whatever the voices are, so a seed reproduces
    the same speed / pitch / pause offsets on any backend.
    """
    amount = min(1.0, max(0.0, float(amount)))
    if amount == 0.0:
        return Variation(rvc_seed=seed)
    if seed is None:
        seed = time.time_ns() % 2**31
    import numpy as np

    rng = np.random.default_rng(seed)
    rvc_seed = int(rng.integers(0, 2**31 - 1))
    speed, pitch, pause, blend_total, split = rng.uniform(-1.0, 1.0, 5)
    two = bool(rng.integers(0, 2))
    picks = rng.permutation(64)

    blend: tuple[tuple[str, float], ...] = ()
    candidates = blend_candidates(voice, voices, allow)
    total = abs(blend_total) * BLEND_MAX * amount
    if candidates and total > 0:
        chosen = [candidates[i % len(candidates)] for i in picks[:2 if two else 1]]
        if len(chosen) == 2 and chosen[0] != chosen[1]:
            first = total * (0.5 + 0.4 * split)
            blend = ((chosen[0], first), (chosen[1], total - first))
        else:
            blend = ((chosen[0], total),)
    return Variation(amount=amount, seed=seed, rvc_seed=rvc_seed,
                     speed=1.0 + speed * SPEED_JITTER * amount,
                     pitch_offset=pitch * PITCH_JITTER_ST * amount,
                     pause_scale=1.0 + pause * PAUSE_JITTER * amount, blend=blend)


def bounds(amount: float) -> str:
    """What ``amount`` allows, for ``--dry-run``."""
    return (f"variation {amount:.2f} per utterance: speed +-{SPEED_JITTER * amount * 100:.1f}%, "
            f"pitch +-{PITCH_JITTER_ST * amount:.2f} st, explicit pauses "
            f"+-{PAUSE_JITTER * amount * 100:.1f}%, style blend <= {BLEND_MAX * amount * 100:.1f}% "
            f"of 1-2 same-family voices (kokoro), a fresh RVC seed")
