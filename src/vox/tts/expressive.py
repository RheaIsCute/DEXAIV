"""Expressive text layer for typed speech: informal text should *sound* informal.

``analyze(text)`` is a pure function from text to ``ExpressivePiece``s: each piece is the
text the phonemizer should see plus the render hints for it. The RVC stage keeps whatever
prosody the TTS renders, so everything expressive happens here, before or at the TTS call:

* dots and dashes (``...``, ``....``, ``…``, ``--``, ``—``) become an explicit pause between
  pieces, ``PAUSE_BASE_S`` plus ``PAUSE_PER_DOT_S`` per dot beyond three, capped;
* a letter repeated three or more times ("heyyy", "nooooo") marks the clause: the spelling
  is normalised to two repeats for the phonemizer and the clause renders slower, scaled by
  the longest run (``SPEED_AT_3`` .. ``SPEED_AT_6``);
* a word in ALL CAPS (two or more letters, not a known acronym) or repeated ``!``/``?``
  marks the clause for emphasis: lowercase / single mark for the phonemizer, an exclamation
  if the clause has no terminal punctuation, ``EMPHASIS_SPEED`` at render and
  ``EMPHASIS_GAIN_DB`` applied to the converted audio by ``speak``;
* everything else passes through untouched: no slang normalisation, no autocorrect.

A sentence with no marks is returned exactly as ``split_sentences`` yields it, with the
default hints, so a formal text renders exactly as it did before this layer existed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

SENTENCE_GAP_S = 0.2  # the plain piece gap (both backends trim silence; see speak)
PAUSE_BASE_S = 0.25  # three dots, or a dash
PAUSE_PER_DOT_S = 0.12
PAUSE_MAX_S = 1.2

ELONGATION_MIN = 3  # repeats that mark a word
SPEED_AT_3 = 0.92
SPEED_AT_6 = 0.78
EMPHASIS_SPEED = 1.05
EMPHASIS_GAIN_DB = 1.5

# ALL-CAPS words that are spoken as acronyms or are normal in caps: never emphasis.
ACRONYMS = frozenset((
    "TTS", "GPU", "CPU", "OK", "LOL", "AI", "RVC", "USB", "VAC", "PC", "TV", "DJ", "ASAP",
    "FYI", "OMG", "WTF", "BRB", "IDK", "TBH", "IMO", "NSFW", "FPS", "RAM", "VRAM", "API",
    "URL", "HTTP", "ID", "US", "UK", "USA", "EU", "NYC", "VIP", "CEO", "DIY", "FAQ", "HD",
    "UI", "GUI", "CLI", "OS", "IO", "IP"))

_SENTENCE_END = re.compile(  # after the terminator, or after one closing quote / bracket
    r"(?<=[.!?…。！？][\"'”’)\]])\s+|(?<=[.!?…。！？])\s+|\n+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")
_PAUSE = re.compile(r"\.{3,}|…|—|–|-{2,}")
_ELONGATION = re.compile(r"([A-Za-z])\1{2,}")
_CAPS = re.compile(r"\b[A-Z]{2,}(?:'[A-Z]+)?\b")
_REPEATED_PUNCT = re.compile(r"([!?])[!?]+")
_TERMINAL = ".!?…"
_CLAUSE_PUNCT = ",;:"


@dataclass(frozen=True)
class ExpressivePiece:
    """Text for the phonemizer plus how to render it.

    ``pause_after_s`` is the silence inserted after this piece when a piece boundary falls
    here (the last piece of an utterance gets none). ``explicit`` marks a pause that came
    from dots or dashes: such a boundary is never packed away.
    """

    text: str
    speed: float = 1.0
    gain_db: float = 0.0
    pause_after_s: float = SENTENCE_GAP_S
    explicit: bool = False

    @property
    def plain(self) -> bool:
        return self.speed == 1.0 and self.gain_db == 0.0 and not self.explicit

    def same_hints(self, other: ExpressivePiece) -> bool:
        return self.speed == other.speed and self.gain_db == other.gain_db


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def pause_seconds(marker: str) -> float:
    """Pause for one dots/dash run: three dots (or any dash) = base, +120 ms per extra dot."""
    dots = len(marker) if marker.startswith(".") else 3
    return min(PAUSE_MAX_S, PAUSE_BASE_S + PAUSE_PER_DOT_S * max(0, dots - 3))


def elongation_speed(run: int) -> float:
    """Render speed for the longest letter run in a clause: 0.92 at 3 repeats, 0.78 at 6+."""
    per_repeat = (SPEED_AT_3 - SPEED_AT_6) / 3.0
    return max(SPEED_AT_6, min(SPEED_AT_3, SPEED_AT_3 - per_repeat * (run - ELONGATION_MIN)))


def is_marked(text: str) -> bool:
    """Does any expressive rule apply to this text (ignoring pauses)?"""
    if _ELONGATION.search(text) or _REPEATED_PUNCT.search(text):
        return True
    return any(m.group(0) not in ACRONYMS for m in _CAPS.finditer(text))


def _with_exclamation(text: str) -> str:
    text = text.rstrip()
    if not text or text[-1] in _TERMINAL:
        return text
    if text[-1] in _CLAUSE_PUNCT:
        return text[:-1] + "!"
    return text + "!"


def analyze_clause(clause: str) -> ExpressivePiece:
    """Hints for one clause (no pause markers inside). Pure."""
    text = clause
    emphasis = False
    speed = 1.0

    if _REPEATED_PUNCT.search(text):  # "!!!" / "??" -> one mark, emphasis
        text = _REPEATED_PUNCT.sub(r"\1", text)
        emphasis = True

    def lower_caps(match: re.Match) -> str:
        nonlocal emphasis
        word = match.group(0)
        if word in ACRONYMS:
            return word
        emphasis = True
        return word.lower()

    text = _CAPS.sub(lower_caps, text)

    runs = [len(m.group(0)) for m in _ELONGATION.finditer(text)]
    if runs:
        text = _ELONGATION.sub(r"\1\1", text)
        speed = elongation_speed(max(runs))

    gain = 0.0
    if emphasis:
        if speed == 1.0:
            speed = EMPHASIS_SPEED
        gain = EMPHASIS_GAIN_DB
        text = _with_exclamation(text)
    return ExpressivePiece(text, speed, gain)


def _segments(sentence: str) -> list[tuple[str, float | None]]:
    """Split one sentence at pause markers: (text, pause after it or None at the end)."""
    out: list[tuple[str, float | None]] = []
    start = 0
    for match in _PAUSE.finditer(sentence):
        out.append((sentence[start:match.start()].strip(), pause_seconds(match.group(0))))
        start = match.end()
    out.append((sentence[start:].strip(), None))
    return out


def analyze(text: str) -> list[ExpressivePiece]:
    """Text -> pieces with render hints, in order. Sentences without marks come back whole
    and untouched; a marked sentence is split at clause punctuation so the hints apply only
    to the clause that carries the mark."""
    pieces: list[ExpressivePiece] = []
    pending: float | None = None  # a pause with nothing before it yet (leading dots)

    def attach(pause: float | None) -> None:
        nonlocal pending
        if pause is None:
            return
        if pieces:
            last = pieces[-1]
            pieces[-1] = replace(last, pause_after_s=max(last.pause_after_s, pause),
                                 explicit=True)
        else:
            pending = pause  # nothing to pause after: dropped

    for sentence in split_sentences(text):
        for segment, pause in _segments(sentence):
            if not segment:
                attach(pause)
                continue
            if is_marked(segment):
                clauses = _CLAUSE_END.split(segment)
                pieces.extend(analyze_clause(c) for c in clauses if c)
            else:
                pieces.append(ExpressivePiece(segment))
            attach(pause)
    return pieces
