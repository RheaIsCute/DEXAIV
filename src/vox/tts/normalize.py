"""Chat text -> text the phonemizer reads well. Runs before ``expressive.analyze``.

``normalize(text, table, laughter)`` is pure. In order:

1. URLs become "link"; emoji are dropped except the laughing ones; ``<3`` is "love" and the
   other emoticons are dropped.
2. Laughter ("lol", "lmao", "hahaha", "xD", the laughing emoji) becomes one spoken "haha"
   set off by commas, or nothing (``laughter="drop"``). Never "L O L".
3. Slang and contractions are expanded from a table (``data/slang_en.json``, merged with
   ``profiles/<name>/slang.json``), whole words only, case-insensitively. A few cheap context
   rules settle the ambiguous ones: "ur" is "you're" before an -ing word or a listed
   adjective and "your" otherwise; "r" is "are" only between two words; "ya" stays "ya" after
   "see / love / thank ...". A question phrase ("wbu") ends the line with "?".
4. Money, percentages, times, ordinals, years and plain numbers are written out in words
   (``num2words``); ``&`` is "and", ``@`` is "at".
5. A lone "i" is "I", sentences start with a capital, and a line without terminal
   punctuation gets "?" when it opens like a question ("are you", "what about") and "."
   otherwise. A line the expressive layer will end with "!" (an emphasised last clause) or
   that ends in a dash pause is left open.

The expressive markers survive by construction: dots and dashes, letter runs, ALL-CAPS
words and repeated ``!``/``?`` are never rewritten here (a caps word is only ever upper-cased
further, never lowered), so ``analyze`` sees the same marks with or without this step.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

from vox.tts.expressive import ACRONYMS, analyze, split_sentences

SHIPPED = Path(__file__).with_name("data") / "slang_en.json"
USER_FILE = "slang.json"  # in the profile directory, same schema as the shipped table
LAUGHTER_MODES = ("haha", "drop")
LAUGH_SPOKEN = "haha"
USER_TEMPLATE = {
    "_help": "Your own chat words, merged over the shipped list (src/vox/tts/data/"
             "slang_en.json). Keys are lowercase; a value ending in ? is a question; an "
             "empty value drops the word; map a word to itself to switch a shipped entry "
             "off. Save, then press Reload in Settings > Typed speech.",
    "words": {"wsg": "what's good?", "tn": "tonight"},
}

_LAUGH = "\ue000"  # placeholder while the rest of the pipeline runs (not a word character)
_QMARK = "\ue001"  # in front of a question phrase that closes its line: comma goes there
_OPEN, _CLOSE = "\ue002", "\ue003"  # around an aside ("ngl"): commas where it meets words

_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_LAUGH_EMOJI = re.compile("[\U0001F602\U0001F923\U0001F62D\U0001F639]")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\u2300-\u23FF"
                    "\uFE0E\uFE0F\u200D\u20E3\U000E0020-\U000E007F]+")
_LOVE = re.compile(r"(?<![\w<])<3+(?![\w])")
_EMOTICON = re.compile(r"(?<![\w:/])[:;=]['^-]?[()\[\]DPpOo/\\|*3]+(?!\w)"
                       r"|(?<!\w)(?:\^_*\^|-_+-|[Tt]_+[Tt]|[oO]_[oO])(?!\w)")
_LAUGH_WORD = re.compile(r"(?<![\w'])(?:l+o+l+(?:o+l+)*z*|l+m+f*a+o+|r+o+f+l+|(?:a*h+a+){2,}h*"
                         r"|(?:h+e+){2,}h*|x+d+)(?![\w'])", re.IGNORECASE)
_CAPS_WORD = re.compile(r"\b[A-Z][A-Z']+\b")

# context rules for the ambiguous table entries
_YOURE_NEXT = re.compile(r"\s+(?:\w+ing|so|too|very|really|such|not|always|never|right|wrong|"
                         r"welcome|kidding|joking|crazy|funny|cute|sweet|nice|mean|late|early|"
                         r"done|good|bad|great|amazing|awesome|beautiful|pretty|smart|dumb|"
                         r"stupid|weird|lucky|sure|ok|okay|fine|here|there|back|gonna|wrong|"
                         r"the best|the worst|a)\b", re.IGNORECASE)
_YA_AFTER = re.compile(r"\b(?:see|love|thank|miss|bet|got|told|catch|hear|get|hate)\s+$",
                       re.IGNORECASE)
_WORD_BEFORE = re.compile(r"[A-Za-z]\s+$")
_WORD_AFTER = re.compile(r"\s+[A-Za-z]")
_PRONOUN = re.compile(r"\s+(?:u|you|we|they|ya|y'?all)\b", re.IGNORECASE)
_LINE_END = re.compile(rf"[\s{_LAUGH}]*(?:\n|$)")

_MONEY = re.compile(r"(?<![\w$£€])([$£€])\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?"
                    r"(?:\s?(k|m|bn|thousand|million|billion)\b)?(?!\d)", re.IGNORECASE)
_PERCENT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s?%")
_CLOCK = re.compile(r"(?<![\w:.])(\d{1,2}):([0-5]\d)(?::[0-5]\d)?(?:\s?([ap])\.?m\.?)?(?![\w:])",
                    re.IGNORECASE)
_HOUR = re.compile(r"(?<![\w:.])(\d{1,2})\s?([ap])\.?m\.?(?![\w])", re.IGNORECASE)
_ORDINAL = re.compile(r"(?<![\w.])(\d+)(st|nd|rd|th)(?!\w)", re.IGNORECASE)
_RANGE = re.compile(r"(?<![\w.])(\d+)\s?-\s?(\d+)(?![\w.])")
_NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\w]|[.,]\d)")
_CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"),
             "£": ("pound", "pounds", "penny", "pence"),
             "€": ("euro", "euros", "cent", "cents")}
_SCALE = {"k": "thousand", "m": "million", "bn": "billion"}

_SPACES = re.compile(r"[ \t\u00a0]+")
_SPACE_BEFORE = re.compile(r"\s+(?=[,;:!?]|\.(?!\.))")
_DUP_COMMA = re.compile(r",(?:\s*,)+")
_COMMA_BEFORE_END = re.compile(r",\s*(?=[.!?]|$)", re.MULTILINE)
_LEAD_PUNCT = re.compile(r"^[\s,;:]+", re.MULTILINE)
_LONE_I = re.compile(r"(?<![\w'])i(?![\w]|\.\w)")
_SENTENCE_START = re.compile(  # not after "...", nor after an initial ("U.S. tomorrow")
    r"(^\s*|(?:(?<!\.)(?<!\b[A-Za-z])\.(?!\.)|[!?]+)[\"')\]]*\s+)([a-z][\w']*)", re.MULTILINE)
# a sentence or clause that opens with one of these gets a comma after it: "Okay, be right back."
_OPENER = re.compile(r"(^|[.!?,]\s+)(okay|yeah|nah|yep|nope|alright|hey)\s+(?=\w)", re.IGNORECASE)
_TRAILING_PAUSE = re.compile(r"(?:\.{3,}|…|—|–|-{2,})\s*$")
_TERMINAL = ".!?…"
_QUESTION = re.compile(
    r"^\W*(?:(?:are|is|am|was|were|do|does|did|can|could|would|will|should|shall|have|has|"
    r"had|may|might|isn't|aren't|wasn't|weren't|don't|doesn't|didn't|can't|couldn't|won't|"
    r"wouldn't|shouldn't|haven't|hasn't)\s+(?:you|i|we|they|he|she|it|that|this|there|these|"
    r"those|y'all|ya|ur|your)\b|(?:what|why|how|where|when|who|which)(?:'s|'re|\s+(?:is|are|"
    r"was|were|do|does|did|can|could|would|will|should|have|has|about|if|you))\b|why\b)",
    re.IGNORECASE)


class SlangError(ValueError):
    """A slang file that cannot be used (bad JSON, wrong shape)."""


@dataclass
class SlangTable:
    """Lowercase token -> spoken form. ``user_entries`` came from the profile's file."""

    words: dict[str, str]
    user_path: Path | None = None
    user_entries: int = 0

    @cached_property
    def pattern(self) -> re.Pattern:
        """One alternation, longest key first, whole words only (a key ending in ``/`` may
        run straight into the next word: "w/friends")."""
        parts = []
        for key in sorted(self.words, key=len, reverse=True):
            guard = r"(?![\w']|-\w|\.\w)" if key[-1].isalnum() else ""
            parts.append(re.escape(key) + guard)
        return re.compile(r"(?<![\w'/])(?<!\w-)(?:" + "|".join(parts) + ")", re.IGNORECASE)


def _read_words(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise SlangError(f"{path.name}: {err}") from err
    if isinstance(data, dict) and isinstance(data.get("words"), dict):
        data = data["words"]
    if not isinstance(data, dict):
        raise SlangError(f"{path.name}: expected an object with a \"words\" mapping")
    words = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue  # comments
        if not isinstance(value, str):
            raise SlangError(f"{path.name}: {key!r} must map to a string, not {value!r}")
        key = " ".join(key.lower().split())
        if key:
            words[key] = value.strip()
    return words


def load_table(user_path: Path | None = None) -> SlangTable:
    """The shipped table with ``user_path`` (when it exists) merged over it."""
    words = _read_words(SHIPPED)
    user = _read_words(user_path) if user_path is not None and user_path.is_file() else {}
    words.update(user)
    return SlangTable(words, user_path, len(user))


@lru_cache(maxsize=8)
def _cached(path: str | None, stamp: tuple[int, int] | None) -> SlangTable:
    return load_table(Path(path) if path else None)


def table_for(user_path: Path | None = None) -> SlangTable:
    """``load_table`` cached until the user file changes (a bad file raises every time)."""
    stamp = None
    if user_path is not None and user_path.is_file():
        st = user_path.stat()
        stamp = (st.st_mtime_ns, st.st_size)
    return _cached(str(user_path) if user_path is not None else None, stamp)


def clear_cache() -> None:
    """Forget cached tables (the GUI's Reload button)."""
    _cached.cache_clear()


def write_user_template(path: Path) -> bool:
    """Create the profile's slang file with two examples; False when it already exists."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(USER_TEMPLATE, indent=2) + "\n", encoding="utf-8")
    return True


# ------------------------------------------------------------------------------ 1-2


def _url(match: re.Match) -> str:
    url = match.group(0)
    tail = len(url) - len(url.rstrip(".,!?;:)]'\""))
    return " link" + (url[-tail:] if tail else "") + " "


def _symbols(text: str) -> str:
    text = _URL.sub(_url, text)
    text = _LAUGH_EMOJI.sub(f" {_LAUGH} ", text)
    text = _EMOJI.sub(" ", text)
    text = _LOVE.sub(" love ", text)
    text = _EMOTICON.sub(" ", text)
    text = _LAUGH_WORD.sub(_LAUGH, text)
    text = re.sub(r"\s*&\s*", " and ", text)
    text = re.sub(r"(?<!\w)@(?=\w)", "at ", text)
    text = re.sub(r"\s@\s", " at ", text)
    text = re.sub(r"(?<!\w)#(?=\w)", "", text)  # hashtags read as the word
    return re.sub(r"[*~]+", "", text)  # markdown-ish *stress* / ~tilde~


# -------------------------------------------------------------------------------- 3


def _shouting(line: str, table: SlangTable) -> bool:
    """A line typed in caps with at least one real (non-slang, non-acronym) caps word."""
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 4 or any(c.islower() for c in letters):
        return False
    return any(w not in ACRONYMS and w.lower().replace("'", "") not in table.words
               for w in _CAPS_WORD.findall(line))


def _cased(token: str, value: str, shouting: bool) -> str:
    if not value:
        return value
    if shouting:
        return value.upper()
    respelling = value.replace("'", "").lower() == token.lower()
    if token.isupper() and len(token) > 1:
        return value.upper() if respelling else value  # DONT -> DON'T keeps the emphasis
    if token[0].isupper():  # Idk -> I don't know, U -> You
        return value[0].upper() + value[1:]
    return value


def _expand_line(line: str, table: SlangTable) -> str:
    shouting = _shouting(line, table)

    def expand(match: re.Match) -> str:
        token = match.group(0)
        key = " ".join(token.lower().split())
        value = table.words[key]
        before, after = line[:match.start()], line[match.end():]
        if key == "ur" and value == "your" and _YOURE_NEXT.match(after):
            value = "you're"
        elif key == "r" and value == "are" and not (
                _WORD_AFTER.match(after) and (_WORD_BEFORE.search(before)
                                              or not before.strip() and _PRONOUN.match(after))):
            return token  # "are" only between words, or opening "r u ok"
        elif key == "ya" and value == "yeah" and _YA_AFTER.search(before):
            return token
        lead = ""
        if value.startswith(","):  # an aside: set off by commas where it touches words
            value = _cased(token, value[1:].strip(), shouting)
            return _OPEN + value + _CLOSE if value else value
        if value.endswith("?"):
            value = value[:-1]
            if _LINE_END.match(after):
                value += "?"
                lead = _QMARK
        value = _cased(token, value, shouting)
        if value and not key[-1].isalnum() and after[:1].isalnum():
            value += " "
        return lead + value

    return table.pattern.sub(expand, line)


# -------------------------------------------------------------------------------- 4


def _words(n: int, to: str = "cardinal") -> str:
    from num2words import num2words

    return num2words(n, to=to).replace(",", "")


def _digits(s: str) -> str:
    return " ".join(_words(int(d)) for d in s)


def _integer(raw: str) -> str:
    digits = raw.replace(",", "")
    if len(digits) > 1 and digits[0] == "0" or len(digits) > 12:
        return _digits(digits)  # codes and phone numbers: one digit at a time
    n = int(digits)
    if "," not in raw and len(digits) == 4 and 1100 <= n <= 2099 and n % 1000 >= 10:
        return _words(n, "year")  # 1998 -> nineteen ninety-eight, 2026 -> twenty twenty-six
    return _words(n)


def _decimal(whole: str, frac: str | None) -> str:
    out = _integer(whole)
    return out + " point " + _digits(frac) if frac else out


def _money(match: re.Match) -> str:
    symbol, whole, cents, scale = match.groups()
    one, many, sub_one, sub_many = _CURRENCY[symbol]
    n = int(whole.replace(",", ""))
    if scale:
        amount = _decimal(whole, cents) + " " + _SCALE.get(scale.lower(), scale.lower())
        return f"{amount} {many}"
    parts = []
    if n or not cents:
        parts.append(f"{_integer(whole)} {one if n == 1 else many}")
    if cents and int(cents.ljust(2, '0')):
        c = int(cents.ljust(2, "0"))
        parts.append(f"{_words(c)} {sub_one if c == 1 else sub_many}")
    return " and ".join(parts)


def _clock(match: re.Match) -> str:
    hour, minute, half = int(match.group(1)), int(match.group(2)), match.group(3)
    if hour > 23:
        return match.group(0)
    words = _words(hour)
    if minute == 0:
        words += "" if half else " o'clock"
    elif minute < 10:
        words += " oh " + _words(minute)
    else:
        words += " " + _words(minute)
    return words + (f" {half.upper()}M" if half else "")


def _numbers(text: str) -> str:
    if not any(c.isdigit() for c in text):
        return text
    try:
        import num2words  # noqa: F401
    except ImportError:  # optional: espeak reads plain digits itself
        return text
    text = _MONEY.sub(_money, text)
    text = _PERCENT.sub(lambda m: _decimal(*(m.group(1).split(".") + [None])[:2]) + " percent",
                        text)
    text = _CLOCK.sub(_clock, text)
    text = _HOUR.sub(lambda m: f"{_words(int(m.group(1)))} {m.group(2).upper()}M", text)
    text = _ORDINAL.sub(lambda m: _words(int(m.group(1)), "ordinal"), text)
    text = _RANGE.sub(lambda m: f"{_integer(m.group(1))} to {_integer(m.group(2))}", text)
    return _NUMBER.sub(lambda m: _decimal(m.group(1), m.group(2)), text)


# -------------------------------------------------------------------------------- 5


def _set_off(text: str, mark: str, spoken: str) -> str:
    """Replace ``mark`` (around or in place of ``spoken``) with commas where it touches a
    word: "ok lol brb" -> "ok, haha, brb"."""
    text = re.sub(rf"(?<=\w)\s*{mark}", f", {mark}", text)
    text = re.sub(rf"{mark}\s*(?=\w)", f"{mark}, ", text)
    return text.replace(mark, spoken)


def _laughter(text: str, mode: str) -> str:
    text = re.sub(rf"{_LAUGH}(?:[\s,]*{_LAUGH})+", _LAUGH, text)
    if mode == "drop":
        return text.replace(_LAUGH, " ")
    return _set_off(text, _LAUGH, LAUGH_SPOKEN)


def _tidy(line: str) -> str:
    line = re.sub(rf"(?<=\w)\s*{_QMARK}", ", ", line).replace(_QMARK, "")
    line = re.sub(rf"(?<=\w)\s*{_OPEN}", ", ", line).replace(_OPEN, "")
    line = re.sub(rf"{_CLOSE}\s*(?=\w)", ", ", line).replace(_CLOSE, "")
    line = _SPACES.sub(" ", line)
    line = _SPACE_BEFORE.sub("", line)
    line = _DUP_COMMA.sub(",", line)
    line = _COMMA_BEFORE_END.sub("", line)
    line = re.sub(r"([.!?])\s*,", r"\1", line)
    line = _LEAD_PUNCT.sub("", line).strip()
    return _OPENER.sub(r"\1\2, ", line)


def _capitalise(line: str) -> str:
    line = _LONE_I.sub("I", line)

    def upper(match: re.Match) -> str:
        word = match.group(2)
        if any(c.isupper() for c in word[1:]):
            return match.group(0)  # iPhone, eBay
        return match.group(1) + word[0].upper() + word[1:]

    return _SENTENCE_START.sub(upper, line)


def _terminate(line: str) -> str:
    """Give an open line its terminal mark (see the module docstring)."""
    stripped = line.rstrip()
    if not stripped or _TRAILING_PAUSE.search(stripped):
        return stripped
    core = stripped.rstrip("\"')]")
    if core and core[-1] in _TERMINAL:
        return stripped
    stripped = stripped.rstrip(",;:").rstrip()
    if not stripped:
        return stripped
    pieces = analyze(stripped)
    if pieces and pieces[-1].gain_db > 0:
        return stripped  # emphasised last clause: the expressive layer ends it with "!"
    sentences = split_sentences(stripped)
    last = sentences[-1] if sentences else stripped
    return stripped + ("?" if _QUESTION.match(last) else ".")


def normalize(text: str, table: SlangTable | None = None, laughter: str = "haha") -> str:
    """Chat text -> speakable text (module docstring). ``laughter`` is "haha" or "drop"."""
    if laughter not in LAUGHTER_MODES:
        raise ValueError(f"laughter must be one of {LAUGHTER_MODES}, not {laughter!r}")
    table = table if table is not None else table_for(None)
    lines = []
    for line in _symbols(text).splitlines():
        line = _expand_line(_SPACES.sub(" ", line), table)
        line = _numbers(line)
        line = _tidy(_laughter(line, laughter))
        if line and any(c.isalnum() for c in line):
            lines.append(_terminate(_capitalise(line)))
    return "\n".join(lines)
