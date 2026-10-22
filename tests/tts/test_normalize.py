"""``normalize``: slang table, laughter, casing, punctuation, numbers, emoji, user overrides,
and that the expressive markers come through untouched."""

import json
import time

import pytest

from vox.tts import normalize as norm
from vox.tts.expressive import analyze
from vox.tts.normalize import SlangError, load_table, normalize

SHIPPED = json.loads(norm.SHIPPED.read_text(encoding="utf-8"))["words"]


def spoken_form(value: str) -> str:
    return value.lstrip(",").rstrip("?").strip()


# ------------------------------------------------------------------- 1. the slang table


@pytest.mark.parametrize("key, value", sorted(SHIPPED.items()))
def test_every_shipped_entry_expands_between_words(key, value):
    out = normalize(f"well {key} then")
    assert spoken_form(value).lower() in out.lower(), out
    if spoken_form(value).lower() != key:
        assert f" {key} " not in f" {out.lower()} ".replace(",", " ")


@pytest.mark.parametrize("text, expected", [
    ("idk wbu", "I don't know, what about you?"),
    ("im so tired rn ngl", "I'm so tired right now, not gonna lie."),
    ("ngl im tired", "Not gonna lie, I'm tired."),
    ("Dont do that", "Don't do that."),
    ("i DONT care", "I DON'T care"),  # caps contraction keeps its emphasis (no "." added)
    ("IDK WBU", "I don't know, what about you?"),  # caps abbreviations are not shouting
    ("WHY R U LIKE THIS", "WHY ARE YOU LIKE THIS"),  # a shouted line stays shouted
    ("w/ u", "With you."),
    ("w/o u", "Without you."),
    ("w/friends", "With friends."),
    ("no prob", "No problem."),
    ("ty sm!!", "Thank you so much!!"),
    ("gr8, see u b4 2nite", "Great, see you before tonight."),
    ("see ya tmrw", "See ya tomorrow."),  # "ya" after "see" is "you", left as typed
    ("ya i know", "Yeah, I know."),
    ("y tho", "Why though?"),
    ("k", "Okay."),
    ("kk cool", "Okay, cool."),
    ("afk brb", "AFK be right back."),
    ("wyd tonight", "What are you doing tonight?"),
])
def test_casual_lines(text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("ur so funny", "You're so funny."),
    ("ur going to love it", "You're going to love it."),
    ("ur the best", "You're the best."),
    ("ur mom called", "Your mom called."),
    ("is that ur car", "Is that your car?"),
])
def test_ur_is_youre_only_before_an_ing_word_or_an_adjective(text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("how r u", "How are you?"),
    ("r u ok", "Are you okay?"),
    ("plan r", "Plan r."),  # not between two words: a letter
    ("the letter u.s.", "The letter u.s."),
])
def test_single_letter_entries_need_word_context(text, expected):
    assert normalize(text) == expected


def test_words_containing_a_slang_key_are_untouched():
    assert normalize("bury the umbrella, kurt") == "Bury the umbrella, kurt."
    assert normalize("u-turn") == "U-turn."
    assert normalize("y'all ready") == "Y'all ready."


# ----------------------------------------------------------------------- 2. laughter


@pytest.mark.parametrize("token", ["lol", "LOL", "lmao", "lmaooo", "lmfao", "rofl", "haha",
                                   "hahaha", "HAHAHA", "hehe", "lolol", "xD",
                                   "😂", "😭", "🤣"])
def test_laughter_is_said_as_haha_never_spelled(token):
    assert normalize(f"ok {token} brb") == "Okay, haha, be right back."
    assert normalize(token) == "Haha."


def test_laughter_can_be_dropped_and_repeats_collapse():
    assert normalize("lol ok brb", laughter="drop") == "Okay, be right back."
    assert normalize("lol", laughter="drop") == ""
    assert normalize("that was funny lmao 😂😂😂") == "That was funny, haha."
    with pytest.raises(ValueError, match="laughter"):
        normalize("lol", laughter="loud")


# ------------------------------------------------------------- 3. casing / punctuation


@pytest.mark.parametrize("text, expected", [
    ("i think i'm fine", "I think I'm fine."),
    ("hi. how are you", "Hi. How are you?"),
    ("wait... what", "Wait... what."),  # no capital after dots: the same sentence goes on
    ("my iPhone died", "My iPhone died."),
    ("I'm going to the U.S. tmrw", "I'm going to the U.S. tomorrow."),
    ("Hello there!", "Hello there!"),
    ("really?", "Really?"),
    ("are you serious", "Are you serious?"),
    ("what about the car", "What about the car?"),
    ("what a day", "What a day."),
    ("so -- ", "So --"),  # a trailing dash pause stays open
    ("hey,", "Hey."),
    ("that is SO not what i said", "That is SO not what I said"),  # expressive adds "!"
])
def test_casing_and_terminal_punctuation(text, expected):
    assert normalize(text) == expected


def test_caps_emphasis_is_never_lowered():
    for text in ("that is SO good", "NOOOO", "you DON'T say", "i am SO DONE"):
        caps = [w for w in text.split() if w.isupper() and len(w) > 1]
        out = normalize(text)
        assert all(w in out for w in caps), out


def test_formal_text_is_unchanged():
    for text in ("Hello, this is a test of the text to speech path.",
                 "The quick brown fox jumps over the lazy dog.",
                 "Is it working? Yes. Thanks!"):
        assert normalize(text) == text


def test_lines_are_normalised_on_their_own():
    assert normalize("idk\nwbu") == "I don't know.\nWhat about you?"


# ------------------------------------------------------------ 4. numbers and symbols


@pytest.mark.parametrize("text, expected", [
    ("3pm", "Three PM."),
    ("at 3 pm", "At three PM."),
    ("10:30", "Ten thirty."),
    ("10:05am", "Ten oh five AM."),
    ("12:00", "Twelve o'clock."),
    ("$5", "Five dollars."),
    ("$1", "One dollar."),
    ("$5.50", "Five dollars and fifty cents."),
    ("$0.99", "Ninety-nine cents."),
    ("$5k", "Five thousand dollars."),
    ("€20", "Twenty euros."),
    ("50%", "Fifty percent."),
    ("2.5%", "Two point five percent."),
    ("the 1st 2nd 3rd 21st", "The first second third twenty-first."),
    ("in 2026", "In twenty twenty-six."),
    ("in 2005", "In two thousand and five."),
    ("1,000 ppl", "One thousand people."),
    ("3.5 volts", "Three point five volts."),
    ("2-3 days", "Two to three days."),
    ("i have 2 cats and 4 dogs", "I have two cats and four dogs."),  # digits stay numbers
    ("code 007", "Code zero zero seven."),
    ("rock & roll", "Rock and roll."),
    ("meet @ 5", "Meet at five."),
    ("ping @sam", "Ping at sam."),
    ("#blessed", "Blessed."),
    ("this is *so* good", "This is so good."),
])
def test_numbers_and_symbols(text, expected):
    assert normalize(text) == expected


def test_digits_inside_words_are_left_alone():
    assert normalize("my ps5 and mp3s") == "My ps5 and mp3s."


# ------------------------------------------------------------------- 5. emoji and urls


@pytest.mark.parametrize("text, expected", [
    ("nice 🔥🔥", "Nice."),
    ("love it ❤️", "Love it."),
    ("thumbs 👍🏽 up", "Thumbs up."),
    ("ok :)", "Okay."),
    ("sad :( ;) :D :P =)", "Sad."),
    ("i <3 u", "I love you."),
    ("look https://example.com/x?y=1 now", "Look link now."),
    ("see www.example.com.", "See link."),
    ("🔥", ""),
])
def test_emoji_emoticons_and_urls(text, expected):
    assert normalize(text) == expected


def test_times_are_not_eaten_as_emoticons():
    assert normalize("at 10:30 :)") == "At ten thirty."


# ---------------------------------------------------------------- user slang overrides


def test_user_file_is_merged_over_the_shipped_table(tmp_path):
    user = tmp_path / "slang.json"
    user.write_text(json.dumps({"_help": "x", "words": {
        "fr": "fr", "bestie": "best friend", "LMK": "let me know soon", "meh": ""}}))
    table = load_table(user)
    assert table.user_entries == 4 and table.words["lmk"] == "let me know soon"
    assert normalize("fr bestie lmk meh", table) == "Fr best friend let me know soon."
    assert normalize("fr", load_table(None)) == "For real."


def test_flat_user_mapping_is_accepted_and_bad_files_are_errors(tmp_path):
    user = tmp_path / "slang.json"
    user.write_text(json.dumps({"bestie": "best friend"}))
    assert load_table(user).words["bestie"] == "best friend"
    user.write_text("{not json")
    with pytest.raises(SlangError, match="slang.json"):
        load_table(user)
    user.write_text(json.dumps({"words": {"x": 3}}))
    with pytest.raises(SlangError, match="string"):
        load_table(user)
    assert load_table(tmp_path / "missing.json").user_entries == 0


def test_table_cache_follows_the_file(tmp_path):
    user = tmp_path / "slang.json"
    user.write_text(json.dumps({"words": {"bestie": "best friend"}}))
    first = norm.table_for(user)
    assert norm.table_for(user) is first
    user.write_text(json.dumps({"words": {"bestie": "bestie", "zz": "sleeping"}}))
    second = norm.table_for(user)
    assert second is not first and second.user_entries == 2


def test_template_is_created_once_and_loads(tmp_path):
    path = tmp_path / "profile" / "slang.json"
    assert norm.write_user_template(path) is True
    assert norm.write_user_template(path) is False
    assert normalize("wsg", load_table(path)) == "What's good?"


# ------------------------------------------------------- the expressive layer survives

EXPRESSIVE_INPUTS = [
    "hey... hey", "wait.... what did you say", "so -- yeah", "so — yeah", "so… yeah",
    "well-known fact", "...hey", "hey... ..... you", "nooooo waaaay", "noooooo",
    "i told you yesterday, and it was soooo obvious, but whatever", "good food, see you soon",
    "that is SO not what i said", "that is SO wrong.", "is that SO?",
    "that is SO wrong, honestly", "the TTS path is fine", "my GPU is OK", "LOL", "the US",
    "I said so", "at 2PM then", "Hello There", "you DON'T say", "are you serious??", "hey!!!",
    "what?!?!", "hey... hey!! are you fr right now", "NOOOOOO",
    "Hello, this is a test of the text to speech path.",
    'Is it "working?" Yes. New line, 3.5 volts; done.', "gonna be lit fr fr, no cap",
    "teh quick brown fox",
]


def hints(pieces):
    return [(p.speed, p.gain_db, p.pause_after_s, p.explicit) for p in pieces]


@pytest.mark.parametrize("text", EXPRESSIVE_INPUTS)
def test_every_expressive_test_input_keeps_its_hints(text):
    """Every input of tests/tts/test_expressive.py renders with the same speed, gain and
    pauses whether or not it was normalised first."""
    assert hints(analyze(normalize(text))) == hints(analyze(text))


def test_expressive_marks_are_still_in_the_normalised_text():
    out = normalize("hey... heyyy!! that was SO good -- right??")
    for mark in ("...", "yyy", "!!", "SO", "--", "??"):
        assert mark in out


# ------------------------------------------------------------------------- latency


def test_normalisation_is_fast():
    text = ("omg ngl that was SO good!! idk wbu lol 😂 brb, meet @ 3pm for $5 & 2 ppl... "
            "heyyy u r gr8 tbh") * 2
    normalize(text)  # warm: num2words import, compiled table
    best = min(_timed(text) for _ in range(5))
    assert best < 0.005, f"{best * 1000:.2f} ms"


def _timed(text):
    start = time.perf_counter()
    normalize(text)
    return time.perf_counter() - start


@pytest.mark.parametrize("tag", ["[laugh]", "[laughs]", "[LAUGH]", "*laughs*", "(laughs)",
                                 "[laughing]", "[ laughter ]"])
def test_an_explicit_laugh_insert_is_said_like_lol(tag):
    """``[laugh]`` and its spellings are the typed laugh insert (docs/DEVIATIONS.md §14 B2):
    the same spoken "haha" as lol, set off by commas, dropped with laughter="drop"."""
    assert normalize(f"that was wild {tag} anyway") == "That was wild, haha, anyway."
    assert normalize(f"ok {tag} lol") == "Okay, haha."  # repeats collapse into one
    assert normalize(f"that was wild {tag} anyway", laughter="drop") == "That was wild anyway."
