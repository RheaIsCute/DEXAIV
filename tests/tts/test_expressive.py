"""``analyze``: the five expressive behaviours, the acronym allowlist, and pass-through."""

import pytest

from vox.tts import expressive
from vox.tts.expressive import ExpressivePiece, analyze

PLAIN = ExpressivePiece("x")


def hints(piece):
    return piece.speed, piece.gain_db, piece.pause_after_s, piece.explicit


# ------------------------------------------------------------------ 1. dots and dashes


@pytest.mark.parametrize("marker, pause", [
    ("...", 0.25), ("....", 0.37), (".....", 0.49), ("…", 0.25), ("--", 0.25), ("—", 0.25),
    ("." * 20, 1.2)])
def test_pause_scales_with_dot_count_and_is_capped(marker, pause):
    assert expressive.pause_seconds(marker) == pytest.approx(pause)


def test_dots_at_a_sentence_end_become_an_explicit_pause():
    pieces = analyze("hey... hey")
    assert [p.text for p in pieces] == ["hey", "hey"]
    assert pieces[0].explicit and pieces[0].pause_after_s == pytest.approx(0.25)
    assert not pieces[1].explicit and pieces[1].pause_after_s == expressive.SENTENCE_GAP_S


def test_mid_sentence_dots_split_the_piece_there():
    pieces = analyze("wait.... what did you say")
    assert [p.text for p in pieces] == ["wait", "what did you say"]
    assert pieces[0].explicit and pieces[0].pause_after_s == pytest.approx(0.37)
    assert pieces[0].speed == 1.0 and pieces[0].gain_db == 0.0


def test_dashes_and_ellipsis_character_pause_too():
    assert analyze("so -- yeah")[0].pause_after_s == pytest.approx(0.25)
    assert [p.text for p in analyze("so — yeah")] == ["so", "yeah"]
    assert [p.text for p in analyze("so… yeah")] == ["so", "yeah"]
    assert [p.text for p in analyze("well-known fact")] == ["well-known fact"]  # one hyphen


def test_leading_dots_are_dropped_and_repeated_markers_keep_the_longest():
    assert [p.text for p in analyze("...hey")] == ["hey"]
    pieces = analyze("hey... ..... you")
    assert [p.text for p in pieces] == ["hey", "you"]
    assert pieces[0].pause_after_s == pytest.approx(0.49)


# ------------------------------------------------------------------ 2. elongation


@pytest.mark.parametrize("run, speed", [(3, 0.92), (4, 0.92 - 0.14 / 3), (5, 0.92 - 0.28 / 3),
                                        (6, 0.78), (9, 0.78)])
def test_elongation_speed_ladder(run, speed):
    assert expressive.elongation_speed(run) == pytest.approx(speed)


def test_elongated_words_are_normalised_and_slowed():
    (piece,) = analyze("nooooo waaaay")
    assert piece.text == "noo waay"
    assert piece.speed == pytest.approx(expressive.elongation_speed(5))  # longest run wins
    assert piece.gain_db == 0.0
    (piece,) = analyze("noooooo")  # six repeats: the floor
    assert piece.text == "noo" and piece.speed == pytest.approx(0.78)


def test_elongation_splits_a_long_sentence_at_the_clause():
    pieces = analyze("i told you yesterday, and it was soooo obvious, but whatever")
    assert [p.text for p in pieces] == [
        "i told you yesterday,", "and it was soo obvious,", "but whatever"]
    assert [p.speed for p in pieces] == [1.0, pytest.approx(expressive.elongation_speed(4)), 1.0]


def test_two_repeats_are_not_elongation():
    (piece,) = analyze("good food, see you soon")
    assert piece.text == "good food, see you soon" and piece.speed == 1.0


# ------------------------------------------------------------------ 3. emphasis caps


def test_caps_word_is_lowercased_emphasised_and_gets_an_exclamation():
    (piece,) = analyze("that is SO not what i said")
    assert piece.text == "that is so not what i said!"
    assert piece.speed == pytest.approx(1.05) and piece.gain_db == 1.5


def test_caps_keeps_existing_terminal_punctuation_and_replaces_clause_commas():
    assert analyze("that is SO wrong.")[0].text == "that is so wrong."
    assert analyze("is that SO?")[0].text == "is that so?"
    pieces = analyze("that is SO wrong, honestly")
    assert [p.text for p in pieces] == ["that is so wrong!", "honestly"]
    assert [p.gain_db for p in pieces] == [1.5, 0.0]


@pytest.mark.parametrize("text", ["the TTS path is fine", "my GPU is OK", "LOL", "the US"])
def test_acronym_allowlist_is_not_emphasis(text):
    (piece,) = analyze(text)
    assert piece.text == text and piece.plain


def test_single_capital_letters_and_mixed_tokens_are_not_emphasis():
    assert analyze("I said so")[0].plain
    assert analyze("at 2PM then")[0].plain
    assert analyze("Hello There")[0].plain


def test_caps_with_apostrophe_lowercases_the_whole_word():
    assert analyze("you DON'T say")[0].text == "you don't say!"


# ------------------------------------------------------------------ 4. repeated punctuation


def test_repeated_punctuation_keeps_one_mark_and_emphasises():
    (piece,) = analyze("are you serious??")
    assert piece.text == "are you serious?"
    assert piece.speed == pytest.approx(1.05) and piece.gain_db == 1.5
    (piece,) = analyze("hey!!!")
    assert piece.text == "hey!" and piece.gain_db == 1.5
    assert analyze("what?!?!")[0].text == "what?"


def test_the_acceptance_sentence():
    pieces = analyze("hey... hey!! are you fr right now")
    assert [p.text for p in pieces] == ["hey", "hey!", "are you fr right now"]
    assert hints(pieces[0]) == (1.0, 0.0, pytest.approx(0.25), True)
    assert hints(pieces[1]) == (pytest.approx(1.05), 1.5, expressive.SENTENCE_GAP_S, False)
    assert pieces[2].plain  # "fr" is slang and stays slang


def test_emphasis_and_elongation_in_one_clause():
    (piece,) = analyze("NOOOOOO")
    assert piece.text == "noo!"
    assert piece.speed == pytest.approx(0.78) and piece.gain_db == 1.5  # slow wins, gain stays


# ------------------------------------------------------------------ 5. pass-through


@pytest.mark.parametrize("text", [
    "Hello, this is a test of the text to speech path.",
    'Is it "working?" Yes. New line, 3.5 volts; done.',
    "gonna be lit fr fr, no cap",
    "teh quick brown fox",  # no autocorrect
])
def test_plain_text_passes_through_untouched(text):
    pieces = analyze(text)
    assert all(p.plain for p in pieces)
    assert [p.text for p in pieces] == expressive.split_sentences(text)
    assert all(p.pause_after_s == expressive.SENTENCE_GAP_S for p in pieces)


def test_empty_text_gives_no_pieces():
    assert analyze("  \n ") == []
    assert analyze("...") == []
