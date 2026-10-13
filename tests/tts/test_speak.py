"""Sentence splitting, crossfade, auto-pitch math and the orchestration (models mocked)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vox.audio.resample import save_audio
from vox.config.schemas import ModelConfig, ModelOverrides, RuntimeParams
from vox.config.store import Profile
from vox.tts import speak
from vox.tts.base import TtsBackend, TtsError

# ---------------------------------------------------------------------------- splitter


def test_split_sentences():
    text = 'Hello there.  Is it "working?" Yes!\nNew line… ¿Qué tal? Bien.'
    assert speak.split_sentences(text) == [
        "Hello there.", 'Is it "working?"', "Yes!", "New line…", "¿Qué tal?", "Bien."]
    assert speak.split_sentences("  \n ") == []
    assert speak.split_sentences("no terminator, 3.5 volts") == ["no terminator, 3.5 volts"]


def test_short_text_stays_in_one_piece():
    assert speak.split_text("One. Two. Three.") == ["One. Two. Three."]


def test_pieces_keep_whole_sentences_under_the_budget():
    sentences = [f"Sentence number {n} is here." for n in range(40)]
    pieces = speak.split_text(" ".join(sentences), max_seconds=10, chars_per_second=10)
    assert len(pieces) > 1 and all(len(p) <= 100 for p in pieces)
    assert " ".join(pieces) == " ".join(sentences)
    assert all(p.endswith(".") for p in pieces)


def test_overlong_sentence_is_cut_at_clauses_then_words():
    clauses = "first part of it, second part of it; third part of it: " + "word " * 30 + "end."
    pieces = speak.split_text(clauses, max_seconds=4, chars_per_second=10)
    assert all(len(p) <= 40 for p in pieces)
    assert " ".join(pieces).split() == clauses.split()
    assert pieces[0] == "first part of it, second part of it;"


def test_default_budget_is_thirty_seconds():
    assert speak.MAX_PIECE_S == 30.0
    pieces = speak.split_text("A sentence of some words. " * 100)
    assert len(pieces) > 1
    assert all(len(p) <= speak.MAX_PIECE_S * speak.CHARS_PER_SECOND for p in pieces)


# --------------------------------------------------------------------------- crossfade


def test_crossfade_overlaps_ten_ms_and_is_smooth():
    sr = 48000
    a, b = np.ones(sr, np.float32), np.full(sr, 0.5, np.float32)
    out = speak.crossfade_concat([a, b, a], sr)
    n = sr // 100
    assert len(out) == 3 * sr - 2 * n and out.dtype == np.float32
    seam = out[sr - n:sr]
    assert seam[0] == 1.0 and seam[-1] == 0.5 and np.all(np.diff(seam) <= 0)
    assert np.array_equal(speak.crossfade_concat([a], sr), a)


def test_crossfade_tolerates_tiny_pieces():
    out = speak.crossfade_concat([np.ones(3, np.float32), np.zeros(0, np.float32)], 48000)
    assert len(out) == 3


# -------------------------------------------------------------------------- auto pitch


@pytest.mark.parametrize("tts_f0, target, expected", [
    (200.0, 200.0, 0), (100.0, 200.0, 12), (220.0, 110.0, -12), (113.1, 200.0, 10),
    (208.9, 200.0, -1), (195.0, 200.0, 0), (0.0, 200.0, 0), (150.0, 0.0, 0),
    (50.0, 1000.0, 24), (1000.0, 50.0, -24)])
def test_auto_pitch(tts_f0, target, expected):
    assert speak.auto_pitch(tts_f0, target) == expected


class FakeExtractor:
    """Returns a scripted F0 track per call instead of analysing audio."""

    def __init__(self, *tracks):
        self.tracks = list(tracks)
        self.seen = []

    def extract(self, audio):
        self.seen.append(audio)
        return torch.tensor(self.tracks.pop(0))


def test_median_ignores_unvoiced_frames_and_pools_pieces():
    extractor = FakeExtractor([0.0, 100.0, 0.0, 120.0], [0.0, 140.0, 0.0])
    audio = [np.zeros(640, np.float32), np.zeros(480, np.float32)]
    assert speak.median_voiced_f0(extractor, audio, torch.device("cpu")) == 120.0
    assert [len(a) for a in extractor.seen] == [640, 480]
    assert speak.median_voiced_f0(FakeExtractor([0.0, 0.0]), audio[:1], torch.device("cpu")) == 0.0


def test_target_hint_prefers_profile_then_model_config_then_200():
    config = ModelConfig(name="Voice")
    assert speak.target_f0_hint(None, config) == 200.0
    assert speak.target_f0_hint(Profile(), config) == 200.0
    config.defaults = {speak.TARGET_F0_KEY: 120}
    assert speak.target_f0_hint(Profile(), config) == 120.0
    profile = Profile(model_overrides=ModelOverrides(
        models={"Voice": {speak.TARGET_F0_KEY: 230.5}, "Other": {speak.TARGET_F0_KEY: 90}}))
    assert speak.target_f0_hint(profile, config) == 230.5


# ----------------------------------------------------------------------- orchestration


class ToneBackend(TtsBackend):
    name = "tone"
    default_voice = "sine"

    def __init__(self):
        super().__init__()
        self.texts = []

    def unavailable_reason(self):
        return None

    def voices(self):
        return ["sine"]

    def synthesize(self, text, out_wav_path):
        self.texts.append(text)
        save_audio(out_wav_path, 0.1 * np.ones(24000, np.float32), 24000)  # 1 s at 24 kHz
        return 24000


class FakeConverter:
    def __init__(self, extractor):
        self.extractor = extractor
        self.calls = []

    def convert(self, audio_16k, params):
        self.calls.append((len(audio_16k), params))
        return np.repeat(audio_16k, 3)  # 16 kHz -> 48 kHz


@pytest.fixture
def fake_engine(monkeypatch):
    """Replace model loading and conversion; returns the fake converter and load arguments."""
    converter = FakeConverter(FakeExtractor([0.0, 100.0], [100.0], [100.0]))
    loads = []

    def load_engine(models_dir, model_name, shape, params, precision, device, warmup_hops):
        loads.append(SimpleNamespace(model=model_name, params=params, warmup_hops=warmup_hops))
        return SimpleNamespace(
            device=torch.device("cpu"), params=params or RuntimeParams(),
            loaded=SimpleNamespace(model=SimpleNamespace(tgt_sr=48000)))

    monkeypatch.setattr(speak, "load_engine", load_engine)
    monkeypatch.setattr(speak.FullFileConverter, "from_engine",
                        classmethod(lambda cls, engine: converter))
    return converter, loads


def test_speak_measures_pitch_and_writes_the_output(fake_engine, tmp_path):
    converter, loads = fake_engine
    backend = ToneBackend()
    params = RuntimeParams(f0_up_key=7.0, index_rate=0.4, protect=0.2)  # 7 = mic calibration
    out = tmp_path / "out.wav"
    result = speak.speak("Hello there. General greeting.", backend, tmp_path, "Voice", params,
                         target_f0=200.0, out=out)

    assert backend.texts == ["Hello there. General greeting."]
    assert loads[0].model == "Voice" and loads[0].warmup_hops == 0
    assert result.auto_pitch and result.tts_median_f0 == 100.0 and result.pitch == 12.0
    (samples, used), = converter.calls
    assert samples == 16000  # resampled to what the converter expects
    assert (used.f0_up_key, used.index_rate, used.protect) == (12.0, 0.4, 0.2)
    assert result.sr == 48000 and result.seconds == 1.0 and result.tts_seconds == 1.0
    assert result.path == out and out.is_file()


def test_explicit_pitch_skips_the_measurement(fake_engine, tmp_path):
    converter, _ = fake_engine
    result = speak.speak("Hi.", ToneBackend(), tmp_path, "Voice", pitch=-3)
    assert not result.auto_pitch and result.pitch == -3.0 and result.path is None
    assert converter.calls[0][1].f0_up_key == -3.0
    assert len(converter.extractor.tracks) == 3  # the extractor was never asked


def test_model_without_f0_gets_no_transposition(fake_engine, tmp_path):
    converter, _ = fake_engine
    converter.extractor = None
    assert speak.speak("Hi.", ToneBackend(), tmp_path, "Voice").pitch == 0.0


def test_long_text_is_converted_piecewise_with_one_pitch(fake_engine, tmp_path):
    converter, _ = fake_engine
    backend = ToneBackend()
    result = speak.speak("First sentence is here. Second sentence is here. Third one.",
                         backend, tmp_path, "Voice", max_piece_s=2.2)  # 30 characters

    assert backend.texts == ["First sentence is here.", "Second sentence is here.", "Third one."]
    assert result.pieces == 3 and result.pitch == 12.0
    gap = int(16000 * speak.PIECE_GAP_S)
    assert [n for n, _ in converter.calls] == [16000 + gap, 16000 + gap, 16000]
    assert {p.f0_up_key for _, p in converter.calls} == {12.0}
    fade = int(48000 * speak.CROSSFADE_MS / 1000)
    assert len(result.audio) == 3 * (3 * 16000) + 2 * 3 * gap - 2 * fade


def test_empty_text_is_an_error(fake_engine, tmp_path):
    with pytest.raises(TtsError, match="empty"):
        speak.speak("  \n", ToneBackend(), tmp_path, "Voice")
