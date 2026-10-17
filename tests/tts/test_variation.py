"""Per-utterance variation: bounds, reproducibility, variation=0 is the old path, and one
draw applies to every piece of an utterance."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vox.audio.resample import save_audio
from vox.config.schemas import RuntimeParams
from vox.tts import speak, variation
from vox.tts.backends import KokoroBackend
from vox.tts.base import TtsBackend
from vox.tts.variation import Variation, blend_candidates, draw

AF = ["af_alloy", "af_bella", "af_heart", "af_nicole", "af_sarah", "am_adam", "bf_emma"]


# --------------------------------------------------------------------------- the draw


def test_zero_draws_nothing_and_keeps_the_callers_seed():
    assert draw(0.0, 0, "af_heart", AF) == Variation(rvc_seed=0)
    assert draw(0.0, None) == Variation(rvc_seed=None)
    assert draw(0.0, 5).neutral and draw(-1.0, 5).neutral


@pytest.mark.parametrize("amount", [0.1, 0.35, 1.0, 3.0])
def test_every_offset_stays_inside_its_bound(amount):
    a = min(amount, 1.0)
    for seed in range(300):
        v = draw(amount, seed, "af_heart", AF)
        assert v.amount == a
        assert abs(v.speed - 1.0) <= variation.SPEED_JITTER * a + 1e-12
        assert abs(v.pitch_offset) <= variation.PITCH_JITTER_ST * a + 1e-12
        assert abs(v.pause_scale - 1.0) <= variation.PAUSE_JITTER * a + 1e-12
        assert 1 <= len(v.blend) <= 2 or (not v.blend)
        assert sum(w for _, w in v.blend) <= variation.BLEND_MAX * a + 1e-12
        assert all(name.startswith("af_") and name != "af_heart" for name, _ in v.blend)
        assert all(w > 0 for _, w in v.blend)


def test_a_seed_reproduces_the_draw_and_no_seed_is_fresh():
    assert draw(0.35, 42, "af_heart", AF) == draw(0.35, 42, "af_heart", AF)
    assert draw(0.35, 42).speed == draw(0.35, 42, "af_heart", AF).speed  # voices do not shift it
    a, b = draw(0.35, 1), draw(0.35, 2)
    assert (a.speed, a.pitch_offset, a.rvc_seed) != (b.speed, b.pitch_offset, b.rvc_seed)
    first = draw(0.35)
    time.sleep(0.001)
    assert draw(0.35).seed != first.seed


def test_blend_candidates_follow_the_voice_family_or_an_allowlist():
    assert blend_candidates("af_heart", AF) == ["af_alloy", "af_bella", "af_nicole", "af_sarah"]
    assert blend_candidates("am_adam", AF) == []
    assert blend_candidates("af_heart", AF, ["af_bella", "af_heart", "zz_none"]) == ["af_bella"]
    assert blend_candidates("en-US-AriaNeural", ["en-US-AriaNeural", "en-US-GuyNeural"]) == []
    assert blend_candidates("af_heart", []) == []
    assert all(not draw(1.0, s, "af_heart", AF, ["af_bella"]).blend
               or {n for n, _ in draw(1.0, s, "af_heart", AF, ["af_bella"]).blend} == {"af_bella"}
               for s in range(50))


def test_describe_and_bounds_text():
    assert "off" in Variation(rvc_seed=0).describe()
    text = draw(0.35, 3, "af_heart", AF).describe()
    assert "seed 3" in text and "speed x" in text
    assert "+-1.4%" in variation.bounds(0.35)


# -------------------------------------------------------------------- kokoro blending


def test_kokoro_style_blend_is_a_weighted_mix():
    styles = {"af_heart": np.ones((2, 1, 3), np.float32),
              "af_bella": np.zeros((2, 1, 3), np.float32),
              "af_sarah": np.full((2, 1, 3), 3.0, np.float32)}
    backend = KokoroBackend("af_heart")
    backend._kokoro = SimpleNamespace(get_voices=lambda: list(styles),
                                      get_voice_style=lambda name: styles[name])
    assert backend.style() is styles["af_heart"]
    mixed = backend.style((("af_bella", 0.1), ("af_sarah", 0.05), ("unknown", 0.5)))
    assert mixed.dtype == np.float32
    assert np.allclose(mixed, 0.85 * 1 + 0.1 * 0 + 0.05 * 3)
    assert KokoroBackend.blends and not speak.TtsBackend.blends


# ------------------------------------------------------------------ through speak()


class StretchBackend(TtsBackend):
    """Renders 1 s / speed of a tone, records every call."""

    name = "stretch"
    default_voice = "af_heart"
    blends = True

    def __init__(self):
        super().__init__()
        self.calls = []

    def unavailable_reason(self):
        return None

    def voices(self):
        return list(AF)

    def synthesize(self, text, out_wav_path, speed=1.0, blend=()):
        self.calls.append((text, speed, blend))
        save_audio(out_wav_path, 0.1 * np.ones(round(24000 / speed), np.float32), 24000)
        return 24000


class SeededConverter:
    """16 kHz -> 48 kHz plus a little noise from the torch RNG, so the seed shows."""

    def __init__(self, extractor):
        self.extractor = extractor
        self.calls = []

    def convert(self, audio_16k, params, rms_frame_ms=None, f0_contour_scale=1.0):
        self.calls.append((len(audio_16k), params))
        noise = torch.rand(len(audio_16k) * 3).numpy() * 1e-3
        return (np.repeat(audio_16k, 3) + noise).astype(np.float32)


class ConstantExtractor:
    def __init__(self, hz=100.0):
        self.hz = hz
        self.seen = 0

    def extract(self, audio):
        self.seen += 1
        return torch.full((len(audio) // 160,), self.hz)


@pytest.fixture
def engine(monkeypatch):
    converter = SeededConverter(ConstantExtractor())

    def load_engine(models_dir, model_name, shape, params, precision, device, warmup_hops):
        return SimpleNamespace(device=torch.device("cpu"), params=params or RuntimeParams(),
                               loaded=SimpleNamespace(model=SimpleNamespace(tgt_sr=48000)))

    monkeypatch.setattr(speak, "load_engine", load_engine)
    monkeypatch.setattr(speak.FullFileConverter, "from_engine",
                        classmethod(lambda cls, engine: converter))
    return converter


TEXT = "hey... hey!! are you fr right now"


def test_variation_zero_is_the_old_path_byte_for_byte(engine, tmp_path):
    old = speak.speak(TEXT, StretchBackend(), tmp_path, "Voice", seed=0)
    calls_old = list(engine.calls)
    engine.calls.clear()
    backend = StretchBackend()
    new = speak.speak(TEXT, backend, tmp_path, "Voice", seed=0, variation=draw(0.0, 0))
    assert np.array_equal(old.audio, new.audio)
    assert [(n, p) for n, p in engine.calls] == calls_old
    assert [(s, b) for _, s, b in backend.calls] == [(1.0, ()), (pytest.approx(1.05), ()),
                                                     (1.0, ())]
    assert new.pitch_offset == 0.0 and new.pitch == 12.0


def test_one_draw_applies_to_every_piece(engine, tmp_path):
    var = Variation(amount=0.5, seed=1, rvc_seed=9, speed=1.02, pitch_offset=0.13,
                    pause_scale=0.9, blend=(("af_bella", 0.04),))
    backend = StretchBackend()
    result = speak.speak(TEXT, backend, tmp_path, "Voice", variation=var)
    assert [s for _, s, _ in backend.calls] == [pytest.approx(1.02), pytest.approx(1.05 * 1.02),
                                                pytest.approx(1.02)]
    assert {b for _, _, b in backend.calls} == {(("af_bella", 0.04),)}
    # explicit pause after the dots is scaled, the plain sentence gap is not
    piece = round(16000 / 1.02)
    assert engine.calls[0][0] == piece + round(16000 * 0.25 * 0.9)
    assert engine.calls[1][0] == round(16000 / (1.05 * 1.02)) + round(16000 * speak.PIECE_GAP_S)
    assert [p.f0_up_key for _, p in engine.calls] == [pytest.approx(12.13)] * 3
    assert result.pitch == 12.0 and result.pitch_offset == pytest.approx(0.13)
    assert result.variation is var


def test_blend_is_not_sent_to_a_backend_that_cannot_blend(engine, tmp_path):
    backend = StretchBackend()
    backend.blends = False
    speak.speak("Hi.", backend, tmp_path, "Voice", variation=draw(1.0, 3, "af_heart", AF))
    assert backend.calls[0][2] == ()


def test_two_varied_renders_differ_but_keep_pitch_and_length(engine, tmp_path):
    a = speak.speak(TEXT, StretchBackend(), tmp_path, "Voice", variation=draw(0.35, 1))
    b = speak.speak(TEXT, StretchBackend(), tmp_path, "Voice", variation=draw(0.35, 2))
    assert a.pitch == b.pitch  # the transposition itself is measured, not drawn
    assert abs(a.pitch_offset - b.pitch_offset) <= 2 * variation.PITCH_JITTER_ST * 0.35
    assert len(a.audio) != len(b.audio) or not np.array_equal(a.audio, b.audio)
    assert abs(len(a.audio) / len(b.audio) - 1.0) < 0.10


def test_a_model_without_f0_gets_no_pitch_offset(engine, tmp_path):
    engine.extractor = None
    result = speak.speak("Hi.", StretchBackend(), tmp_path, "Voice",
                         variation=Variation(amount=1, seed=1, rvc_seed=1, pitch_offset=0.3))
    assert result.pitch == 0.0 and result.pitch_offset == 0.0
    assert engine.calls[0][1].f0_up_key == 0.0
