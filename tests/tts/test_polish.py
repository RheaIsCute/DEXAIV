"""Typed-speech polish (Step 3 of the live-quality work): short-frame loudness mix, F0
contour widening and Kokoro voice blends. All off by default; the plain path is unchanged."""

import math

import numpy as np
import pytest
import torch

from vox.config.schemas import RuntimeParams
from vox.offline.convert import _change_rms, scale_contour
from vox.tts import speak
from vox.tts.backends import parse_voice_blend
from vox.tts.base import TtsError


def syllables(seconds: float, sr: int, rate_hz: float = 4.0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    envelope = 0.05 + 0.25 * (0.5 + 0.5 * np.sin(2 * math.pi * rate_hz * t))
    return (envelope * np.sin(2 * math.pi * 200 * t)).astype(np.float32)


def frame_db(x: np.ndarray, sr: int) -> np.ndarray:
    hop = sr // 100
    n = len(x) // hop
    return 10 * np.log10(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)


def test_short_rms_frames_carry_syllable_stress_the_1s_frames_flatten():
    source = syllables(3.0, 16000)
    flat = (0.2 * np.sin(2 * math.pi * 220 * np.arange(3 * 48000) / 48000)).astype(np.float32)
    upstream = _change_rms(source, 16000, flat, 48000, 0.35)
    short = _change_rms(source, 16000, flat, 48000, 0.35, frame_ms=40.0)
    swing = {k: np.ptp(frame_db(v, sr)[50:-50]) for k, v, sr in
             (("source", source, 16000), ("upstream", upstream, 48000),
              ("short", short, 48000))}
    assert swing["upstream"] < 3.0  # 1 s frames: the syllables are averaged away
    assert swing["short"] > 0.5 * swing["source"]  # 40 ms frames: they come through


def test_scale_contour_widens_around_the_median_in_semitones():
    rng = np.random.default_rng(0)
    semis = rng.normal(0, 2.0, 400)
    f0 = 200.0 * 2 ** (semis / 12)
    f0[::7] = 0.0  # unvoiced frames stay unvoiced
    pitchf = torch.tensor(f0, dtype=torch.float32)
    coarse, wide = scale_contour(pitchf, 1.2)
    voiced = pitchf > 0
    assert torch.equal(wide[~voiced], pitchf[~voiced]) and torch.all(coarse[~voiced] == 1)
    def st(x):
        return 12 * torch.log2(x[voiced] / x[voiced].median())

    assert wide[voiced].median() == pytest.approx(pitchf[voiced].median().item(), rel=1e-4)
    assert st(wide).std() == pytest.approx(1.2 * st(pitchf).std().item(), rel=1e-3)
    _, same = scale_contour(pitchf, 1.0)
    assert torch.allclose(same, pitchf)


def test_parse_voice_blend():
    assert parse_voice_blend("af_heart") == [("af_heart", 1.0)]
    blend = parse_voice_blend("af_heart*0.7 + af_bella*0.3")
    assert [n for n, _ in blend] == ["af_heart", "af_bella"]
    assert sum(w for _, w in blend) == pytest.approx(1.0) and blend[0][1] == pytest.approx(0.7)
    assert parse_voice_blend("a*2+b*2") == [("a", 0.5), ("b", 0.5)]
    for bad in ("af_heart*x", "*0.5", "a*-1"):
        with pytest.raises(TtsError):
            parse_voice_blend(bad)


class RecordingConverter:
    def __init__(self):
        self.extractor = None
        self.calls = []

    def convert(self, audio_16k, params, *options):
        self.calls.append((params.volume_envelope, options))
        return np.repeat(audio_16k, 3)


def test_polish_reaches_the_converter_and_the_default_does_not():
    rendered = [np.ones(1600, np.float32)]
    converter = RecordingConverter()
    speak.convert_rendered(converter, torch.device("cpu"), 48000, rendered, RuntimeParams(),
                           pitch=0)
    assert converter.calls == [(1.0, ())]  # plain: the old two-argument call
    polish = speak.TypedPolish(envelope=0.35, rms_frame_ms=40.0, contour_scale=1.2)
    speak.convert_rendered(converter, torch.device("cpu"), 48000, rendered, RuntimeParams(),
                           pitch=0, polish=polish)
    assert converter.calls[-1] == (0.35, (40.0, 1.2))
    assert speak.TypedPolish().plain and not polish.plain
