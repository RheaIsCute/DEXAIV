"""The index experiment's verdict rule and ABX writer, and the formant check's envelope
estimator (benchmarks/formant_index.py, benchmarks/formant_check.py). CPU only."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

import formant_check as fc
import formant_index as fi


def test_abx_threshold_is_13_of_16():
    assert fi.abx_threshold(16) == 13
    p13 = sum(math.comb(16, j) for j in range(13, 17)) / 2 ** 16
    assert p13 == pytest.approx(0.0106, abs=1e-4)
    assert fi.abx_threshold(16, alpha=0.05) == 12  # the looser, conventional bar


def test_verdict_rule():
    assert fi.verdict(0.10, 0.30) == "inert"
    assert fi.verdict(0.40, 0.30) == "changes output: needs the ABX"
    assert fi.verdict(0.40, 0.30, heard=True) == "heard"
    assert fi.verdict(0.40, 0.30, heard=False) == "not heard"


def test_write_abx_is_balanced_and_keeps_the_key_apart(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    a.write_bytes(b"AAAA")
    b.write_bytes(b"BBBB")
    key = fi.write_abx(a, b, tmp_path / "abx", seed=3)
    answers = key["answers"]
    assert len(answers) == 16 and sorted(answers.values()).count("A") == 8
    for trial, ans in answers.items():
        assert (tmp_path / "abx" / f"{trial}_X.wav").read_bytes() == (b"AAAA" if ans == "A" else b"BBBB")
    sheet = (tmp_path / "abx" / "abx.md").read_text()
    assert "trial16" in sheet and "AAAA" not in sheet
    assert json.loads((tmp_path / "abx" / "answer_key.json").read_text())["answers"] == answers
    again = fi.write_abx(a, b, tmp_path / "abx", seed=3)  # rewritable, same seed same key
    assert again["answers"] == answers


@pytest.mark.parametrize("st", [-2.0, 1.5])
def test_envelope_shift_recovers_a_resampled_spectrum(st):
    """Noise through a two-resonance filter, and the same with the resonances moved by f:
    the estimator reports f (the formant bench's envelope measure)."""
    from scipy.signal import lfilter

    sr = 32000
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(sr * 4)

    def voice(scale: float) -> np.ndarray:
        y = noise
        for fc_hz, bw in ((700.0, 120.0), (1800.0, 200.0)):
            r = math.exp(-math.pi * bw * scale / sr)
            theta = 2 * math.pi * fc_hz * scale / sr
            y = lfilter([1.0], [1.0, -2 * r * math.cos(theta), r * r], y)
        return (0.1 * y / np.abs(y).max()).astype(np.float32)

    f = 2 ** (st / 12)
    shift = fc.envelope_shift(voice(1.0), voice(f), sr)
    assert 12 * math.log2(shift) == pytest.approx(st, abs=0.3)


def test_framewise_cents_ignores_unvoiced_and_octave_errors():
    a = np.full(100, 200.0)
    b = a * 2 ** (10 / 1200)
    b[:10] = 0.0
    b[50] = 400.0
    cents, within = fc.framewise_cents(a, b)
    assert cents == pytest.approx(10.0, abs=0.01) and within == 1.0
