"""Seam-rate modulation prominence (offline/metrics.py): tracks imposed AM depth
monotonically at the hop rate, and is specific to that rate."""

import numpy as np
import pytest

from vox.offline.metrics import band_envelopes, impose_am, seam_prominence, third_octave_bands

SR = 16000


def voice_like(seconds: float = 3.0, f0: float = 150.0, seed: int = 0) -> np.ndarray:
    """Harmonics up to 6 kHz with a 4 Hz syllabic swell and a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi)) / k
            for k in range(1, int(6000 // f0)))
    syllables = 0.6 + 0.4 * np.cos(2 * np.pi * 4.0 * t)
    return (0.2 * x * syllables + 0.005 * rng.standard_normal(len(t))).astype(np.float32)


def test_bands_stop_below_the_nyquist():
    bands = third_octave_bands(SR)
    assert bands[0][0] < 200 < bands[0][1] and bands[-1][1] < 0.45 * SR
    assert len(third_octave_bands(48000)) == 17 and len(bands) < 17
    env = band_envelopes(voice_like(0.5), SR)
    assert env.shape == (len(bands), 500) and (env >= 0).all()


@pytest.mark.parametrize("hop_ms, span_db", [(20.0, 15.0), (60.0, 8.0)])
def test_prominence_tracks_imposed_am_depth_monotonically(hop_ms, span_db):
    """At 16.7 Hz (hop 60) the synthetic talker's 4 Hz swell has harmonics inside the
    floor region, so the span is smaller there; the order must hold at both rates."""
    base = voice_like()
    rate = 1000.0 / hop_ms
    values = [seam_prominence(impose_am(base, SR, rate, d), SR, hop_ms)["seam_db"]
              for d in (0.0, 0.02, 0.05, 0.10, 0.20)]
    assert all(np.isfinite(values))
    assert all(b > a for a, b in zip(values, values[1:])), values
    assert values[-1] - values[0] > span_db  # 20 % depth is far above the floor


def test_depth_reads_the_imposed_depth():
    base = voice_like()
    clean = seam_prominence(base, SR, 20.0)["seam_depth_pct"]
    for d in (0.05, 0.10, 0.20):
        read = seam_prominence(impose_am(base, SR, 50.0, d), SR, 20.0)["seam_depth_pct"]
        assert abs(read - 100 * d) < 0.35 * 100 * d + clean, (d, read, clean)
    assert clean < 3.0


def test_prominence_is_specific_to_the_hop_rate():
    base = voice_like()
    at_50 = impose_am(base, SR, 50.0, 0.1)
    clean = seam_prominence(base, SR, 20.0)["seam_db"]
    assert seam_prominence(at_50, SR, 20.0)["seam_db"] > clean + 10
    # the 50 Hz AM is not a 16.7 Hz seam (hop 60): its fundamental reads near the clean signal
    assert abs(seam_prominence(at_50, SR, 60.0)["seam_db"]
               - seam_prominence(base, SR, 60.0)["seam_db"]) < 4.0


def test_short_or_silent_input_is_nan_not_an_error():
    assert np.isnan(seam_prominence(np.zeros(100, dtype=np.float32), SR, 20.0)["seam_db"])
    out = seam_prominence(np.zeros(SR, dtype=np.float32), SR, 20.0)
    assert np.isnan(out["seam_db"]) and out["blocks"] >= 0
