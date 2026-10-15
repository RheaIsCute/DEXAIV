"""Live microphone conditioning: high-pass, auto level, trim and soft limit (no models)."""

import math

import numpy as np
import pytest

from vox.audio.conditioning import ConditioningConfig, InputConditioner, soft_limit

SR, BLOCK = 48000, 960  # 20 ms hops


def tone(seconds: float, freq: float, dbfs: float, start: int = 0) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(start, start + n) / SR
    return (math.sqrt(2) * 10 ** (dbfs / 20) * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def speechlike(seconds: float, dbfs: float, seed: int = 0) -> np.ndarray:
    """Harmonic 'syllables' at ~4 Hz with a 20 dB swing: an RMS level that moves like speech."""
    t = np.arange(int(seconds * SR)) / SR
    voice = sum(np.sin(2 * math.pi * 120 * k * t) / k for k in range(1, 12))
    envelope = 10 ** ((-10 + 10 * np.sin(2 * math.pi * 4 * t)) / 20)
    x = voice * envelope
    x *= 10 ** (dbfs / 20) / np.sqrt(np.mean(x * x))
    return x.astype(np.float32)


def run(cond: InputConditioner, x: np.ndarray) -> np.ndarray:
    out = [cond.process(x[i:i + BLOCK]).copy() for i in range(0, len(x) - BLOCK + 1, BLOCK)]
    return np.concatenate(out)


def rms_db(x: np.ndarray) -> float:
    return 20 * math.log10(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) + 1e-12)


def test_disabled_chain_is_the_identity():
    cond = InputConditioner(SR, BLOCK, ConditioningConfig(enabled=False))
    x = tone(0.1, 200, -30)[:BLOCK]
    assert cond.process(x) is x and cond.gain_db == 0.0


def test_highpass_removes_thumps_keeps_the_voice_and_is_seamless():
    cfg = ConditioningConfig(auto_level=False)
    low = run(InputConditioner(SR, BLOCK, cfg), tone(2.0, 25, -30))
    assert rms_db(low[SR:]) < -30 - 15  # a 25 Hz desk thump: down > 15 dB
    voice = run(InputConditioner(SR, BLOCK, cfg), tone(2.0, 300, -30))
    assert abs(rms_db(voice[SR:]) + 30) < 0.3  # 300 Hz: untouched
    # block-by-block equals filtering the whole signal (the state carries over)
    from scipy.signal import butter, sosfilt

    x = speechlike(1.0, -30)
    whole = sosfilt(butter(2, 70, btype="highpass", fs=SR, output="sos"), x.astype(np.float64))
    streamed = run(InputConditioner(SR, BLOCK, cfg), x)
    np.testing.assert_allclose(streamed, whole[:len(streamed)], atol=1e-6)


def test_auto_level_lifts_a_quiet_mic_toward_the_target_within_the_cap():
    cond = InputConditioner(SR, BLOCK, ConditioningConfig())
    out = run(cond, speechlike(6.0, -42))
    assert cond.agc_db == pytest.approx(18.0, abs=0.5)  # wants +22, capped at +18
    assert rms_db(out[3 * SR:]) == pytest.approx(-42 + 18, abs=1.0)
    cond = InputConditioner(SR, BLOCK, ConditioningConfig())
    out = run(cond, speechlike(6.0, -30))
    assert rms_db(out[3 * SR:]) == pytest.approx(-20, abs=1.5)  # in range: reaches it


def test_auto_level_turns_a_hot_mic_down_fast():
    cond = InputConditioner(SR, BLOCK, ConditioningConfig())
    run(cond, speechlike(0.5, -12))
    assert cond.agc_db < -6  # half a second in, already well on the way down to -8


def test_gain_is_held_in_pauses_below_the_noise_floor():
    cfg = ConditioningConfig(noise_floor_dbfs=-60.0)
    cond = InputConditioner(SR, BLOCK, cfg)
    run(cond, speechlike(4.0, -40))
    held = cond.agc_db
    rng = np.random.default_rng(0)
    room = (10 ** (-62 / 20) * rng.standard_normal(3 * SR)).astype(np.float32)
    run(cond, room)  # three seconds of room tone at -62 dBFS: below floor + 6 dB
    assert cond.agc_db == pytest.approx(held, abs=1e-9)


def test_trim_alone_when_auto_level_is_off_and_toggles_ramp_smoothly():
    cond = InputConditioner(SR, BLOCK, ConditioningConfig(auto_level=False, trim_db=6.0))
    out = run(cond, tone(1.0, 300, -30))
    assert rms_db(out[SR // 2:]) == pytest.approx(-24, abs=0.2)
    # switching auto level on mid-stream: the gain ramps across one block, no step
    x = tone(2.0, 300, -40)
    first = run(cond, x[:SR])
    cond.set_config(ConditioningConfig(auto_level=True, trim_db=6.0))
    second = run(cond, x[SR:])
    joined = np.concatenate([first, second])
    max_slope = 2 * math.pi * 300 / SR * np.abs(joined).max()
    assert np.abs(np.diff(joined)).max() <= max_slope * 1.05


def test_soft_limit_is_transparent_below_the_knee_and_never_passes_the_ceiling():
    x = np.linspace(-3, 3, 10001)
    y = soft_limit(x, -3.0)
    knee, ceiling = 10 ** (-6 / 20), 10 ** (-3 / 20)
    quiet = np.abs(x) <= knee
    np.testing.assert_array_equal(y[quiet], x[quiet])
    assert np.abs(y).max() < ceiling
    assert np.all(np.diff(y) >= 0)  # monotonic
    assert np.abs(np.diff(y)).max() <= np.abs(np.diff(x)).max() * 1.0001  # slope <= 1


def test_resize_keeps_state_for_a_live_hop_change():
    cond = InputConditioner(SR, BLOCK, ConditioningConfig())
    run(cond, speechlike(3.0, -40))
    gain = cond.agc_db
    cond.resize(1440)
    assert cond.agc_db == gain
    assert cond.process(np.zeros(1440, np.float32)).shape == (1440,)
