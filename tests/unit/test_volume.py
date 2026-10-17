"""Volume controls: the ramped gain, the speech level, clip matching, and where each gain
sits (output callback, clip player)."""

import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from vox.audio import volume as vol
from vox.audio.devices import DeviceInfo
from vox.audio.rings import SPSCRing
from vox.audio.streams import DuplexAudio, StreamConfig
from vox.engine.override import ClipOverride

OUT = DeviceInfo(7, "Speakers", "Windows WASAPI", 0, 2, 48000.0, 0.0, 0.003)


# ---------------------------------------------------------------------- the math


def test_db_and_clamp():
    assert vol.db_to_gain(0) == 1.0
    assert vol.db_to_gain(-6) == pytest.approx(0.5012, abs=1e-4)
    assert vol.db_to_gain(6) == pytest.approx(1.9953, abs=1e-4)
    assert vol.clamp(-40, vol.OUTPUT_DB_RANGE) == -24.0 and vol.clamp(9, vol.OUTPUT_DB_RANGE) == 6
    assert vol.clamp(-20, vol.TYPED_DB_RANGE) == -12.0 and vol.clamp(20, vol.TYPED_DB_RANGE) == 12


def test_unity_gain_never_touches_the_signal():
    gain = vol.RampedGain(480)
    block = np.linspace(-1, 1, 333, dtype=np.float32)
    before = block.copy()
    assert gain.unity
    gain.apply(block)
    assert np.array_equal(block, before)


@pytest.mark.parametrize("block", [64, 128, 480, 2048])
def test_a_change_ramps_over_the_ramp_length_whatever_the_block(block):
    gain = vol.RampedGain(480)
    gain.target = 0.25
    out = []
    for _ in range(2048 // block + 1):
        buf = np.ones(block, dtype=np.float32)
        gain.apply(buf)
        out.append(buf)
    out = np.concatenate(out)
    assert out[0] == pytest.approx(1 - 0.75 / 480, abs=1e-6)
    assert out[479] == pytest.approx(0.25, abs=1e-6)
    assert np.all(np.diff(out[:480]) < 0)  # monotonic, no step anywhere
    assert np.max(np.abs(np.diff(out[:480]))) == pytest.approx(0.75 / 480, rel=1e-3)
    assert np.allclose(out[480:], 0.25) and gain.now == 0.25


def test_a_new_target_mid_ramp_restarts_from_where_it_is():
    gain = vol.RampedGain(100)
    gain.target = 0.0
    buf = np.ones(50, dtype=np.float32)
    gain.apply(buf)
    assert gain.now == pytest.approx(0.5)
    gain.target = 1.0
    buf = np.ones(100, dtype=np.float32)
    gain.apply(buf)
    assert buf[0] == pytest.approx(0.505) and buf[-1] == pytest.approx(1.0)
    assert np.all(np.diff(buf) > 0) and gain.unity


def test_jump_sets_the_gain_without_a_ramp():
    gain = vol.RampedGain(480)
    gain.jump(0.5)
    buf = np.ones(10, dtype=np.float32)
    gain.apply(buf)
    assert np.allclose(buf, 0.5)


def test_speech_level_skips_pauses_and_follows_the_voice():
    level = vol.SpeechLevel(tau_s=1.0)
    assert level.dbfs is None
    level.add(np.full(960, 1e-4, dtype=np.float32), 0.02)  # -80 dBFS: a pause
    assert level.dbfs is None
    level.add(np.full(960, 0.1, dtype=np.float32), 0.02)
    assert level.dbfs == pytest.approx(-20.0, abs=0.01)
    for _ in range(500):  # 10 s at -6 dBFS: the old level is forgotten
        level.add(np.full(960, 0.5, dtype=np.float32), 0.02)
    assert level.dbfs == pytest.approx(-6.02, abs=0.05)


def tone(seconds, amp, sr=48000):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_active_level_ignores_silence_and_fades():
    speech = tone(1.0, 0.1)  # RMS 0.0707: -23.0 dBFS
    padded = np.concatenate([np.zeros(48000, np.float32), speech, np.zeros(96000, np.float32)])
    expected = 20 * math.log10(0.1 / math.sqrt(2))
    assert vol.active_level_dbfs(speech, 48000) == pytest.approx(expected, abs=0.1)
    assert vol.active_level_dbfs(padded, 48000) == pytest.approx(expected, abs=0.1)
    assert vol.active_level_dbfs(np.zeros(48000, np.float32), 48000) is None
    assert vol.active_level_dbfs(np.zeros(10, np.float32), 48000) is None


def test_match_gain_is_bounded():
    assert vol.match_gain_db(-30.0, -20.0) == 10.0
    assert vol.match_gain_db(-70.0, -20.0) == vol.MATCH_LIMIT_DB[1]
    assert vol.match_gain_db(0.0, -40.0) == vol.MATCH_LIMIT_DB[0]
    assert vol.match_gain_db(None, -20.0) == 0.0


# ------------------------------------------------------------ the output callback


def make_audio():
    cfg = StreamConfig(input=None, output=OUT)
    ring = SPSCRing(48000)
    return DuplexAudio(cfg, SPSCRing(48000), ring, 960, threading.Event()), ring


def play(audio, ring, block):
    ring.write(block)
    out = np.zeros((len(block), 2), dtype=np.float32)
    audio._output_callback(out, len(block), None, SimpleNamespace(output_underflow=False))
    return out


def test_output_volume_at_unity_leaves_the_output_untouched():
    audio, ring = make_audio()
    block = np.linspace(-0.5, 0.5, 480, dtype=np.float32)
    out = play(audio, ring, block)
    assert np.array_equal(out[:, 0], block) and np.array_equal(out[:, 1], block)


def test_output_volume_ramps_holds_and_the_meter_follows():
    audio, ring = make_audio()
    half = np.full(480, 0.5, dtype=np.float32)
    audio.volume.target = vol.db_to_gain(-6.0)
    ramped = play(audio, ring, half)[:, 0]
    assert np.all(np.diff(ramped) <= 0)
    held = play(audio, ring, half)[:, 0]
    assert np.allclose(held, 0.5 * vol.db_to_gain(-6.0))
    audio.health.output_peak = 0.0
    play(audio, ring, half)
    assert audio.health.output_peak == pytest.approx(0.5 * vol.db_to_gain(-6.0), abs=1e-6)


def test_output_boost_is_clipped_at_full_scale():
    audio, ring = make_audio()
    audio.volume.jump(vol.db_to_gain(6.0))
    out = play(audio, ring, np.full(480, 0.9, dtype=np.float32))[:, 0]
    assert np.allclose(out, 1.0) and audio.health.output_peak == pytest.approx(1.0)


# ---------------------------------------------------------------- the clip player


def test_typed_volume_scales_the_clip_only_and_works_mid_clip():
    override = ClipOverride(480)
    override.volume.jump(0.5)
    override.submit(np.full(2400, 0.4, dtype=np.float32))
    live = np.full(480, 0.2, dtype=np.float32)
    first = live.copy()
    override.mix(first)  # live fades out over this hop, the clip comes in at half
    assert np.allclose(first - 0.2 * override._fade_out, 0.2)
    second = live.copy()
    override.mix(second)
    assert np.allclose(second, 0.2)  # clip alone, at the typed volume
    override.volume.target = 1.0  # moved while the clip plays: ramped, no step
    third = live.copy()
    override.mix(third)
    assert np.all(np.diff(third) >= 0) and third[-1] == pytest.approx(0.4)
    assert np.max(np.abs(np.diff(np.concatenate([second, third])))) < 0.001


def test_typed_volume_at_unity_is_the_old_mix():
    a, b = ClipOverride(480), ClipOverride(480)
    clip = np.linspace(-0.3, 0.3, 1500, dtype=np.float32)
    b.volume.target = 0.5
    b.volume.jump(1.0)
    for o in (a, b):
        o.submit(clip)
    for _ in range(4):
        x, y = np.full(480, 0.1, np.float32), np.full(480, 0.1, np.float32)
        a.mix(x)
        b.mix(y)
        assert np.array_equal(x, y)
