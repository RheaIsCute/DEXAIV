"""Cut attribution: the event log, the modelled output ring, the output tap and the
live-mic diagnostics in offline.metrics (no models)."""

import math

import numpy as np
import pytest

from vox.audio.rings import SPSCRing
from vox.audio.streams import OutputTap
from vox.offline import metrics
from vox.offline.simulate import _PlayoutModel
from vox.profiling.timers import EventLog, StageTimers

SR = 48000


def test_event_log_keeps_kinds_apart_and_wraps():
    log = EventLog(capacity=4)
    log.add("gate_close", 10, 0.2, -50.0)
    log.add("underrun", 11, 0.22, 480, count=2)
    log.add("gate_open", 15, 0.3, -40.0)
    assert log.counts == {"gate_open": 1, "gate_close": 1, "underrun": 2}
    first = log.since(0)
    assert [e["kind"] for e in first] == ["gate_close", "underrun", "gate_open"]
    assert first[1]["value"] == 480 and first[1]["hop"] == 11
    for n in range(5):
        log.add("gate_close", 20 + n, 1.0 + n)
    assert log.total == 8 and len(log.since(0)) == 4  # ring keeps the last four
    assert log.since(7)[0]["hop"] == 24
    log.reset()
    assert log.total == 0 and log.counts["gate_close"] == 0


def test_timers_reset_clears_the_event_log():
    timers = StageTimers(("a",), "cpu")
    timers.events.add("underrun", 1, 0.0, 10)
    timers.reset()
    assert timers.events.total == 0


def test_ring_stamps_underruns():
    ring = SPSCRing(64)
    ring.write(np.ones(4, np.float32))
    out = np.zeros(8, np.float32)
    assert ring.read(out) == 4
    assert ring.underruns == 1 and ring.underrun_samples == 4 and ring.last_underrun_at > 0


def test_playout_model_counts_a_late_hop_and_refills():
    block, fill = 960, 2400  # 20 ms hops, 50 ms fill
    model = _PlayoutModel(fill, block, SR)
    assert model.hop_written(0.0) == 0  # playback starts here
    t = 0.0
    for _ in range(10):  # on time: one hop per 20 ms
        t += 0.020
        assert model.hop_written(t) == 0
    t += 0.020 + 0.080  # this hop is 80 ms late: the 50 ms (+ one hop) cushion runs out
    missing = model.hop_written(t)
    assert missing == pytest.approx(SR * 0.030, abs=2)
    t += 0.020  # topped back up: the next hop is fine again
    assert model.hop_written(t) == 0


def test_output_tap_marks_underrun_filled_samples():
    tap = OutputTap(SR, 0.01)  # 480 samples
    tap.write(np.ones(300, np.float32), 300)
    tap.write(np.zeros(300, np.float32), 100)  # 200 samples zero-filled; only 180 fit
    assert tap.pos == 480
    assert tap.from_ring[:400].all() and not tap.from_ring[400:].any()
    tap.write(np.ones(10, np.float32), 10)  # full: ignored
    assert tap.pos == 480


def test_zero_runs_finds_exact_digital_silence_only():
    x = np.ones(1000, np.float32)
    x[100:300] = 0.0
    x[500:505] = 0.0
    x[700:800] = 1e-9  # quiet is not zero
    x[990:] = 0.0
    assert metrics.zero_runs(x, 10) == [(100, 300), (990, 1000)]


def test_phrases_merge_short_gaps():
    t = np.arange(4 * SR) / SR
    x = (0.1 * np.sin(2 * math.pi * 200 * t)).astype(np.float32)
    x[int(1.0 * SR):int(1.1 * SR)] = 0  # 100 ms gap: bridged
    x[int(2.0 * SR):int(3.0 * SR)] = 0  # 1 s gap: splits
    spans = metrics.phrases(x, SR)
    assert len(spans) == 2
    assert spans[0][0] == 0 and abs(spans[0][1] - 2.0 * SR) < 0.03 * SR
    assert abs(spans[1][0] - 3.0 * SR) < 0.03 * SR


def test_band_ratio_tells_hiss_from_a_low_tone():
    rng = np.random.default_rng(0)
    t = np.arange(SR) / SR
    tone = (0.1 * np.sin(2 * math.pi * 300 * t)).astype(np.float32)
    hiss = (0.1 * rng.standard_normal(SR)).astype(np.float32)
    assert metrics.hf_lf_p90(tone, SR) < -40
    # white noise: 9 kHz of band against 0.9 kHz -> +10 dB on average, the 90th
    # percentile of the per-frame ratio a little above it
    assert 9.5 < metrics.hf_lf_p90(hiss, SR) < 13.0


def test_voiced_ratio_over_a_mask():
    f0 = np.array([0, 100, 120, 0, 0, 110], dtype=np.float32)
    mask = np.array([True, True, True, True, False, False])
    assert metrics.voiced_ratio(f0, mask) == pytest.approx(0.5)
