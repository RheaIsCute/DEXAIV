"""SPSC ring: wraparound, underrun zero-fill, overrun recovery, latency trimming."""

import numpy as np

from vox.audio.rings import SPSCRing


def test_capacity_rounds_to_power_of_two_and_wraps():
    ring = SPSCRing(1000)
    assert ring.capacity == 1024
    data = np.arange(1500, dtype=np.float32)
    for i in range(0, 1500, 300):
        ring.write(data[i:i + 300])
        out = np.zeros(300, dtype=np.float32)
        assert ring.read(out) == 300
        assert np.array_equal(out, data[i:i + 300])
    assert ring.available() == 0 and ring.underruns == 0 and ring.overruns == 0


def test_underrun_fills_zeros_and_counts():
    ring = SPSCRing(64)
    ring.write(np.ones(10, dtype=np.float32))
    out = np.full(16, 7.0, dtype=np.float32)
    assert ring.read(out) == 10
    assert np.all(out[:10] == 1) and np.all(out[10:] == 0)
    assert ring.underruns == 1 and ring.underrun_samples == 6


def test_overrun_is_detected_by_the_reader_and_skipped():
    ring = SPSCRing(64, guard=8)
    for k in range(5):  # 5 x 20 = 100 samples into a 64-sample ring, nobody reading
        ring.write(np.full(20, float(k), dtype=np.float32))
    out = np.zeros(20, dtype=np.float32)
    ring.read(out)
    assert ring.overruns == 1 and ring.overrun_samples > 0
    # whatever we got is intact (no half-overwritten region): values are from recent writes
    assert set(np.unique(out)) <= {2.0, 3.0, 4.0}


def test_keep_latest_and_fill():
    ring = SPSCRing(256)
    ring.write(np.arange(100, dtype=np.float32))
    assert ring.fill() == 100
    assert ring.keep_latest(30) == 70
    out = np.zeros(30, dtype=np.float32)
    ring.read(out)
    assert out[0] == 70 and out[-1] == 99


def test_request_drop_is_applied_by_the_reader():
    """The writer may not move the read position; it asks, and the next read discards."""
    ring = SPSCRing(64)
    ring.write(np.arange(40, dtype=np.float32))
    ring.request_drop(25)
    assert ring.drop_pending and ring.available() == 40  # nothing moved yet
    out = np.zeros(5, dtype=np.float32)
    assert ring.read(out) == 5
    assert out.tolist() == [25.0, 26.0, 27.0, 28.0, 29.0]
    assert not ring.drop_pending and ring.available() == 10 and ring.underruns == 0
    ring.request_drop(1000)  # more than is queued: clamps, then the read underruns
    assert ring.read(out) == 0 and ring.underruns == 1
    ring.request_drop(-3)
    assert not ring.drop_pending
