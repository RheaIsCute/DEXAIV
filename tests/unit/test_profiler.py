import numpy as np
import pytest
import torch

from vox.profiling.report import environment_fingerprint, format_report, stage_table
from vox.profiling.stats import StatsRing, headroom
from vox.profiling.timers import AllocProbe, StageTimers


def test_stats_ring_percentiles_and_wraparound():
    ring = StatsRing(1, capacity=8)
    for v in range(1, 5):
        ring.push(0, v)
    assert ring.summary(0).p50 == pytest.approx(2.5)
    for v in range(5, 21):  # wraps: only the last 8 samples (13..20) remain
        ring.push(0, v)
    np.testing.assert_array_equal(ring.samples(0), np.arange(13, 21, dtype=np.float32))
    assert ring.summary(0).max == 20
    # skip counts from the very first push, so samples already overwritten cost nothing
    np.testing.assert_array_equal(ring.samples(0, skip=14), np.arange(15, 21))
    assert ring.summary(0, skip=100).count == 0


def test_headroom():
    assert headroom(20.0, 10.0) == 2.0
    assert np.isnan(headroom(20.0, float("nan")))


def _matmul_loop(device: torch.device, hops: int = 40) -> StageTimers:
    timers = StageTimers(("small", "large"), device, capacity=64)
    a = torch.randn(64, 64, device=device)
    b = torch.randn(1024, 1024, device=device)
    for _ in range(hops):
        timers.begin_hop()
        with timers.stage("small"):
            a @ a
        with timers.stage("large"):
            for _ in range(4):
                b @ b
        if device.type == "cuda":
            torch.cuda.synchronize()  # stands in for the hop's one download
        timers.end_hop()
    return timers


def test_host_timers_on_dummy_matmul_loop():
    timers = _matmul_loop(torch.device("cpu"))
    table = stage_table(timers, skip=5)
    assert set(table) == {"small", "large", "total"}
    for row in table.values():
        host = row["host"]
        assert host["count"] == 35
        assert 0 < host["p50"] <= host["p95"] <= host["p99"] <= host["max"]
    assert table["large"]["host"]["p50"] > table["small"]["host"]["p50"]
    assert table["total"]["host"]["p50"] >= table["large"]["host"]["p50"]
    assert timers.counters.hops == 40


@pytest.mark.gpu
def test_cuda_events_read_one_hop_late():
    timers = _matmul_loop(torch.device("cuda"))
    large = timers.index("large")
    # Events are harvested one hop late: before flush the GPU ring trails the host ring.
    assert timers.gpu.total(large) == timers.host.total(large) - 1
    timers.flush()
    assert timers.gpu.total(large) == timers.host.total(large)
    table = stage_table(timers, skip=5)
    assert table["large"]["gpu"]["p50"] > table["small"]["gpu"]["p50"] > 0
    report = format_report(timers, hop_ms=20.0, skip=5)
    assert "gpu p50" in report and "headroom" in report and "large" in report


def test_disabled_timers_record_nothing():
    timers = StageTimers(("a",), "cpu", enabled=False)
    timers.begin_hop()
    with timers.stage("a"):
        pass
    timers.end_hop()
    assert timers.host.total(0) == 0


def test_counters_and_report_text():
    timers = _matmul_loop(torch.device("cpu"), hops=12)
    timers.counters.output_underruns += 2
    text = format_report(timers, hop_ms=20.0, title="unit", notes={"overlap": "30 ms"})
    assert "output_underruns: 2" in text and "overlap=30 ms" in text


def test_environment_fingerprint_has_versions():
    info = environment_fingerprint()
    assert info["torch"] == torch.__version__ and "python" in info
    if torch.cuda.is_available():
        assert info["gpu"] and info["cuda"] and info["driver"]


@pytest.mark.gpu
def test_alloc_probe_separates_temporaries_from_growth():
    device = torch.device("cuda")
    probe = AllocProbe(device, capacity=32)
    keep = torch.zeros(1 << 16, device=device)
    for _ in range(20):
        probe.begin_hop()
        keep.add_(1)  # in place: no allocation
        (keep * 2).sum()  # temporaries: served from the pool, no growth
        probe.end_hop()
    summary = probe.summary(skip=4)
    assert summary["pool_allocs_per_hop"] >= 1
    assert summary["driver_mallocs"] == 0 and summary["live_drift_bytes"] == 0
