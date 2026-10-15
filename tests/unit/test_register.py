"""Auto pitch: the register tracker (docs/DEVIATIONS.md §9), on synthetic F0 tracks."""

import math

import numpy as np
import pytest
import torch

from vox.pitch.base import REGISTER_WARM_FRAMES, RegisterTracker

SHIFT = 2  # fresh frames per hop (a 20 ms hop)
TAIL = 32  # frames the extractor hands over per hop (RMVPE-like)
OFF = TAIL - SHIFT  # output k is input frame k + OFF


def track(f0_hz: np.ndarray, fixed_st: float = 0.0, **settings):
    """Feed a 100 Hz F0 track hop by hop and return each hop's fresh-frame ratios (st)."""
    tracker = RegisterTracker("cpu")
    tracker.fresh = SHIFT
    tracker.set(True, settings.get("target_hz", 200.0), settings.get("follow_s", 2.0),
                settings.get("pitch_range", 1.0))
    fixed = torch.tensor(2.0 ** (fixed_st / 12.0))
    f0 = torch.from_numpy(f0_hz.astype(np.float32))
    out = []
    for n in range(0, len(f0) - TAIL + 1, SHIFT):
        out.append(12 * torch.log2(tracker.ratio(f0[n:n + TAIL], fixed)[-SHIFT:]))
    return torch.cat(out).numpy(), tracker


def test_fixed_until_warm_then_lands_the_register_on_the_target():
    f0 = np.full(1000, 100.0)  # 10 s at 100 Hz
    st, tracker = track(f0, fixed_st=3.0)
    assert np.all(st[:REGISTER_WARM_FRAMES - SHIFT] == pytest.approx(3.0))  # the manual pitch
    assert np.all(st[REGISTER_WARM_FRAMES:] == pytest.approx(12.0, abs=0.01))  # 100 -> 200 Hz
    assert tracker.offset_st == pytest.approx(12.0, abs=0.01)


def test_register_follows_a_change_at_the_time_constant_and_ignores_pauses():
    f0 = np.concatenate([np.full(500, 100.0), np.zeros(300), np.full(600, 150.0)])
    st, _ = track(f0, follow_s=2.0)
    settled = st[400 - OFF]
    assert settled == pytest.approx(12.0, abs=0.05)
    pause = st[500 - OFF:800 - OFF]  # unvoiced frames carry the ratio; the level stays
    assert pause.min() == pytest.approx(settled, abs=1e-3)
    assert pause.max() == pytest.approx(settled, abs=1e-3)
    # 150 -> 200 Hz wants +4.98 st (7 st less); after one time constant (200 voiced
    # frames) ~63% of the way, after the 600 frames (3 tau) ~95%
    want = 12 * math.log2(200 / 150)
    one_tau = st[1000 - OFF]
    assert 12.0 - 0.75 * (12 - want) < one_tau < 12.0 - 0.5 * (12 - want)
    assert st[-1] == pytest.approx(want + (12 - want) * math.exp(-3), abs=0.15)


def test_intonation_is_kept_at_range_one_and_squeezed_below():
    t = np.arange(800) / 100.0
    f0 = 100.0 * 2 ** (0.25 * np.sin(2 * math.pi * 0.5 * t))  # +-3 st around 100 Hz
    st1, _ = track(f0)
    st_half, _ = track(f0, pitch_range=0.5)
    out1 = 12 * np.log2(f0[OFF:OFF + len(st1)]) + st1
    out_half = 12 * np.log2(f0[OFF:OFF + len(st_half)]) + st_half
    assert np.ptp(out1[-200:]) == pytest.approx(6.0, abs=0.3)  # intonation untouched
    assert np.ptp(out_half[-200:]) == pytest.approx(3.0, abs=0.3)  # halved
    assert np.mean(out_half[-200:]) == pytest.approx(12 * math.log2(200.0), abs=0.2)


def test_fixed_mode_is_an_exact_no_op_with_the_same_ops():
    tracker = RegisterTracker("cpu")
    tracker.fresh = SHIFT
    tracker.set(False, 200.0, 2.0, 1.0)
    f0 = torch.tensor([0.0, 100.0, 110.0, 0.0, 120.0])
    fixed = torch.tensor(1.5)
    assert torch.equal(tracker.ratio(f0, fixed), torch.full((5,), 1.5))
    assert math.isnan(tracker.offset_st)
    tracker.reset()
    assert float(tracker.heard) == 0.0
