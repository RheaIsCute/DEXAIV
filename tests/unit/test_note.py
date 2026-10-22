"""Held-note detection and what it switches (docs/DEVIATIONS.md §14, the Natural preset),
on synthetic F0 tracks; no models."""

import math

import numpy as np
import torch

from vox.pitch.base import NoteDetector, RegisterTracker, postprocess_f0, soft_ceiling

SHIFT = 2  # fresh frames per hop (a 20 ms hop)
TAIL = 32


def detect(f0_hz: np.ndarray, fresh: int = SHIFT, **settings):
    """Feed a 100 Hz F0 track hop by hop; return per-hop (note, frozen) flags."""
    det = NoteDetector("cpu")
    det.fresh = fresh
    det.set(True, settings.get("min_ms", 400.0), settings.get("tolerance_st", 1.0),
            settings.get("freeze_ms", 100.0))
    f0 = torch.from_numpy(f0_hz.astype(np.float32))
    flags = []
    for n in range(0, len(f0) - TAIL + 1, fresh):
        note, frozen = det.update(f0[n:n + TAIL])
        flags.append((bool(note), bool(frozen)))
    return flags, det


def test_a_steady_tone_becomes_a_note_after_min_ms_and_freezes_earlier():
    f0 = np.full(200, 220.0)  # 2 s
    flags, det = detect(f0)
    notes = [n for n, _ in flags]
    frozen = [f for _, f in flags]
    # hop k has fed (k + 1) * SHIFT fresh frames to the detector
    assert (notes.index(True) + 1) * SHIFT == 40  # 400 ms of the run
    assert (frozen.index(True) + 1) * SHIFT == 10  # 100 ms
    assert all(notes[notes.index(True):])
    assert det.run.item() == len(flags) * SHIFT


def test_vibrato_inside_the_tolerance_is_one_note_and_a_jump_is_not():
    t = np.arange(150) / 100.0
    vibrato = 220.0 * 2 ** (0.4 / 12 * np.sin(2 * math.pi * 6 * t))  # +-0.4 st at 6 Hz
    flags, _ = detect(vibrato)
    assert any(n for n, _ in flags)
    jumpy = np.concatenate([np.full(30, 220.0), np.full(30, 220.0 * 2 ** (2 / 12))] * 3)
    flags, _ = detect(jumpy)  # 300 ms steps two semitones apart: never a 400 ms run
    assert not any(n for n, _ in flags)


def test_unvoiced_frames_end_the_run_and_disabled_never_reports():
    # the detector sees only the fresh frames: hop k feeds frames 30 + 2k and 31 + 2k
    f0 = np.concatenate([np.full(80, 220.0), np.zeros(3), np.full(80, 220.0)])
    flags, det = detect(f0)
    notes = [n for n, _ in flags]
    assert notes.index(True) == 19  # frames 30..69: 40 frames of the first run
    assert notes[24] and not notes[25]  # hop 25 feeds the gap (frames 80, 81): run over
    assert not any(notes[25:46]) and notes[46]  # the second run earns its own 400 ms
    assert det.run.item() == 1 + 2 * (len(flags) - 1 - 26)  # hop 26 fed its first frame
    det2 = NoteDetector("cpu")
    det2.fresh = SHIFT
    det2.set(False, 400.0, 1.0, 100.0)
    f = torch.from_numpy(np.full(TAIL, 220.0, dtype=np.float32))
    for _ in range(40):
        note, frozen = det2.update(f)
    assert not note and not frozen and det2.run.item() == 80.0  # the run still counts


def test_a_frozen_register_stops_following_and_the_range_override_applies():
    tracker = RegisterTracker("cpu")
    tracker.fresh = SHIFT
    tracker.set(True, 200.0, 2.0, 1.0)
    fixed = torch.tensor(1.0)
    f0 = torch.full((TAIL,), 100.0)
    for _ in range(60):  # warm up on 100 Hz
        tracker.ratio(f0, fixed)
    level = tracker.level.item()
    high = torch.full((TAIL,), 300.0)
    tracker.ratio(high, fixed, freeze=torch.tensor(True))
    assert tracker.level.item() == level  # held still
    tracker.ratio(high, fixed, freeze=torch.tensor(False))
    assert tracker.level.item() > level  # follows again
    # range override: 0.5 halves the distance from the register, 1.0 keeps it
    st_full = 12 * torch.log2(tracker.ratio(high, fixed, torch.tensor(1.0),
                                            torch.tensor(True))[-1] * 300.0 / 200.0)
    st_half = 12 * torch.log2(tracker.ratio(high, fixed, torch.tensor(0.5),
                                            torch.tensor(True))[-1] * 300.0 / 200.0)
    above = 12 * (math.log2(300.0) - tracker.level.item())
    assert st_full.item() == pytest_approx(above, 0.05)
    assert st_half.item() == pytest_approx(0.5 * above, 0.05)


def test_note_ceiling_switch_is_a_where_on_the_same_ops():
    """What the pipeline does: ``ceiling = where(note, note_ceiling, ceiling)`` feeds
    ``postprocess_f0``; a 420 Hz transposed frame is bent under a 330 Hz ceiling and left
    nearer its pitch under 360."""
    f0 = torch.tensor([210.0, 0.0, 210.0])
    ratio = torch.tensor(2.0)
    for note, ceiling_hz in ((False, 330.0), (True, 360.0)):
        ceiling = torch.where(torch.tensor(note), torch.tensor(360.0), torch.tensor(330.0))
        assert ceiling.item() == ceiling_hz
        _, out = postprocess_f0(f0, ratio, ceiling=ceiling)
        assert out[1].item() == 0.0
        assert out[0].item() == soft_ceiling(torch.tensor(420.0), ceiling_hz).item()
        assert out[0].item() < 420.0
    assert soft_ceiling(torch.tensor(420.0), 360.0) > soft_ceiling(torch.tensor(420.0), 330.0)


def pytest_approx(value, abs_tol):
    import pytest

    return pytest.approx(value, abs=abs_tol)
