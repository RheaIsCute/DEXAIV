"""Laugh guard and ceiling lift (pitch/base.py ``LaughGuard``, docs/DEVIATIONS.md §15)."""

import math

import torch

from vox.pitch.base import LaughGuard, RegisterTracker

FRESH = 8  # hop 80 ms


def tracker():
    r = RegisterTracker("cpu")
    r.fresh = FRESH
    r.set(True, 200.0, 2.0, 1.0)
    return r


def guard(on=True):
    g = LaughGuard("cpu")
    g.fresh = FRESH
    g.set(on, 9.0, 8.0, 600.0)
    return g


def hop(hz, voiced=FRESH):
    f0 = torch.zeros(32)
    f0[-FRESH:][:voiced] = hz
    return f0


def feed(r, g, f0):
    w = g.update(f0, r.level, r.heard) if g is not None else None
    r.ratio(f0, torch.tensor(1.0), weight=w)
    return w


def test_off_is_an_exact_no_op():
    r, g = tracker(), guard(on=False)
    for _ in range(20):
        w = feed(r, g, hop(110.0))
    assert float(w) == 1.0 and float(g.lift) == 0.0
    c = torch.tensor(330.0)
    assert float(g.ceiling(c)) == 330.0
    assert math.isinf(float(g.ceiling(torch.tensor(math.inf))))


def test_a_laugh_an_octave_up_barely_moves_the_register():
    """1 s of voiced calls at 272 Hz on a 110 Hz register (report 19 §3.3): unguarded the
    register climbs ~0.5 oct; guarded it stays within a semitone."""
    moves = {}
    for on in (False, True):
        r, g = tracker(), guard(on)
        for _ in range(30):
            feed(r, g, hop(110.0))
        start = float(r.level)
        for _ in range(12):
            feed(r, g, hop(272.0))
        moves[on] = 12.0 * (float(r.level) - start)
    assert moves[False] > 4.0
    assert abs(moves[True]) < 1.0


def test_flicker_lifts_the_ceiling_and_speech_lets_it_down():
    r, g = tracker(), guard()
    for _ in range(30):
        feed(r, g, hop(110.0))
    assert float(g.lift) < 0.05
    # calls: voiced 4 frames, unvoiced 4 frames, every hop -> ~25 flips/s at hop 80
    for _ in range(15):
        f0 = hop(150.0, voiced=4)
        w = feed(r, g, f0)
    assert float(g.rate) > 15.0 and float(w) < 0.05
    assert float(g.lift) > 0.8
    lifted = float(g.ceiling(torch.tensor(330.0)))
    assert 500.0 < lifted <= 600.0
    for _ in range(30):
        feed(r, g, hop(110.0))
    assert float(g.lift) < 0.05 and float(g.rate) < 3.0
