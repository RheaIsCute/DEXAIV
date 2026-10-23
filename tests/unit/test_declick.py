"""In-speech de-clicker (audio/conditioning.py ``Declicker``, docs/DEVIATIONS.md §15)."""

import dataclasses

import numpy as np

from vox.audio.conditioning import ConditioningConfig, Declicker, InputConditioner

SR = 48000


def vowel(n, amp=0.05, f=150.0):
    t = np.arange(n) / SR
    return amp * (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 3 * f * t))


def click(rng, n=240, amp=0.6):
    """5 ms of broadband noise, decaying fast: a keystroke's push impulse."""
    return amp * rng.standard_normal(n) * np.exp(-np.arange(n) / 48.0)


def run(x, block, ratio=4.0):
    d = Declicker(SR, ratio)
    out, retro_hits = [], 0
    for i in range(0, len(x), block):
        gain, lookback = d.process(x[i:i + block])
        y = x[i:i + block] * gain
        if lookback is not None:  # what the engine does: rewrite the samples already passed on
            retro_hits += 1
            done = np.concatenate(out) if out else np.zeros(0)
            n = min(len(lookback), len(done))
            done[len(done) - n:] *= lookback[len(lookback) - n:]
            out = [done]
        out.append(y)
    return np.concatenate(out), d, retro_hits


def test_a_click_inside_a_vowel_is_clamped_and_the_vowel_is_kept():
    rng = np.random.default_rng(0)
    x = vowel(SR)
    at = int(0.5 * SR)
    x[at:at + 240] += click(rng, amp=1.2)
    y, d, _ = run(x, 3840)
    assert len(d.clicks) == 1 and abs(d.clicks[0][0] - 0.5) < 0.002
    win = slice(at, at + 240)
    before = 10 * np.log10(np.mean(x[win] ** 2))
    after = 10 * np.log10(np.mean(y[win] ** 2))
    assert before - after > 10.0  # the clack is tamed...
    assert after - 10 * np.log10(np.mean(vowel(SR) ** 2)) < 9.0  # ...to ~the vowel + 6-8 dB
    keep = np.r_[0:at - 200, at + 1000:SR]
    assert np.allclose(y[keep], x[keep])  # the rest is untouched


def test_a_plosive_burst_followed_by_voicing_is_left_alone():
    """/p/: closure (silence), burst, then aspiration and the vowel within 20 ms."""
    rng = np.random.default_rng(1)
    x = np.concatenate([vowel(SR // 2), np.zeros(int(0.06 * SR)), click(rng, 240, 0.3),
                        0.02 * rng.standard_normal(int(0.015 * SR)), vowel(SR // 2)])
    y, d, _ = run(x, 3840)
    assert d.clicks == []
    assert np.array_equal(y, x)


def test_a_click_whose_decision_needs_the_next_block_is_clamped_retroactively():
    """At hop 20 every decision lands in a later block: the look-back carries the gain,
    and the result equals processing the whole signal as one block."""
    rng = np.random.default_rng(2)
    x = vowel(SR)
    at = int(0.5 * SR) - 300  # 6 ms before a 20 ms block boundary
    x[at:at + 240] += click(rng)
    y20, d20, hits = run(x, 960)
    y_all, d_all, _ = run(x, len(x))
    assert hits >= 1 and len(d20.clicks) == 1
    assert np.allclose(y20, y_all, atol=1e-9)


def test_off_by_default_and_the_conditioner_reports_no_lookback():
    assert ConditioningConfig().declick is False
    c = InputConditioner(SR, 960, ConditioningConfig(auto_level=False))
    assert c.declicker is None
    c.process(np.zeros(960, np.float32))
    assert c.retro is None


def test_the_conditioner_splits_the_lookback_across_the_rnnoise_delay():
    """Through a chain delay d the look-back's last d samples land in this block's output
    and the rest moves d samples later in ``retro``."""
    cfg = ConditioningConfig(auto_level=False, declick=True, highpass_hz=0.0)
    c = InputConditioner(SR, 960, cfg)
    c._out[:] = 1.0
    look = np.linspace(0.1, 1.0, 1440)
    c._place_lookback(look.copy(), 960)
    assert np.allclose(c._out[:960], look[480:])
    assert np.allclose(c.retro[:960], 1.0) and np.allclose(c.retro[960:], look[:480])
    c2 = InputConditioner(SR, 960, dataclasses.replace(cfg))
    c2._place_lookback(look.copy(), 0)
    assert np.allclose(c2.retro, look)
