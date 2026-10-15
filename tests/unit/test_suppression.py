"""RNNoise suppression on the mic path: state across hops, alignment, the live toggle."""

import math
import tracemalloc

import numpy as np
import pytest

from vox.audio import suppression
from vox.audio.conditioning import ConditioningConfig, InputConditioner

pytestmark = pytest.mark.skipif(suppression.unavailable_reason() is not None,
                                reason=f"RNNoise: {suppression.unavailable_reason()}")
SR, BLOCK = 48000, 960
DELAY, WARMUP = suppression.DELAY, int(SR * suppression.WARMUP_MS / 1000)
FADE = int(SR * suppression.FADE_MS / 1000)


def noisy_voice(seconds: float, seed: int = 0) -> np.ndarray:
    """A harmonic 'vowel' with a 4 Hz envelope over steady fan-like noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    voice = sum(np.sin(2 * math.pi * 150 * k * t) / k for k in range(1, 10))
    voice *= 0.05 * (0.6 + 0.4 * np.sin(2 * math.pi * 4 * t))
    return (voice + 0.01 * rng.standard_normal(len(t))).astype(np.float64)


def run(sup: suppression.Suppressor, x: np.ndarray, block: int = BLOCK) -> np.ndarray:
    return np.concatenate([sup.process(x[i:i + block]).copy()
                           for i in range(0, len(x) - block + 1, block)])


def test_idle_is_the_identity_and_costs_nothing():
    sup = suppression.Suppressor(BLOCK)
    x = noisy_voice(0.1)[:BLOCK]
    assert sup.process(x) is x and not sup.active and sup.delay_samples == 0


def test_state_carries_across_hops_whatever_the_hop():
    """RNNoise runs frame by frame on one state: 20 ms and 10 ms hops give the same audio,
    and a state that skipped the first half gives something else."""
    x = noisy_voice(1.0)
    a = run(suppression.Suppressor(960, enabled=True), x, 960)
    b = run(suppression.Suppressor(480, enabled=True), x, 480)
    np.testing.assert_allclose(a, b, atol=1e-7)
    late = run(suppression.Suppressor(960, enabled=True), x[SR // 2:], 960)
    assert np.abs(late - a[SR // 2:]).max() > 1e-3


def test_partial_mix_keeps_the_dry_part_aligned_with_the_denoised_one():
    """mix 0 is the dry input through the same 20 ms delay: exactly, so a partial mix of
    the two never comb-filters."""
    x = noisy_voice(0.5)
    out = run(suppression.Suppressor(BLOCK, enabled=True, mix=0.0), x)
    np.testing.assert_allclose(out[DELAY:], x[:len(out) - DELAY], atol=1e-12)
    full = run(suppression.Suppressor(BLOCK, enabled=True), x)
    noise_floor = np.sqrt(np.mean((full[SR // 4:] - out[SR // 4:]) ** 2))
    assert noise_floor > 0.003  # mix 1 really removes something (the 0.01 RMS noise)


def test_switching_on_warms_up_unheard_then_crossfades_without_a_click():
    x = noisy_voice(1.0, seed=3)
    sup = suppression.Suppressor(BLOCK)
    head = run(sup, x[:4 * BLOCK])  # idle
    np.testing.assert_array_equal(head, x[:4 * BLOCK])
    sup.set(enabled=True)
    rest = run(sup, x[4 * BLOCK:])
    np.testing.assert_array_equal(rest[:WARMUP], x[4 * BLOCK:4 * BLOCK + WARMUP])  # warm-up
    assert sup.delay_samples == DELAY
    # after the fade the output is the denoised (delayed) signal of a never-idle state's kind
    out = np.concatenate([head, rest])
    steps = np.abs(np.diff(out))
    fade = slice(4 * BLOCK + WARMUP, 4 * BLOCK + WARMUP + FADE + 1)
    assert steps[fade].max() < 2.0 * np.percentile(np.abs(np.diff(x)), 99.9)
    sup.set(enabled=False)
    tail = run(sup, x[:3 * BLOCK])  # fades back to the undelayed dry signal in 20 ms
    np.testing.assert_array_equal(tail[FADE:], x[FADE:3 * BLOCK])
    assert not sup.active


def test_mix_changes_glide_instead_of_stepping():
    x = noisy_voice(0.6, seed=5)
    sup = suppression.Suppressor(BLOCK, enabled=True, mix=1.0)
    first = run(sup, x[:10 * BLOCK])
    sup.set(mix=0.0)
    second = run(sup, x[10 * BLOCK:20 * BLOCK])
    joined = np.concatenate([first, second])
    step = np.abs(np.diff(joined[10 * BLOCK - 2:10 * BLOCK + 2])).max()
    assert step < 2.0 * np.percentile(np.abs(np.diff(x)), 99.9)


def test_process_allocates_nothing_in_steady_state():
    sup = suppression.Suppressor(BLOCK, enabled=True, mix=0.5)
    x = noisy_voice(0.5)
    blocks = [x[i:i + BLOCK] for i in range(0, len(x) - BLOCK + 1, BLOCK)]
    for b in blocks[:3]:
        sup.process(b)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        for b in blocks[3:]:
            sup.process(b)
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()
    assert peak < BLOCK * 4  # less than one float32 block: no per-hop array


def test_conditioner_runs_it_after_the_high_pass_and_reports_its_delay():
    cond = InputConditioner(SR, BLOCK, ConditioningConfig(auto_level=False))
    assert cond.suppressor is not None and cond.delay_ms == 0.0
    cond.set_config(ConditioningConfig(auto_level=False, suppression=True,
                                       suppression_mix=0.5))
    assert cond.suppressor.enabled and cond.suppressor.mix == 0.5
    x = noisy_voice(0.4).astype(np.float32)
    for i in range(0, len(x) - BLOCK + 1, BLOCK):
        cond.process(x[i:i + BLOCK])
    assert cond.delay_ms == pytest.approx(20.0)
    cond.set_config(ConditioningConfig(auto_level=False))
    for i in range(0, 2 * BLOCK, BLOCK):
        cond.process(x[i:i + BLOCK])
    assert cond.delay_ms == 0.0
    assert InputConditioner(32000, 640).suppressor is None  # 48 kHz only
