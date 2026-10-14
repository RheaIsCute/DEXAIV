"""The clip override mixer: a typed clip replaces the live hop, with fades at both ends."""

import numpy as np

from vox.engine.override import ClipOverride

BLOCK = 8


def live() -> np.ndarray:
    return np.ones(BLOCK, dtype=np.float32)


def run(override: ClipOverride, hops: int) -> list[np.ndarray]:
    out = []
    for _ in range(hops):
        block = live()
        override.mix(block)
        out.append(block)
    return out


def test_idle_leaves_the_live_hop_alone():
    override = ClipOverride(BLOCK)
    block = live()
    assert override.mix(block) is False and not override.active
    assert np.array_equal(block, live())


def test_clip_takes_over_then_live_fades_back():
    override = ClipOverride(BLOCK)
    clip = np.full(2 * BLOCK + 3, 0.5, dtype=np.float32)
    override.submit(clip)
    assert override.active
    first, second, third, after, later = run(override, 5)

    ramp = np.linspace(0.0, 1.0, BLOCK, endpoint=False, dtype=np.float32)
    assert np.allclose(first, (1.0 - ramp) + 0.5)  # live fades out under the clip's first hop
    assert np.allclose(second, 0.5)  # then the clip alone
    assert np.allclose(third[:3], 0.5) and np.allclose(third[3:], 0.0)  # clip ends mid-hop
    assert np.allclose(after, ramp)  # live fades back in over the next hop
    assert np.array_equal(later, live())
    assert override.played == 1 and not override.active


def test_mix_reports_ownership_per_hop():
    override = ClipOverride(BLOCK)
    override.submit(np.zeros(BLOCK + 1, dtype=np.float32))
    assert [override.mix(live()) for _ in range(4)] == [True, True, False, False]


def test_a_second_clip_replaces_the_first_without_unmuting():
    override = ClipOverride(BLOCK)
    override.submit(np.full(4 * BLOCK, 0.25, dtype=np.float32))
    run(override, 2)
    override.submit(np.full(BLOCK, 0.75, dtype=np.float32))
    replaced, after = run(override, 2)
    assert np.allclose(replaced, 0.75)  # no live audio leaks in at the swap
    assert after[0] == 0.0 and override.played == 2


def test_clip_right_after_another_keeps_live_muted():
    override = ClipOverride(BLOCK)
    override.submit(np.full(BLOCK, 0.5, dtype=np.float32))
    run(override, 1)  # clip spent; a fade-in is due on the next hop
    override.submit(np.full(BLOCK, 0.25, dtype=np.float32))
    (block,) = run(override, 1)
    assert np.allclose(block, 0.25)


def test_cancel_stops_the_clip_and_fades_live_in():
    override = ClipOverride(BLOCK)
    override.submit(np.full(10 * BLOCK, 0.5, dtype=np.float32))
    run(override, 2)
    override.cancel()
    block = live()
    assert override.mix(block) is False and not override.active
    assert block[0] == 0.0 and 0.0 < block[-1] < 1.0
    assert np.array_equal(run(override, 1)[0], live())


def test_cancel_drops_a_clip_that_has_not_started():
    override = ClipOverride(BLOCK)
    override.submit(np.ones(BLOCK, dtype=np.float32))
    override.cancel()
    assert np.array_equal(run(override, 1)[0], live()) and override.played == 0


def test_resize_keeps_a_clip_in_flight():
    override = ClipOverride(BLOCK)
    override.submit(np.full(3 * BLOCK, 0.5, dtype=np.float32))
    run(override, 1)
    override.resize(2 * BLOCK)
    block = np.ones(2 * BLOCK, dtype=np.float32)
    assert override.mix(block) is True and np.allclose(block, 0.5)
    block = np.ones(2 * BLOCK, dtype=np.float32)
    override.mix(block)
    assert block[0] == 0.0 and block.shape == (2 * BLOCK,)  # fade-in at the new hop size
