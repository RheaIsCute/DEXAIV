"""F0 post-processing and the rolling pitch cache, against the canonical numpy formulas."""

import numpy as np
import pytest
import torch

from vox.pitch import make_extractor
from vox.pitch.base import (
    clamp_voicing,
    f0_to_coarse,
    interp_unvoiced,
    postprocess_f0,
    semitones_to_ratio,
)
from vox.pitch.pitch_cache import PitchCache
from vox.pitch.rmvpe import RmvpeExtractor


def coarse_reference(f0: np.ndarray) -> np.ndarray:
    """The five lines every RVC codebase carries (rvc-webui infer/vc/pipeline.py)."""
    f0_mel_min = 1127 * np.log(1 + 50 / 700)
    f0_mel_max = 1127 * np.log(1 + 1100 / 700)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    return np.rint(f0_mel).astype(np.int64)


def test_f0_to_coarse_matches_the_trained_in_contract():
    f0 = np.concatenate(([0.0, 10.0, 49.9, 50.0, 1100.0, 1500.0],
                         np.random.default_rng(0).uniform(40, 1200, 2000)))
    ours = f0_to_coarse(torch.from_numpy(f0)).numpy()
    np.testing.assert_array_equal(ours, coarse_reference(f0.copy()))
    assert ours[0] == 1 and ours.min() == 1 and ours.max() == 255


def test_transposition_is_in_semitones():
    assert semitones_to_ratio(12) == pytest.approx(2.0)
    f0 = torch.tensor([0.0, 100.0, 200.0])
    coarse, cont = postprocess_f0(f0, semitones_to_ratio(12))
    np.testing.assert_allclose(cont.numpy(), [0.0, 200.0, 400.0])
    assert coarse[0] == 1 and coarse[1] < coarse[2]  # unvoiced stays bucket 1


def interp_reference(f0: np.ndarray) -> np.ndarray:
    f0 = f0.copy()
    uv = f0 == 0
    if (~uv).any():
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return f0


@pytest.mark.parametrize("seed", range(6))
def test_interp_unvoiced_matches_np_interp(seed):
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(80, 300, 64)
    f0[rng.random(64) < 0.5] = 0
    if seed == 0:
        f0[:5] = 0  # leading gap: hold the first voiced value
    if seed == 1:
        f0[-7:] = 0  # trailing gap: hold the last voiced value
    ours = interp_unvoiced(torch.from_numpy(f0)).numpy()
    np.testing.assert_allclose(ours, interp_reference(f0), rtol=1e-9)


def test_interp_unvoiced_edge_cases():
    assert interp_unvoiced(torch.zeros(10)).abs().sum() == 0  # nothing voiced: unchanged
    voiced = torch.tensor([100.0, 110.0, 120.0])
    assert torch.equal(interp_unvoiced(voiced), voiced)
    np.testing.assert_allclose(interp_unvoiced(torch.tensor([0.0, 0.0, 90.0, 0.0])).numpy(),
                               [90.0, 90.0, 90.0, 90.0])


def test_clamp_voicing_removes_out_of_range_estimates():
    f0 = torch.tensor([0.0, 55.0, 70.0, 150.0, 300.0, 520.0])
    np.testing.assert_array_equal(clamp_voicing(f0, 60.0, 300.0).numpy(),
                                  [0.0, 0.0, 70.0, 150.0, 300.0, 0.0])
    # the calibrated clamp is applied before transposition
    _, cont = postprocess_f0(f0, 2.0, f0_min=60.0, f0_max=300.0)
    np.testing.assert_array_equal(cont.numpy(), [0, 0, 140, 300, 600, 0])


def test_rmvpe_tail_rounding():
    extractor = RmvpeExtractor()
    for block_16k in (160, 320, 480, 800, 1600, 4320):
        assert extractor.tail_samples(block_16k) == 4960  # 310 ms: one 32-frame tile
    assert extractor.tail_samples(4321) == 2 * 5120 - 160
    assert extractor.tail_samples(4320) // 160 + 1 == 32


def test_fcpe_floor_and_factory():
    fcpe = make_extractor("fcpe")
    assert fcpe.tail_samples(320) == 4800 and fcpe.tail_samples(8000) == 8800
    assert make_extractor("rmvpe").name == "rmvpe"
    with pytest.raises(ValueError):
        make_extractor("harvest")


def test_pitch_cache_shift_and_interior_splice():
    """The rtrvc pattern: shift by one hop, write fresh[3:-1] to the tail."""
    convert_frames, shift = 20, 2
    cache = PitchCache(convert_frames, shift, p_len=20, device="cpu")
    reference_f = np.zeros(convert_frames + 1, dtype=np.float32)
    reference_c = np.ones(convert_frames + 1, dtype=np.int64)
    rng = np.random.default_rng(0)
    for _ in range(15):
        fresh_f = rng.uniform(80, 300, 8).astype(np.float32)
        fresh_c = rng.integers(1, 255, 8)
        cache.roll_and_write(torch.from_numpy(fresh_c), torch.from_numpy(fresh_f))
        reference_f[:-shift] = reference_f[shift:].copy()
        reference_c[:-shift] = reference_c[shift:].copy()
        reference_f[4 - len(fresh_f):] = fresh_f[3:-1]
        reference_c[4 - len(fresh_c):] = fresh_c[3:-1]
        np.testing.assert_array_equal(cache.pitchf.numpy(), reference_f)
        np.testing.assert_array_equal(cache.pitch.numpy(), reference_c)
    assert cache.pitch_view.shape == (1, 20) and cache.pitchf_view.shape == (1, 20)
    np.testing.assert_array_equal(cache.pitchf_view[0].numpy(), reference_f[-20:])


def test_pitch_cache_pointers_are_stable_and_zero_forgets():
    cache = PitchCache(30, 3, p_len=30, device="cpu")
    pointers = (cache.pitch.data_ptr(), cache.pitchf.data_ptr(), cache.pitchf_view.data_ptr())
    for _ in range(4):
        cache.roll_and_write(torch.full((10,), 120), torch.full((10,), 200.0))
    assert pointers == (cache.pitch.data_ptr(), cache.pitchf.data_ptr(),
                        cache.pitchf_view.data_ptr())
    assert cache.pitchf.max() == 200.0
    cache.zero()
    assert cache.pitchf.abs().sum() == 0 and torch.all(cache.pitch == 1)  # 1 = unvoiced


def test_pitch_cache_tail_longer_than_cache_keeps_the_newest_frames():
    cache = PitchCache(convert_frames=10, shift_frames=1, p_len=10, device="cpu")
    fresh = torch.arange(32, dtype=torch.float32)
    cache.roll_and_write(fresh.long(), fresh)
    np.testing.assert_array_equal(cache.pitchf.numpy(), np.arange(20, 31))  # ends at fresh[-2]
