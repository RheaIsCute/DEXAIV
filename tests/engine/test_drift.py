"""Feature / F0 drift harness and the frozen ContentVec norm, on the test model (CPU or
GPU; needs the assets)."""

import dataclasses

import numpy as np
import pytest
import torch
from tests.conftest import MODELS, TEST_MODEL

from vox.config import defaults
from vox.config.schemas import RuntimeParams
from vox.features.contentvec import FrozenGroupNorm
from vox.offline.drift import measure_drift
from vox.offline.simulate import StreamSimulator, load_engine

pytestmark = pytest.mark.assets
SR = 48000


@pytest.fixture(scope="module")
def cache():
    return {}


@pytest.fixture(scope="module")
def engine(device, cache):
    return load_engine(MODELS, TEST_MODEL, defaults.DEFAULT_SHAPE, RuntimeParams(),
                       device=device, cache=cache, warmup_hops=2)


def test_drift_by_age_runs_and_is_ordered(engine, clip_48k):
    clip = clip_48k[: 4 * SR]
    result = measure_drift(engine, clip, ages_ms=(0, 20, 80), start_s=1.2)
    assert result.ages_ms == (0, 20, 80) and result.frames > 50
    for age in result.ages_ms:
        f, p = result.feature[age], result.f0[age]
        assert f["n"] > 50 and 0.0 <= f["cos_mean"] < 1.0 and f["cos_p95"] >= f["cos_mean"]
        assert p["n"] > 0 and np.isfinite(p["cents_rmse"]) and 0.0 <= p["flip_rate"] <= 1.0
        assert 0.0 <= p["voicing_agreement"] <= 1.0
    # the newest frame has no right context: its features drift most against whole-file
    assert result.feature[0]["cos_mean"] > result.feature[80]["cos_mean"]
    assert "age ms" in result.table()


def test_drift_rejects_ages_outside_the_window(engine, clip_48k):
    with pytest.raises(ValueError):
        measure_drift(engine, clip_48k[:SR], ages_ms=(15,))
    with pytest.raises(ValueError):
        measure_drift(engine, clip_48k[:SR], ages_ms=(10 * engine.plan.p_len,))


def test_frozen_norm_matches_window_norm_on_its_own_statistics(engine, clip_48k):
    """Statistics measured on one window reproduce that window's GroupNorm output through
    the frozen affine (the two differ only on other inputs); and the switch is a static-key
    change for the graphs."""
    embedder = engine.pipeline.embedder
    assert embedder.has_group_norm and embedder.norm_mode == "window"
    from vox.audio.resample import resample_array

    wave = torch.from_numpy(resample_array(clip_48k[: 2 * SR], SR, 16000)).to(engine.device)
    stats = embedder.measure_norm_stats(wave)
    assert stats["mean"].shape == (512,) and (stats["var"] >= 0).all()
    layer0 = embedder.model.feature_extractor.conv_layers[0]
    with torch.inference_mode():
        y = layer0.conv(wave.reshape(1, 1, -1))
        window = layer0.layer_norm(y)
        frozen = FrozenGroupNorm(layer0.layer_norm, stats["mean"], stats["var"]).to(
            engine.device)(y)
    assert torch.allclose(window, frozen, atol=1e-3, rtol=1e-3)
    key_window = engine.pipeline.static_key()
    embedder.set_norm("frozen", stats)
    try:
        assert embedder.norm_mode == "frozen"
        assert engine.pipeline.static_key() != key_window
        with torch.inference_mode():
            feats = embedder.embed(wave.unsqueeze(0))
        assert torch.isfinite(feats).all()
    finally:
        embedder.set_norm("window")
    assert engine.pipeline.static_key() == key_window
    with pytest.raises(ValueError):
        embedder.set_norm("frozen")  # no statistics


def test_frozen_norm_streams_clean(device, cache, clip_48k, tmp_path):
    """A frozen-norm engine builds (statistics stored where load_engine looks), streams
    finite audio and leaves the shared embedder back in window mode afterwards."""
    from vox.features.contentvec import ContentVec

    base = load_engine(MODELS, TEST_MODEL, defaults.DEFAULT_SHAPE, RuntimeParams(),
                       device=device, cache=cache, warmup_hops=0)
    from vox.audio.resample import resample_array

    wave = torch.from_numpy(resample_array(clip_48k[: 3 * SR], SR, 16000)).to(device)
    stats = base.pipeline.embedder.measure_norm_stats(wave)
    path = ContentVec.norm_stats_path(MODELS)
    existed = path.is_file()
    backup = path.read_bytes() if existed else None
    try:
        ContentVec.save_norm_stats(MODELS, stats)
        frozen = load_engine(MODELS, TEST_MODEL, defaults.DEFAULT_SHAPE,
                             RuntimeParams(contentvec_norm="frozen"), device=device,
                             cache=cache, warmup_hops=2)
        assert frozen.pipeline.embedder.norm_mode == "frozen"
        result = StreamSimulator(frozen).run(clip_48k[: 2 * SR])
        assert np.isfinite(result.audio).all() and result.pointers_stable
        assert 0.005 < np.sqrt(np.mean(result.audio[SR:] ** 2)) < 0.5
        frozen.graphs.release()
    finally:
        if existed:
            path.write_bytes(backup)
        else:
            path.unlink(missing_ok=True)
        base.pipeline.embedder.set_norm("window")
