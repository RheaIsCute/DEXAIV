"""Formant shift on real weights (docs/DEVIATIONS.md §12): the streamed and the full-file
renders keep the pitch and move the resonances by the factor, the seams stay as they are,
and the per-step synth-graph cache swaps steps without a capture and replays like eager."""

import dataclasses
import math

import numpy as np
import pytest
import torch
from scipy.linalg import solve_toeplitz
from tests.conftest import MODELS, TEST_MODEL

from vox.audio.resample import resample_array
from vox.config import defaults
from vox.config.schemas import RuntimeParams
from vox.engine.graphs import FORMANT_GRAPHS_KEPT
from vox.offline.convert import FullFileConverter
from vox.offline.metrics import boundary_click_ratio
from vox.offline.simulate import StreamSimulator, load_engine

pytestmark = [pytest.mark.assets, pytest.mark.gpu]
SR = 48000


@pytest.fixture(scope="module")
def cache():
    return {}


@pytest.fixture(scope="module")
def engine(device, cache):
    eng = load_engine(MODELS, TEST_MODEL, defaults.DEFAULT_SHAPE, RuntimeParams(),
                      device=device, cache=cache, graphs="on")
    yield eng
    eng.graphs.release()


def f0_median(engine, audio: np.ndarray, sr: int) -> float:
    y = resample_array(audio, sr, 16000)
    with torch.inference_mode():
        f0 = engine.pipeline.extractor.extract(torch.from_numpy(y).to(engine.device))
    f0 = f0.cpu().numpy()
    return float(np.median(f0[f0 > 0]))


def formants(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """Median F1 / F2 (LPC order 12 at 10 kHz) over the loudest 30% of 25 ms frames."""
    y = resample_array(audio, sr, 10000).astype(np.float64)
    win, hop, order = 250, 100, 12
    starts = range(0, len(y) - win, hop)
    energy = np.array([np.sum(y[i:i + win] ** 2) for i in starts])
    f1, f2 = [], []
    for e, i in zip(energy, starts):
        if e < np.percentile(energy, 70):
            continue
        seg = y[i:i + win] * np.hamming(win)
        seg = np.append(seg[0], seg[1:] - 0.63 * seg[:-1])
        r = np.correlate(seg, seg, "full")[win - 1:win + order + 1]
        a = solve_toeplitz(r[:order], -r[1:order + 1])
        roots = np.roots(np.concatenate(([1.0], a)))
        roots = roots[np.imag(roots) > 0]
        freq = np.angle(roots) * 10000 / (2 * np.pi)
        bw = -10000 / np.pi * np.log(np.abs(roots)) / 2
        f = np.sort(freq[(freq > 200) & (freq < 4500) & (bw < 400)])
        if len(f) >= 2:
            f1.append(f[0])
            f2.append(f[1])
    return float(np.median(f1)), float(np.median(f2))


@pytest.fixture(scope="module")
def streams(engine, clip_48k):
    """The test clip streamed at formant -2, 0, +2 (same seed), each aligned to the input."""
    out = {}
    sim = StreamSimulator(engine)
    for st in (-2.0, 0.0, 2.0):
        engine.pipeline.set_params(dataclasses.replace(engine.params, formant_st=st))
        engine.graphs.sync_formant()
        result = sim.run(clip_48k, seed=0)
        out[st] = result
    engine.pipeline.set_params(engine.params)
    engine.graphs.sync_formant()
    return out


@pytest.mark.parametrize("st", [-2.0, 2.0])
def test_streamed_formant_keeps_the_pitch(engine, streams, clip_48k, st):
    n = len(clip_48k)
    base = f0_median(engine, streams[0.0].aligned[:n], SR)
    shifted = f0_median(engine, streams[st].aligned[:n], SR)
    assert abs(1200 * math.log2(shifted / base)) <= 10.0


@pytest.mark.parametrize("st", [-2.0, 2.0])
def test_streamed_formant_moves_f1_and_f2_by_the_factor(streams, clip_48k, st):
    n = len(clip_48k)
    b1, b2 = formants(streams[0.0].aligned[:n], SR)
    s1, s2 = formants(streams[st].aligned[:n], SR)
    expected = 2 ** (st / 12)
    for ratio in (s1 / b1, s2 / b2):
        assert ratio == pytest.approx(expected, rel=0.15)
    assert (s2 > b2) == (st > 0)


@pytest.mark.parametrize("st", [-2.0, 2.0])
def test_formant_adds_no_clicks_at_hop_seams(streams, st):
    block = streams[0.0].plan.block
    base = boundary_click_ratio(streams[0.0].audio, block)
    assert boundary_click_ratio(streams[st].audio, block) <= base + 0.1


def test_full_file_formant_keeps_pitch_and_moves_formants(engine, clip_48k):
    conv = FullFileConverter.from_engine(engine)
    audio = resample_array(clip_48k, SR, 16000)
    sr = engine.loaded.model.tgt_sr
    renders = {}
    for st in (0.0, 2.0):
        torch.manual_seed(0)
        renders[st] = conv.convert(audio, dataclasses.replace(engine.params, formant_st=st))
    assert len(renders[2.0]) == len(renders[0.0])
    cents = 1200 * math.log2(f0_median(engine, renders[2.0], sr) /
                             f0_median(engine, renders[0.0], sr))
    assert abs(cents) <= 10.0
    assert formants(renders[2.0], sr)[1] / formants(renders[0.0], sr)[1] == \
        pytest.approx(2 ** (2 / 12), rel=0.15)


def test_formant_steps_swap_cached_graphs_and_replay_like_eager(engine, clip_48k):
    graphs, pipe = engine.graphs, engine.pipeline
    assert graphs.active and graphs.synth_step == 0.0
    before = graphs.formant_captures
    pipe.set_params(dataclasses.replace(engine.params, formant_st=1.5))
    assert graphs.formant_needs_capture()
    assert graphs.sync_formant() == "captured" and graphs.synth_step == 1.5
    assert not graphs.stale()  # only the synth graph changed: no full recapture
    pipe.set_params(dataclasses.replace(engine.params, formant_st=0.0))
    assert not graphs.formant_needs_capture()
    assert graphs.sync_formant() == "cached"  # back to a step already captured
    pipe.set_params(dataclasses.replace(engine.params, formant_st=1.5))
    assert graphs.sync_formant() == "cached"
    blocks = torch.from_numpy(clip_48k[:12 * engine.plan.block].reshape(12, -1)).to(engine.device)
    assert graphs.verify(blocks)["log_mel_l1"] < 0.01  # the 1.5 st graph replays like eager
    for st in (-1.0, -2.0, -3.0, 2.5, 3.0):  # more steps than the cache keeps
        pipe.set_params(dataclasses.replace(engine.params, formant_st=st))
        graphs.sync_formant()
    assert len(graphs._synth_cache) == FORMANT_GRAPHS_KEPT
    assert graphs.formant_captures - before == 1 + 5
    pipe.set_params(engine.params)
    graphs.sync_formant()
