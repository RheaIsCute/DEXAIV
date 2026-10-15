"""Engine-level acceptance tests on real weights (docs/research/13, Phases 1-2).

They need the assets from ``scripts/fetch_assets.py --all`` and skip without them. The
synthesizer draws fresh noise on every render, so audio is compared by log-mel distance
against measured tolerances, never sample by sample.

Regenerate the committed golden log-mel (an explicit, reviewed action) with::

    VOX_UPDATE_GOLDEN=1 pytest tests/engine -k golden
"""

import dataclasses
import os

import numpy as np
import pytest
import torch
from tests.conftest import FIXTURES, MODELS, TEST_MODEL

from vox.config import defaults
from vox.config.schemas import RuntimeParams
from vox.offline.convert import render_full
from vox.offline.metrics import boundary_click_ratio, compare, log_mel
from vox.offline.simulate import StreamSimulator, load_engine

pytestmark = pytest.mark.assets
GOLDEN = FIXTURES / "golden_full_base48k_libri1272_8s.npz"
SR = 48000
# Measured on the reference machine (docs/DEVIATIONS.md): new-seed noise floor ~0.31,
# same-seed same-GPU repeat ~0.005. A gross pipeline error lands well above 0.6.
GOLDEN_TOLERANCE = 0.42
# Streamed vs full-file in speech frames: measured 0.56 at the default plan. This is a
# regression guard, not an equivalence claim (see docs/research/14 Q14).
STREAM_SPEECH_TOLERANCE = 0.70
SETTLE_S = 1.1  # default plan: 0.56 s of context plus margin


@pytest.fixture(scope="module")
def cache():
    return {}


@pytest.fixture(scope="module")
def engine(device, cache):
    return load_engine(MODELS, TEST_MODEL, defaults.DEFAULT_SHAPE, RuntimeParams(),
                       device=device, cache=cache)


@pytest.fixture(scope="module")
def full(engine, clip_48k):
    return render_full(engine, clip_48k, seed=0)


@pytest.fixture(scope="module")
def streamed(engine, clip_48k):
    return StreamSimulator(engine).run(clip_48k, sync_debug=True, seed=0)


def test_full_file_render_matches_golden_log_mel(full):
    mel = log_mel(full, SR).numpy()
    if os.environ.get("VOX_UPDATE_GOLDEN"):
        np.savez_compressed(GOLDEN, log_mel=mel.astype(np.float16))
        pytest.skip("golden updated")
    golden = np.load(GOLDEN)["log_mel"].astype(np.float32)
    assert mel.shape == golden.shape
    distance = float(np.abs(mel - golden).mean())
    assert distance < GOLDEN_TOLERANCE, f"log-mel L1 vs golden = {distance:.3f}"


def test_full_file_render_is_sane(full, clip_48k):
    assert len(full) == len(clip_48k) and np.isfinite(full).all()
    assert 0.01 < np.sqrt(np.mean(full**2)) < 0.5 and np.abs(full).max() <= 1.0


def test_seeded_full_file_render_is_reproducible(engine, clip_48k, full):
    again = render_full(engine, clip_48k[: 3 * SR], seed=0)
    first = render_full(engine, clip_48k[: 3 * SR], seed=0)
    assert compare(first, again, SR)["l1"] < 0.05


# docs/DEVIATIONS.md §7, fix 1: at the 20 ms default the overlap equals the hop, so every
# emitted sample but the seams is a mix of two renders that are partly uncorrelated; the
# seam metric (seam vs mid-hop roughness) reads that as 1.3-1.4x. The pre-fix geometry
# (30 ms overlap at the 20 ms hop, a three-render blend, ``overlap_beyond_hop``) reads ~1.0x
# and passes. Which one ships is a listening decision (out/ab/reference.wav vs
# sola_blend3.wav); until it is made these record the regression. Not strict: the ratio
# moves with the decoder-noise seed alone (8 s clip, seeds 0-2: 1.20-1.34 on the 48 kHz test
# voice, 1.18-1.45 on the 40 kHz one, docs/DEVIATIONS.md §8), so one seed can land under the
# 1.25 guard without anything having changed.
OVERLAP_EQUALS_HOP = pytest.mark.xfail(
    strict=False, reason="20 ms overlap == 20 ms hop: seam roughness ~1.2-1.45x, see "
                         "docs/DEVIATIONS.md §7 (pending the listening A/B)")


@OVERLAP_EQUALS_HOP
def test_stream_has_no_hop_boundary_clicks(streamed, full):
    ratio = boundary_click_ratio(streamed.audio, streamed.plan.block)
    control = boundary_click_ratio(full, streamed.plan.block)
    assert ratio < 1.25, f"boundary roughness {ratio:.2f}x mid-hop (full-file: {control:.2f}x)"


def test_pre_clamp_overlap_geometry_has_no_hop_boundary_clicks(device, cache, clip_48k):
    """The A/B alternative to the 20 ms default: 30 ms overlap at the 20 ms hop."""
    shape = dataclasses.replace(defaults.DEFAULT_SHAPE, crossfade_ms=30, overlap_beyond_hop=True)
    engine = load_engine(MODELS, TEST_MODEL, shape, RuntimeParams(), device=device, cache=cache)
    assert engine.plan.sola_buffer > engine.plan.block
    result = StreamSimulator(engine).run(clip_48k, seed=0)
    assert boundary_click_ratio(result.audio, engine.plan.block) < 1.25


def test_stream_stays_close_to_full_file_render(streamed, full, clip_48k):
    quality = compare(full, streamed.aligned[: len(clip_48k)], SR, lag_search_ms=15.0,
                      start_s=SETTLE_S)
    assert quality["l1_speech"] < STREAM_SPEECH_TOLERANCE, quality
    assert abs(quality["lag_ms"]) <= 10.0  # nominal delay accounts for the stream timing


def test_stream_output_shape_and_timing(streamed, clip_48k):
    plan = streamed.plan
    assert len(streamed.audio) == streamed.hops * plan.block
    assert streamed.hops == -(-len(clip_48k) // plan.block) + -(-plan.stream_delay // plan.block)
    assert np.isfinite(streamed.audio).all() and np.abs(streamed.audio).max() <= 1.0
    assert streamed.audio[: plan.block].max() < 0.05  # first hop fades in from silence
    assert 0 <= streamed.sola_offsets_ms.min() and streamed.sola_offsets_ms.max() <= 10.0
    assert len(streamed.sola_offsets_ms) == streamed.hops


@pytest.mark.gpu
def test_steady_state_hop_is_sync_free_and_allocation_stable(streamed):
    """The ``streamed`` fixture ran under torch's sync-debug mode in ``error`` mode: any
    ``.item()``, ``.cpu()`` or mask-indexing inside the hop would have raised."""
    assert streamed.pointers_stable  # every persistent buffer kept its address
    alloc = streamed.alloc
    assert alloc["driver_mallocs"] == 0  # no cudaMalloc after warmup
    assert alloc["reserved_growth_bytes"] == 0 and alloc["live_drift_bytes"] == 0


def test_profiler_report_covers_every_stage(engine, streamed):
    from vox.profiling.report import format_report, stage_table

    table = stage_table(engine.timers, skip=50)
    assert {"resample_in", "f0", "embed", "synth", "sola", "download", "total"} <= set(table)
    stages = sum(table[s]["host"]["p50"] for s in table if s != "total")
    assert stages <= table["total"]["host"]["p50"] * 1.05
    if engine.device.type == "cuda":
        assert table["synth"]["gpu"]["count"] > 0
    assert "headroom" in format_report(engine.timers, hop_ms=20.0, skip=50)


def test_pitch_cache_matches_full_window_f0(engine, clip_48k):
    """Tail-only F0 spliced through the rolling cache agrees with full-file extraction.

    Each cached frame is compared at the frame it was estimated for. With
    ``f0_trim_tail`` = t the cache ends t frames before the window does (t = 1 upstream;
    the default t = 2 puts F0 10 ms later against the features: docs/DEVIATIONS.md §8)."""
    from vox.audio.resample import resample_array

    plan, pipeline = engine.plan, engine.pipeline
    pipeline.reset()
    trim = pipeline.pitch_cache.trim_tail
    interior = plan.f0_tail_16k // 160 + 1 - 3 - trim  # frames each hop rewrites
    settled = []
    hops = len(clip_48k) // plan.block
    for n in range(hops):
        pipeline.upload(clip_48k[n * plan.block:(n + 1) * plan.block])
        pipeline.process_hop()
        # The frames about to leave the rewrite zone carry their final value.
        cache = pipeline.pitch_cache.pitchf
        settled.append(cache[-interior:-interior + plan.shift_frames].clone())
    stream_f0 = torch.cat(settled).cpu().numpy()
    # After hop n the window ends at frame (n+1)*shift and cache[-1] is the frame ``trim``
    # before it, so the first settled frame of hop 0 is frame ``shift - interior + 1 - trim``.
    first = plan.shift_frames - interior + 1 - trim
    audio_16k = torch.from_numpy(resample_array(clip_48k, SR, 16000)).to(engine.device)
    full_f0 = pipeline.extractor.extract(audio_16k).cpu().numpy()
    frames = np.arange(first, first + len(stream_f0))
    keep = (frames >= 50) & (frames < len(full_f0))  # skip the stream's start-up
    a, b = stream_f0[keep], full_f0[frames[keep]]
    both = (a > 0) & (b > 0)
    cents = 1200 * np.abs(np.log2(a[both] / b[both]))
    stats = {"voicing_agreement": float(((a > 0) == (b > 0)).mean()),
             "voiced_share": float(both.mean()), "cents_p50": float(np.median(cents)),
             "cents_p95": float(np.percentile(cents, 95)),
             "octave_errors": float((cents > 600).mean())}
    print("pitch cache vs full-file F0:", stats)
    # Measured on the reference machine: agreement 0.970, cents p50 2.7 / p95 38, no octave
    # errors. Disagreements sit on voiced/unvoiced edges, where a 310 ms tail and the whole
    # file legitimately decide differently; an off-by-one splice would drop agreement far
    # below these guards.
    assert stats["voicing_agreement"] > 0.95, stats
    assert stats["voiced_share"] > 0.3 and stats["octave_errors"] < 0.005, stats
    assert stats["cents_p50"] < 5.0 and stats["cents_p95"] < 50.0, stats


@pytest.mark.parametrize("shape, params", [
    (dict(block_ms=10, extra_ms=100), {}),  # window shorter than the F0 tail
    (dict(block_ms=50, crossfade_ms=20), dict(sola_mode="pv")),
    (dict(block_ms=30), dict(f0_up_key=12.0, f0_unvoiced="interp", volume_envelope=0.5)),
    (dict(), dict(f0_method="fcpe")),
    # the Step 2 A/B settings (docs/DEVIATIONS.md §7), all at once
    (dict(embed_pad_ms=15.0, decoder_context_ms=80.0), dict(noise_roll=True, f0_bridge_ms=60.0)),
])
def test_other_plans_and_params_run_clean(device, cache, clip_48k, shape, params):
    engine = load_engine(MODELS, TEST_MODEL, dataclasses.replace(defaults.DEFAULT_SHAPE, **shape),
                         RuntimeParams(**params), device=device, cache=cache)
    result = StreamSimulator(engine).run(clip_48k[: 3 * SR])
    assert np.isfinite(result.audio).all() and result.pointers_stable
    assert 0.005 < np.sqrt(np.mean(result.audio[SR:] ** 2)) < 0.5


@pytest.fixture(scope="module")
def stream_40k(device, cache, clip_48k):
    engine = load_engine(MODELS, "base_f0_40k_v2", defaults.DEFAULT_SHAPE, RuntimeParams(),
                         device=device, cache=cache)
    return engine.plan, StreamSimulator(engine).run(clip_48k[: 3 * SR])


def test_40k_model_resamples_to_the_device_rate(stream_40k):
    plan, result = stream_40k
    assert plan.model_sr == 40000 and plan.decoder_out * 48000 == plan.out_device * 40000
    assert len(result.audio) == result.hops * plan.block and np.isfinite(result.audio).all()


@OVERLAP_EQUALS_HOP
def test_40k_model_has_no_hop_boundary_clicks(stream_40k):
    plan, result = stream_40k
    assert boundary_click_ratio(result.audio, plan.block) < 1.3


@pytest.mark.gpu
def test_graph_replay_matches_eager_and_removes_pool_allocations(engine, clip_48k):
    """Graphs are the default fast path: the engine must be replaying, the replayed stream
    must match an eager stream of the same seed, and the hop must allocate nothing."""
    from vox.engine.pipeline import GPU_STAGES

    assert engine.mode == "graphs", engine.graphs.report
    assert set(engine.graphs.report.stages) >= set(GPU_STAGES) | {"sola"}
    assert engine.graphs.report.verify["passed"]
    sim = StreamSimulator(engine)
    graphed = sim.run(clip_48k[: 4 * SR], seed=3)
    assert graphed.alloc["pool_allocs_per_hop"] == 0
    engine.graphs.release()
    try:
        assert engine.mode == "eager"
        eager = sim.run(clip_48k[: 4 * SR], seed=3)
    finally:
        engine.graphs.capture()
    assert engine.mode == "graphs"
    distance = compare(eager.audio, graphed.audio, SR, start_s=1.5)
    # same seed, same noise draws, same state: only kernel-level rounding differs
    assert distance["l1"] < 0.05, distance


def test_gate_streams_speech_silence_speech_without_clicks(engine, clip_48k):
    """speech -> 1.5 s silence -> speech through the real engine with the gate on: the gate
    must close and reopen, transitions must be click-free, and closed hops must be cheap."""
    from vox.offline.metrics import transition_click_ratio

    plan = engine.plan
    speech = clip_48k[: 3 * SR]
    audio = np.concatenate([speech, np.zeros(int(1.5 * SR), np.float32), speech])
    result = StreamSimulator(engine).run(audio, gate=True)
    assert result.gate_transitions >= 2, result.gate_transitions
    closed = ~result.gate_open
    assert closed.sum() >= 40  # most of the 1.5 s of silence (75 hops) minus the hangover
    toggles = np.flatnonzero(np.diff(result.gate_open.astype(np.int8)) != 0) + 1
    ratio = transition_click_ratio(result.audio, toggles * plan.block, plan.block)
    assert ratio < 1.25, f"click ratio at gate transitions {ratio:.2f}"
    assert boundary_click_ratio(result.audio, plan.block) < 1.25
    # silence really is silence once the drain is over
    silent_hops = np.flatnonzero(closed)[3:-1]
    for h in silent_hops[:10]:
        assert np.abs(result.audio[h * plan.block:(h + 1) * plan.block]).max() == 0.0
    # a closed hop only advances the 16 kHz ring: far below the cost of a converted hop
    assert np.median(result.closed_hop_ms) < 2.0, np.median(result.closed_hop_ms)


def test_gate_close_keeps_the_pitch_history_moving_with_the_ring(engine, clip_48k):
    """A gate close no longer wipes the pitch cache: closed hops shift it one hop each, in
    step with the 16 kHz ring, so after the reopen the voiced history lines up with the
    features ContentVec computes from that ring."""
    pipeline, plan = engine.pipeline, engine.plan
    pipeline.reset()
    for n in range(60):
        pipeline.upload(clip_48k[(100 + n) * plan.block:(101 + n) * plan.block])
        pipeline.process_hop()
    before = pipeline.pitch_cache.pitchf.clone()
    assert (before > 0).float().mean() > 0.3  # speech: the history is mostly voiced
    pipeline.forget_speech()  # the drain hop
    closed = 5
    for _ in range(closed):
        pipeline.upload(np.zeros(plan.block, np.float32))
        pipeline.advance_context()
    after = pipeline.pitch_cache.pitchf
    moved = closed * plan.shift_frames
    assert torch.equal(after[:-moved], before[moved:])
    assert not after[-moved:].any()
    assert pipeline.feats.abs().sum() > 0  # features are recomputed each hop, not wiped


def test_hot_params_apply_between_hops_without_reallocating(engine, clip_48k):
    pipeline, plan = engine.pipeline, engine.plan
    pointers = {k: v.data_ptr() for k, v in pipeline.buffers().items()}
    pipeline.reset()
    block = clip_48k[4 * SR: 4 * SR + plan.block]

    def median_pitch(key):
        pipeline.set_params(dataclasses.replace(engine.params, f0_up_key=key))
        for _ in range(40):  # refill the whole pitch cache at this key
            pipeline.upload(block)
            pipeline.process_hop()
        voiced = pipeline.pitch_cache.pitchf[pipeline.pitch_cache.pitchf > 0]
        return voiced.median().item() if voiced.numel() else float("nan")

    low, high = median_pitch(0.0), median_pitch(12.0)
    pipeline.set_params(engine.params)
    assert high == pytest.approx(2 * low, rel=0.05)
    assert pointers == {k: v.data_ptr() for k, v in pipeline.buffers().items()}
