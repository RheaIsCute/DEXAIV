import pytest

from vox.engine import chunker
from vox.engine.chunker import ChunkPlanError


def default(**kw):
    args = dict(block_ms=20, crossfade_ms=30, extra_ms=500, sola_search_ms=10,
                device_sr=48000, model_sr=48000)
    args.update(kw)
    return chunker.plan(**args)


def test_the_chunking_law():
    p = default()
    # convert = extra + crossfade + sola_search + block, in 16 kHz samples
    assert p.convert_16k == 16 * (500 + 30 + 10 + 20) == 8960
    assert p.convert_16k % 160 == 0
    assert p.convert_frames == 56
    assert p.skip_head == 50  # extra frames dropped
    assert p.return_length == 6  # block + overlap + search frames vocoded
    assert p.skip_head + p.return_length == p.convert_frames
    assert p.decoder_out == 6 * 480
    assert p.out_device == p.block + p.sola_buffer + p.sola_search == 2880
    assert p.shift_frames == p.block_16k // 160 == 2
    assert p.p_len == p.convert_frames


def test_rvc_webui_worked_example():
    """docs/research/02 §2: block 160 ms, extra 2.5 s, crossfade 80 ms, 40 k model."""
    p = default(block_ms=160, crossfade_ms=80, extra_ms=2500, model_sr=40000)
    assert p.sola_buffer == 4 * p.zc  # 80 ms requested, capped at 40 ms
    assert p.skip_head == 250
    assert p.return_length == 16 + 4 + 1
    assert p.decoder_out == 21 * 400


@pytest.mark.parametrize("ms, units", [(4, 1), (10, 1), (14, 1), (15, 2), (16, 2), (25, 3),
                                       (32, 3), (35, 4)])
def test_sizes_snap_to_10ms_units_half_up(ms, units):
    assert default(block_ms=ms).block == units * 480


def test_overlap_capped_at_40ms_and_latency_uses_the_cap():
    p = default(crossfade_ms=100)
    assert p.crossfade == 10 * 480 and p.sola_buffer == 4 * 480
    assert p.algorithmic_latency_ms == 20 + 40 + 10
    assert p.stream_delay == p.sola_buffer + p.sola_search


@pytest.mark.parametrize("device_sr, model_sr", [(48000, 32000), (48000, 40000),
                                                 (44100, 48000), (48000, 48000)])
def test_rates_stay_in_exact_integer_ratio(device_sr, model_sr):
    p = default(device_sr=device_sr, model_sr=model_sr)
    assert p.zc == device_sr // 100
    assert p.block_16k * p.zc == 160 * p.block  # the 16 k mirror never drifts vs the device
    assert p.decoder_out * device_sr == p.out_device * model_sr


@pytest.mark.parametrize("block_ms", [10, 20, 30, 50, 100])
@pytest.mark.parametrize("extra_ms", [0, 100, 300, 500, 2500])
def test_embedder_always_covers_the_window(block_ms, extra_ms):
    p = default(block_ms=block_ms, extra_ms=extra_ms)
    assert p.hubert_frames == (p.convert_16k - 400) // 320 + 1
    assert 2 * (p.hubert_frames + 1) >= p.convert_frames == p.p_len


def test_f0_tail_longer_than_window_extends_the_ring():
    small = default(block_ms=10, extra_ms=100, f0_tail_16k=4960)
    assert small.convert_16k == 16 * 150 and small.history_16k == 4960
    big = default(f0_tail_16k=4960)
    assert big.history_16k == big.convert_16k


def test_hubert_phase_flag():
    assert default(block_ms=20).hubert_phase_stable
    assert not default(block_ms=10).hubert_phase_stable
    assert not default(block_ms=30).hubert_phase_stable


@pytest.mark.parametrize("kw", [dict(block_ms=0), dict(crossfade_ms=-1), dict(extra_ms=-5),
                                dict(device_sr=44101), dict(sola_search_ms=0)])
def test_invalid_inputs(kw):
    with pytest.raises(ChunkPlanError):
        default(**kw)


def test_plan_is_frozen_and_pure_math():
    p = default()
    with pytest.raises(AttributeError):
        p.block = 1
    with open(chunker.__file__, encoding="utf-8") as fh:
        source = fh.read()
    assert "import torch" not in source and "import numpy" not in source


def test_decoder_context_is_clamped_to_the_available_history():
    assert default().dec_context == 0  # upstream behaviour by default
    assert default(decoder_context_ms=80).dec_context == 8
    assert default(extra_ms=50, decoder_context_ms=200).dec_context == 5  # skip_head is 5
