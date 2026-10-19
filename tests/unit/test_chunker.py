import pytest

from vox.engine import chunker
from vox.engine.chunker import ChunkPlanError


def default(**kw):
    args = dict(block_ms=20, crossfade_ms=20, extra_ms=500, sola_search_ms=10,
                device_sr=48000, model_sr=48000)
    args.update(kw)
    return chunker.plan(**args)


def test_the_chunking_law():
    p = default()
    # convert = extra + crossfade + sola_search + block, in 16 kHz samples
    assert p.convert_16k == 16 * (500 + 20 + 10 + 20) == 8800
    assert p.convert_16k % 160 == 0
    assert p.convert_frames == 55
    assert p.skip_head == 50  # extra frames dropped
    assert p.return_length == 5  # block + overlap + search frames vocoded
    assert p.skip_head + p.return_length == p.convert_frames
    assert p.decoder_out == 5 * 480
    assert p.out_device == p.block + p.sola_buffer + p.sola_search == 2400
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
    p = default(block_ms=50, crossfade_ms=100)
    assert p.crossfade == 10 * 480 and p.sola_buffer == 4 * 480
    assert p.algorithmic_latency_ms == 50 + 40 + 10
    assert p.stream_delay == p.sola_buffer + p.sola_search
    assert "capped at 40 ms" in p.overlap_note


def test_overlap_beyond_hop_is_the_unclamped_ab_geometry():
    p = default(crossfade_ms=30, overlap_beyond_hop=True)
    assert p.sola_buffer == 30 * 48 > p.block and p.overlap_note == ""
    assert p.out_device == p.block + p.sola_buffer + p.sola_search


@pytest.mark.parametrize("extra_ms", [500, 1000])
@pytest.mark.parametrize("crossfade_ms", [20, 30])
def test_embed_pad_makes_contentvec_reach_the_end_of_the_window(extra_ms, crossfade_ms):
    """Without the pad the last 50 Hz frame stops short and is held for the final decoded
    frames; 240 replicated samples (15 ms) give 2 x frames >= the window's 100 Hz frames."""
    plain = default(extra_ms=extra_ms, crossfade_ms=crossfade_ms, overlap_beyond_hop=True)
    padded = default(extra_ms=extra_ms, crossfade_ms=crossfade_ms, overlap_beyond_hop=True,
                     embed_pad_16k=240)
    assert 2 * plain.hubert_frames < plain.convert_frames
    assert 2 * padded.hubert_frames >= padded.convert_frames
    assert padded.convert_16k == plain.convert_16k  # the window itself is unchanged


def test_from_shape_reads_every_shape_field():
    from vox.config.schemas import ShapeParams

    shape = ShapeParams(crossfade_ms=30, decoder_context_ms=80, overlap_beyond_hop=True,
                        embed_pad_ms=15)
    p = chunker.from_shape(shape, 32000, f0_tail_16k=4960)
    assert (p.sola_buffer, p.dec_context, p.embed_pad_16k, p.f0_tail_16k) == (1440, 8, 240, 4960)
    assert p.model_sr == 32000


@pytest.mark.parametrize("block_ms, crossfade_ms, used", [(20, 30, 20), (20, 20, 20),
                                                          (10, 30, 10), (30, 20, 20),
                                                          (20, 100, 20)])
def test_overlap_never_exceeds_the_hop(block_ms, crossfade_ms, used):
    p = default(block_ms=block_ms, crossfade_ms=crossfade_ms)
    assert p.sola_buffer == used * 48 and p.sola_buffer <= p.block
    assert p.out_device == p.block + p.sola_buffer + p.sola_search
    assert ("cannot exceed the hop" in p.overlap_note) == (crossfade_ms > block_ms)
    assert p.overlap_note == "" or p.overlap_note.startswith(f"crossfade {crossfade_ms} ms")


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
    small = default(block_ms=10, extra_ms=100, f0_tail_16k=4960)  # overlap clamped to 10
    assert small.convert_16k == 16 * 130 and small.history_16k == 4960
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


# ------------------------------------------------------------ right context (lookahead)


@pytest.mark.parametrize("right_ms", [0, 20, 40, 80, 100])
def test_right_context_extends_the_window_and_moves_the_kept_region(right_ms):
    """convert = extra + right + crossfade + search + block; the kept region ends ``right``
    before the window's end; the decoder renders those frames too and drops them."""
    p = default(extra_ms=500, decoder_context_ms=80, extra_right_ms=right_ms)
    r = right_ms // 10
    assert p.right == r * 480 and p.right_frames == r
    assert p.convert_16k == 16 * (500 + right_ms + 20 + 10 + 20)
    assert p.convert_frames == 55 + r and p.p_len == p.convert_frames
    assert p.return_length == 5  # the kept region itself is unchanged
    assert p.skip_head == 50 and p.skip_head + p.return_length + p.right_frames == p.convert_frames
    assert p.dec_context == 8
    assert p.decode_frames == 5 + 8 + r
    assert p.out_device == 2400 and p.decoder_out == 5 * 480  # emitted sizes unchanged
    # latency: the stream trails the input by the overlap, the search and the lookahead
    assert p.stream_delay == p.sola_buffer + p.sola_search + p.right
    assert p.algorithmic_latency_ms == 20 + 20 + 10 + right_ms
    assert p.right_ms == right_ms
    assert p.hubert_phase_stable


def test_right_context_from_shape_and_rounding():
    from vox.config.schemas import ShapeParams

    p = chunker.from_shape(ShapeParams(extra_right_ms=40), 48000)
    assert p.right_frames == 4 and p.right == 4 * 480
    assert default(extra_right_ms=14).right_frames == 1  # 10 ms units, half up
    assert default(extra_right_ms=15).right_frames == 2
    with pytest.raises(ChunkPlanError):
        default(extra_right_ms=-10)


def test_right_context_at_hop_40_and_with_the_f0_tail():
    p = default(block_ms=40, crossfade_ms=40, extra_ms=1000, extra_right_ms=40,
                f0_tail_16k=4960, decoder_context_ms=80)
    assert p.sola_buffer == 4 * 480 and p.return_length == 4 + 4 + 1
    assert p.skip_head == p.convert_frames - 9 - 4
    assert p.history_16k == p.convert_16k  # the window is longer than the F0 tail
    assert p.algorithmic_latency_ms == 40 + 40 + 10 + 40


# ------------------------------------------------------------------- output-ring fill rule


@pytest.mark.parametrize("block_ms, legacy, hop_plus", [(20, 50, 40), (40, 90, 60), (60, 130, 80),
                                                       (200, 410, 220)])
def test_fill_rules(block_ms, legacy, hop_plus):
    assert chunker.fill_target_ms(block_ms, 10.0) == legacy
    assert chunker.fill_target_ms(block_ms, 10.0, "legacy") == legacy
    assert chunker.fill_target_ms(block_ms, 10.0, "hop_plus", 20.0) == hop_plus
    assert chunker.fill_target_ms(block_ms, 10.0, "hop_plus", 30.0) == block_ms + 30
    with pytest.raises(ChunkPlanError):
        chunker.fill_target_ms(block_ms, 10.0, "twice")
    assert chunker.FILL_RULES == ("legacy", "hop_plus")
