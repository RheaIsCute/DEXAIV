"""Pipeline math that must match the canonical formulas: x2 features, protect, k-NN blend,
the incremental resampler, weight-norm fusion and the synthesizer's tail-only decode."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from vox.audio.resample import IncrementalResampler16k, OutputResampler
from vox.engine.pipeline import protect_blend, upsample_features
from vox.retrieval import gpu_knn, index_loader
from vox.retrieval.gpu_knn import GpuIndex
from vox.synthesis.nets.synthesizer import Synthesizer, fuse_weight_norm


@pytest.mark.parametrize("n_in, n_out", [(27, 56), (28, 57), (28, 56), (5, 12), (5, 11), (6, 6)])
def test_upsample_features_equals_dup_interpolate_truncate(n_in, n_out):
    feats = torch.randn(1, n_in, 8)
    reference = torch.cat((feats, feats[:, -1:, :]), 1)
    reference = F.interpolate(reference.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
    out = torch.full((1, n_out, 8), float("nan"))
    upsample_features(feats, out)
    assert torch.equal(out, reference[:, :n_out])


def test_protect_blend_matches_upstream():
    feats, feats0 = torch.randn(1, 12, 4), torch.randn(1, 12, 4)
    pitchf = torch.tensor([[0, 0, 120, 130, 0.5, 0, 200, 0, 0, 90, 95, 0]], dtype=torch.float32)
    protect = 0.33
    pitchff = pitchf.clone()  # rvc-webui Pipeline.vc
    pitchff[pitchf > 0] = 1
    pitchff[pitchf < 1] = protect
    reference = feats * pitchff.unsqueeze(-1) + feats0 * (1 - pitchff.unsqueeze(-1))
    assert torch.allclose(protect_blend(feats, feats0, pitchf, protect), reference, atol=1e-5)
    # voiced frames keep the blended features exactly; unvoiced move toward feats0
    assert torch.allclose(protect_blend(feats, feats0, pitchf, 0.0)[0, 0], feats0[0, 0])
    # the in-place hot-path form
    out = feats.clone()
    protect_blend(out, feats0, pitchf[0], torch.tensor(protect), out=out)
    assert torch.allclose(out, reference, atol=1e-5)


def knn_reference(query: np.ndarray, vectors: np.ndarray, index_rate: float) -> np.ndarray:
    """rvc-webui: k=8 exact search, w = 1/d^2 on squared-L2 scores, linear blend."""
    import faiss

    flat = faiss.IndexFlatL2(vectors.shape[1])
    flat.add(vectors)
    score, ix = flat.search(query, k=8)
    weight = np.square(1 / score)
    weight /= weight.sum(axis=1, keepdims=True)
    retrieved = np.sum(vectors[ix] * np.expand_dims(weight, axis=2), axis=1)
    return retrieved * index_rate + (1 - index_rate) * query


def test_gpu_knn_blend_matches_canonical_numpy(tmp_path):
    import faiss

    rng = np.random.default_rng(0)
    dim, n = 64, 400
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    index = faiss.index_factory(dim, "IVF4,Flat")
    index.train(vectors)
    index.add(vectors)
    path = tmp_path / "added_IVF4_Flat.index"
    faiss.write_index(index, str(path))

    loaded = index_loader.load(path, dim, "cpu")
    assert (loaded.ntotal, loaded.dim) == (n, dim)
    query = rng.standard_normal((30, dim)).astype(np.float32)
    feats = torch.from_numpy(query.copy()).unsqueeze(0)
    skip = 10
    gpu_knn.blend(feats, loaded, 0.75, skip_offset=skip)
    # context frames are untouched, the rest follow the canonical blend
    np.testing.assert_array_equal(feats[0, :skip].numpy(), query[:skip])
    order = np.argsort(index.reconstruct_n(0, n)[:, 0])  # IVF reorders vectors on disk
    assert order is not None
    np.testing.assert_allclose(feats[0, skip:].numpy(), knn_reference(query[skip:], vectors, 0.75),
                               rtol=2e-3, atol=2e-4)


def test_index_loader_validates_and_caps(tmp_path):
    import faiss

    vectors = np.random.default_rng(1).standard_normal((300, 32)).astype(np.float32)
    index = faiss.index_factory(32, "IVF2,Flat")
    index.train(vectors)
    faiss.write_index(index, str(tmp_path / "empty.index"))
    index.add(vectors)
    faiss.write_index(index, str(tmp_path / "added.index"))
    faiss.write_index(index, str(tmp_path / "trained_x.index"))

    with pytest.raises(index_loader.IndexFileError, match="dim"):
        index_loader.load(tmp_path / "added.index", 768, "cpu")
    with pytest.raises(index_loader.IndexFileError, match="empty"):
        index_loader.load(tmp_path / "empty.index", 32, "cpu")
    with pytest.raises(index_loader.IndexFileError, match="trained_"):
        index_loader.load(tmp_path / "trained_x.index", 32, "cpu")
    capped = index_loader.load(tmp_path / "added.index", 32, "cpu", max_vectors=100)
    assert capped.ntotal == 100
    assert torch.allclose(capped.norms, (capped.vectors**2).sum(1))


def test_retrieve_of_an_indexed_vector_returns_it():
    vectors = torch.randn(50, 16)
    index = GpuIndex(vectors, (vectors**2).sum(1))
    out = gpu_knn.retrieve(vectors[:5].clone(), index)
    assert torch.allclose(out, vectors[:5], atol=1e-3)  # zero distance dominates the weights


@pytest.mark.parametrize("device_sr, block_ms", [(48000, 20), (48000, 10), (44100, 30)])
def test_incremental_resampler_matches_one_shot(device_sr, block_ms):
    """Streaming block by block must equal resampling the whole signal at once."""
    import torchaudio

    block = device_sr // 100 * (block_ms // 10)
    block_16k = 160 * (block_ms // 10)
    hops = 30
    t = torch.arange(hops * block) / device_sr
    signal = 0.5 * torch.sin(2 * torch.pi * 220 * t) + 0.2 * torch.sin(2 * torch.pi * 3100 * t)
    whole = torchaudio.transforms.Resample(device_sr, 16000)(signal)

    resampler = IncrementalResampler16k(device_sr, block, "cpu")
    assert resampler.out_len == block_16k + 160
    ring = torch.zeros(hops * block_16k)
    for hop in range(hops):
        fresh = resampler.push(signal[hop * block:(hop + 1) * block])
        ring[:-block_16k] = ring[block_16k:].clone()
        ring[-resampler.out_len:] = fresh
    # everything except the newest 10 ms (still waiting for right-hand context) is exact
    settled = slice(2 * block_16k, hops * block_16k - 160)
    assert torch.allclose(ring[settled], whole[settled], atol=2e-5)


def test_output_resampler_lengths():
    assert OutputResampler(48000, 48000, "cpu").identity
    out = OutputResampler(40000, 48000, "cpu").run(torch.randn(6 * 400))
    assert out.shape[0] == 6 * 480
    assert OutputResampler(32000, 48000, "cpu").run(torch.randn(6 * 320)).shape[0] == 6 * 480


@pytest.mark.parametrize("layer", [torch.nn.Conv1d(4, 6, 3), torch.nn.ConvTranspose1d(4, 6, 4, 2)])
def test_fuse_weight_norm_equals_torch_weight_norm(layer):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        normed = torch.nn.utils.weight_norm(layer)
    with torch.no_grad():
        normed.weight_g.mul_(1.7)  # make g differ from ||v||
    x = torch.randn(1, 4, 9)
    expected = normed(x)
    state = fuse_weight_norm({f"m.{k}": v.half() for k, v in normed.state_dict().items()})
    assert set(state) == {"m.weight", "m.bias"} and state["m.weight"].dtype == torch.float32
    plain = type(layer)(4, 6, layer.kernel_size[0], layer.stride[0])
    plain.load_state_dict({k[2:]: v for k, v in state.items()})
    assert torch.allclose(plain(x), expected, atol=2e-3)  # half-precision storage round-off


def test_fuse_weight_norm_handles_parametrization_keys():
    layer = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(3, 5, 3))
    x = torch.randn(1, 3, 7)
    state = fuse_weight_norm({f"m.{k}": v for k, v in layer.state_dict().items()})
    plain = torch.nn.Conv1d(3, 5, 3)
    plain.load_state_dict({k[2:]: v for k, v in state.items()})
    assert torch.allclose(plain(x), layer(x), atol=1e-6)


TINY = [33, 8, 16, 16, 32, 2, 2, 3, 0, "1", [3], [[1, 3, 5]], [5, 4], 32, [9, 8], 3, 8, 2000]


@pytest.mark.parametrize("emb_dim, f0", [(768, True), (256, True), (768, False), (256, False)])
def test_synthesizer_variants_and_tail_only_shapes(emb_dim, f0):
    """All four upstream variants build; skip_head/return_length decode exactly the tail."""
    torch.manual_seed(0)
    net = Synthesizer(TINY, emb_dim, f0).eval()
    assert net.upp == 20 and net.has_f0 is f0
    frames = 60
    feats = torch.randn(1, frames, emb_dim)
    lengths, sid = torch.tensor([frames]), torch.tensor([1])
    pitch = torch.randint(1, 255, (1, frames)) if f0 else None
    pitchf = torch.rand(1, frames) * 200 + 80 if f0 else None
    with torch.no_grad():
        full = net.infer(feats, lengths, pitch, pitchf, sid)
        tail = net.infer(feats, lengths, pitch, pitchf, sid, skip_head=40, return_length=20)
    assert full.shape == (1, 1, frames * 20) and tail.shape == (1, 1, 20 * 20)
    assert torch.isfinite(full).all() and full.abs().max() <= 1.0
    assert ("emb_pitch" in dict(net.enc_p.named_children())) is f0


def test_decoder_context_keeps_the_tail_length_and_position():
    torch.manual_seed(0)
    net = Synthesizer(TINY, 768, True).eval()
    frames = 60
    feats = torch.randn(1, frames, 768)
    lengths, sid = torch.tensor([frames]), torch.tensor([0])
    pitch = torch.randint(1, 255, (1, frames))
    pitchf = torch.rand(1, frames) * 200 + 80
    with torch.no_grad():
        outs = [net.infer(feats, lengths, pitch, pitchf, sid, 40, 20, dec_context=c)
                for c in (0, 8, 100)]  # 100 is clamped to skip_head
    assert all(o.shape == (1, 1, 20 * 20) for o in outs)
    assert all(torch.isfinite(o).all() for o in outs)
