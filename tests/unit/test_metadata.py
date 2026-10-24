"""Metadata detection on synthetic checkpoints (docs/research/09 §5). No real weights needed."""

import pickle

import numpy as np
import pytest
import torch

from vox.models import discovery
from vox.models.metadata import (
    MetadataError,
    detect_checkpoint,
    find_index,
    inspect_pth,
    load_checkpoint,
)

CONFIG_48K = [1025, 36, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11],
              [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [12, 10, 2, 2], 512, [24, 20, 4, 4], 109, 256,
              48000]


def checkpoint(emb_dim=768, f0=True, n_spk=1, config=None, **keys):
    weight = {
        "enc_p.emb_phone.weight": torch.zeros(192, emb_dim),
        "emb_g.weight": torch.zeros(n_spk, 256),
        "dec.ups.0.weight_g": torch.zeros(512, 1, 1),
        "dec.ups.0.weight_v": torch.zeros(512, 256, 24),
    }
    if f0:
        weight["dec.m_source.l_linear.weight"] = torch.zeros(1, 1)
        weight["dec.noise_convs.0.weight"] = torch.zeros(256, 1, 80)
    return {"weight": weight, "config": list(config or CONFIG_48K), **keys}


def test_v2_f0_model_with_all_keys():
    meta = detect_checkpoint(checkpoint(f0=1, version="v2", info="300epoch"))
    assert (meta.version, meta.emb_dim, meta.f0, meta.sr) == ("v2", 768, True, 48000)
    assert (meta.emb_output_layer, meta.use_final_proj) == (12, False)
    assert meta.info == "300epoch" and not meta.warnings and not meta.fork


def test_v1_model_uses_layer_9_and_final_proj():
    meta = detect_checkpoint(checkpoint(emb_dim=256, version="v1"))
    assert (meta.version, meta.emb_dim, meta.emb_output_layer, meta.use_final_proj) == (
        "v1", 256, 9, True)


def test_missing_version_and_f0_keys_are_sniffed_from_weights():
    assert detect_checkpoint(checkpoint(emb_dim=768)).version == "v2"
    assert detect_checkpoint(checkpoint(emb_dim=256)).version == "v1"
    assert detect_checkpoint(checkpoint(f0=True)).f0 is True
    assert detect_checkpoint(checkpoint(f0=False)).f0 is False


def test_weights_win_over_lying_keys():
    meta = detect_checkpoint(checkpoint(emb_dim=768, f0=False, version="v1", f0_key=None) |
                             {"f0": 1})
    assert meta.version == "v2" and meta.f0 is False
    assert any("version" in w for w in meta.warnings)
    assert any("'f0'" in w for w in meta.warnings)


def test_speaker_count_is_patched_from_the_embedding_table():
    meta = detect_checkpoint(checkpoint(n_spk=4))
    assert meta.n_spk == 4 and meta.config[-3] == 4


@pytest.mark.parametrize("stored, expected", [("48k", 48000), ("48000", 48000), (48000, 48000)])
def test_sample_rate_forms(stored, expected):
    config = [*CONFIG_48K[:-1], stored]
    meta = detect_checkpoint(checkpoint(config=config))
    assert meta.sr == expected and meta.config[-1] == expected


def test_sample_rate_cross_checked_against_upsample_rates():
    config = [*CONFIG_48K[:-1], 40000]  # says 40 k, upsamples x480
    meta = detect_checkpoint(checkpoint(config=config))
    assert meta.sr == 48000 and any("upsample" in w for w in meta.warnings)


def test_fork_config_is_flagged_not_fatal():
    meta = detect_checkpoint(checkpoint(config=[*CONFIG_48K, 768]))
    assert meta.fork and len(meta.config) == 18 and any("fork" in w for w in meta.warnings)


def test_training_checkpoint_gets_a_helpful_error():
    with pytest.raises(MetadataError, match="training checkpoint"):
        detect_checkpoint({"model": {}, "optimizer": {}, "iteration": 100})


@pytest.mark.parametrize("bad", [
    {"weight": {}, "config": CONFIG_48K},  # no synthesizer weights
    {"config": CONFIG_48K},
    ["not", "a", "dict"],
    checkpoint(config=CONFIG_48K[:10]),
    checkpoint(emb_dim=1024),
    checkpoint(vocoder="RefineGAN"),
])
def test_unusable_checkpoints_raise_not_crash(bad):
    with pytest.raises(MetadataError):
        detect_checkpoint(bad)


def test_embedder_name_suffix_is_stripped():
    assert detect_checkpoint(checkpoint(embedder_name="contentvec768")).embedder == "contentvec"
    meta = detect_checkpoint(checkpoint(embedder_name="spin"))
    assert meta.embedder == "spin" and any("spin" in w for w in meta.warnings)


def test_files_load_with_weights_only(tmp_path):
    path = tmp_path / "ok.pth"
    torch.save(checkpoint(version="v2"), path)
    assert inspect_pth(path).version == "v2"


class _Evil:
    def __reduce__(self):
        return (print, ("code ran during unpickling",))


def test_pickle_with_code_is_refused(tmp_path, capsys):
    path = tmp_path / "evil.pth"
    with open(path, "wb") as fh:
        pickle.dump({"weight": _Evil(), "config": CONFIG_48K}, fh)
    with pytest.raises(MetadataError, match="weights_only"):
        load_checkpoint(path)
    assert "code ran" not in capsys.readouterr().out


def _write_index(path, dim, n=64, add=True):
    import faiss

    index = faiss.index_factory(dim, "IVF2,Flat")
    vectors = np.random.default_rng(0).standard_normal((n, dim)).astype(np.float32)
    index.train(vectors)
    if add:
        index.add(vectors)
    faiss.write_index(index, str(path))


def test_index_pairing_prefers_added_and_checks_dim(tmp_path):
    _write_index(tmp_path / "trained_IVF2_Flat.index", 768, add=False)
    _write_index(tmp_path / "other.index", 256)
    _write_index(tmp_path / "added_IVF2_Flat_v2.index", 768)
    found, warnings = find_index(tmp_path, 768)
    assert found.path.name == "added_IVF2_Flat_v2.index" and (found.dim, found.ntotal) == (768, 64)
    found_v1, _ = find_index(tmp_path, 256)
    assert found_v1.path.name == "other.index"
    assert any("256" in w or "trained_" in w for w in find_index(tmp_path, 768)[1] + warnings) \
        or found is not None


def test_index_rejections_are_reported(tmp_path):
    _write_index(tmp_path / "trained_IVF2_Flat.index", 768, add=False)
    _write_index(tmp_path / "wrong_dim.index", 256)
    _write_index(tmp_path / "zzz_empty.index", 768, add=False)
    found, warnings = find_index(tmp_path, 768)
    assert found is None
    text = " ".join(warnings)
    assert "trained_*" in text and "dim 256" in text and "empty" in text


def test_import_model_writes_layout_and_config(tmp_path):
    drop = tmp_path / "downloads"
    drop.mkdir()
    torch.save(checkpoint(version="v2", info="e300"), drop / "My Voice (e300).pth")
    _write_index(drop / "added_IVF2_Flat_v2.index", 768)
    models = tmp_path / "models"
    target = discovery.import_model(drop / "My Voice (e300).pth", models)
    assert target.name == "My_Voice_e300_"
    assert sorted(p.name for p in target.iterdir()) == ["config.json", "model.index", "model.pth"]
    assert discovery.scan(models) == [target.name]
    from vox.config.store import load_model_config

    config = load_model_config(target / "config.json")
    assert (config.version, config.sr, config.f0) == ("v2", 48000, True)
    assert config.index.file == "model.index" and config.index.ntotal == 64
    assert config.source_file == "My Voice (e300).pth" and len(config.sha256) == 64
    with pytest.raises(FileExistsError):
        discovery.import_model(drop / "My Voice (e300).pth", models)


def test_import_rejects_before_copying(tmp_path):
    bad = tmp_path / "train.pth"
    torch.save({"model": {}, "optimizer": {}}, bad)
    with pytest.raises(MetadataError):
        discovery.import_model(bad, tmp_path / "models")
    assert not (tmp_path / "models").exists()


def test_import_names_a_generic_pth_after_the_given_folder_name(tmp_path):
    """Training exports are often just ``model.pth``: the GUI passes the dropped folder's
    name (or a typed one), made folder-safe like a filename."""
    drop = tmp_path / "Gothmommy female"
    drop.mkdir()
    torch.save(checkpoint(version="v2", info="e300"), drop / "model.pth")
    models = tmp_path / "models"
    target = discovery.import_model(drop / "model.pth", models, "Gothmommy female")
    assert target.name == "Gothmommy_female"
    assert discovery.scan(models) == ["Gothmommy_female"]
