"""The quality-metric plumbing of benchmarks/quality.py (Phase 0.1): WER alignment, F0
statistics, speaker-cosine summaries, blind pairs and the table. Pure CPU; the model-backed
scorers are stubbed."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

import quality
from quality import (
    Pair,
    QualityKit,
    Scores,
    TargetSet,
    cosine,
    f0_stats,
    load_script,
    speaker_summary,
    table,
    word_errors,
    words,
    write_pairs,
)

# ------------------------------------------------------------------------------ WER


def test_words_ignore_case_and_punctuation():
    assert words("She sells sea-shells; by the SEASHORE!") == \
        ["she", "sells", "sea", "shells", "by", "the", "seashore"]
    assert words("don’t") == ["don't"]  # curly apostrophes are plain ones


def test_word_errors_counts_each_edit_kind():
    ref = words("the quick brown fox jumps over the lazy dog")
    assert word_errors(ref, ref) == {"wer": 0.0, "substitutions": 0, "deletions": 0,
                                     "insertions": 0, "reference_words": 9,
                                     "hypothesis_words": 9}
    hyp = words("the quick brown fox jumped over lazy dog now")  # 1 sub, 1 del, 1 ins
    e = word_errors(ref, hyp)
    assert (e["substitutions"], e["deletions"], e["insertions"]) == (1, 1, 1)
    assert e["wer"] == pytest.approx(3 / 9)
    assert word_errors([], ["x"])["wer"] == float("inf")
    assert word_errors(ref, [])["wer"] == 1.0


def test_wer_matches_a_brute_force_edit_distance():
    rng = np.random.default_rng(1)
    vocab = list("abcdefg")
    for _ in range(40):
        ref = list(rng.choice(vocab, rng.integers(1, 9)))
        hyp = list(rng.choice(vocab, rng.integers(0, 9)))
        d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
        d[:, 0] = np.arange(len(ref) + 1)
        d[0, :] = np.arange(len(hyp) + 1)
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                              d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
        e = word_errors(ref, hyp)
        assert e["substitutions"] + e["deletions"] + e["insertions"] == d[-1, -1]


# ------------------------------------------------------------------------- F0 stats


def test_f0_stats_median_spread_voicing_and_flips():
    f0 = np.zeros(600)  # 6 s at 100 Hz
    f0[100:400] = 200.0 * 2 ** (np.linspace(-0.25, 0.25, 300))  # a half-octave glide
    f0[200] = 400.0  # one octave error inside the glide: two flips (up, back down)
    s = f0_stats(f0)
    assert s["median_hz"] == pytest.approx(200.0, rel=0.02)
    assert s["iqr_st"] == pytest.approx(3.0, abs=0.2)  # quartiles sit +-1.5 st around the median
    assert s["voiced_ratio"] == pytest.approx(0.5)
    assert s["voiced_s"] == pytest.approx(3.0)
    assert s["flips_per_min"] == pytest.approx(2 * 60 / 3.0)
    mask = np.zeros(600, dtype=bool)
    mask[100:400] = True
    assert f0_stats(f0, mask)["voiced_ratio"] == 1.0
    assert f0_stats(np.zeros(100))["median_hz"] == 0.0


def test_f0_stats_does_not_count_glides_as_flips():
    f0 = 100.0 * 2 ** (np.linspace(0, 1, 100))  # one octave over a second: 0.12 st per frame
    assert f0_stats(f0)["flips_per_min"] == 0.0


# --------------------------------------------------------------------- speaker cosine


def test_speaker_summary_centroid_min_and_self_similarity():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    out = speaker_summary(np.array([1.0, 1.0, 0.0]), [a, b])
    assert out["cos"] == pytest.approx(1.0)  # on the centroid
    assert out["cos_min"] == pytest.approx(np.sqrt(0.5))
    assert out["target_self"] == pytest.approx(0.0)
    assert cosine(a, 3 * a) == pytest.approx(1.0)
    assert all(np.isnan(v) for v in speaker_summary(a, []).values())


# ------------------------------------------------------------------------ blind pairs


def test_write_pairs_randomises_xy_and_keeps_the_key_apart(tmp_path):
    files = {}
    for name in ("ref", "v1", "v2"):
        p = tmp_path / f"{name}.wav"
        p.write_bytes(name.encode())
        files[name] = p
    pairs = [Pair("ref", "v1", "seams"), Pair("ref", "v2", "hiss")] * 6
    key = write_pairs(pairs, files, tmp_path / "pairs", seed=3, title="t")
    assert set(key["pairs"]) == {f"pair{n:02d}" for n in range(1, 13)}
    flips = []
    for name, entry in key["pairs"].items():
        for label in ("X", "Y"):
            assert (tmp_path / "pairs" / f"{name}_{label}.wav").read_bytes() == \
                entry[label].encode()
        flips.append(entry["X"] != "ref")
    assert any(flips) and not all(flips)  # X is sometimes the reference, sometimes not
    sheet = (tmp_path / "pairs" / "pairs.md").read_text()
    assert "pair01_X.wav" in sheet and "ref" not in sheet.split("\n", 4)[-1]  # no answers
    saved = json.loads((tmp_path / "pairs" / "answer_key.json").read_text())
    assert saved["pairs"] == key["pairs"]
    # the same seed gives the same assignment (a sheet can be regenerated)
    again = write_pairs(pairs, files, tmp_path / "pairs2", seed=3)
    assert [e["X"] for e in again["pairs"].values()] == [e["X"] for e in key["pairs"].values()]


# -------------------------------------------------------------------- kit and table


class StubEmbedder:
    backend = "stub"

    def embed(self, audio, sr):
        return np.array([float(np.mean(audio)), float(np.std(audio)), 1.0])


class StubTranscriber:
    def text(self, audio, sr):
        return "the quick brown fox"


class StubNaturalness:
    def score(self, audio, sr):
        return 3.5


def make_kit(tmp_path):
    kit = QualityKit(tmp_path / "models", f0_track=lambda a, sr: np.full(len(a) // 480, 180.0))
    kit.transcriber, kit.speaker, kit.naturalness = StubTranscriber(), StubEmbedder(), \
        StubNaturalness()
    kit._cache_dir = tmp_path / "cache"
    return kit


def test_kit_scores_everything_and_the_table_reads(tmp_path):
    kit = make_kit(tmp_path)
    targets = TargetSet("M", "proxy: vox say", [np.array([0.0, 1.0, 1.0]),
                                                 np.array([0.0, 1.2, 1.0])])
    audio = np.random.default_rng(0).standard_normal(48000).astype(np.float32)
    s = kit.score(audio, 48000, reference_text="The quick brown fox.", mic_wer=0.1,
                  targets=targets, seed_pair=(audio, audio))
    assert s.wer == 0.0 and s.wer_delta == pytest.approx(-0.1)
    assert s.target_source == "proxy: vox say" and 0 < s.speaker_cos <= 1
    assert s.seed_floor == pytest.approx(1.0, abs=1e-6)  # identical renders: the ceiling
    assert s.utmos == 3.5 and s.f0["median_hz"] == 180.0
    out = table({"reference": s, "variant": Scores()}, extra=(("click", "click", "{:.2f}"),))
    assert "| reference | 0.0% | -10.0% |" in out and "| variant | - | - |" in out
    assert "microphone WER against the same script: 10.0%" in out
    assert "proxy: vox say" in out
    d = s.as_dict()
    assert d["wer_delta"] == s.wer_delta and "f0" in d


def test_targets_prefer_held_out_recordings_and_cache_their_embeddings(tmp_path):
    import soundfile as sf

    kit = make_kit(tmp_path)
    folder = tmp_path / "target" / "M"
    folder.mkdir(parents=True)
    sf.write(str(folder / "a.wav"), np.full(16000, 0.5, np.float32), 16000)
    sf.write(str(folder / "b.wav"), np.full(16000, 0.25, np.float32), 16000)
    calls = []
    ts = kit.targets("M", proxy=lambda: calls.append(1) or [], targets_dir=folder)
    assert ts.source == "held-out" and len(ts.embeddings) == 2 and not calls
    assert len(list((tmp_path / "cache" / "targets").glob("*.npz"))) == 2
    kit.speaker = None  # cached: the embedder is not needed again
    assert kit.targets("M", targets_dir=folder).embeddings[0].tolist() == \
        ts.embeddings[0].tolist()
    kit.speaker = StubEmbedder()
    proxied = kit.targets("M", proxy=lambda: [(np.ones(8000, np.float32), 16000)],
                          targets_dir=tmp_path / "missing")
    assert proxied.source == "proxy: vox say" and proxied.available
    assert not kit.targets("M", targets_dir=tmp_path / "missing").available


def test_kit_with_nothing_to_compare_against_leaves_nans(tmp_path):
    kit = make_kit(tmp_path)
    s = kit.score(np.zeros(4800, np.float32), 48000)
    assert np.isnan(s.wer) and np.isnan(s.speaker_cos) and s.target_source == "none"
    assert s.utmos == 3.5 and s.f0["voiced_ratio"] == 1.0


def test_load_script_drops_comments_and_joins_lines(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("# a comment\nOne line.\n\nTwo lines.\n", encoding="utf-8")
    assert load_script(p) == "One line. Two lines."
    script = load_script(ROOT / "benchmarks" / "corpus" / "speech_script.txt")
    assert 120 <= len(words(script)) <= 220 and not any(ch.isdigit() for ch in script)


def test_quality_never_offers_the_metrics_the_report_rules_out():
    source = (ROOT / "benchmarks" / "quality.py").read_text(encoding="utf-8").lower()
    body = source.split('"""', 2)[2]  # past the module docstring, which names them
    for banned in ("mcd", "pesq", "nisqa", "f0_rmse"):
        assert banned not in body
    assert quality.FLIP_SEMITONES >= 6.0
