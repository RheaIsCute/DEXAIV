"""The raw-mic take protocol (benchmarks/matrix.py) and the per-model register estimators
(benchmarks/target_f0.py): plans, labels, thresholds, pairs, take analysis on a synthetic
take; bucket inversion, embedding movement, fidelity. CPU only; scorers stubbed."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

import matrix as mx
import target_f0 as tf
from quality import Scores
from vox.pitch.base import f0_to_coarse

# ------------------------------------------------------------------------------ plan


def test_plan_covers_every_cell_with_one_session_per_input_and_one_rebuild_per_hop():
    runs = mx.plan_runs()
    assert len(runs) == 16 and len({r.label for r in runs}) == 16
    assert runs[0].label == "shure_gateoff_agcoff_hop20"
    # ordered input -> hop -> (gate, agc): the device changes once, the hop twice
    keys = [r.input_key for r in runs]
    assert keys == sorted(keys, key=keys.index) and keys[:8] == ["shure"] * 8
    hops = [r.hop_ms for r in runs if r.input_key == "shure"]
    assert hops == [20.0] * 4 + [40.0] * 4
    assert mx.plan_runs({"x": "Dev"}, gates=(True,), agcs=(True,), hops=(20,))[0].label == \
        "x_gateon_agcon_hop20"


def test_gate_threshold_follows_the_auto_level_state():
    on = mx.Run("shure", "Shure MV6", True, True, 20.0)
    off = mx.Run("shure", "Shure MV6", True, False, 20.0)
    assert on.gate_threshold() == -40.0 and off.gate_threshold() == -55.0
    assert off.gate_threshold(-38.0, -50.0) == -50.0
    assert "auto level OFF" in off.describe() and "hop 20 ms" in off.describe()


def test_default_pairs_change_one_thing_each_and_skip_missing_runs():
    runs = mx.plan_runs()
    pairs = mx.default_pairs(runs)
    names = {(p.a, p.b) for p in pairs}
    assert ("shure_gateon_agcon_hop20", "shure_gateoff_agcon_hop20") in names
    assert ("shure_gateon_agcon_hop20", "shure_gateon_agcoff_hop20") in names
    assert ("shure_gateon_agcon_hop20", "shure_gateon_agcon_hop40") in names
    assert ("shure_gateoff_agcoff_hop20", "shure_gateon_agcon_hop20") in names
    assert ("shure_gateon_agcon_hop20", "broadcast_gateon_agcon_hop20") in names
    assert len(pairs) == 9
    for p in pairs:
        a, b = p.a.split("_"), p.b.split("_")
        changed = sum(x != y for x, y in zip(a, b))
        cross = p.a.startswith("shure") and p.b.startswith("broadcast")
        assert changed == 1 or changed == 2 and "everything" in p.what or cross
    # only the shure hop-20 cells recorded: the hop and the cross-input pairs drop out
    partial = [r for r in runs if r.input_key == "shure" and r.hop_ms == 20.0]
    assert {p.what.split(":")[1].strip()[:4] for p in mx.default_pairs(partial)} == \
        {"gate", "auto", "ever"}


# --------------------------------------------------------------------- take analysis


class StubKit:
    enable = frozenset({"wer", "speaker", "utmos", "f0"})

    def __init__(self):
        self.transcribed = []

    def transcript_of(self, audio, sr):
        self.transcribed.append(len(audio))
        return "the quick brown fox"

    def score(self, audio, sr, *, reference_text=None, mic_wer=float("nan"), targets=None,
              seed_pair=None, mask=None):
        return Scores(wer=0.25, wer_mic=mic_wer, utmos=3.0, f0={"median_hz": 200.0})


def synthetic_take(folder: Path, label: str, sr: int = 48000, hop_ms: float = 20.0):
    """A 4 s take: two 1 s phrases of a 150 Hz tone on the mic; the output follows but the
    gate closed 200 ms into the second phrase (a cut) and the stream trails by 30 ms."""
    n = 4 * sr
    t = np.arange(n) / sr
    mic = np.zeros(n, np.float32)
    for start in (0.5, 2.5):
        s, e = int(start * sr), int((start + 1.0) * sr)
        mic[s:e] = 0.3 * np.sin(2 * np.pi * 150 * t[s:e])
    out = np.zeros(n, np.float32)
    delay = int(0.03 * sr)
    out[delay:] = 0.3 * np.sin(2 * np.pi * 220 * t[:-delay]) * (mic[:-delay] != 0)
    cut_s, cut_e = int((2.5 + 0.2) * sr) + delay, int((2.5 + 0.4) * sr) + delay
    out[cut_s:cut_e] = 0.0
    block = int(sr * hop_ms / 1000)
    hops = n // block
    meta = {"model": "Mom", "params": {}, "shape": {}, "gate": {}, "conditioning": {},
            "plan": {"hop_ms": hop_ms}, "seconds": 4.0, "stream_delay_ms": 30.0,
            "run": {"input_key": label.split("_")[0],
                    "input_spec": {"shure": "Shure MV6", "broadcast": "NVIDIA Broadcast"}
                    [label.split("_")[0]], "gate": "gateon" in label, "agc": "agcon" in label,
                    "hop_ms": hop_ms},
            "events": [{"kind": "gate_open", "hop": 25, "t": 0.5, "value": -30.0},
                       {"kind": "gate_close", "hop": 95, "t": 1.9, "value": -60.0},
                       {"kind": "gate_open", "hop": 125, "t": 2.5, "value": -30.0},
                       {"kind": "gate_close", "hop": 135, "t": 2.7, "value": -60.0},
                       {"kind": "gate_open", "hop": 145, "t": 2.9, "value": -30.0}],
            "event_counts_in_take": {"gate_open": 3, "gate_close": 2, "underrun": 0},
            "gate_open_per_hop": "1" * hops}
    for track, audio in (("output", out), ("mic_raw", mic), ("mic_conditioned", mic)):
        sf.write(str(folder / f"{label}_{track}.wav"), audio, sr, subtype="FLOAT")
    (folder / f"{label}.json").write_text(json.dumps(meta), encoding="utf-8")


def test_analyze_take_reads_gate_behaviour_and_scores_the_output(tmp_path):
    label = "shure_gateon_agcon_hop20"
    synthetic_take(tmp_path, label)
    kit = StubKit()
    scores, info = mx.analyze_take(tmp_path, label, kit, "The quick brown fox.")
    g = scores.extra
    assert g["openings"] == 3 and g["closures"] == 2 and g["phrases"] == 2
    assert g["closes_in_phrases"] == 1  # the close at 2.7 s sits inside the second phrase
    # the 200 ms cut, aligned by the 30 ms delay, plus the 40 ms analysis windows' edges
    assert 180 <= g["speech_lost_ms"] <= 300
    assert g["openings_per_min"] == pytest.approx(45.0)
    assert scores.wer_mic == 0.0 and scores.wer == 0.25 and "click" in g
    assert kit.transcribed  # the mic was transcribed here because none was given
    assert info["label"] == label and info["meta"]["plan"] == {"hop_ms": 20.0}
    assert mx.recorded_runs(tmp_path) == [mx.Run("shure", "Shure MV6", True, True, 20.0)]


def test_analyze_matrix_transcribes_each_microphone_once(tmp_path):
    for label in ("shure_gateon_agcon_hop20", "shure_gateoff_agcon_hop20"):
        synthetic_take(tmp_path, label)
    kit = StubKit()
    rows = mx.analyze_matrix(tmp_path, kit, "the quick brown fox", log=lambda *_: None)
    assert set(rows) == {"shure_gateon_agcon_hop20", "shure_gateoff_agcon_hop20"}
    assert len(kit.transcribed) == 1  # one mic transcript for the shared microphone
    assert all(r.wer_mic == 0.0 for r in rows.values())


def test_analyze_without_a_kit_still_reports_the_gate(tmp_path):
    label = "broadcast_gateoff_agcoff_hop40"
    synthetic_take(tmp_path, label, hop_ms=40.0)
    scores, _ = mx.analyze_take(tmp_path, label, None, None)
    assert np.isnan(scores.wer) and scores.extra["openings"] == 3


# ------------------------------------------------------------------------- target F0


def test_bucket_to_hz_inverts_the_coarse_pitch_mapping():
    import torch

    hz = np.array([60.0, 100.0, 200.0, 440.0, 1000.0])
    buckets = f0_to_coarse(torch.from_numpy(hz)).numpy()
    back = tf.bucket_to_hz(buckets)
    assert np.all(np.abs(12 * np.log2(back / hz)) < 0.35)  # within a bucket's width


def test_embedding_register_finds_where_the_table_moved():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((256, 192))
    fine = base.copy()
    import torch

    centre = int(f0_to_coarse(torch.tensor([220.0])).item())
    for b in range(centre - 6, centre + 7):  # ~220 Hz +- half an octave trained
        fine[b] += rng.standard_normal(192) * 0.5
    reg = tf.embedding_register(fine, base)
    assert 150 < reg["median_hz"] < 300 and reg["p10_hz"] < reg["median_hz"] < reg["p90_hz"]
    assert reg["peak_ratio"] > 3.0
    assert tf.embedding_register(base, base) is None  # nothing moved
    assert tf.embedding_register(base[:10], base[:10]) is None


def test_fidelity_and_its_cost_prefer_the_faithful_shift():
    f0_in = np.zeros(1000)
    f0_in[200:800] = 200.0
    good = f0_in * 2 ** (4 / 12)
    bad = good.copy()
    bad[300:500] = 0.0  # voicing lost
    bad[600] = good[600] * 2  # an octave flip
    rg, rb = tf.fidelity(f0_in, 2 ** (4 / 12), good), tf.fidelity(f0_in, 2 ** (4 / 12), bad)
    assert rg["cents_error"] == pytest.approx(0.0, abs=1e-6) and rg["voicing_kept"] == 1.0
    assert rb["voicing_kept"] < 1.0 and rb["flips_per_min"] > 0
    assert tf.fidelity_cost(rg) < tf.fidelity_cost(rb)
    assert tf.fidelity_cost({"cents_error": float("nan"), "voicing_kept": 1.0,
                             "flips_per_min": 0.0}) == float("inf")


def test_calibration_writes_the_rendered_value_and_keeps_the_rest():
    cal = tf.Calibration("Mom", "af_heart", "speech_script.txt", rendered_hz=201.4,
                         tts_hz=198.0, sweep_best_st=4.0, sweep_hz=250.0,
                         embedding={"median_hz": 180.0, "p10_hz": 120.0, "p90_hz": 260.0,
                                    "peak_ratio": 5.0})
    d = cal.as_defaults()
    assert d["target_f0_median_hz"] == 201.4
    assert d["target_f0_calibration"]["sweep_hz"] == 250.0
    dis = cal.disagreement_st()
    assert dis["sweep"] == pytest.approx(12 * np.log2(250 / 201.4))
    assert dis["embedding"] == pytest.approx(12 * np.log2(180 / 201.4))
    assert "Mom" in tf.table([cal]) and "201 Hz" in tf.table([cal])
    assert "shift" in tf.sweep_table(cal)
