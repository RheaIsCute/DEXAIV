"""Session recording: aligned tracks, float WAVs, the JSON beside them, the length cap."""

import json

import numpy as np
import soundfile as sf

from vox.audio.recorder import SessionRecorder

SR, BLOCK = 48000, 960


def test_tracks_stay_aligned_and_exact(tmp_path):
    rec = SessionRecorder(SR)
    for n in range(50):
        raw = np.full(BLOCK, n, np.float32) * 1e-3
        rec.add(raw, raw * 2, np.zeros(BLOCK, np.float32) if n % 10 == 0 else raw * 3, n % 10 != 0)
        raw[:] = 99  # the recorder copied: later reuse of the worker's buffers changes nothing
    assert rec.seconds == 50 * BLOCK / SR
    paths = rec.save(tmp_path, "take", {"model": "Mom"})
    out, sr = sf.read(str(paths["output"]), dtype="float32")
    raw, _ = sf.read(str(paths["mic_raw"]), dtype="float32")
    cond, _ = sf.read(str(paths["mic_conditioned"]), dtype="float32")
    assert sr == SR and len(out) == len(raw) == len(cond) == 50 * BLOCK
    assert sf.info(str(paths["output"])).subtype == "FLOAT"
    assert raw[7 * BLOCK] == np.float32(7e-3) and cond[7 * BLOCK] == np.float32(14e-3)
    assert not out[10 * BLOCK:11 * BLOCK].any()  # digital zeros survive exactly
    meta = json.loads(paths["meta"].read_text())
    assert meta["model"] == "Mom" and meta["gate_open_per_hop"][:11] == "01111111110"
    assert meta["files"]["output"] == "take_output.wav"


def test_take_stops_growing_at_the_cap(tmp_path):
    rec = SessionRecorder(SR, max_seconds=0.1)  # 5 hops
    for _ in range(20):
        rec.add(*(np.ones(BLOCK, np.float32),) * 3, True)
    assert rec.full and rec.samples == 5 * BLOCK
