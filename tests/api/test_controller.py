"""EngineController against a fake session: commands, state document, events, persistence.

No audio device, GPU or TTS engine is touched: the session is replaced, the device list is
canned, and the models directory holds empty checkpoints with hand-written configs.
"""

import dataclasses
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from vox.api import controller as ctl
from vox.api.controller import ApiError, EngineController
from vox.api.events import METERS, PERF, STATE, TOAST
from vox.audio.devices import DeviceInfo
from vox.engine.override import ClipOverride

DEVICES = [
    DeviceInfo(3, "Mic A", "Windows WASAPI", 2, 0, 48000.0, 0.003, 0.0, default_input=True),
    DeviceInfo(4, "Mic B", "Windows WASAPI", 1, 0, 48000.0, 0.003, 0.0),
    DeviceInfo(7, "Speakers", "Windows WASAPI", 0, 2, 48000.0, 0.0, 0.003, default_output=True),
    DeviceInfo(8, "CABLE Input", "Windows WASAPI", 0, 2, 48000.0, 0.0, 0.001),
]


class FakeSession:
    """Just enough of ``LiveSession`` for the controller to drive."""

    created: ClassVar[list["FakeSession"]] = []
    fail_start: ClassVar[Exception | None] = None

    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.calls: list[tuple] = []
        self.warnings = ["cable not pinned"]
        self.started_at = 0.0
        self.stopped = False
        self.engine = SimpleNamespace(mode="graphs", pipeline=SimpleNamespace(extractor=object()))
        plan = SimpleNamespace(block_ms=cfg.shape.block_ms, extra_ms=cfg.shape.extra_ms)
        override = ClipOverride(960)
        self.worker = SimpleNamespace(
            params=cfg.params, plan=plan, hold=False, override=override,
            play_clip=override.submit,
            stats=SimpleNamespace(gate_rms_dbfs=-20.0, gate_open=True, gate_vad=0.9,
                                  input_gain_db=14.0))
        self.audio = SimpleNamespace(health=SimpleNamespace(input_peak=0.5, output_peak=0.5))
        self.out_ring = SimpleNamespace(underruns=0)
        self.errors: list[str] = []
        FakeSession.created.append(self)

    def start(self):
        if FakeSession.fail_start is not None:
            raise FakeSession.fail_start
        self.calls.append(("start",))

    def stop(self):
        self.stopped = True

    def set_params(self, **changes):
        self.calls.append(("params", changes))
        self.worker.params = dataclasses.replace(self.worker.params, **changes)

    def set_gate(self, **changes):
        self.calls.append(("gate", changes))

    def set_conditioning(self, **changes):
        self.calls.append(("conditioning", changes))

    def set_shape(self, **changes):
        self.calls.append(("shape", changes))

    def describe(self):
        return "fake live"

    def snapshot(self):
        row = {"p50": 7.0, "p95": 8.0, "p99": 9.0, "gpu_p50": 7.5, "gpu_p99": 8.5}
        return {"t": 12.0, "stages": {"f0": row, "sola": {"p50": 0.1, "p95": 0.2, "p99": 0.3}},
                "total": {"p50": 15.0, "p95": 16.0, "p99": 17.0},
                "latency": {"total": 120.0, "hop": 20.0}, "underruns": 2, "out_fill_ms": 40.0,
                "gate_open": True, "mode": "graphs", "hops": 100, "gated_hops": 10,
                "rebuilds": 0, "errors": list(self.errors), "headroom": float("nan")}


def write_index(path, dim, n=4):
    import faiss

    index = faiss.IndexFlatL2(dim)
    index.add(np.zeros((n, dim), dtype=np.float32))
    faiss.write_index(index, str(path))


def add_model(models, name, sr=48000, index=True):
    folder = models / name
    folder.mkdir(parents=True)
    (folder / "model.pth").write_bytes(b"")
    config = {"name": name, "version": "v2", "sr": sr, "f0": True, "emb_dim": 768, "n_spk": 1}
    if index:
        write_index(folder / "model.index", 768)
        config["index"] = {"file": "model.index", "dim": 768, "ntotal": 4}
    (folder / "config.json").write_text(json.dumps(config))


@pytest.fixture
def env(tmp_path, monkeypatch):
    models = tmp_path / "models"
    add_model(models, "Alpha")
    add_model(models, "Beta", sr=32000, index=False)
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "model_overrides.json").write_text(json.dumps(
        {"default_model": "Beta", "models": {"Alpha": {"f0_up_key": 7.0}}}))
    monkeypatch.setattr(ctl.audio_devices, "list_devices", lambda *a, **k: list(DEVICES))
    FakeSession.created.clear()
    FakeSession.fail_start = None
    made: list[EngineController] = []

    def make(**kwargs) -> EngineController:
        controller = EngineController(models, profile, session_factory=FakeSession,
                                      log=lambda *a: None, **kwargs)
        made.append(controller)
        settle(controller)
        return controller

    yield SimpleNamespace(make=make, models=models, profile=profile)
    for controller in made:
        controller.shutdown()


def settle(controller, timeout=10.0):
    """Wait until the control thread has run everything queued."""
    end = time.time() + timeout
    while time.time() < end:
        if controller._jobs.empty() and controller.busy is None:
            return
        time.sleep(0.01)
    raise TimeoutError(controller.busy)


def drain(q, kind):
    out = []
    while not q.empty():
        k, data = q.get_nowait()
        if k == kind:
            out.append(data)
    return out


# ------------------------------------------------------------------------------ state


def test_initial_state_document(env):
    state = env.make().state()
    assert state["engine"]["status"] == "stopped" and state["model"]["name"] == "Beta"
    assert [m["name"] for m in state["models"]] == ["Alpha", "Beta"]
    assert state["gate"] == {"enabled": True, "threshold_dbfs": -48.0, "noise_floor_dbfs": -60.0}
    assert state["shape"]["block_ms"] == 20.0 and state["profile"]["name"] == "profile"
    assert state["devices"]["input"]["name"] == "Mic A"
    assert state["devices"]["output"]["name"] == "Speakers"
    assert [d["name"] for d in state["devices"]["outputs"]] == ["Speakers", "CABLE Input"]
    assert state["devices"]["outputs"][1]["cable"] is True
    for stub in ("suppression", "monitor", "auto_tune"):
        assert state[stub] == {"enabled": False, "available": False}
    json.dumps(state)  # the whole document must be JSON


def test_refresh_reinitialises_portaudio_only_while_stopped(env, monkeypatch):
    calls = []
    monkeypatch.setattr(ctl.audio_devices, "reinitialize", lambda: calls.append(1))
    c = env.make()
    c.refresh_devices()
    settle(c)
    assert calls == [1]
    c.start()
    settle(c)
    c.refresh_devices()
    settle(c)
    assert calls == [1] and c.status == "running"  # refused, stream untouched


def test_saved_device_missing_from_fresh_list_is_shown_not_replaced(env):
    c = env.make(input="Gone Mic")
    devices = c.state()["devices"]
    assert devices["input"]["name"] == "Gone Mic" and devices["input"]["missing"] is True
    assert devices["inputs"][0]["missing"] is True
    assert devices["output"]["name"] == "Speakers"


def test_model_argument_wins_and_unknown_falls_back(env):
    assert env.make(model="Alpha").model == "Alpha"
    assert env.make(model="Nope").model == "Beta"


def test_pitch_is_layered_on_the_calibrated_value(env):
    c = env.make(model="Alpha")
    params = c.state()["params"]
    assert params["pitch"] == 7.0 and params["pitch_is_calibrated"] is True
    c.set_param("pitch", 9)
    params = c.state()["params"]
    assert params["pitch"] == 9.0 and params["calibrated_pitch"] == 7.0
    assert params["pitch_is_calibrated"] is False
    c.reset_pitch()
    assert c.state()["params"]["pitch_is_calibrated"] is True
    # the calibrated profile file itself is never rewritten
    saved = json.loads((env.profile / "model_overrides.json").read_text())
    assert saved["models"]["Alpha"] == {"f0_up_key": 7.0}


def test_index_options_mark_the_paired_and_the_mismatched(env):
    write_index(env.models / "Alpha" / "added_other.index", 768)
    write_index(env.models / "Alpha" / "wrong.index", 256)
    (env.models / "Alpha" / "broken.index").write_bytes(b"not an index")
    c = env.make(model="Alpha")
    options = {o["file"]: o for o in c.state()["model"]["index_options"]}
    assert options["model.index"]["label"] == "model.index (auto)"
    assert options["model.index"]["selected"] and options["added_other.index"]["selectable"]
    assert options["wrong.index"]["label"].endswith("(mismatch)")
    assert not options["wrong.index"]["selectable"] and not options["broken.index"]["selectable"]
    assert options[""]["label"] == "No index (off)"
    with pytest.raises(ApiError, match="does not match"):
        c.set_index("Alpha", "wrong.index")
    with pytest.raises(ApiError, match="no index file"):
        c.set_index("Alpha", "missing.index")
    c.set_index("Alpha", "")
    settle(c)
    assert c.state()["model"]["index"] == "" and c.registry.index_path("Alpha") is None
    c.set_index("Alpha", "added_other.index")
    settle(c)
    assert c.registry.index_path("Alpha").name == "added_other.index"


# -------------------------------------------------------------------------- lifecycle


def test_start_stop_and_what_the_session_is_given(env):
    c = env.make(model="Alpha", output="CABLE")
    c.set_param("protect", 0.2)
    c.start()
    settle(c)
    session = FakeSession.created[-1]
    assert c.state()["engine"]["status"] == "running" and c.state()["engine"]["mode"] == "graphs"
    assert c.state()["engine"]["warnings"] == ["cable not pinned"]
    cfg = session.cfg
    assert (cfg.model, cfg.output, cfg.input) == ("Alpha", "CABLE", None)
    assert cfg.params.f0_up_key == 7.0 and cfg.params.protect == 0.2
    assert cfg.registry is c.registry and cfg.cache is c.cache
    # an index exists and the rate is 0: it is loaded anyway, then set back to exactly 0
    assert cfg.params.index_rate == ctl.INDEX_PRIME
    assert ("params", {"index_rate": 0.0}) in session.calls
    c.stop()
    settle(c)
    assert session.stopped and c.state()["engine"]["status"] == "stopped"


def test_no_index_means_no_priming(env):
    c = env.make(model="Beta")
    c.start()
    settle(c)
    assert FakeSession.created[-1].cfg.params.index_rate == 0.0


def test_a_failed_start_is_reported_not_raised(env):
    c = env.make()
    FakeSession.fail_start = RuntimeError("device is busy")
    c.start()
    settle(c)
    engine = c.state()["engine"]
    assert engine["status"] == "error" and "device is busy" in engine["error"]
    assert FakeSession.created[-1].stopped and c.session is None
    FakeSession.fail_start = None
    c.start()
    settle(c)
    assert c.state()["engine"]["status"] == "running" and c.state()["engine"]["error"] is None


def test_hot_parameters_reach_the_running_session(env):
    c = env.make(model="Alpha")
    c.start()
    settle(c)
    session = FakeSession.created[-1]
    session.calls.clear()
    assert c.set_param("pitch", 99) == {"pitch": 24.0}  # clamped to the trained range
    c.set_param("index_rate", 0.5)
    c.set_gate(threshold_dbfs=-55)
    c.set_gate(enabled=False)
    assert session.calls == [
        ("params", {"f0_up_key": 24.0}), ("params", {"index_rate": 0.5}),
        ("gate", {"enabled": True, "threshold_dbfs": -55.0}),
        ("gate", {"enabled": False, "threshold_dbfs": -55.0})]
    assert len(FakeSession.created) == 1  # nothing restarted
    with pytest.raises(ApiError, match="unknown parameter"):
        c.set_param("volume", 1)


def test_input_level_defaults_on_is_hot_and_persists(env):
    c = env.make(model="Alpha")
    assert c.state()["input_level"]["auto"] is True and c.state()["input_level"]["gain_db"] == 0
    c.start()
    settle(c)
    session = FakeSession.created[-1]
    cfg = session.cfg.conditioning
    assert cfg.enabled and cfg.auto_level and cfg.trim_db == 0.0
    session.calls.clear()
    assert c.set_input_level(gain_db=40) == {"auto": True, "gain_db": 24.0}  # clamped
    c.set_input_level(auto=False)
    assert session.calls == [("conditioning", {"auto_level": True, "trim_db": 24.0}),
                             ("conditioning", {"auto_level": False, "trim_db": 24.0})]
    assert len(FakeSession.created) == 1  # hot: nothing restarted
    c.shutdown()
    again = env.make(model="Alpha")
    assert again.state()["input_level"] == {"auto": False, "gain_db": 24.0,
                                            "range_db": [-12.0, 24.0]}


def test_shape_changes_recapture_in_place_and_are_rounded(env):
    c = env.make(model="Alpha")
    c.start()
    settle(c)
    session = FakeSession.created[-1]
    assert c.set_shape(block_ms=34)["block_ms"] == 30.0  # the chunker's 10 ms units
    assert c.state()["shape"]["block_ms"] == 30.0 and c.state()["engine"]["recapturing"]
    c.set_shape(f0_method="fcpe", extra_ms=500)
    assert session.calls[-2:] == [("shape", {"f0_method": None, "block_ms": 30.0}),
                                  ("shape", {"f0_method": "fcpe", "extra_ms": 500.0})]
    assert len(FakeSession.created) == 1
    with pytest.raises(ApiError, match="unknown pitch tracker"):
        c.set_shape(f0_method="crepe")
    with pytest.raises(ApiError, match="does not work"):
        c.set_shape(block_ms=-10)


def test_restart_class_changes_reopen_the_session(env):
    c = env.make(model="Alpha")
    c.start()
    settle(c)
    c.set_device("output", "CABLE Input")
    settle(c)
    c.set_exclusive(True)
    settle(c)
    c.set_graphs(False)
    settle(c)
    c.select_model("Beta")
    settle(c)
    assert len(FakeSession.created) == 5 and all(s.stopped for s in FakeSession.created[:-1])
    cfg = FakeSession.created[-1].cfg
    assert (cfg.model, cfg.output, cfg.exclusive, cfg.precision.graphs) == \
        ("Beta", "CABLE Input", True, "off")
    assert c.state()["devices"]["output"]["name"] == "CABLE Input"
    with pytest.raises(ApiError, match="no model named"):
        c.select_model("Gamma")
    with pytest.raises(ApiError, match="device kind"):
        c.set_device("monitor", "x")


def test_changes_survive_a_restart_of_the_app(env):
    c = env.make(model="Alpha")
    c.set_param("pitch", 3)
    c.set_gate(threshold_dbfs=-52)
    c.set_shape(block_ms=40)
    c.set_tts(voice="am_adam")
    c.select_model("Alpha")
    settle(c)
    c.shutdown()
    again = env.make()
    state = again.state()
    assert state["model"]["name"] == "Alpha" and state["params"]["pitch"] == 3.0
    assert state["gate"]["threshold_dbfs"] == -52.0 and state["shape"]["block_ms"] == 40.0
    assert state["tts"]["voice"] == "am_adam"


# ---------------------------------------------------------------------------- commands


def test_command_dispatch_and_errors(env):
    c = env.make()
    assert c.command("get_state")["engine"]["status"] == "stopped"
    assert c.command("set_param", {"name": "protect", "value": 0.1}) == {"protect": 0.1}
    with pytest.raises(ApiError, match="unknown command"):
        c.command("format_disk")
    with pytest.raises(ApiError, match="bad arguments"):
        c.command("set_param", {"nome": "pitch"})
    for name in ("set_suppression", "set_monitor", "set_auto_tune"):
        with pytest.raises(ApiError, match="not built"):
            c.command(name, {"enabled": True})
    with pytest.raises(ApiError, match="Phase 5"):
        c.command("calibrate")
    with pytest.raises(ApiError, match="type something"):
        c.command("speak", {"text": "   "})
    with pytest.raises(ApiError, match="no longer kept"):
        c.command("replay", {"id": 5})
    assert set(ctl.COMMANDS.values()) <= set(dir(c))


def test_uploads_are_confined_and_typed(env):
    c = env.make()
    path = c.upload_path("..\\..\\evil\\voice.pth")
    assert path.parent == c._uploads and path.name == "voice.pth"
    with pytest.raises(ApiError, match="only .pth and .index"):
        c.upload_path("run.exe")
    toasts = c.bus.subscribe()
    c.import_model("never_uploaded.pth")
    settle(c)
    assert any("was not uploaded" in t["message"] for t in drain(toasts, TOAST))


def test_export_report_writes_json(env, tmp_path):
    c = env.make(reports_dir=tmp_path / "reports")
    c.start()
    settle(c)
    path = c.export_report()["path"]
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    assert report["model"] == "Beta" and report["describe"] == "fake live"
    assert report["snapshot"]["headroom"] is None  # NaN became null


# ------------------------------------------------------------------------------ events


def test_events_while_running(env):
    c = env.make(model="Alpha")
    events = c.bus.subscribe()
    c.start()
    settle(c)
    time.sleep(1.3)
    seen = []
    while not events.empty():
        seen.append(events.get_nowait())
    meters = [d for k, d in seen if k == METERS and d["running"]]
    perf = [d for k, d in seen if k == PERF]
    states = [d for k, d in seen if k == STATE]
    assert 10 <= len(meters) <= 25  # ~15 Hz
    assert meters[0]["in_db"] == -20.0 and meters[0]["gate_open"] is True
    assert meters[0]["in_gain_db"] == 14.0
    assert math.isclose(meters[0]["out_db"], 20 * math.log10(0.5), abs_tol=1e-6)
    assert 1 <= len(perf) <= 4  # ~2 Hz
    p = perf[-1]
    assert p["latency_ms"] == 120.0 and p["underruns"] == 2 and p["hop_ms"] == 20.0
    assert p["stages"] == [{"key": "f0", "label": "F0 RMVPE", "p50": 7.5, "p99": 8.5},
                           {"key": "sola", "label": "SOLA", "p50": 0.1, "p99": 0.3}]
    assert math.isclose(p["headroom_pct"], 15.0)
    assert [s["engine"]["status"] for s in states][-1] == "running"
    assert len(c.state()["perf_history"]) == len(perf)


def test_an_engine_error_stops_the_session_and_is_shown(env):
    c = env.make()
    c.start()
    settle(c)
    FakeSession.created[-1].errors.append("RuntimeError: CUDA out of memory")
    time.sleep(1.0)
    settle(c)
    engine = c.state()["engine"]
    assert engine["status"] == "error" and "out of memory" in engine["error"]
    assert FakeSession.created[-1].stopped


def test_speaking_state_follows_the_override(env):
    c = env.make()
    c.start()
    settle(c)
    worker = FakeSession.created[-1].worker
    assert c.state()["tts"]["speaking"] is False
    clip = np.zeros(4800, dtype=np.float32)
    c._clips[1] = clip
    c.replay(1)
    settle(c)
    assert worker.override.active and c.state()["tts"]["speaking"] is True
    c.cancel_speech()
    worker.override.mix(np.zeros(960, dtype=np.float32))
    assert c.state()["tts"]["speaking"] is False


def test_clean_replaces_non_finite_numbers():
    assert ctl._clean({"a": float("nan"), "b": [1.0, float("inf")], "c": "x"}) == \
        {"a": None, "b": [1.0, None], "c": "x"}
