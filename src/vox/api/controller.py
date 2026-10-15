"""EngineController: every command a UI can send, and the state it gets back.

Control path only (docs/research/12 Â§3.12, docs/gui/GUI_SPEC.md). The controller wraps the
same ``LiveSession`` that ``vox run`` uses and owns nothing the engine does not already
own: it resolves parameters from the profile, starts and stops sessions, forwards hot
parameters, and publishes three event streams on an ``EventBus`` (state on change, meters
at ~15 Hz, profiler snapshots at ~2 Hz). A UI holds no engine state of its own.

Commands are plain methods, also reachable by name through ``command`` (that is what the
local HTTP transport calls). Quick ones (a slider) apply at once; slow ones (start, model
switch, rendering a typed clip, importing a model) run in order on one control thread and
report progress through the state document.

What the user changes is kept in ``gui_state.json`` next to the profile files, layered on
top of them: the calibrated profile is never rewritten from here.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import queue
import shutil
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from vox.api.control import BROADCAST_MARKER, LiveConfig, LiveSession, resolve_suppression
from vox.api.events import METERS, PERF, STATE, TOAST, EventBus
from vox.audio import devices as audio_devices
from vox.audio import suppression as rnnoise
from vox.audio.conditioning import ConditioningConfig
from vox.audio.resample import resample_array
from vox.config.schemas import RuntimeParams, ShapeParams
from vox.config.store import load_json, load_profile, resolve_runtime_params, save_json
from vox.engine import chunker
from vox.engine.gate import GateConfig
from vox.models import discovery
from vox.models.metadata import MetadataError
from vox.models.registry import LoadedModel, ModelRegistry
from vox.pitch import EXTRACTORS

STATE_FILE = "gui_state.json"
TTS_DIR = "_tts"
METER_HZ = 15.0
PERF_EVERY = 8  # meter ticks between profiler snapshots (~2 Hz)
PERF_HISTORY = 120  # 60 s of profiler snapshots
CLIP_HISTORY = 20
CLIP_FADE_MS = 5.0
HOLD_SETTLE_HOPS = 4  # hops between closing the live voice and starting a clip's GPU render
METER_FLOOR_DB = -120.0
INDEX_PRIME = 1e-6  # see _start: keeps the index loaded so its rate stays a hot parameter
PREVIEW_TEXT = "Hi, this is a quick preview of this voice."
STAGE_LABELS = {"f0": "F0", "embed": "ContentVec", "index": "Index kNN", "synth": "Synth",
                "sola": "SOLA"}
HOT_PARAMS = {"pitch": "f0_up_key", "index_rate": "index_rate", "protect": "protect",
              "ceiling": "f0_ceiling_hz", "target_hz": "target_f0_hz",
              "pitch_range": "pitch_range"}
PARAM_RANGES = {"pitch": (-24.0, 24.0), "index_rate": (0.0, 1.0), "protect": (0.0, 0.5),
                "ceiling": (0.0, 1100.0),  # ceiling 0 = off
                "target_hz": (60.0, 600.0), "pitch_range": (0.0, 1.5)}
INPUT_GAIN_RANGE = (-12.0, 24.0)  # the input-gain trim slider, dB
# Speak / Sing presets, per model (docs/DEVIATIONS.md §8). Each keeps its own pitch, index
# rate, protect and F0 ceiling; Speak is what the model entry always held. Sing starts from
# these until the user moves a slider in Sing: the Speak pitch shifted down (a sung line
# sits 5-9 semitones above the speaking voice), less index (speech vectors smear sung
# vowels), more protect, and the soft F0 ceiling on. Sung notes also get a gentler input
# chain: auto level boosts at most +10 dB and the trim slider is out (0 dB).
PRESETS = ("speak", "sing")
SING_PITCH_OFFSET = -5.0  # semitones from the Speak pitch; ``sing_pitch_offset`` overrides
SING_DEFAULTS = {"index_rate": 0.3, "protect": 0.35, "ceiling": 360.0}
SING_MAX_GAIN_DB = 10.0
BROADCAST_SING_WARNING = ("Broadcast's noise removal can cut sustained notes — use your mic "
                          "directly or turn its noise removal off for singing.")


class ApiError(Exception):
    """A command that cannot be carried out; the message is shown to the user as is."""


class _Registry(ModelRegistry):
    """The model registry plus the user's index choice per model.

    ``choice[name]`` is a file name in the model's folder, ``""`` for no index, or absent
    for the index paired at import.
    """

    def __init__(self, models_dir: Path) -> None:
        super().__init__(models_dir)
        self.choice: dict[str, str] = {}

    def index_path(self, name: str) -> Path | None:
        chosen = self.choice.get(name)
        if chosen is None:
            chosen = self.config(name).index.file
        path = self.models_dir / name / chosen if chosen else None
        return path if path is not None and path.is_file() else None

    def get(self, name: str, device) -> LoadedModel:
        loaded = super().get(name, device)
        loaded.index_path = self.index_path(name)
        return loaded


def _clean(value: Any) -> Any:
    """JSON has no NaN or infinity: turn them into null, recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _reveal_in_explorer(path: Path) -> None:
    """Open a File Explorer window on ``path``'s folder with the file selected."""
    import subprocess

    subprocess.Popen(f'explorer /select,"{Path(path)}"')


def _db(amplitude: float) -> float:
    return 20.0 * math.log10(amplitude) if amplitude > 1e-6 else METER_FLOOR_DB


class EngineController:
    def __init__(self, models_dir: Path, profile_dir: Path, *, model: str | None = None,
                 input: str | int | None = None, output: str | int | None = None,
                 device: str | None = None, input_file: Path | None = None,
                 loop_file: bool = False, reports_dir: Path | None = None,
                 bus: EventBus | None = None, session_factory=LiveSession, log=print,
                 recordings_dir: Path | None = None, reveal=None) -> None:
        self.models_dir = Path(models_dir)
        self.profile_dir = Path(profile_dir)
        self.reports_dir = Path(reports_dir) if reports_dir else self.profile_dir / "reports"
        # Takes of the Record button (mic + output WAVs); ``reveal`` shows a saved file
        # (default: Windows Explorer with the file selected).
        self.recordings_dir = (Path(recordings_dir) if recordings_dir
                               else self.models_dir.parent / "recordings")
        self._reveal = reveal or _reveal_in_explorer
        self._last_take: str | None = None
        self.device = device
        self.input_file = input_file
        self.loop_file = loop_file
        self.bus = bus or EventBus()
        self.session_factory = session_factory
        self._print = log

        self.profile = load_profile(self.profile_dir)
        self.registry = _Registry(self.models_dir)
        self.cache: dict = {}  # embedder / extractors / VAD, shared by every session
        self.store = self._load_store()
        for name, entry in self.store["models"].items():
            if "index" in entry:
                self.registry.choice[name] = entry["index"]
        if input is not None:
            self.store["input"] = str(input)
        if output is not None:
            self.store["output"] = str(output)

        self.session: LiveSession | None = None
        self.status = "stopped"  # stopped | starting | running | stopping | error
        self.error: str | None = None
        self.busy: str | None = None  # what the control thread is doing, for the UI
        self.warnings: list[str] = []
        self.log_lines: deque[str] = deque(maxlen=200)
        self.perf: dict[str, Any] | None = None
        self.perf_history: deque[dict[str, Any]] = deque(maxlen=PERF_HISTORY)
        self._rebuilds_seen = 0
        self._recapturing = False
        self._warned_broadcast_sing = False  # the toast is shown once per app run

        self._tts = None
        self._tts_speaking_until = 0.0
        self._tts_last: dict[str, Any] | None = None
        self._clips: dict[int, np.ndarray] = {}
        self._history: deque[dict[str, Any]] = deque(maxlen=CLIP_HISTORY)
        self._clip_id = 0
        self._offline: tuple[Any, Any] | None = None  # (key, engine) for stopped-mode TTS
        self._side_stream = None
        self._uploads = Path(tempfile.mkdtemp(prefix="vox_upload_"))
        self._models_cache: list[dict[str, Any]] | None = None
        self._index_meta: dict[tuple, tuple[int, int] | None] = {}

        names = self.registry.names()
        wanted = (model, self.store.get("model"), self.profile.model_overrides.default_model)
        self.model: str | None = next((name for name in wanted if name in names),
                                      names[0] if names else None)

        self._lock = threading.RLock()
        self._pa_lock = threading.RLock()
        self._device_cache: list = []
        self._device_error: str | None = None
        self._jobs: queue.Queue = queue.Queue()
        self._closing = threading.Event()
        self._job_thread = threading.Thread(target=self._job_loop, name="vox-control",
                                            daemon=True)
        self._pub_thread = threading.Thread(target=self._publish_loop, name="vox-events",
                                            daemon=True)
        self._job_thread.start()
        self._pub_thread.start()
        self._submit("Scanning devices", self._refresh_devices)

    # ------------------------------------------------------------------ persistence

    def _load_store(self) -> dict[str, Any]:
        """What the user changed in the window. ``None`` / absent = not changed: the value
        then comes from the profile and the engine defaults, never from a saved copy."""
        store: dict[str, Any] = {"model": None, "models": {}, "input": None, "output": None,
                                 "exclusive": None, "gate": {}, "shape": {},
                                 "f0_method": None, "graphs": None, "tts": {},
                                 "input_level": {}, "suppression": {}}
        path = self.profile_dir / STATE_FILE
        if path.is_file():
            try:
                store.update(load_json(path))
            except (OSError, ValueError) as err:
                self._log(f"ignoring unreadable {path.name}: {err}")
        return store

    def _save_store(self) -> None:
        """Only the user's choices are written: unset (None) keys and empty groups are left
        out, so a new engine default is never shadowed by a stale saved value."""
        def compact(value: Any) -> Any:
            if isinstance(value, dict):
                kept = {k: compact(v) for k, v in value.items()}
                return {k: v for k, v in kept.items() if v is not None and v != {}}
            return value

        try:
            save_json(self.profile_dir / STATE_FILE, compact(self.store))
        except OSError as err:
            self._log(f"could not save {STATE_FILE}: {err}")

    def _model_store(self, name: str) -> dict[str, Any]:
        return self.store["models"].setdefault(name, {})

    # ------------------------------------------------------------- resolved settings

    def _config(self, name: str | None = None):
        name = name or self.model
        if name is None:
            raise ApiError("no voice model is imported yet")
        try:
            return self.registry.config(name)
        except MetadataError as err:
            raise ApiError(str(err)) from err

    def _preset(self, name: str | None = None) -> str:
        name = name or self.model
        entry = self.store["models"].get(name, {}) if name else {}
        return "sing" if entry.get("preset") == "sing" else "speak"

    def _preset_store(self, name: str) -> dict[str, Any]:
        """Where the active preset's slider values live: the model entry itself for Speak
        (as before presets existed), its ``sing`` sub-entry for Sing."""
        entry = self._model_store(name)
        return entry.setdefault("sing", {}) if self._preset(name) == "sing" else entry

    def _sing_pitch(self, speak_pitch: float) -> float:
        offset = self.store.get("sing_pitch_offset")
        offset = SING_PITCH_OFFSET if offset is None else float(offset)
        low, high = PARAM_RANGES["pitch"]
        return min(high, max(low, speak_pitch + offset))

    def _params(self, name: str | None = None, calibrated: bool = False) -> RuntimeParams:
        """Profile + model config, then what the user changed here (unless ``calibrated``),
        then the Sing preset when it is the model's active one."""
        config = self._config(name)
        overrides: dict[str, Any] = {"f0_method": self.store.get("f0_method")}
        entry = self.store["models"].get(config.name, {})
        if not calibrated:
            overrides.update({field: entry.get(key) for key, field in HOT_PARAMS.items()})
        params = resolve_runtime_params(self.profile, config, overrides)
        if entry.get("target_hz") is None:  # the model's register, as vox say measures it
            from vox.tts.speak import target_f0_hint

            params = dataclasses.replace(params, target_f0_hz=target_f0_hint(self.profile, config))
        if not calibrated and entry.get("auto_pitch"):
            params = dataclasses.replace(params, pitch_mode="auto")
        if calibrated or self._preset(config.name) != "sing":
            return params
        sing = {**SING_DEFAULTS, "pitch": self._sing_pitch(params.f0_up_key),
                "target_hz": params.target_f0_hz, "pitch_range": params.pitch_range,
                **entry.get("sing", {})}
        return dataclasses.replace(params, **{field: float(sing[key])
                                              for key, field in HOT_PARAMS.items()})

    def _pitch_default(self, params: RuntimeParams) -> float:
        """What "back to default" sets the pitch to: the calibrated value in Speak, the
        Speak pitch shifted by the Sing offset in Sing."""
        if self._preset() == "sing":
            entry = self.store["models"].get(self._config().name, {})
            speak = entry.get("pitch")
            if speak is None:
                speak = self._params(calibrated=True).f0_up_key
            return self._sing_pitch(speak)
        return self._params(calibrated=True).f0_up_key

    def _shape(self) -> ShapeParams:
        shape = self.profile.runtime.shape()
        user = {k: v for k, v in self.store["shape"].items() if k in ("block_ms", "extra_ms")}
        if "block_ms" in user:  # saved before hops snapped to 20 ms (docs/DEVIATIONS.md §8)
            user["block_ms"] = chunker.snap_block_ms(user["block_ms"])
        return dataclasses.replace(shape, **user)

    def _precision(self):
        precision = self.profile.runtime.precision()
        if self.store.get("graphs"):
            precision = dataclasses.replace(precision, graphs=self.store["graphs"])
        return precision

    def _gate(self) -> GateConfig:
        cal = self.profile.calibration
        user = self.store["gate"]
        return GateConfig(enabled=user.get("enabled", True),
                          threshold_dbfs=user.get("threshold_dbfs", cal.silence_gate_dbfs),
                          vad_threshold=cal.vad.threshold, hangover_ms=cal.vad.release_ms,
                          use_vad=cal.vad.engine == "silero")

    def _conditioning(self) -> ConditioningConfig:
        """The mic input chain: auto level (on unless switched off here) and the input-gain
        slider as a trim on top of calibration's input gain. The Sing preset caps the auto
        level's boost and leaves the trim out, so held notes do not ride the limiter."""
        cal = self.profile.calibration
        user = self.store["input_level"]
        sing = self.model is not None and self._preset() == "sing"
        config = ConditioningConfig(auto_level=user.get("auto", True),
                                    trim_db=cal.input_gain_db + (
                                        0.0 if sing else user.get("gain_db", 0.0)),
                                    noise_floor_dbfs=cal.noise_floor_dbfs,
                                    suppression_mix=self._suppression_mix())
        if sing:
            config = dataclasses.replace(config, max_gain_db=SING_MAX_GAIN_DB)
        return config

    # suppression (RNNoise) -----------------------------------------------------------

    def _input_name(self) -> str | None:
        """The microphone in use (or about to be): the session's, else the selected one."""
        session = self.session
        if session is not None and getattr(session, "in_device", None) is not None:
            return session.in_device.name
        current = self._devices().get("input")
        return current["name"] if current else self.store.get("input")

    def _suppression_mix(self) -> float:
        mix = self.store["suppression"].get("mix")
        return 1.0 if mix is None else float(mix)

    def _suppression_choice(self, input_name: str | None) -> bool | None:
        """The user's explicit on/off for this microphone, None if never switched. Kept per
        input so the NVIDIA Broadcast mic keeps its own default (off, TARGET.md)."""
        return self.store["suppression"].get("by_input", {}).get(input_name or "")

    def _suppression_state(self) -> dict[str, Any]:
        reason = rnnoise.unavailable_reason(self._shape().device_sr)
        session = self.session
        name = self._input_name()
        if session is not None and self.status == "running":
            enabled, why = session.suppression, getattr(session, "suppression_why", "")
        else:
            enabled, why = resolve_suppression(self._suppression_choice(name),
                                               self.profile.calibration.rnnoise_enabled, name)
        why = {"explicit": "your choice for this microphone",
               "profile": "the profile's default"}.get(why, why)
        return {"enabled": bool(enabled) and reason is None, "available": reason is None,
                "reason": reason or why, "mix": self._suppression_mix()}

    def _warn_broadcast_sing(self, input_name: str | None = None) -> None:
        """Once per run: Broadcast's own noise removal and sustained notes do not mix."""
        if self._warned_broadcast_sing or self.model is None or self._preset() != "sing":
            return
        name = input_name or self._input_name() or ""
        if BROADCAST_MARKER in name.lower():
            self._warned_broadcast_sing = True
            self.toast(BROADCAST_SING_WARNING, "warn")

    # ------------------------------------------------------------------------ state

    def _log(self, message: str) -> None:
        self.log_lines.append(str(message))
        self._print(message)

    def toast(self, message: str, kind: str = "info") -> None:
        self.bus.publish(TOAST, {"message": message, "kind": kind})

    def _refresh_devices(self) -> None:
        """Re-read the device list.

        Every PortAudio call made from here (enumeration, opening and closing streams, clip
        playback) runs on the control thread, and state documents are built from this
        cached copy. PortAudio's WASAPI backend sets COM up on the thread that first
        touches it; opening a stream from any other thread then fails with an
        "unanticipated host error" (docs/DEVIATIONS.md, G5).
        """
        with self._pa_lock:
            try:
                self._device_cache = audio_devices.list_devices()
                self._device_error = None
            except Exception as err:  # noqa: BLE001
                self._device_cache, self._device_error = [], str(err)

    def _rescan_devices(self) -> None:
        """Re-initialise PortAudio so newly plugged devices appear.

        Refused while audio is open: terminating PortAudio under a running stream would
        kill it, so the user stops first (the GUI says so).
        """
        if self.session is not None or self.status in ("starting", "running", "stopping"):
            raise ApiError("Stop the engine before refreshing devices")
        if self._speaking():
            raise ApiError("Wait for the clip to finish before refreshing devices")
        with self._pa_lock:
            audio_devices.reinitialize()
        self._refresh_devices()

    def refresh_devices(self) -> dict[str, Any]:
        return self._submit("Scanning devices", self._rescan_devices)

    def _devices(self) -> dict[str, Any]:
        found = self._device_cache
        if self._device_error:
            return {"error": self._device_error, "inputs": [], "outputs": [], "input": None,
                    "output": None}
        ins = [d for d in found if d.max_input_channels > 0]
        outs = [d for d in found if d.max_output_channels > 0]

        def entry(d) -> dict[str, Any]:
            return {"index": d.index, "name": d.name, "cable": d.is_cable,
                    "default": d.default_input or d.default_output}

        def missing(kind: str, pool) -> dict[str, Any] | None:
            """The saved device when it is absent from the fresh list, else None."""
            wanted = self.store.get(kind)
            if wanted is None or any(d.name == wanted or str(d.index) == str(wanted)
                                     for d in pool):
                return None
            return {"index": None, "name": str(wanted), "cable": False, "default": False,
                    "missing": True}

        def current(kind: str, pool) -> dict[str, Any] | None:
            wanted = self.store.get(kind)
            for d in pool:
                if wanted is not None and (d.name == wanted or str(d.index) == str(wanted)):
                    return entry(d)
            gone = missing(kind, pool)
            if gone is not None:
                return gone  # never silently substitute another device
            for d in pool:
                if d.default_input if kind == "input" else d.default_output:
                    return entry(d)
            return entry(pool[0]) if pool else None

        inputs = [entry(d) for d in ins]
        outputs = [entry(d) for d in outs]
        for kind, listed in (("input", inputs), ("output", outputs)):
            gone = missing(kind, ins if kind == "input" else outs)
            if gone is not None and not (kind == "input" and self.input_file):
                listed.insert(0, gone)
        return {"inputs": inputs, "outputs": outputs,
                "input": current("input", ins), "output": current("output", outs),
                "input_file": self.input_file.name if self.input_file else None}

    def _index_info(self, path: Path) -> tuple[int, int] | None:
        """(dim, ntotal) of an index file, read once per file version; None if unreadable."""
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key not in self._index_meta:
            try:
                import faiss

                index = faiss.read_index(str(path))
                self._index_meta[key] = (index.d, index.ntotal)
            except Exception:  # noqa: BLE001
                self._index_meta[key] = None
        return self._index_meta[key]

    def _index_options(self, name: str, config) -> list[dict[str, Any]]:
        directory = self.models_dir / name
        paired = config.index.file
        chosen = self.registry.choice.get(name, paired)
        options = []
        for path in sorted(directory.glob("*.index")):
            if path.name == paired:
                info = (config.index.dim, config.index.ntotal)
            else:
                info = self._index_info(path)
            usable = info is not None and info[0] == config.emb_dim and info[1] > 0
            label = path.name + (" (auto)" if path.name == paired else "")
            if not usable:
                label += " (mismatch)" if info else " (unreadable)"
            options.append({"file": path.name, "label": label, "selectable": usable,
                            "selected": path.name == chosen, "vectors": info[1] if info else 0})
        options.append({"file": "", "label": "No index (off)", "selectable": True,
                        "selected": not chosen, "vectors": 0})
        return options

    def _models(self) -> list[dict[str, Any]]:
        if self._models_cache is None:
            models = []
            for name in self.registry.names():
                try:
                    config = self.registry.config(name)
                except (MetadataError, OSError, ValueError) as err:
                    self._log(f"skipping model {name}: {err}")
                    continue
                size = sum(p.stat().st_size for p in (self.models_dir / name).iterdir()
                           if p.is_file())
                options = self._index_options(name, config)
                selected = next(o for o in options if o["selected"])
                models.append({
                    "name": name, "version": config.version, "sr": config.sr, "f0": config.f0,
                    "speakers": config.n_spk, "size_mb": size / 2**20,
                    "index_options": options, "index": selected["file"],
                    "index_vectors": selected["vectors"], "warnings": list(config.warnings),
                    "test_voice": name.startswith("base_"),
                })
            self._models_cache = models
        return [{**m, "active": m["name"] == self.model} for m in self._models_cache]

    def _tts_state(self) -> dict[str, Any]:
        from vox.tts.backends import BACKENDS

        backend = self.store["tts"].get("backend") or "kokoro"
        cls = BACKENDS.get(backend, BACKENDS["kokoro"])
        reason = cls(cache_dir=self.models_dir / TTS_DIR).unavailable_reason()
        return {"backend": cls.name, "voice": self.store["tts"].get("voice") or cls.default_voice,
                "available": reason is None, "reason": reason, "backends": list(BACKENDS),
                "speaking": self._speaking(), "last": self._tts_last,
                "history": list(self._history)}

    def _speaking(self) -> bool:
        session = self.session
        if (session is not None and session.worker is not None
                and session.worker.override.active):
            return True
        return time.monotonic() < self._tts_speaking_until

    def state(self) -> dict[str, Any]:
        with self._lock:
            models = self._models()
            active = next((m for m in models if m["active"]), None)
            params = None
            if active is not None:
                params = self._params()
                default_pitch = self._pitch_default(params)
                preset = self._preset()
            shape, gate, precision = self._shape(), self._gate(), self._precision()
            session = self.session
            running = self.status == "running" and session is not None
            calibrated_f0 = self.profile.voice_profile.f0_extractor.f0_min_hz is not None
            sing = params is not None and preset == "sing"
            import torch

            cuda = torch.cuda.is_available()
            return _clean({
                "engine": {
                    "status": self.status, "error": self.error, "busy": self.busy,
                    "mode": session.engine.mode if running else None,
                    "recapturing": self._recapturing,
                    "warnings": list(self.warnings),
                    "started": session.started_at if running else None,
                },
                "model": active,
                "params": None if params is None else {
                    "pitch": params.f0_up_key, "index_rate": params.index_rate,
                    "protect": params.protect, "f0_method": params.f0_method,
                    "ceiling_hz": params.f0_ceiling_hz,
                    "auto_pitch": params.pitch_mode == "auto",
                    "target_hz": params.target_f0_hz, "pitch_range": params.pitch_range,
                    "preset": preset, "presets": list(PRESETS),
                    "calibrated_pitch": default_pitch,
                    "pitch_is_calibrated": params.f0_up_key == default_pitch,
                    "f0_min_hz": params.f0_min_hz, "f0_max_hz": params.f0_max_hz,
                    "f0_calibrated": calibrated_f0,
                },
                "gate": {"enabled": gate.enabled, "threshold_dbfs": gate.threshold_dbfs,
                         "noise_floor_dbfs": self.profile.calibration.noise_floor_dbfs},
                "input_level": {"auto": self.store["input_level"].get("auto", True),
                                "gain_db": 0.0 if sing
                                else self.store["input_level"].get("gain_db", 0.0),
                                "trim_locked": sing,
                                "max_gain_db": self._conditioning().max_gain_db,
                                "range_db": list(INPUT_GAIN_RANGE)},
                "suppression": self._suppression_state(),
                # Not built in the engine yet: shown, not switchable (docs/DEVIATIONS.md).
                "monitor": {"enabled": False, "available": False},
                "auto_tune": {"enabled": False, "available": False},
                "shape": {"block_ms": shape.block_ms, "extra_ms": shape.extra_ms,
                          "device_sr": shape.device_sr},
                "graphs": precision.graphs != "off" and cuda,
                "graphs_available": cuda,
                "f0_methods": list(EXTRACTORS),
                "exclusive": bool(self.store.get("exclusive")),
                "devices": self._devices(),
                "tts": self._tts_state(),
                "recording": {"active": running
                              and getattr(session, "recording", None) is not None,
                              "last": self._last_take},
                "models": models,
                "models_dir": str(self.models_dir),
                "profile": {"name": self.profile_dir.name,
                            "calibrated": self.profile.voice_profile.created or None},
                "app": {"version": _version(),
                        "gpu": torch.cuda.get_device_name(0) if cuda else "CPU"},
                "perf_history": list(self.perf_history),
            })

    def publish_state(self) -> None:
        self.bus.publish(STATE, self.state())

    # --------------------------------------------------------------------- commands

    def command(self, name: str, args: dict[str, Any] | None = None) -> Any:
        """Run one command by name. Raises ``ApiError`` with a user-facing message."""
        handler = COMMANDS.get(name)
        if handler is None:
            raise ApiError(f"unknown command {name!r}")
        try:
            return getattr(self, handler)(**(args or {}))
        except TypeError as err:
            raise ApiError(f"{name}: bad arguments ({err})") from err

    def _submit(self, label: str, fn, *args) -> dict[str, Any]:
        self._jobs.put((label, fn, args))
        return {"queued": label}

    def _job_loop(self) -> None:
        while True:
            try:
                label, fn, args = self._jobs.get(timeout=0.2)
            except queue.Empty:
                if self._closing.is_set():
                    return  # shutdown: only once everything queued has run
                continue
            self.busy = label
            self.publish_state()
            try:
                fn(*args)
            except ApiError as err:
                self.toast(str(err), "error")
            except Exception as err:  # noqa: BLE001
                self._log(f"{label} failed: {type(err).__name__}: {err}")
                self.toast(f"{label} failed: {err}", "error")
            finally:
                self.busy = None
                self.publish_state()

    # lifecycle ---------------------------------------------------------------------

    def start(self) -> dict[str, Any]:
        return self._submit("Starting", self._start)

    def stop(self) -> dict[str, Any]:
        return self._submit("Stopping", self._stop)

    def _start(self) -> None:
        if self.session is not None:
            return
        config = self._config()
        params = self._params()
        start_params = params
        if self.registry.index_path(config.name) is not None and params.index_rate <= 0:
            # The engine only loads an index when the rate is above zero at start. Loading
            # it anyway keeps "index rate" a hot parameter instead of a restart.
            start_params = dataclasses.replace(params, index_rate=INDEX_PRIME)
        cfg = LiveConfig(
            model=config.name, models_dir=self.models_dir, shape=self._shape(),
            params=start_params, precision=self._precision(), gate=self._gate(),
            conditioning=self._conditioning(),
            input=self.store.get("input"), output=self.store.get("output"),
            exclusive=bool(self.store.get("exclusive")), input_file=self.input_file,
            loop_file=self.loop_file, device=self.device, registry=self.registry,
            cache=self.cache, profile_dir=self.profile_dir,
            suppression=self._suppression_choice(self._input_name()),
            suppression_profile=self.profile.calibration.rnnoise_enabled)
        self.status, self.error = "starting", None
        self.perf, self._rebuilds_seen, self._recapturing = None, 0, False
        self.perf_history.clear()
        self.publish_state()
        self._warm_tts()
        session = self.session_factory(cfg, log=self._log)
        try:
            with self._pa_lock:
                session.start()
        except Exception as err:  # noqa: BLE001
            with contextlib.suppress(Exception), self._pa_lock:
                session.stop()
            self.status, self.error = "error", f"{type(err).__name__}: {err}"
            self._log(f"start failed: {self.error}")
            return
        if start_params is not params:
            session.set_params(index_rate=params.index_rate)
        self.warnings = list(session.warnings)
        self.session = session
        self.status = "running"
        self.store["model"] = config.name
        self._save_store()
        self._warn_broadcast_sing()

    def _stop(self) -> None:
        session, self.session = self.session, None
        if session is None:
            if self.status != "error":
                self.status = "stopped"
            return
        self.status = "stopping"
        self.publish_state()
        if getattr(session, "recording", None) is not None:  # stopping mid-take keeps it
            self._save_take(session)
        try:
            with self._pa_lock:
                session.stop()
        finally:
            self.status = "stopped"
            self._refresh_devices()

    def _restart(self) -> None:
        if self.session is not None:
            self._stop()
            self._start()

    # recording ---------------------------------------------------------------------

    def record(self, enabled: bool) -> dict[str, Any]:
        """Start or stop a take: raw mic, conditioned mic and converted output as aligned
        WAVs plus a JSON of the settings and gate events, in ``recordings/``. Stopping
        opens Explorer on the output file."""
        session = self.session
        if enabled:
            if session is None or self.status != "running":
                raise ApiError("Start the engine to record")
            session.start_recording()
            self.publish_state()
            return {"recording": True}
        if session is None or session.recording is None:
            return {"recording": False}
        return self._submit("Saving recording", self._save_take, session)

    def _save_take(self, session) -> None:
        paths = session.stop_recording(self.recordings_dir)
        if paths is None:
            return
        self._last_take = str(paths["output"])
        self._log(f"recording saved: {paths['output']}")
        self.toast(f"Recording saved: {paths['output'].name}")
        try:
            self._reveal(paths["output"])
        except Exception as err:  # noqa: BLE001 - the take is saved either way
            self._log(f"could not open Explorer: {err}")

    # hot parameters ----------------------------------------------------------------

    def set_param(self, name: str, value: float) -> dict[str, Any]:
        if name not in HOT_PARAMS:
            raise ApiError(f"unknown parameter {name!r}")
        low, high = PARAM_RANGES[name]
        value = min(high, max(low, float(value)))
        with self._lock:
            self._preset_store(self._config().name)[name] = value  # the active preset's
            session = self.session
            if session is not None:
                session.set_params(**{HOT_PARAMS[name]: value})
            self._save_store()
        self.publish_state()
        return {name: value}

    def reset_pitch(self) -> dict[str, Any]:
        """Back to the calibrated transposition (Speak) or the Speak pitch plus the Sing
        offset (Sing)."""
        with self._lock:
            self._preset_store(self._config().name).pop("pitch", None)
            value = self._params().f0_up_key
            if self.session is not None:
                self.session.set_params(f0_up_key=value)
            self._save_store()
        self.publish_state()
        return {"pitch": value}

    def set_auto_pitch(self, enabled: bool) -> dict[str, Any]:
        """Auto pitch for the current model (docs/DEVIATIONS.md section 9): the talker's
        register is measured live and transposed onto ``target_hz``; off = the fixed
        pitch offset."""
        with self._lock:
            entry = self._model_store(self._config().name)
            if enabled:
                entry["auto_pitch"] = True
            else:
                entry.pop("auto_pitch", None)
            params = self._params()
            if self.session is not None:
                self.session.set_params(pitch_mode=params.pitch_mode)
            self._save_store()
        self.publish_state()
        return {"auto_pitch": params.pitch_mode == "auto", "target_hz": params.target_f0_hz}

    def set_preset(self, preset: str) -> dict[str, Any]:
        """Speak / Sing for the current model: its own pitch, index rate, protect and F0
        ceiling, and the input chain's singing limits. Hot: nothing is recaptured."""
        if preset not in PRESETS:
            raise ApiError(f"unknown preset {preset!r}")
        with self._lock:
            entry = self._model_store(self._config().name)
            if preset == "sing":
                entry["preset"] = "sing"
            else:
                entry.pop("preset", None)
            params = self._params()
            session = self.session
            if session is not None:
                session.set_params(**{field: getattr(params, field)
                                      for field in HOT_PARAMS.values()})
                config = self._conditioning()
                session.set_conditioning(auto_level=config.auto_level, trim_db=config.trim_db,
                                         max_gain_db=config.max_gain_db)
            self._save_store()
        self._warn_broadcast_sing()
        self.publish_state()
        return {"preset": preset, "pitch": params.f0_up_key}

    def set_gate(self, enabled: bool | None = None,
                 threshold_dbfs: float | None = None) -> dict[str, Any]:
        with self._lock:
            if enabled is not None:
                self.store["gate"]["enabled"] = bool(enabled)
            if threshold_dbfs is not None:
                self.store["gate"]["threshold_dbfs"] = min(0.0, max(-90.0, float(threshold_dbfs)))
            gate = self._gate()
            if self.session is not None:
                self.session.set_gate(enabled=gate.enabled, threshold_dbfs=gate.threshold_dbfs)
            self._save_store()
        self.publish_state()
        return {"enabled": gate.enabled, "threshold_dbfs": gate.threshold_dbfs}

    def set_input_level(self, auto: bool | None = None,
                        gain_db: float | None = None) -> dict[str, Any]:
        """Auto level on/off and the input-gain trim (dB): hot, applied between hops."""
        with self._lock:
            user = self.store["input_level"]
            if auto is not None:
                user["auto"] = bool(auto)
            if gain_db is not None:
                low, high = INPUT_GAIN_RANGE
                user["gain_db"] = min(high, max(low, float(gain_db)))
            config = self._conditioning()
            if self.session is not None:
                self.session.set_conditioning(auto_level=config.auto_level,
                                              trim_db=config.trim_db)
            self._save_store()
        self.publish_state()
        return {"auto": config.auto_level, "gain_db": user.get("gain_db", 0.0)}

    def set_suppression(self, enabled: bool | None = None,
                        mix: float | None = None) -> dict[str, Any]:
        """RNNoise on the mic (after the high-pass, before auto level and the gate): on/off
        for the current microphone, and the wet/dry ``mix``. Hot, crossfaded (~20 ms)."""
        reason = rnnoise.unavailable_reason(self._shape().device_sr)
        if enabled and reason is not None:
            raise ApiError(f"Noise suppression is unavailable: {reason}")
        with self._lock:
            user = self.store["suppression"]
            if enabled is not None:
                user.setdefault("by_input", {})[self._input_name() or ""] = bool(enabled)
            if mix is not None:
                user["mix"] = min(1.0, max(0.0, float(mix)))
            session = self.session
            if session is not None:
                changes: dict[str, Any] = {"suppression_mix": self._suppression_mix()}
                if enabled is not None:
                    changes["suppression"] = bool(enabled)
                session.set_conditioning(**changes)
                if enabled is not None:
                    session.suppression = bool(enabled)
                    session.suppression_why = "switched here"
            self._save_store()
            result = self._suppression_state()
        self.publish_state()
        return result

    def set_monitor(self, enabled: bool) -> None:
        raise ApiError("Headphone monitoring is not built into the engine yet")

    def set_auto_tune(self, enabled: bool) -> None:
        raise ApiError("Automatic hop tuning is not built into the engine yet")

    # restart-class -----------------------------------------------------------------

    def set_shape(self, block_ms: float | None = None, extra_ms: float | None = None,
                  f0_method: str | None = None) -> dict[str, Any]:
        """Hop / context / pitch tracker: re-captured in place by the worker (brief mute)."""
        with self._lock:
            if f0_method is not None and f0_method not in EXTRACTORS:
                raise ApiError(f"unknown pitch tracker {f0_method!r}")
            shape = self._shape()
            changes = {k: float(v) for k, v in (("block_ms", block_ms), ("extra_ms", extra_ms))
                       if v is not None}
            if "block_ms" in changes:  # whole 20 ms HuBERT frames: hubert_phase_stable
                changes["block_ms"] = chunker.snap_block_ms(changes["block_ms"])
            shape = dataclasses.replace(shape, **changes)
            if self.model is not None:
                try:  # reject a plan the chunker would refuse before the worker sees it
                    plan = chunker.plan(shape.block_ms, shape.crossfade_ms, shape.extra_ms,
                                        shape.sola_search_ms, shape.device_sr,
                                        self._config().sr)
                except ValueError as err:
                    raise ApiError(f"that combination does not work: {err}") from err
                # The chunker works in whole 10 ms units: keep what will really run.
                rounded = {"block_ms": plan.block_ms, "extra_ms": plan.extra_ms}
                changes = {k: rounded[k] for k in changes}
                shape = dataclasses.replace(shape, **changes)
            self.store["shape"].update(changes)
            if f0_method is not None:
                self.store["f0_method"] = f0_method
            session = self.session
            if session is not None:
                method = f0_method if session.engine.pipeline.extractor is not None else None
                session.set_shape(f0_method=method, **changes)
                self._recapturing = True
            self._save_store()
        self.publish_state()
        return {"block_ms": shape.block_ms, "extra_ms": shape.extra_ms, "f0_method": f0_method}

    def set_graphs(self, enabled: bool) -> dict[str, Any]:
        self.store["graphs"] = "auto" if enabled else "off"
        self._save_store()
        self.publish_state()
        return self._submit("Re-capturing", self._restart)

    def set_exclusive(self, enabled: bool) -> dict[str, Any]:
        self.store["exclusive"] = bool(enabled)
        self._save_store()
        self.publish_state()
        return self._submit("Reopening audio", self._restart)

    def set_device(self, kind: str, name: str | None) -> dict[str, Any]:
        if kind not in ("input", "output"):
            raise ApiError(f"unknown device kind {kind!r}")
        self.store[kind] = name or None
        self._save_store()
        if kind == "input" and name:
            self._warn_broadcast_sing(name)
        self.publish_state()
        return self._submit("Reopening audio", self._restart)

    def select_model(self, name: str) -> dict[str, Any]:
        self._config(name)
        return self._submit(f"Loading {name}", self._select_model, name)

    def _select_model(self, name: str) -> None:
        self.model = name
        self.store["model"] = name
        self._offline = None
        self._save_store()
        self._restart()

    def set_index(self, model: str, file: str) -> dict[str, Any]:
        config = self._config(model)
        options = {o["file"]: o for o in self._index_options(model, config)}
        if file not in options:
            raise ApiError(f"{model} has no index file named {file!r}")
        if not options[file]["selectable"]:
            raise ApiError(f"{file} does not match this model and cannot be used")
        return self._submit("Switching index", self._set_index, model, file)

    def _set_index(self, model: str, file: str) -> None:
        self.registry.choice[model] = file
        self._model_store(model)["index"] = file
        self._models_cache = None
        self._offline = None
        self._save_store()
        if model == self.model:
            self._restart()

    # models ------------------------------------------------------------------------

    def upload_path(self, filename: str) -> Path:
        """Where the transport should write an uploaded file (kept until it is imported)."""
        safe = Path(filename).name
        if not safe.lower().endswith((".pth", ".index")):
            raise ApiError("only .pth and .index files can be imported")
        return self._uploads / safe

    def import_model(self, pth: str, index: str | None = None,
                     name: str | None = None) -> dict[str, Any]:
        """Import an uploaded ``.pth`` (and an ``.index`` dropped with it) as one voice."""
        return self._submit("Importing", self._import_model, pth, index, name)

    def _import_model(self, pth: str, index: str | None, name: str | None) -> None:
        pth_path = self._uploads / Path(pth).name
        index_path = self._uploads / Path(index).name if index else None
        try:
            if not pth_path.is_file():
                raise ApiError(f"{pth} was not uploaded")
            # Stage the pair alone in a folder: import pairs an index from the .pth's folder.
            stage = Path(tempfile.mkdtemp(prefix="pair_", dir=self._uploads))
            shutil.move(str(pth_path), stage / pth_path.name)
            if index_path is not None and index_path.is_file():
                shutil.move(str(index_path), stage / index_path.name)
            try:
                target = discovery.import_model(stage / pth_path.name, self.models_dir, name)
            except (MetadataError, FileExistsError) as err:
                raise ApiError(str(err)) from err
            finally:
                shutil.rmtree(stage, ignore_errors=True)
        finally:
            for leftover in (pth_path, index_path):
                if leftover is not None:
                    leftover.unlink(missing_ok=True)
        self._models_cache = None
        config = self.registry.config(target.name)
        paired = f"index paired ({config.index.ntotal} vectors)" if config.index.file \
            else "no index"
        self.toast(f"Imported {target.name}: {config.version}, {config.sr // 1000} kHz, {paired}")
        for warning in config.warnings:
            self.toast(warning, "warn")
        if self.model is None:
            self.model = target.name

    def add_index(self, model: str, index: str) -> dict[str, Any]:
        """An uploaded ``.index`` joins a model's folder and becomes its index (Browseâ€¦)."""
        self._config(model)
        return self._submit("Adding index", self._add_index, model, index)

    def _add_index(self, model: str, index: str) -> None:
        source = self._uploads / Path(index).name
        if not source.is_file():
            raise ApiError(f"{index} was not uploaded")
        target = self.models_dir / model / source.name
        shutil.move(str(source), target)
        self._models_cache = None
        options = {o["file"]: o for o in self._index_options(model, self._config(model))}
        if not options[target.name]["selectable"]:
            raise ApiError(f"{target.name} does not match {model}; it is listed but not used")
        self._set_index(model, target.name)

    # typed speech --------------------------------------------------------------------

    def set_tts(self, voice: str | None = None, backend: str | None = None) -> dict[str, Any]:
        from vox.tts.backends import BACKENDS

        with self._lock:
            if backend is not None:
                if backend not in BACKENDS:
                    raise ApiError(f"unknown TTS backend {backend!r}")
                if backend != self.store["tts"].get("backend"):
                    self.store["tts"] = {"backend": backend}
                    self._tts = None
            if voice is not None:
                self.store["tts"]["voice"] = voice
                self._tts = None
            self._save_store()
        self.publish_state()
        return dict(self.store["tts"])

    def tts_voices(self) -> list[str]:
        from vox.tts.base import TtsError

        try:
            return self._backend().voices()
        except TtsError as err:
            raise ApiError(str(err)) from err

    def _backend(self):
        from vox.tts.backends import pick_backend
        from vox.tts.base import TtsError

        if self._tts is None:
            tts = self.store["tts"]
            try:
                self._tts = pick_backend(tts.get("backend") or "kokoro", tts.get("voice"),
                                         self.models_dir / TTS_DIR)
            except TtsError as err:
                raise ApiError(str(err)) from err
        return self._tts

    def speak(self, text: str) -> dict[str, Any]:
        if not text or not text.strip():
            raise ApiError("type something to say first")
        self._config()
        return self._submit("Rendering speech", self._speak, text.strip(), None)

    def preview(self, model: str) -> dict[str, Any]:
        """Speak a fixed sentence in any imported voice (the play button on a model card)."""
        self._config(model)
        return self._submit(f"Previewing {model}", self._speak, PREVIEW_TEXT, model)

    def replay(self, id: int) -> dict[str, Any]:
        clip = self._clips.get(int(id))
        if clip is None:
            raise ApiError("that clip is no longer kept")
        return self._submit("Replaying", self._play, clip)

    def cancel_speech(self) -> dict[str, Any]:
        session = self.session
        self._tts_speaking_until = 0.0
        if session is not None:
            session.worker.override.cancel()
            self.publish_state()
            return {}
        return self._submit("Stopping speech", self._stop_playback)

    def _stop_playback(self) -> None:
        with contextlib.suppress(Exception), self._pa_lock:
            import sounddevice as sd

            sd.stop()
        self._tts_speaking_until = 0.0

    def _engine_for(self, model: str):
        """The engine whose loaded models render a clip for ``model``.

        The live session's own engine when it is running that voice; otherwise an engine
        built without warmup or graphs, kept until the voice changes. Another voice than the
        live one gets its own registry so the live model stays the resident one.
        """
        session = self.session
        if session is not None and session.cfg.model == model:
            return session.engine, True
        from vox.offline.simulate import load_engine

        key = (model, self.registry.choice.get(model))
        if self._offline is None or self._offline[0] != key:
            registry = self.registry
            if model != self.model:
                registry = _Registry(self.models_dir)
                registry.choice = self.registry.choice
            params = self._params(model)
            if registry.index_path(model) is not None and params.index_rate <= 0:
                params = dataclasses.replace(params, index_rate=INDEX_PRIME)
            engine = load_engine(self.models_dir, model, self._shape(), params,
                                 self._precision(), self.device, warmup_hops=0,
                                 registry=registry, cache=self.cache, graphs="off",
                                 gate=GateConfig(enabled=False, use_vad=False), log=self._log,
                                 profile_dir=self.profile_dir)
            self._offline = (key, engine)
        return self._offline[1], False

    @contextlib.contextmanager
    def _beside_live(self, engine, live: bool):
        """Render on a second CUDA stream while live, so the clip's kernels do not queue
        ahead of the hop on the stream the worker replays its graphs on."""
        if not live or engine.device.type != "cuda":
            yield
            return
        import torch

        if self._side_stream is None:
            self._side_stream = torch.cuda.Stream(device=engine.device)
        with torch.cuda.stream(self._side_stream):
            yield
            self._side_stream.synchronize()

    def _warm_tts(self) -> None:
        """Load the TTS engine before going live, so the first typed clip does not pay for
        it (a second or two of CPU that the live hop would feel). Never downloads."""
        from vox.tts.base import TtsError

        try:
            self._backend().load()
        except (ApiError, TtsError, OSError) as err:
            self._log(f"tts not preloaded: {err}")

    def _speak(self, text: str, model: str | None) -> None:
        from vox.offline.convert import FullFileConverter
        from vox.tts import speak as tts
        from vox.tts.base import TtsError

        model = model or self._config().name
        config = self._config(model)
        backend = self._backend()
        session = self.session if self.status == "running" else None
        ring = session.out_ring if session is not None else None
        under_start = ring.underruns if ring else 0
        started = time.perf_counter()
        try:
            pieces = tts.split_pieces(text)
            rendered = tts.render_tts(backend, pieces)
        except TtsError as err:
            raise ApiError(str(err)) from err
        tts_s = time.perf_counter() - started
        under_tts = ring.underruns if ring else 0
        converting = time.perf_counter()
        try:
            if session is not None:
                # The live voice goes quiet now and the GPU renders the clip instead. Let
                # the fade-out hops reach the output ring before the GPU gets busy, so a
                # late hop during the render can only ever delay silence.
                session.worker.hold = True
                time.sleep(HOLD_SETTLE_HOPS * session.worker.plan.block_ms / 1000.0)
            engine, same_voice = self._engine_for(model)
            params = session.worker.params if same_voice else self._params(model)
            target = tts.target_f0_hint(self.profile, config)
            with self._beside_live(engine, session is not None):
                result = tts.convert_rendered(
                    FullFileConverter.from_engine(engine), engine.device,
                    engine.loaded.model.tgt_sr, rendered, params, None, target, seed=None,
                    pieces=pieces, polish=tts.DEFAULT_POLISH)
            convert_s = time.perf_counter() - converting
            device_sr = self._shape().device_sr
            clip = resample_array(result.audio, result.sr, device_sr)
            fade = min(len(clip) // 2, int(device_sr * CLIP_FADE_MS / 1000.0))
            if fade:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                clip[:fade] *= ramp
                clip[-fade:] *= ramp[::-1]
            self._clip_id += 1
            entry = {"id": self._clip_id, "text": text, "model": model, "voice": backend.voice,
                     "backend": backend.name, "pitch": result.pitch,
                     "seconds": len(clip) / device_sr, "at": time.time(),
                     "render_ms": 1000.0 * (time.perf_counter() - started),
                     "tts_ms": 1000.0 * tts_s, "convert_ms": 1000.0 * convert_s}
            if len(self._history) == self._history.maxlen:
                self._clips.pop(self._history[-1]["id"], None)
            self._history.appendleft(entry)
            self._clips[entry["id"]] = clip
            self._tts_last = entry
            self._play(clip)  # from here the clip itself keeps the live voice closed
        finally:
            if session is not None:
                session.worker.hold = False
        if ring is not None:
            self._log(f"speech: {entry['seconds']:.1f} s clip, tts {1000 * tts_s:.0f} ms, "
                      f"convert {1000 * convert_s:.0f} ms (live held), underruns during tts "
                      f"{under_tts - under_start}, during convert {ring.underruns - under_tts}")

    def _play(self, clip: np.ndarray) -> None:
        """Through the live output path when live (the mic is muted for the clip), else
        straight to the selected output device."""
        session = self.session
        device_sr = self._shape().device_sr
        if session is not None and self.status == "running":
            session.worker.play_clip(clip)
        else:
            import sounddevice as sd

            try:
                with self._pa_lock:
                    target = audio_devices.resolve_device(self.store.get("output"), "output")
                    sd.play(clip, device_sr, device=target.index)
            except Exception as err:
                raise ApiError(f"could not play on the output device: {err}") from err
        self._tts_speaking_until = time.monotonic() + len(clip) / device_sr

    # reports / calibration ---------------------------------------------------------

    def export_report(self) -> dict[str, Any]:
        """Write the current profiler view to a JSON report file. Returns its path."""
        from vox.profiling.report import environment_fingerprint

        session = self.session
        record = {
            "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "environment": environment_fingerprint(),
            "engine": self.state()["engine"],
            "model": self.model,
            "describe": session.describe() if session is not None else None,
            "perf": self.perf,
            "history": list(self.perf_history),
            "snapshot": _clean(session.snapshot()) if session is not None else None,
        }
        path = self.reports_dir / f"gui_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        save_json(path, _clean(record))
        self.toast(f"Report written to {path}")
        return {"path": str(path)}

    def calibrate(self) -> dict[str, Any]:
        raise ApiError("Voice calibration is Phase 5 and is not built yet; "
                       "set the pitch by ear for now")

    def get_state(self) -> dict[str, Any]:
        return self.state()

    def shutdown(self) -> None:
        """Stop everything this controller started. The CLI stays usable afterwards."""
        self._submit("Stopping", self._stop)  # streams close on the thread that opened them
        self._closing.set()
        for thread in (self._job_thread, self._pub_thread):
            if thread is not threading.current_thread():
                thread.join(timeout=5.0)
        shutil.rmtree(self._uploads, ignore_errors=True)

    # ----------------------------------------------------------------------- events

    def _publish_loop(self) -> None:
        tick = 0
        speaking = False
        while not self._closing.wait(1.0 / METER_HZ):
            tick += 1
            session = self.session
            running = self.status == "running" and session is not None
            try:
                if running:
                    self.bus.publish(METERS, self._meters(session))
                    if tick % PERF_EVERY == 0:
                        self._publish_perf(session)
                elif tick % PERF_EVERY == 0:
                    self.bus.publish(METERS, {"running": False, "in_db": METER_FLOOR_DB,
                                              "out_db": METER_FLOOR_DB, "gate_open": False,
                                              "speaking": self._speaking()})
            except Exception as err:  # noqa: BLE001
                self._log(f"events: {type(err).__name__}: {err}")
            now = self._speaking()
            if now != speaking:
                speaking = now
                self.publish_state()

    def _meters(self, session) -> dict[str, Any]:
        stats, health = session.worker.stats, session.audio.health
        out_peak, health.output_peak = health.output_peak, 0.0
        health.input_peak = 0.0
        take = getattr(session, "recording", None)
        return _clean({"running": True, "in_db": stats.gate_rms_dbfs, "out_db": _db(out_peak),
                       "in_gain_db": stats.input_gain_db,
                       "gate_open": stats.gate_open, "vad": stats.gate_vad,
                       "speaking": session.worker.override.active,
                       "rec_s": take.seconds if take is not None else None})

    def _publish_perf(self, session) -> None:
        snap = session.snapshot()
        if snap["errors"]:
            self.error = "; ".join(snap["errors"])
            self.status = "error"
            self._log(f"engine error: {self.error}")
            self._submit("Stopping", self._stop_after_error)
        plan = session.worker.plan
        method = session.worker.params.f0_method.upper()
        stages = []
        for key, label in STAGE_LABELS.items():
            row = snap["stages"].get(key)
            if row is not None:
                # GPU-event time when there is one: under graphs the host time of a stage is
                # only the launch of its replay.
                stages.append({"key": key, "label": f"F0 {method}" if key == "f0" else label,
                               "p50": row.get("gpu_p50", row["p50"]),
                               "p99": row.get("gpu_p99", row["p99"])})
        total = snap["total"]
        p99 = total["p99"]
        perf = _clean({
            "t": snap["t"], "latency_ms": snap["latency"]["total"], "latency": snap["latency"],
            "underruns": snap["underruns"], "hop_ms": plan.block_ms, "extra_ms": plan.extra_ms,
            "stages": stages, "p50": total["p50"], "p99": p99,
            "headroom_pct": 100.0 * (1.0 - p99 / plan.block_ms) if math.isfinite(p99) else None,
            "out_fill_ms": snap["out_fill_ms"], "gate_open": snap["gate_open"],
            "mode": snap["mode"], "hops": snap["hops"], "gated_hops": snap["gated_hops"],
            "rebuilds": snap["rebuilds"], "auto_pitch_st": snap.get("auto_pitch_st"),
            **_system(),
        })
        self.perf = perf
        self.perf_history.append({"t": perf["t"], "p50": perf["p50"], "p99": perf["p99"]})
        self.bus.publish(PERF, perf)
        if self._recapturing and snap["rebuilds"] > self._rebuilds_seen:
            self._recapturing = False
            self.publish_state()
        self._rebuilds_seen = snap["rebuilds"]

    def _stop_after_error(self) -> None:
        error = self.error
        self._stop()
        self.status, self.error = "error", error


COMMANDS = {
    "start": "start", "stop": "stop", "get_state": "get_state", "shutdown": "shutdown",
    "set_param": "set_param", "reset_pitch": "reset_pitch", "set_gate": "set_gate",
    "set_input_level": "set_input_level", "record": "record",
    "set_suppression": "set_suppression", "set_preset": "set_preset",
    "set_auto_pitch": "set_auto_pitch",
    "set_monitor": "set_monitor",
    "set_auto_tune": "set_auto_tune", "set_shape": "set_shape", "set_graphs": "set_graphs",
    "set_exclusive": "set_exclusive", "set_device": "set_device",
    "select_model": "select_model", "set_index": "set_index", "import_model": "import_model",
    "add_index": "add_index", "set_tts": "set_tts", "tts_voices": "tts_voices",
    "speak": "speak", "preview": "preview", "replay": "replay",
    "cancel_speech": "cancel_speech", "export_report": "export_report",
    "calibrate": "calibrate", "refresh_devices": "refresh_devices",
}


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("rvc-vox")
    except PackageNotFoundError:
        return "dev"


_PROCESS = None


def _system() -> dict[str, Any]:
    """GPU / CPU load for the Performance screen. Every probe is optional."""
    global _PROCESS
    info: dict[str, Any] = {"gpu_pct": None, "vram_gb": None, "cpu_pct": None}
    try:
        import psutil

        if _PROCESS is None:
            _PROCESS = psutil.Process()
            _PROCESS.cpu_percent(None)
        info["cpu_pct"] = _PROCESS.cpu_percent(None) / (psutil.cpu_count() or 1)
    except Exception:  # noqa: BLE001, S110 - an optional probe
        pass
    try:
        import torch

        if torch.cuda.is_available():
            info["vram_gb"] = torch.cuda.memory_reserved() / 2**30
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            info["gpu_pct"] = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:  # noqa: BLE001, S110 - an optional probe
        pass
    return info
