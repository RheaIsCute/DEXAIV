"""Live check of the volume controls: a mic take looped through the engine into the output
device while the output and typed-speech volumes move, with typed clips over the live voice.

Reads the OUT meter the GUI shows (METERS events) per phase, counts underruns, records the
output as a take for listening, and restarts the controller on the same profile to show the
values persist. Works on a copy of the profile, so the user's saved GUI state is untouched.

    .venv\\Scripts\\python.exe scripts\\volume_check.py --input-file takes\\mic\\x.wav
        --output 23 --model Mom
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from vox.api.controller import EngineController
from vox.api.events import METERS, EventBus

ROOT = Path(__file__).resolve().parents[1]


def wait(c: EngineController, status: str, timeout: float = 120.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        engine = c.state()["engine"]
        if engine["status"] == status and engine["busy"] is None:
            return
        if engine["status"] == "error":
            raise RuntimeError(engine["error"])
        time.sleep(0.2)
    raise TimeoutError(f"engine never reached {status!r}")


def idle(c: EngineController, timeout: float = 60.0) -> None:
    end = time.time() + timeout
    while time.time() < end and (c.busy is not None or not c._jobs.empty()):
        time.sleep(0.05)


def phase(c, bus_q, name, seconds, rows, underruns_at):
    """Collect OUT meter readings for ``seconds``; keep the loud ones (speech peaks)."""
    while not bus_q.empty():
        bus_q.get_nowait()
    end = time.time() + seconds
    outs = []
    while time.time() < end:
        try:
            kind, data = bus_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if kind == METERS and data.get("running") and data.get("out_db") is not None:
            outs.append(data["out_db"])
    session = c.session
    loud = sorted(outs)[-max(1, len(outs) // 4):]  # top quarter: peaks of speech
    row = {"phase": name, "readings": len(outs),
           "out_peak_db_p75": round(statistics.median(loud), 1) if loud else None,
           "underruns": session.out_ring.underruns - underruns_at,
           "volume": c.state().get("volume")}
    rows.append(row)
    print(f"{name:34s} OUT p75 {row['out_peak_db_p75']} dB, underruns {row['underruns']}")
    return session.out_ring.underruns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", type=Path, required=True)
    ap.add_argument("--output", default="23")
    ap.add_argument("--model", default="Mom")
    ap.add_argument("--typed", default="0,-12,12", help="typed-speech volumes to play a clip at")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "volume_check")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="vox_volume_"))
    profile = work / "profile"
    shutil.copytree(ROOT / "profiles" / "default", profile)
    for name in ("gui_state.json",):
        (profile / name).unlink(missing_ok=True)  # the engine defaults, not the user's state

    if not hasattr(EngineController, "set_volume"):  # older code: clips only, no volumes
        EngineController.set_volume = lambda self, **kw: None
    bus = EventBus()
    log = []
    c = EngineController(ROOT / "models", profile, model=args.model, output=int(args.output),
                         input_file=args.input_file, loop_file=True, bus=bus,
                         recordings_dir=args.out, log=lambda m: log.append(m))
    q = bus.subscribe()
    c.set_gate(enabled=False)  # every hop converted: the meter follows the voice only
    c.start()
    wait(c, "running")
    time.sleep(3.0)
    c.record(True)
    rows: list[dict] = []
    u = c.session.out_ring.underruns
    u = phase(c, q, "output 0 dB", 5, rows, u)
    c.set_volume(output_db=-12)
    u = phase(c, q, "output -12 dB", 5, rows, u)
    c.set_volume(output_db=6)
    u = phase(c, q, "output +6 dB", 5, rows, u)
    c.set_volume(output_db=-24)
    u = phase(c, q, "output -24 dB", 5, rows, u)
    c.set_volume(output_db=0)
    u = phase(c, q, "output back to 0 dB", 4, rows, u)
    text = "This is a typed line, spoken over the live voice to check its level."
    for typed in [int(v) for v in args.typed.split(",")]:
        c.set_volume(typed_db=typed)
        c.speak(text)
        idle(c)
        time.sleep(0.2)
        u = phase(c, q, f"typed clip, typed {typed:+d} dB", 3.0, rows, u)
        while c.session.worker.override.active:
            time.sleep(0.1)
        time.sleep(1.0)
    c.set_volume(typed_db=0)
    c.speak(text)
    time.sleep(1.5)
    c.set_volume(typed_db=-12)  # moved mid-clip: ramped
    idle(c)
    u = phase(c, q, "typed 0 -> -12 dB mid-clip", 3.0, rows, u)
    while c.session.worker.override.active:
        time.sleep(0.1)
    c.set_volume(output_db=-6, typed_db=3, auto_match=True)
    u = phase(c, q, "output -6 (persist check)", 3, rows, u)
    c.record(False)
    idle(c)
    snapshot = c.session.snapshot() if hasattr(c.session, "snapshot") else {}
    level = getattr(c.session, "voice_level_dbfs", None)
    c.stop()
    wait(c, "stopped")
    c.shutdown()

    again = EngineController(ROOT / "models", profile, model=args.model, log=lambda m: None)
    persisted = again.state().get("volume")
    again.shutdown()
    total_underruns = sum(r["underruns"] for r in rows)
    report = {"rows": rows, "persisted_after_restart": persisted, "live_level_dbfs": level,
              "total_underruns": total_underruns,
              "speech_log": [m for m in log if m.startswith(("speech", "speech level"))],
              "takes": sorted(str(p) for p in args.out.glob("take_*")),
              "underflows": snapshot.get("output_underflows") if snapshot else None,
              "underrun_events": [e for e in snapshot.get("events", []) if e.get("kind") == "underrun"]
              if snapshot else None}
    (args.out / "volume_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"persisted": persisted, "total_underruns": total_underruns,
                      "live_level_dbfs": level}, indent=2))
    for m in report["speech_log"]:
        print(m)
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
