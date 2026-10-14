"""``vox gui``: the control API on a local port, plus the window that talks to it.

The engine side (controller + HTTP transport) runs in this process. The window is the
separate ``ui/app.py`` process, started by path and given only the URL: nothing under
``src/vox`` imports the UI, and the UI process never loads the engine. When the window
closes, the session stops and this command returns; the CLI is untouched either way.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def add_gui_parser(sub, repo_root: Path, default_models: Path, default_profile: Path) -> None:
    p = sub.add_parser("gui", help="the DEXAIV window: live, models, performance, settings")
    p.add_argument("--models-dir", type=Path, default=default_models)
    p.add_argument("--profile", type=Path, default=default_profile)
    p.add_argument("--model", default=None, help="voice to select (default: the last one used)")
    p.add_argument("--input", default=None, help="input device index or name fragment")
    p.add_argument("--output", default=None, help="output device index or name fragment")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    p.add_argument("--input-file", type=Path, default=None,
                   help="software microphone: feed this WAV instead of a device")
    p.add_argument("--loop-file", action="store_true")
    p.add_argument("--port", type=int, default=0, help="local port (default: any free one)")
    p.add_argument("--start", action="store_true", help="go live as soon as it opens")
    p.add_argument("--no-window", action="store_true",
                   help="serve the API and print its URL without opening the window")
    p.add_argument("--browser", action="store_true",
                   help="open the UI in the default browser instead of its own window")
    p.set_defaults(func=run, repo_root=repo_root)


def run(args: argparse.Namespace) -> int:
    from vox.api.controller import EngineController
    from vox.api.server import ApiServer

    ui_dir = args.repo_root / "ui"
    controller = EngineController(
        args.models_dir, args.profile, model=args.model, input=args.input, output=args.output,
        device=args.device, input_file=args.input_file, loop_file=args.loop_file,
        reports_dir=args.repo_root / "benchmarks" / "results")
    server = ApiServer(controller, ui_dir / "web", port=args.port)
    server.start()
    print(f"control API on {server.url}", flush=True)
    if args.start:
        controller.start()
    try:
        if args.no_window:
            while True:
                time.sleep(3600.0)
        command = [sys.executable, str(ui_dir / "app.py"), server.url]
        if args.browser:
            command.append("--browser")
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        return 0
    finally:
        server.stop()
        controller.shutdown()
