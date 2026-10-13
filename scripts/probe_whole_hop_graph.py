"""Q16 probe: capture the WHOLE hop as one CUDA graph and try to replay it, under a timeout.

Phase 2 found that whole-hop replay never returns (docs/DEVIATIONS.md R2). This script
re-tries with the Phase 3 changes in place: no RNG inside the capture (decoder noise is
pre-filled), 3 warmup iterations on a side stream at the exact shapes, and replay on the
capture (current) stream. It runs the attempt in a child process so a hang is killed and
recorded instead of wedging the session.

    python scripts/probe_whole_hop_graph.py [--timeout 90] [--replays 200]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def child(replays: int, stages: str = "") -> int:
    import numpy as np
    import torch

    from vox.engine.graphs import StageGraphs
    from vox.engine.warmup import test_signal
    from vox.offline.simulate import load_engine

    engine = load_engine(ROOT / "models", "base_f0_48k_v2", graphs="off", log=print)
    pipeline, sola, plan = engine.pipeline, engine.sola, engine.plan
    pipeline.timers.enabled = False
    signal = test_signal(400 * plan.block, plan.device_sr)
    for n in range(60):  # real content in every buffer
        pipeline.upload(signal[n * plan.block:(n + 1) * plan.block])
        sola.stitch(pipeline.process_hop())
    torch.cuda.synchronize()

    def timed(fn, n):
        times = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000)
        return np.percentile(times[max(1, n // 10):], [50, 95, 99]).round(2).tolist()

    print("eager hop:", timed(lambda: sola.stitch(pipeline.process_hop()), 100), flush=True)
    fused = tuple(stages.split("+")) if stages else True
    graphs = StageGraphs(pipeline, sola, "on", whole_hop=fused)
    t = time.perf_counter()
    graphs.capture()
    print(f"whole-hop capture ok in {(time.perf_counter() - t) * 1000:.0f} ms; "
          f"stages: {graphs.report.stages}", flush=True)
    print("replaying once...", flush=True)
    t = time.perf_counter()
    with torch.inference_mode():
        pipeline.process_hop()
        sola.stitch(pipeline.out)
    torch.cuda.synchronize()
    print(f"first replay returned in {(time.perf_counter() - t) * 1000:.1f} ms", flush=True)
    print("whole-hop replay:",
          timed(lambda: sola.stitch(pipeline.process_hop()), replays), flush=True)
    graphs.verify(torch.from_numpy(signal[100 * plan.block:110 * plan.block]
                                   .reshape(10, plan.block)).cuda())
    print("verify:", graphs.report.verify, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--stages", default="",
                        help="fuse only these consecutive stages, e.g. f0+embed (default: all)")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        return child(args.replays, args.stages)
    cmd = [sys.executable, __file__, "--child", "--replays", str(args.replays),
           "--stages", args.stages]
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        out, hung, code = proc.stdout + proc.stderr, False, proc.returncode
    except subprocess.TimeoutExpired as err:
        out = (err.stdout or b"").decode(errors="replace") if isinstance(err.stdout, bytes) \
            else (err.stdout or "")
        out += (err.stderr or b"").decode(errors="replace") if isinstance(err.stderr, bytes) \
            else (err.stderr or "")
        hung, code = True, None
    elapsed = time.perf_counter() - start
    lines = [ln for ln in out.splitlines() if "Warning" not in ln and ln.strip()]
    print("\n".join(lines[-25:]))
    verdict = "HUNG (killed)" if hung else ("ok" if code == 0 else f"failed (exit {code})")
    print(f"\nwhole-hop graph probe: {verdict} after {elapsed:.1f} s")
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "verdict": verdict,
              "elapsed_s": elapsed, "tail": lines[-25:]}
    results = ROOT / "benchmarks" / "results" / "whole_hop_probe.jsonl"
    results.parent.mkdir(parents=True, exist_ok=True)
    with open(results, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return 0 if not hung else 1


if __name__ == "__main__":
    sys.exit(main())
