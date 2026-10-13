"""Profiler text report, JSON export and environment fingerprint (docs/research/04 §5.3, 10 §7).

Control path only: everything here may synchronize or touch the filesystem.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vox.profiling.stats import headroom
from vox.profiling.timers import TOTAL, StageTimers


def environment_fingerprint(device: torch.device | str | None = None) -> dict[str, Any]:
    """GPU / driver / CUDA / torch versions: recorded in every benchmark result."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import torchaudio

        info["torchaudio"] = torchaudio.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        index = torch.device(device).index if device is not None else None
        index = torch.cuda.current_device() if index is None else index
        props = torch.cuda.get_device_properties(index)
        info.update(
            gpu=props.name,
            vram_mb=props.total_memory // (1024 * 1024),
            compute_capability=f"{props.major}.{props.minor}",
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            tf32_cudnn=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            driver=_nvidia_driver_version(),
        )
    return info


def _nvidia_driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def stage_table(timers: StageTimers, skip: int = 0) -> dict[str, dict[str, Any]]:
    """Per-stage host and GPU-event percentiles (ms) as plain data."""
    timers.flush()
    table: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(timers.stages):
        host = timers.host.summary(i, skip)
        gpu = timers.gpu.summary(i, skip)
        if host.count == 0:
            continue
        row: dict[str, Any] = {"host": host.__dict__}
        if gpu.count:
            row["gpu"] = gpu.__dict__
        table[name] = row
    return table


def format_report(
    timers: StageTimers,
    *,
    hop_ms: float,
    title: str = "rt-profiler",
    skip: int = 0,
    notes: dict[str, Any] | None = None,
    alloc: dict[str, float] | None = None,
    sola_offsets_ms: np.ndarray | None = None,
) -> str:
    """The rt-profiler table. ``host`` is wall time, ``gpu`` is CUDA-event device time."""
    table = stage_table(timers, skip)
    total = table.get(TOTAL, {}).get("host", {})
    hops = total.get("count", 0)
    header = f"=== {title} | {hops} hops (after {skip} warmup) | hop={hop_ms:.1f} ms"
    if notes:
        header += " | " + ", ".join(f"{k}={v}" for k, v in notes.items())
    lines = [header + " ==="]
    lines.append(f"{'stage':<16}{'host p50':>9}{'p95':>8}{'p99':>8}{'max':>8}"
                 f"   {'gpu p50':>8}{'p95':>8}{'p99':>8}{'max':>8}   ms")
    rule = "-" * len(lines[-1])
    lines.append(rule)

    def row(name: str) -> str:
        host = table[name]["host"]
        text = (f"{name:<16}{host['p50']:>9.2f}{host['p95']:>8.2f}{host['p99']:>8.2f}"
                f"{host['max']:>8.2f}")
        gpu = table[name].get("gpu")
        if gpu:
            text += (f"   {gpu['p50']:>8.2f}{gpu['p95']:>8.2f}{gpu['p99']:>8.2f}"
                     f"{gpu['max']:>8.2f}")
        return text

    for name in table:
        if name != TOTAL:
            lines.append(row(name))
    if TOTAL in table:
        lines.append(rule)
        lines.append(row(TOTAL))
        ratio = headroom(hop_ms, total["p99"])
        lines.append(f"headroom (hop / p99 total) = {ratio:.2f}x   "
                     f"jitter (p99 - p50) = {total['p99'] - total['p50']:.2f} ms")
    counters = timers.counters.as_dict()
    lines.append("  ".join(f"{k}: {v}" for k, v in counters.items()))
    if sola_offsets_ms is not None and sola_offsets_ms.size:
        p50, p95 = np.percentile(sola_offsets_ms, [50, 95])
        lines.append(f"sola offset p50={p50:.2f} ms p95={p95:.2f} ms")
    if alloc:
        lines.append(
            f"allocator: driver mallocs={alloc['driver_mallocs']}  "
            f"reserved growth={alloc['reserved_growth_bytes']} B  "
            f"live drift={alloc['live_drift_bytes']} B  "
            f"pool allocs/hop={alloc['pool_allocs_per_hop']:.0f} (eager temporaries)"
        )
    return "\n".join(lines)


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=_jsonable) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


if __name__ == "__main__":
    json.dump(environment_fingerprint(), sys.stdout, indent=2)
