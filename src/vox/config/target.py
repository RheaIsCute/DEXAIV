"""The one supported machine (TARGET.md): assert it, size for it, stop paying for generality.

Control path only. ``select_device`` is the only way the engine picks its compute device:
it resolves to the RTX 5060 Ti or raises ``TargetError`` with a message that says what was
found instead. The iGPU and virtual display adapters are never CUDA devices, so CUDA
enumeration cannot pick them, but a second NVIDIA card or a renamed device would be
caught by the name / capability / VRAM checks. The pure-CPU path exists only for unit
tests, behind ``VOX_ALLOW_CPU=1``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

GPU_NAME_FRAGMENT = "5060 Ti"
COMPUTE_CAPABILITY = (12, 0)  # Blackwell
MIN_VRAM_GIB = 15.0  # 16311 MiB reported; leave room for driver rounding
TORCH_CUDA = "12.8"  # torch 2.7.1+cu128 (requirements/runtime.txt)
CPU_ENV_FLAG = "VOX_ALLOW_CPU"

# Memory sizing (TARGET.md rule 3): ~12 GiB usable for the engine after Windows/driver
# overhead on the 16 GiB card. The index cap is what a brute-force k-NN index may occupy.
GIB = 1 << 30
ENGINE_BUDGET_BYTES = 12 * GIB
INDEX_BUDGET_BYTES = 2 * GIB


class TargetError(RuntimeError):
    """The compute device is not the supported machine's GPU."""


@dataclass(frozen=True)
class TargetReport:
    device: torch.device
    name: str
    capability: tuple[int, int]
    vram_gib: float
    torch_cuda: str

    def line(self) -> str:
        return (f"hardware check: {self.device} = {self.name}, cc {self.capability[0]}."
                f"{self.capability[1]}, {self.vram_gib:.1f} GiB, torch {torch.__version__}: "
                f"OK (TARGET.md)")


def cpu_allowed() -> bool:
    return os.environ.get(CPU_ENV_FLAG, "") not in ("", "0", "false", "no")


def _describe(index: int) -> tuple[str, tuple[int, int], float]:
    props = torch.cuda.get_device_properties(index)
    return props.name, (props.major, props.minor), props.total_memory / GIB


def check_cuda_device(index: int) -> TargetReport:
    """Raise ``TargetError`` unless ``cuda:index`` is the RTX 5060 Ti of TARGET.md."""
    name, capability, vram = _describe(index)
    problems = []
    if GPU_NAME_FRAGMENT.lower() not in name.lower():
        problems.append(f"name {name!r} does not contain {GPU_NAME_FRAGMENT!r}")
    if capability != COMPUTE_CAPABILITY:
        problems.append(f"compute capability {capability} != {COMPUTE_CAPABILITY}")
    if vram < MIN_VRAM_GIB:
        problems.append(f"VRAM {vram:.1f} GiB < {MIN_VRAM_GIB} GiB")
    if (torch.version.cuda or "") != TORCH_CUDA:
        problems.append(f"torch {torch.__version__} is built for CUDA {torch.version.cuda}, "
                        f"not {TORCH_CUDA}")
    if problems:
        raise TargetError(
            f"cuda:{index} is not the supported GPU: " + "; ".join(problems)
            + ". This project runs on one machine only (see TARGET.md: RTX 5060 Ti, "
              "16 GiB, torch 2.7.1+cu128).")
    return TargetReport(torch.device("cuda", index), name, capability, vram,
                        torch.version.cuda or "")


def select_device(requested: torch.device | str | None = None) -> tuple[torch.device, str]:
    """Resolve the compute device and verify it. Returns ``(device, check line)``.

    ``None`` or ``"cuda"`` picks the RTX 5060 Ti among the CUDA devices; ``"cuda:N"`` must
    be it; ``"cpu"`` is allowed only with ``VOX_ALLOW_CPU=1`` (unit tests).
    """
    requested = torch.device(requested) if requested is not None else None
    if requested is not None and requested.type == "cpu":
        if not cpu_allowed():
            raise TargetError(
                f"the CPU path is for unit tests only; set {CPU_ENV_FLAG}=1 to use it "
                "(TARGET.md rule 1)")
        return requested, f"hardware check: cpu (unit-test path, {CPU_ENV_FLAG}=1)"
    if not torch.cuda.is_available():
        if requested is None and cpu_allowed():
            return torch.device("cpu"), f"hardware check: cpu (no CUDA; {CPU_ENV_FLAG}=1)"
        raise TargetError("no CUDA device is visible to torch; the RTX 5060 Ti is required "
                          "(TARGET.md rule 1). Check the NVIDIA driver and that torch is the "
                          "cu128 build.")
    if requested is not None and requested.index is not None:
        report = check_cuda_device(requested.index)
        return report.device, report.line()
    candidates = [i for i in range(torch.cuda.device_count())
                  if GPU_NAME_FRAGMENT.lower() in _describe(i)[0].lower()]
    if not candidates:
        found = ", ".join(f"cuda:{i} {_describe(i)[0]}" for i in range(torch.cuda.device_count()))
        raise TargetError(f"no {GPU_NAME_FRAGMENT!r} among the CUDA devices ({found}); see "
                          "TARGET.md rule 1")
    report = check_cuda_device(candidates[0])
    return report.device, report.line()


def index_vector_cap(emb_dim: int, budget_bytes: int = INDEX_BUDGET_BYTES) -> int:
    """How many FP32 index vectors of ``emb_dim`` fit the index budget (plus their norms)."""
    return max(1, budget_bytes // (4 * (emb_dim + 1)))


def memory_line(device: torch.device) -> str:
    """One line of allocator state against the engine budget, for the startup log."""
    if device.type != "cuda":
        return "memory: cpu"
    reserved = torch.cuda.memory_reserved(device) / GIB
    allocated = torch.cuda.memory_allocated(device) / GIB
    free, total = torch.cuda.mem_get_info(device)
    note = "" if reserved <= ENGINE_BUDGET_BYTES / GIB else "  OVER BUDGET"
    return (f"memory: reserved {reserved:.2f} GiB (allocated {allocated:.2f}) of a "
            f"{ENGINE_BUDGET_BYTES / GIB:.0f} GiB engine budget; device free "
            f"{free / GIB:.1f} / {total / GIB:.1f} GiB{note}")
