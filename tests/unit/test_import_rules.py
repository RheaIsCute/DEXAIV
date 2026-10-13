"""The dependency rules of docs/research/12 §2 and §4, enforced by scanning imports."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "vox"
CONTROL = {"vox.offline", "vox.api", "vox.calibration", "vox.cli"}
RULES = {
    # stage packages never reach up into the engine or the control path
    "pitch": CONTROL | {"vox.engine", "vox.models"},
    "features": CONTROL | {"vox.engine", "vox.models"},
    "retrieval": CONTROL | {"vox.engine", "vox.models"},
    "audio": CONTROL | {"vox.engine", "vox.models"},
    # synthesis may read detection results from models.metadata, nothing else upward
    "synthesis": CONTROL | {"vox.engine", "vox.models.registry", "vox.models.discovery"},
    # the hot path receives already-constructed objects; it never imports the control path
    "engine": CONTROL | {"vox.models"},
    "profiling": CONTROL | {"vox.engine", "vox.models"},
    "config": CONTROL | {"vox.engine", "vox.models"},
}


def imports_of(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


@pytest.mark.parametrize("package", sorted(RULES))
def test_package_respects_dependency_direction(package):
    violations = []
    for path in sorted((SRC / package).rglob("*.py")):
        for name in imports_of(path):
            if any(name == bad or name.startswith(bad + ".") for bad in RULES[package]):
                violations.append(f"{path.relative_to(SRC)} imports {name}")
    assert not violations, "\n".join(violations)


def test_nothing_imports_ui():
    for path in SRC.rglob("*.py"):
        assert not any(name == "ui" or name.startswith("ui.") for name in imports_of(path)), path


def test_chunker_is_pure_and_audio_stays_light():
    assert not {n for n in imports_of(SRC / "engine" / "chunker.py")
                if n.split(".")[0] in ("torch", "numpy", "sounddevice")}
