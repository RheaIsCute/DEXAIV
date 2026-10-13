"""F0 extractors behind the ``PitchExtractor`` interface (GPU)."""

from __future__ import annotations

from vox.pitch.base import PitchExtractor

EXTRACTORS = ("rmvpe", "fcpe")


def make_extractor(name: str) -> PitchExtractor:
    """Construct (not load) an extractor by name. Imports stay lazy: control path only."""
    if name == "rmvpe":
        from vox.pitch.rmvpe import RmvpeExtractor

        return RmvpeExtractor()
    if name == "fcpe":
        from vox.pitch.fcpe import FcpeExtractor

        return FcpeExtractor()
    raise ValueError(f"unknown f0 method {name!r}; choose from {EXTRACTORS}")
