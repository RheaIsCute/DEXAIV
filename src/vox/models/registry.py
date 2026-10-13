"""Model registry: name -> loaded synthesizer + config (docs/research/12 §3.7).

Control path only. A model switch is always stop -> load -> realloc -> start; nothing here
is ever called from the hop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from vox.config.schemas import ModelConfig
from vox.config.store import load_model_config
from vox.models import discovery
from vox.models.metadata import MetadataError
from vox.synthesis.rvc_model import RvcModel


@dataclass
class LoadedModel:
    name: str
    model: RvcModel
    config: ModelConfig
    index_path: Path | None


class ModelRegistry:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir)
        self._loaded: LoadedModel | None = None  # one resident model in V1

    def names(self) -> list[str]:
        return discovery.scan(self.models_dir)

    def config(self, name: str) -> ModelConfig:
        directory = self.models_dir / name
        if not (directory / discovery.MODEL_FILE).is_file():
            raise MetadataError(f"no model named {name!r} in {self.models_dir} "
                                f"(have: {', '.join(self.names()) or 'none'})")
        discovery.ensure_config(directory)
        return load_model_config(directory / discovery.CONFIG_FILE)

    def get(self, name: str, device: torch.device | str) -> LoadedModel:
        device = torch.device(device)
        if (self._loaded and self._loaded.name == name
                and self._loaded.model.device == device):
            return self._loaded
        config = self.config(name)
        directory = self.models_dir / name
        model = RvcModel.load(directory / discovery.MODEL_FILE, device)
        index_path = directory / config.index.file if config.index.file else None
        self._loaded = LoadedModel(name, model, config, index_path)
        return self._loaded
