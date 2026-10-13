# models/

User model storage (docs/research/09 §6). Weights are gitignored.

    models/<Name>/{model.pth, model.index, config.json}
    models/_embedders/contentvec/{config.json, pytorch_model.bin}
    models/_predictors/rmvpe.pt

- `python scripts/fetch_assets.py` downloads the shared embedder and predictor weights.
- `vox models import path/to/voice.pth` copies a model in and writes its detected `config.json`.
