# ui/ (placeholder)

Intentionally empty. A future UI lives here and obeys one contract (docs/research/12, P9/P12):

- it talks to the engine only through `vox.api` (JSON commands and events),
- it imports nothing from `vox.engine`,
- nothing under `src/vox` ever imports from `ui/`; the engine runs with no UI installed.
