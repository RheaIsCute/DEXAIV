# ui/ — the DEXAIV window

A thin client over `vox.api` (docs/gui/GUI_SPEC.md, docs/research/12 P9/P12). It owns no
engine state: it renders the state the API pushes and sends one command per control.

| Path | Role |
|---|---|
| `app.py` | The window: a pywebview shell given the API's URL by `vox gui`. Imports nothing from `vox`. Falls back to the default browser without pywebview or with `--browser`. |
| `web/index.html` | The four screens (Live, Models, Performance, Settings) as one page. |
| `web/styles.css` | The theme: every colour, radius and typeface as variables, from the spec's design system. |
| `web/app.js` | Event stream in (`state`, `meters`, `perf`, `toast`), commands out. No framework. |
| `web/fonts/` | Space Grotesk and JetBrains Mono (SIL Open Font License 1.1), latin subsets from Fontsource. |

Rules, enforced by `tests/unit/test_import_rules.py`:

- `ui/` reaches the engine only through `vox.api` (in practice: only over its HTTP API),
- nothing under `src/vox` imports from `ui/`; the engine runs with no UI installed.

The framework choice is confined to `app.py`: any shell that can show a URL can replace it.

```powershell
pip install -r requirements/gui.txt
vox gui                      # the window
vox gui --browser            # the same pages in the default browser
vox gui --no-window          # only the API; prints its URL
```
