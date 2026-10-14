# DEXAIV GUI — design spec

The approved design lives in `docs/gui/mockups/` as four self-contained HTML files
(open them in a browser; each renders one 1200x780 screen exactly as approved):

- `Live.html` — the home screen
- `Models.html`
- `Performance.html`
- `Settings.html`

These files ARE the spec for layout, spacing, copy, and visual style. Implement what
they show; where behavior isn't visible in static HTML, this document defines it.

## Design system (extract into theme constants, do not hardcode per-widget)

- Ground `#121217`, rail `#0D0D11`, panel `#1A1A21`, border `#24242D`,
  border-strong `#2E2E39`, text `#ECECF1`, muted `#8B8B98`, faint `#5C5C68`
- Accent `#E38BD0` (one accent, used for primary action, sliders, ticks, OUT meter);
  success `#58D6A3`; warn/over-budget `#E25D5D`
- Type: Space Grotesk (UI, bundle the font) + JetBrains Mono (every number, level,
  stat, model spec). Numbers are ALWAYS mono.
- Radii 9–14 px, panels 14. Dark only for V1.

## Screens and behavior

### Live (home)
- START/STOP: starts/stops the engine worker. STOP shows while running. Status dot +
  "Live / Gate open · speaking" reflect engine + gate state (gate closed = "Gate closed",
  dot dims).
- Latency readout: big mono number = live end-to-end estimate from the profiler
  (plan.algorithmic_latency_ms + ring fill + device buffers), refreshed ~2x/s;
  "N dropouts" = underrun counter this session.
- Voice model card: current model from the registry (name, version, sr, index state);
  "Change" navigates to Models.
- Pitch card: value from the profile; slider applies between hops (no restart); the
  "auto · calibrated" badge shows while value == calibrated value, hides once edited.
- Levels card: IN/OUT meters at >=15 fps from the audio rings; the accent tick on the
  IN meter is the gate threshold; the slider below sets it live. Right column: Noise
  gate toggle, Suppression (RNNoise) toggle, both live. Caption shows the calibrated
  noise floor.
- Type-to-speak bar: text + Enter (or Speak) renders TTS via src/vox/tts and plays it
  through the SAME output path as live conversion. While the clip plays, the converted
  mic feed is muted (priority override), then live resumes; the engine never stops.
  The caption row shows TTS voice, auto pitch, and the override note; History opens a
  small popover of past clips with replay.
- Footer: input and output device readouts (click -> Settings).

### Models
- Grid of model cards from the registry. Card: name, auto-detected spec line
  (version · sr · f0), tags, Index dropdown, Use, preview-play.
- Index dropdown per model: auto-paired index selected by default; other compatible
  .index files found in the model's folder; "Browse…"; "No index (off)". A
  dimension-mismatched index is listed but marked and not selectable.
- Import: button + whole-window drag-drop. A dropped .pth imports via
  `vox models import`; a .pth AND .index dropped together import as one voice with the
  index paired. Import runs off the UI thread; card appears with detected metadata.

### Performance
- Live profiler view: header stats (end-to-end estimate, underruns, headroom %),
  per-stage p50 bars with p50/p99 values against the hop budget, 60 s hop-time chart
  (p50 + p99 lines, budget line), System panel (GPU/CPU/VRAM, output ring fill, gate
  state, session time), Export report (writes the bench report file).

### Settings
- Header: profile name + calibration date; "Recalibrate my voice" launches the Phase 5
  calibration flow (stub if not built yet).
- Audio column: mic/output device selects (from vox devices), monitor toggle,
  exclusive-WASAPI toggle, silence gate toggle.
- Conversion column: index rate + protect sliders (live-apply), pitch tracker select
  (RMVPE/FCPE; applying re-captures with a brief gate-closed mute), calibrated F0
  clamp note.
- Engine column (advanced): hop size, context history, CUDA graphs toggle, auto-tune
  hop toggle; the info box text about re-capture is shown as-is.

## Architecture rules (non-negotiable, from docs/research/12)

- The GUI is a THIN client over src/vox/api (EngineController + events). It owns NO
  engine state; it renders state pushed by the event bus and sends commands. The
  backend must keep working headless via the CLI.
- GUI runs in its own process or thread group; never calls into the hot path; meter
  and profiler data arrive via the events channel (snapshots, not per-hop messages).
- Live-applicable params (pitch, index rate, protect, gate, thresholds, suppression)
  apply between hops. Graph-invalidating params (hop, context, F0 method, devices)
  trigger the controlled re-capture path.

## Framework

Use what best renders this spec on Windows with Python 3.13 without fighting the
design: the mockups are plain HTML/CSS, so a webview shell (e.g. pywebview) over a
small local API is the straightest path; a Qt (PySide6) build matching the spec is
acceptable if webview proves problematic. Decide, justify in one paragraph in the PR,
and keep the choice isolated under ui/ so it can be swapped.
