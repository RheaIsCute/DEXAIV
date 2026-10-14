# TTS notes (`vox say`)

What was learned while adding text-to-speech output through the RVC engine. Measured on
2026-10-13: Windows 11, Python 3.13.15, torch 2.7.1+cu128, test model `base_f0_48k_v2`.

## Pipeline

`vox say` -> `vox.tts.backends` renders the text to a WAV -> loaded as mono 16 kHz ->
`FullFileConverter` (the same object `vox convert --mode full` uses, built by `load_engine`
with `warmup_hops=0`) -> WAV at the model's sample rate and/or `sounddevice` playback.
Nothing in `engine/`, `audio/` or `offline/` was changed.

## Backends

| | kokoro (default) | edge (fallback) |
|---|---|---|
| Package | `kokoro-onnx==0.6.1` + `onnxruntime==1.31.0` | `edge-tts==7.3.1` |
| Runs | locally, CPU | Microsoft's service, needs internet |
| Weights | 2 files, about 340 MB, fetched on first use | none |
| Output | 24 kHz mono float | 24 kHz MP3 |
| Voices | 54 (`af_*`, `am_*`, `bf_*`, `ef_dora`, `em_alex`, ...) | 322 (`en-US-AriaNeural`, `es-MX-DaliaNeural`, ...) |

- **The torch `kokoro` package does not install on Python 3.13.** Every release that
  matters depends on `misaki`, whose releases all declare `Requires-Python <3.13`; pip
  fails with "No matching distribution found for misaki". `kokoro-onnx` has no such pin
  and its install changes nothing already in `requirements/runtime.txt` (numpy and torch
  stay where they are). It brings `espeakng-loader`, which bundles espeak-ng, so no system
  install is needed for phonemization.
- Kokoro runs on the CPU: the `onnxruntime` wheel only offers the CPU provider. That is
  fine here (the 2.9 s test sentence rendered in 0.45 s once loaded; loading plus a first
  short render took 2.3 s) and keeps the GPU for RVC.
  `onnxruntime-gpu` was not tried.
- The Kokoro weights come from the `model-files-v1.1` release of
  `thewh1teagle/kokoro-onnx` (`kokoro-v1.0.onnx`, `voices-v1.0.bin`) and land in
  `models/_tts/kokoro/`. Their sha256 digests are pinned in `backends.py` and checked after
  download. `models/_tts/` is already ignored by the existing `models/*/` rule.
  `--list-voices` only needs the small voices file.
- Kokoro's voice name carries the language (`a` American, `b` British, `e` Spanish,
  `f` French, `h` Hindi, `i` Italian, `j` Japanese, `p` Portuguese, `z` Mandarin); the
  backend passes the matching espeak language code. Only the English and the edge paths
  were listened to.
- Kokoro is not bit-reproducible between runs: the same sentence measured 195.9 Hz and
  196.5 Hz median F0 on two runs. `--seed` fixes the RVC side only.
- edge-tts streams MP3. `soundfile` 0.14 (libsndfile 1.2) decodes it, so no ffmpeg is
  needed. Every edge failure (offline, unknown voice) surfaces as one `error:` line.
- When `--backend` is omitted and kokoro is not installed, edge is used and a note is
  printed. If `--voice` was given in that situation it is an error, because voice names
  are backend-specific.

## Pitch

The transposition for TTS input is unrelated to the user's calibrated microphone pitch, so
`vox say` ignores `f0_up_key` from the profile and the model config. Without `--pitch` it
extracts F0 from the rendered TTS audio with the project's extractor (RMVPE by default),
takes the median over voiced frames of the whole utterance, and uses
`round(12 * log2(target / tts_median))`, clamped to ±24.

`target` is `target_f0_median_hz` for the model, read from `profiles/<p>/model_overrides.json`
(`models.<Name>.target_f0_median_hz`) or else from the `defaults` of the model's
`config.json`, or else 200 Hz. No model has this key yet; both files tolerate unknown keys.

Measured on "Hello, this is a test of the text to speech path." with the 200 Hz default:

| Backend / voice | TTS median F0 | Auto pitch |
|---|---|---|
| kokoro `af_heart` | 195.9 Hz | +0 |
| kokoro `am_michael` | 113.1 Hz | +10 |
| edge `en-US-AriaNeural` | 208.9 Hz | -1 |
| edge `en-US-GuyNeural` | 159.7 Hz | +4 |

The 200 Hz default suits a female target. For a male model set `target_f0_median_hz`
(roughly 110-130 Hz) or pass `--pitch`; otherwise a male TTS voice is pushed up ten
semitones. Starting from a TTS voice of the target's gender keeps the shift near zero,
which is where RVC sounds most natural.

## Long text

Text is split at sentence ends into pieces sized by character count (14 characters per
second, so at most 420 characters for the 30 s budget), because the duration is not known
before rendering. Kokoro speaks English at about 16 characters per second, so real pieces
come out shorter than the estimate: a 720-character paragraph became 2 pieces, 44.5 s in
total. A sentence longer than the budget is cut at `, ; :` and then between words. Both
backends trim silence, so 200 ms of silence is appended to every piece but the last before
conversion; the converted pieces are joined with a 10 ms linear crossfade. The pitch is
measured once over all pieces.

## Not verified

- Quality was checked by level and duration and by one playback, not by a listening test
  across voices. `base_f0_48k_v2` is the untrained base generator, not a real voice.
- Three real v2 voices were also run (two at 32 kHz, one at 48 kHz; two with an index,
  at `--index-rate 0` and `0.5`, on kokoro, and one on edge). All produced full-length
  output at a normal level with auto pitch +0 from `af_heart`; nobody has listened to
  judge how close they are to the target voices.
- `--play` ran on the system default device and with `--device` set to that same
  device's name; no second output device was tried.

## Expressive typed speech (`vox.tts.expressive`)

Informal text should sound informal. The RVC stage keeps whatever prosody the TTS renders,
so every expressive decision is made before or at the TTS call, by the pure function
`expressive.analyze(text) -> list[ExpressivePiece]` (text for the phonemizer + render
hints); `speak.split_pieces` packs those pieces under the 30 s budget exactly as
`split_text` packed sentences, except that neighbours with different hints never join and
an explicit pause is never packed away. Rules, each unit-tested in
`tests/tts/test_expressive.py`:

| Mark | Phonemizer sees | Render hint |
|---|---|---|
| `...` `....` `…` `--` `—` | the dots removed; a piece boundary there | pause of 250 ms + 120 ms per dot beyond three, capped at 1.2 s (a dash = 250 ms), inserted as silence between the pieces instead of the usual 200 ms sentence gap |
| letter repeated 3+ times ("heyyy", "nooooo") | two repeats ("heyy", "noo") | speed 0.92 at 3 repeats down to 0.78 at 6+, linear, from the longest run in the clause; a marked sentence is split at `, ; :` so only its clause slows down |
| word in ALL CAPS (2+ letters, not in the `ACRONYMS` allowlist: TTS, GPU, OK, LOL, US, ...) | lowercased; `!` appended if the clause has no terminal punctuation (a trailing `,;:` becomes `!`) | speed 1.05 and +1.5 dB gain on that clause |
| `!!` `??` `?!?!` | one mark | the same speed 1.05 and +1.5 dB |
| anything else | untouched | none: no slang normalisation, no autocorrect |

Speed goes to the backend per piece: kokoro's `create(..., speed=)`, edge-tts's prosody
`rate` (`"-8%"` for 0.92; the parameter is only sent when the speed is not 1.0, so a plain
render is the same request as before). Gain is applied to the **converted** audio of the
piece: the converter peak-normalises its input, so gain before RVC would vanish. After the
pieces are joined the utterance is scaled down only if a gained piece pushed its peak past
0.99. One transposition for the whole utterance, as before.

### Acceptance (2026-10-13, kokoro `af_heart`, `--pitch 0`, seed 0)

`scripts/expressive_cases.py` renders each case and, as a reference, the normalised text
through the pre-change path (`<case>_plain.wav`); it then measures the silent gap between
the first two speech segments (10 ms RMS frames above -40 dBFS) and levels. Wavs are
committed under `out/expressive/` (`base_f0_48k_v2`, 48 kHz) and `out/expressive/mae/`
(`MaeASMR`, a real 32 kHz voice, for listening).

| case | text | pieces | expressive | plain reference |
|---|---|---|---|---|
| dots + emphasis | `hey... hey!! are you fr right now` | `hey` (pause 250 ms) / `hey!` (1.05x, +1.5 dB) / `are you fr right now` | gap at the dots **430 ms** (base) / **530 ms** (Mae); "hey!" +0.9 / +1.7 dB over the first "hey", +1.1 / +1.9 dB over the plain render's "hey!" | gap 280 / 270 ms (`hey. hey! ...` as one piece: kokoro's own sentence pause) |
| elongation | `nooooo waaaay` | `noo waay` at speed 0.83 (run of 5) | **1.14 s** (base) / 1.16 s (Mae) | 0.92 s for `noo waay` at 1.0: the marked text is 24% longer |
| caps | `that is SO not what i said` | `that is so not what i said!` at 1.05, +1.5 dB | 1.56 s, overall level **+1.5 dB** (base) / +1.9 dB (Mae) over the plain render | 1.66 s |
| formal | `Hello, this is a test of the text to speech path.` | one plain piece | 2.92 s, same level, same length | 2.92 s |

- The measured pause at the dots is the inserted 250 ms plus the ~180 ms of trailing
  silence kokoro leaves on a bare "hey" (it trims less than a full sentence end), which is
  why it lands at the ~400 ms the acceptance asked for.
- **Byte-identical formal path.** Kokoro is not bit-reproducible between runs (two renders
  of the formal sentence differ by up to 0.10 in sample value), so the comparison feeds one
  pre-rendered TTS wav through `speak.speak` on the pre-change tree (`feature/tts`, commit
  e1d3ada) and on this tree, same seed, HuBERT FP32: the two output files are
  byte-identical (`cmp`), and the formal sentence runs through the old code path
  (one piece, speed 1.0, gain 0, 200 ms gap, no peak limiter).
- Under the Part 2 defaults (HuBERT autocast ladder) `vox say` still runs HuBERT in FP32:
  the full-file converter builds its engine without warmup, and the ladder is a warmup
  decision on the streaming hop, so the TTS path's output is unchanged.
- Not verified by ear: nobody has listened to the four cases yet; durations, gaps and
  levels are what the table measures. "noo" / "waay" / "soo" is what the spec's
  two-repeat normalisation gives the phonemizer; if espeak reads "soo" as "sue" the
  normalisation should go to the dictionary spelling instead, which needs a word list.
