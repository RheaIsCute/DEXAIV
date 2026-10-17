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

## Typed-speech polish (2026-10-14, A/B only; docs/DEVIATIONS.md §7.5)

Three options for the typed path, all off by default (the plain path is the old call, byte
for byte); the live microphone never sees them:

- `TypedPolish(envelope, rms_frame_ms, contour_scale)` passed to `convert_rendered`:
  `envelope` is the `volume_envelope` of the clip alone; `rms_frame_ms` sets the frames of
  that loudness mix (default rvc-webui's 1 s at a 0.5 s hop; 40 = 40 ms at a 10 ms hop);
  `contour_scale` widens the voiced F0 around the utterance median in semitones
  (`offline.convert.scale_contour`).
- Kokoro voice blends: `--voice "af_heart*0.7+af_bella*0.3"` (weights normalised; any
  number of voices, `tts.backends.parse_voice_blend`).

`vox bench typed --model Mom` renders the report's two sentences once per variant into
`out/ab/typed_<variant>.wav` and measures F0 spread, loudness spread, voicing and HF/LF.
On Mom: contour x1.2 widens the F0 spread 3.0 -> 3.4 st; the blend is darker (+3.0 -> +1.6
dB HF/LF) and narrower (2.7 st); the 40 ms envelope mix narrows the loudness spread
(8.9 -> 8.2 dB), the opposite of what the report expected. Nothing has been listened to.

## Natural typed speech: normalisation and per-utterance variation

Pipeline: raw text -> `normalize.normalize()` -> `expressive.analyze()` (via
`speak.split_pieces`) -> one `variation.draw()` per utterance -> backend -> RVC. `vox say`
and the GUI both take this path; `--no-normalize` / `--variation 0` switch each half off,
and `vox say --dry-run "text"` prints raw -> normalized -> pieces without loading a model.

### Normalisation (`vox.tts.normalize`, pure)

In order: URLs become "link"; emoji are dropped except the laughing ones; `<3` is "love",
other emoticons are dropped. Laughter (`lol`, `lmao`, `lmfao`, `rofl`, `haha...`, `hehe`,
`xD`, 😂 😭 🤣) becomes one "haha" set off by commas, or nothing (`--laughter drop`, GUI
Settings > Typed speech > Laughter); it is never spelled. Then the slang table, then
numbers, then casing and the terminal mark. Every rule is in `tests/tts/test_normalize.py`
(241 cases: one per shipped slang entry plus the tables below).

**Slang table.** `src/vox/tts/data/slang_en.json` (103 entries), with
`profiles/<name>/slang.json` merged over it. Schema:

```json
{"words": {"wsg": "what's good?", "tn": "tonight", "fr": "fr", "meh": ""}}
```

- Keys are matched as whole words, case-insensitively; they may hold spaces (`no prob`),
  digits (`gr8`, `b4`, `2nite`: digits are only read as letters inside such entries) or a
  slash (`w/`, `w/o`, which may run into the next word: `w/friends`). The longest key wins.
- A value ending in `?` is a question phrase (`wbu`, `hbu`, `wyd`, `wdym`): at the end of a
  line it gets the `?` and a comma before it ("I don't know, what about you?"); elsewhere
  neither. A value starting with `,` is an aside (`ngl`, `tbh`, `imo`, `btw`, `omg`, `ofc`,
  `jk`, `istg`): commas where it meets other words ("Not gonna lie, I'm tired.").
- An empty value drops the word; mapping a word to itself switches a shipped entry off.
- A flat `{"word": "spoken"}` object is accepted too; keys starting with `_` are comments.
- The user file is re-read when it changes (mtime), or on Settings > Reload. A broken file
  is reported (GUI note under the buttons, a `vox say` warning) and the shipped list is
  used alone. Settings > Edit slang list creates the file with two examples the first time
  and opens it in the default editor.
- Case: an all-caps contraction keeps its emphasis (`DONT` -> `DON'T`); an all-caps
  abbreviation is not shouting (`IDK` -> "I don't know"); in a line typed in caps with at
  least one real caps word, every expansion is upper-cased (`WHY R U LIKE THIS` ->
  `WHY ARE YOU LIKE THIS`).
- Ambiguous entries are decided by small context rules only: `ur` is "you're" before an
  `-ing` word or a short adjective list ("ur so funny", "ur the best"), else "your";
  `r` is "are" between two words or opening a line before a pronoun ("r u ok"), else left
  alone; `ya` stays "ya" after see/love/thank/miss/... ("see ya"), else "yeah". `ill`,
  `were`, `its`, `well`, `lets` are deliberately not in the table.

**Numbers** use `num2words==0.5.14` (pure Python plus docopt; it installed cleanly on
Windows / Python 3.13 into the main `.venv` and is pinned in `requirements/tts.txt`; without
it digits pass through and espeak reads them). Money (`$5.50` -> "five dollars and fifty
cents", `$5k`, `€`, `£`), percentages, clock times (`10:30` "ten thirty", `10:05am` "ten oh
five AM", `12:00` "twelve o'clock"), `3pm` -> "three PM", ordinals, ranges (`2-3` "two to
three"), years 1100-2099 as years ("twenty twenty-six"), leading-zero codes digit by digit;
digits inside words (`ps5`) are left alone. `&` -> "and", `@` -> "at", `#tag` -> "tag",
`*` and `~` are removed. AM, PM and AFK were added to the expressive acronym allowlist so
espeak spells them without them counting as emphasis.

**Casing and the terminal mark.** A lone `i` is `I`; a sentence starts with a capital (not
after `...`, not after an initial: "the U.S. tomorrow"); a sentence or clause opening with
okay/yeah/nah/yep/nope/alright/hey gets a comma after it. A line without `. ! ? …` gets `?`
when it opens like a question (auxiliary + pronoun: "are you", "r u"; a wh-word + auxiliary
or "about"; "why") and `.` otherwise. A line ending in a dash pause, or whose last clause
the expressive layer will end with `!` (caps / `!!` emphasis), is left open.

**The expressive markers survive.** Dots, dashes, letter runs, caps words and repeated
`!?` are never rewritten (caps are only ever raised).
`test_every_expressive_test_input_keeps_its_hints` runs all 33 inputs of
`test_expressive.py` with and without normalisation: identical speed, gain and pauses.

Why it matters, checked with kokoro's phonemizer (`kokoro_onnx.tokenizer`, espeak-ng):
`$5` reads "dollar five", `IM` reads "ɪm", `fr` is spelled "F R", `lol` is "lahl", and a
lone capital `A` is the article (`A F K` -> "ə F K"; `AFK` / `afk` are spelled correctly).

### Variation (`vox.tts.variation`)

One knob, `variation` 0..1 (default 0.35; `vox say --variation`, GUI slider), drawn once per
utterance from `numpy.random.default_rng(seed)`; `seed` is the clock unless `--seed` is
given (the CLI prints it, so a draw can be reproduced). Every piece of the utterance uses
the same draw.

| what | bound at variation v | at the default 0.35 |
|---|---|---|
| TTS speed | x(1 ± 0.04 v), on top of the expressive speed | ±1.4% |
| Kokoro style | blend of 1-2 other voices with the same prefix (`af_*` for `af_heart`), together ≤ 0.15 v | ≤ 5.25% |
| transposition | ± 0.4 v semitones on top of the auto / manual pitch | ±0.14 st |
| explicit pauses (dots, dashes) | x(1 ± 0.25 v); the plain 200 ms sentence gap is not varied | ±8.75% |
| RVC seed | a fresh 31-bit seed | |

- **kokoro-onnx 0.6.1 takes a style array** in place of a voice name (`create(voice=
  NDArray)`); voice styles are (510, 1, 256) float32. `KokoroBackend.style(blend)` mixes
  them; a render without a blend still passes the voice name, as before. edge-tts voices
  cannot be blended (`TtsBackend.blends = False`). The GUI's allowlist can be set as
  `tts.blend_voices` in `gui_state.json` (default: the voice's own prefix family).
- **Fractional transposition needs no other path:** `f0_up_key` is a float through
  `semitones_to_ratio` (`2 ** (key / 12)`) in the full-file converter.
- **`variation 0` is the old path.** It draws nothing and passes the caller's seed (`vox say`
  0, the GUI None, as before). Verified byte-identical on `Mom`: one pre-rendered kokoro wav
  per piece of "hey... hey!! are you for real right now" through `speak.speak` on `main`
  (af28754) and on this branch with `draw(0.0, 0)`, seed 0: `cmp` finds no difference.
- **Kokoro was never deterministic.** Two renders of "Hi." in one process differ by up to
  0.089 in sample value, and `onnxruntime.set_seed()` does not change that. So "hi" twice
  was never bit-identical, and the two `hi_fixed_*` wavs below differ in bytes too; what
  sounded identical was the delivery (speed, style, transposition, pauses), which is what
  variation now changes.
- **`f0_contour_scale`** (`--f0-contour-scale`, GUI "Livelier intonation" = 1.15; default
  1.0 = off) spreads voiced F0 around the median in semitone space before RVC. On `main`
  it is `TypedPolish.contour_scale` (the converter's `scale_contour`, median per piece);
  the branch's own `WidenedContour` (one median per utterance) was dropped when the two
  met. Off until approved by ear; no wav uses it yet.
- Prep cost (normalise + split into pieces + draw) per utterance, 500 runs of the cases
  below on the 7900X: median 0.066 ms, p99 0.40 ms, max 12 ms (one outlier); the budget was
  5 ms. Each clip's line in the GUI log now carries `prep N ms`.

### Acceptance (2026-10-16, kokoro `af_heart`, model `Mom`, auto pitch, target 200 Hz)

`scripts/natural_cases.py --model Mom` (`--dry-run` prints only the first table and the
timing). Dry run of the acceptance texts:

| raw | normalized | pieces (hints) |
|---|---|---|
| `idk wbu` | `I don't know, what about you?` | one piece, plain |
| `im so tired rn ngl` | `I'm so tired right now, not gonna lie.` | one piece, plain |
| `lol ok brb` | `Haha, okay, be right back.` | one piece, plain |
| `omg that was SO good!!` | `Oh my god, that was SO good!!` | `Oh my god,` plain / `that was so good!` speed 1.05, gain +1.5 dB |
| `hi` | `Hi.` | one piece, plain |
| `ngl i thought the movie was gonna be mid but it was SO good lol, we got there at 7:30 and it cost like $15 each... wbu, u still going tmrw?` | `Not gonna lie, I thought the movie was gonna be mid but it was SO good, haha, we got there at seven thirty and it cost like fifteen dollars each... what about you, you still going tomorrow?` | `Not gonna lie,` plain / `I thought the movie was gonna be mid but it was so good!` speed 1.05, gain +1.5 dB / `haha, we got there at seven thirty and it cost like fifteen dollars each` pause 250 ms / `what about you, you still going tomorrow?` plain |

Rendered wavs in `out/tts_natural/` (variation 0.35 unless "off"):

| file | variation draw | TTS median F0 | auto pitch | offset | length |
|---|---|---|---|---|---|
| idk_wbu.wav | speed x0.998, pauses x0.95, 3.3% af_aoede | 198.9 Hz | +0 | +0.000 | 1.76 s |
| tired_rn_ngl.wav | speed x1.005, pauses x0.98, 1.3% af_aoede | 195.8 Hz | +0 | +0.063 | 2.54 s |
| lol_ok_brb.wav | speed x1.002, pauses x0.91, 1.5% af_river | 190.2 Hz | +1 | +0.134 | 1.98 s |
| omg_so_good.wav | speed x1.013, pauses x0.98, 2.5% af_jessica | 206.4 Hz | -1 | -0.120 | 2.45 s |
| hi_var_1.wav | speed x1.010, pauses x0.95, 2.8% af_nicole + 0.4% af_bella | 208.2 Hz | -1 | +0.032 | 0.80 s |
| hi_var_2.wav | speed x1.001, pauses x1.07, 1.4% af_nova | 208.6 Hz | -1 | +0.044 | 0.80 s |
| hi_var_3.wav | speed x0.986, pauses x0.97, 1.2% af_nicole | 206.4 Hz | -1 | -0.102 | 0.80 s |
| hi_fixed_1.wav | off | 208.6 Hz | -1 | 0 | 0.80 s |
| hi_fixed_2.wav | off | 209.0 Hz | -1 | 0 | 0.80 s |
| long_casual.wav | speed x1.005, pauses x1.01, 0.6% af_sky + 0.5% af_kore | 196.3 Hz | +0 | -0.128 | 11.07 s |

- **The auto pitch is per utterance and rounded, so it is not stable across clips.** "Hi."
  measures 206-209 Hz, astride the 205.9 Hz line where `round(12 log2(200 / f))` flips
  between 0 and -1: in a first run of the same script `hi_var_1` came out +0 and the other
  four "hi" -1. Across sentences the median moves with the content (190-206 Hz above), so
  an excited line is pulled down a semitone and a flat one pushed up. That spread (±1 st) is
  larger than the variation offset (±0.14 st) and predates this work. The fix would be one
  transposition per TTS voice and model (measured once on a reference sentence and kept),
  not one per utterance. Not changed here: it alters the `variation 0` output.
- Not verified by ear yet: the wavs are for listening; durations, medians and offsets are
  what the tables measure. The expressive layer's two-repeat spelling still reaches espeak:
  `soo` is read "sue" and `waay` "wahee" (checked with the phonemizer), so "sooo tired"
  will say "sue tired" until that rule spells to the dictionary word.

### Later (not built)

Kokoro has no emotion control: variation and punctuation are the ceiling here. If more
expressive delivery is wanted, evaluate an emotion-capable TTS backend behind the existing
`TtsBackend` interface (one with an exaggeration / emotion parameter), keeping Kokoro as the
fast default. Other options noted while building this: a stable per-voice transposition
(above); dictionary respelling of elongated words; slang with elongation (`omggg`, `plzzz`
are not matched today).

## Expressive backends: the bake-off (2026-10-17)

Kokoro is deterministic and reads like a newsreader; RVC keeps the source's delivery, so the
TTS engine is the naturalness ceiling. Five expressive engines were set up behind
`TtsBackend` and rendered the same eight lines as Kokoro. Nothing was chosen: the user
picks by ear.

### How they run: one worker process per engine (`vox.tts.external`)

Every candidate pins its own torch / transformers, and some want another Python, so none
goes into the main `.venv`. Each has a virtual environment under `.tts_envs/<env>/` (its
Python downloaded by uv into `.tts_envs/_python`, torch 2.7.1+cu128 forced over the
package pins: the 2.6.0 that Chatterbox pins has no Blackwell kernels) and a worker script
in `src/vox/tts/workers/` that imports nothing from `vox`. `ExternalBackend` starts the
worker on first use and keeps it up; requests and replies are JSON lines
(`workers/_protocol.py`; fd 1 is pointed at stderr inside the worker so a library's prints
cannot corrupt the replies, which go to the worker log in `models/_tts/<name>_worker.log`).
Engine knobs travel as `synthesize(..., controls={...})`. `scripts/setup_tts_envs.py`
rebuilds everything; weights land in `models/_tts/hf` (Hugging Face cache) and
`models/_tts/orpheus`.

| backend | model | env | license (personal use: all OK) | expressive controls |
|---|---|---|---|---|
| `chatterbox` | Resemble AI Chatterbox 0.5B (chatterbox-tts 0.1.7) | py3.11 | MIT | `exaggeration` 0.25-2 (emotion intensity), `cfg_weight` (pace), `temperature`; voice cloning from a reference clip |
| `chatterbox-turbo` | Chatterbox Turbo 350M (same package) | py3.11 | MIT | tags in the text: `[laugh]` `[chuckle]` `[sigh]` `[cough]` ...; no exaggeration |
| `orpheus` | Canopy Orpheus 3B 0.1-ft, unsloth Q6_K GGUF in llama.cpp b11327 (CUDA 13.4 build) + SNAC 24 kHz | py3.12 | Apache-2.0 card; a Llama 3.2 3B fine-tune, so Meta's Llama 3.2 community license applies too | tags in the text: `<laugh>` `<chuckle>` `<sigh>` `<gasp>` `<groan>` `<yawn>` ...; 8 voices (tara default) |
| `qwen3` | Qwen3-TTS-12Hz-1.7B-CustomVoice (qwen-tts 0.1.1) | py3.12 | Apache-2.0 | `instruct`: natural-language delivery ("sad and quiet"); 9 preset speakers, none a native-English woman (Serena used) |
| `qwen3-design` | Qwen3-TTS-12Hz-1.7B-VoiceDesign | py3.12 | Apache-2.0 | the voice itself is a description; `instruct` appended for delivery |
| `f5` | F5-TTS v1 Base (f5-tts 1.1.22) | py3.11 | code MIT, **weights CC-BY-NC-4.0** (non-commercial) | none: it continues the voice *and delivery* of a reference clip; here 9.2 s of the user's own mic script (`models/_tts/f5/ref_user.wav`, transcript by F5's Whisper in `ref_user.txt`), so it speaks the way the user talks and RVC then makes it the model |

Installation snags (all handled by the setup script): Chatterbox's Perth watermarker needs
`pkg_resources` (setuptools < 81) and stamps an inaudible watermark on every render; F5 needs
`datasets >= 3.6` (2.x breaks on pyarrow 26), `huggingface-hub < 1` (transformers 4.57), and
torchcodec removed (built for a newer torch, it breaks Whisper's preprocessing; F5 does not
use it); F5's own transcription path wants ffmpeg, so the worker hands Whisper the decoded
array. Orpheus: llama-server does not stop on Orpheus's end-of-speech token, and a `stop`
string on `<custom_token_2>` never fired; it ran on into a hallucinated text turn. The
worker starts the server with `--override-kv tokenizer.ggml.eos_token_id=int:128258`, which
ends generation exactly there, prompts with Canopy's token ids (start-of-human, text,
end-of-text, end-of-human, start-of-AI, start-of-speech) and reads the codes from the
returned token ids. Each worker renders one sentence while loading (Chatterbox's first
`generate` otherwise takes 26 s).

### Results (`scripts/tts_bakeoff.py --model Mom`, 2026-10-17 01:15)

Eight lines, normalised first (`I don't know, what about you?` / `Haha, okay, be right back.` /
`Oh my god, that was SO good!!` / a question / a sad line with `...` / `WE WON!! I can't
believe we actually won!!` / the long movie line / `Hi.`). Kokoro took its production path
(expressive pieces, variation 0.35, seed 7); each other engine rendered every line `plain`
(default controls) and, where it has controls, `expressive` (set by hand per line:
Chatterbox exaggeration 0.4-0.9 with cfg 0.3-0.5, `[laugh]` / `<laugh>` for "haha" and
`[sigh]` / `<sigh>` before the sad line, a Qwen instruction per mood). Every render then went
through Mom (auto pitch, the shipped typed polish). Render timings were taken **while a live
session ran unheld** (the mic script looped through Mom into VAC Line 1, gate on), which is
also how the underruns were counted.

| backend | load s | warm, short line s (live / GPU idle) | RTF while live | VRAM: torch peak / GPU used MB | live underruns while rendering | Whisper WER after RVC, plain / expressive |
|---|---|---|---|---|---|---|
| kokoro (CPU) | 1.4 | 0.28 / 0.28 | ~0.2 (median render / median length) | 0 / 0 | 0 | 0.01 / — |
| chatterbox | 39.6 | 4.2 / 1.7 | 2.0 | 3287 / 3656 | 35 | 0.08 / 0.15 |
| chatterbox-turbo | 35.1 | 1.3 / 0.98 | 0.8 | 3026 / 3673 | 81 | 0.20 / 0.16 |
| orpheus (llama.cpp) | 18.5 | 3.6 / 2.1 | 1.5 | 510 (SNAC) / 4751 | 890 | 0.03 / 0.02 |
| qwen3 CustomVoice | 36.2 | 8.5 / 3.9 | 4.1 | 4435 / 4972 | 126 | 0.19 / 0.07 |
| qwen3 VoiceDesign | 32.8 | 5.8 / 4.2 | 4.1 | 4375 / 4839 | 100 | 0.14 / 0.14 |
| f5 (user's reference) | 33.4 | 3.1 / 1.6 | 0.9 | 1990 / 1173 | 305 | 0.06 / — |

"Short line" = the median of `idk wbu`, `lol ok brb` and `hi` (warm worker); "GPU idle" =
the same worker's 7-word smoke line with no live session. Load includes one warm-up render.
RTF = render time / audio length. "GPU used" is `nvidia-smi` before vs. with the worker up
(it includes the CUDA context and allocator cache; for Orpheus all of llama-server).

- **Every GPU engine starves the live engine while it renders.** It is a second process, so
  CUDA time-slices the two contexts and the hop graph waits. Today's GUI only holds the
  live voice for the RVC convert (`worker.hold`), after the TTS render, which was free
  while Kokoro ran on the CPU. A GPU engine needs the hold to start before the TTS render
  as well: the live voice is then silent for render + convert, and a late hop only delays
  silence (the hold cushion). That is the Step 3 change, whatever is picked.
- **Budget (about 4 GB):** Chatterbox, Turbo and F5 fit; the two Qwen3 1.7B models sit at
  4.4 GB of tensors; Orpheus takes 4.75 GB with a 4096-token context (2048 is enough for
  about 24 s of speech and should cut that). The 0.6B Qwen3 models do not take
  instructions.
- **CAPS words get spelled out.** Normalisation keeps caps because Kokoro's expressive layer
  turns them into emphasis; F5 read `WE WON` as "double-wee-woe", Turbo and Qwen3 as "to
  be one" / "one". Any non-Kokoro engine needs caps lowered (or mapped to its own emphasis
  control) after the expressive analysis. One-word `Hi.` is too short for Whisper to judge
  (it heard "Fine", "Bye", "Okay"): listen to it.
- Whisper transcripts and per-file WER: `out/tts_bakeoff/asr.json`; render timings per line:
  `results.json`.
