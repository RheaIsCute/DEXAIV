# What is needed to debug and improve the voice conversion

For anyone (person or model) picking up a complaint like "it stutters", "it sounds AI",
"my voice cuts off", or "typed speech sounds robotic": what to collect, where it already
is, and what each thing lets us decide. Written against the engine as of 2026-10-15
(`docs/DEVIATIONS.md` §7-§9).

## 1. The one thing that answers most questions: a Record-button take

Press **Record** on the Live screen, do the thing that sounds wrong, press it again. It
writes to `recordings/` (copy to `takes/live/` to share):

- `*_mic_raw.wav`, `*_mic_conditioned.wav`, `*_output.wav`: sample-aligned, so any
  moment in the output can be matched to what the mic delivered and what the engine saw.
- `*.json`: the model, every parameter (pitch, index, protect, ceiling, auto pitch and its
  target, hop, context, gate threshold, suppression and its mix, auto level and trim), the
  **running** plan, the gate open/close events with times, and the session's underrun count.

Without the take, a verbal description cannot be told apart from a dozen different causes.
With it, most can:

| complaint | first thing to check in the take |
|---|---|
| words or syllables missing at phrase starts | `gate_open` times vs the entrance in `mic_conditioned` (`vox bench onset` does it per entrance); the level at the entrance vs `threshold_dbfs` and the 6 dB onset margin (§8.3) |
| gaps or drops mid-phrase | `event_counts_in_take.underrun` (ring ran dry: compute) vs `gate_close` inside speech (gate chatter: threshold / VAD); the §7 attribution rule is in `benchmarks/bench.py::_cuts` |
| "stutter" / flutter on held notes | hop seams: `vox bench sing` on the mic track gives the SOLA offset histogram and the seam pitch step; compare hop 20 vs 60 (§8.4) |
| cracked / screechy high notes | output F0 vs the model's register: `f0_up_key` (or the auto-pitch offset) times the input F0; the pitch ceiling (§8 S7) and auto pitch (§9) |
| muffled held vowels | `mic_raw` itself: on the NVIDIA Broadcast mic the denoiser can notch 1.5-4 kHz mid-note (§8.5 could not measure this; needs a raw-mic take) |
| too quiet / pumping / distorted | `mic_conditioned` vs `mic_raw`: auto level gain (`input_gain_db` in the stats), the -3 dBFS limiter, trim (§7 L5) |
| latency | the session line in the JSON (`device latency`, `fill target`) and the Performance screen estimate; suppression adds 20 ms |

## 2. The microphone itself

The engine has never been run on a raw microphone by the user: every take so far is the
**NVIDIA Broadcast** virtual mic, which denoises, gates to digital silence and (per the
external report) carves harmonics off sustained notes before the engine sees anything. For
any singing or "sounds AI" question we need **one take with the Shure MV6 selected
directly** (Settings > Microphone), same material, so Broadcast's processing can be
separated from ours. The only raw-mic audio we have is the 13 Oct OBS capture of the
Realtek mic (`takes/mic/obs_default_mic_2026-10-13.wav`), speech only.

Also useful once, not per take: the mic's level while speaking normally (the IN meter with
auto level off), and whether the room has a steady noise (fan) or intermittent noise
(keyboard, other voices).

Since 2026-10-17 this is a command: `vox bench take --matrix` records the whole
input x gate x auto-level x hop matrix with the Record pipeline while the user reads
`benchmarks/corpus/speech_script.txt` (16 cells of about a minute; `--only` or `--hops 20`
for a subset), labels the files, scores them and writes blind pairs (docs/DEVIATIONS.md §11).

**Held-out target recordings.** The speaker-similarity metric needs 2-3 clips of the real
target voice that were *not* in the model's training set, in `takes/target/<Model>/`. Without
them every similarity number is against the model's own `vox say` renders and says so
("proxy"). Ask for the clips before trusting any similarity ranking.

## 3. Listening verdicts, per file

Metrics and ears have disagreed twice (§8.8): the seam-roughness ratio and the log-mel
distance both preferred the renders the user liked least. So for any quality change we
need the user to listen to the A/B pair and say which, with where in the clip and what
("warble on 'true' at 6 s", "the 's' sounds lispy"). The pairs live in `takes/ab/` with
their meaning in `takes/README.md`. Three rules that make the verdicts usable:

1. One change per pair, same input, same seed (`vox bench ab --variants a,b`).
2. Say it per file, not per batch ("these three are meh" cannot be acted on).
3. A reference point: which existing file is the best so far, so new ones are compared to it.

## 4. For the "stutter" in speech (the open item)

Every mic render of the speech script stutters to the user, typed renders never do, and
`raw_input` (input chain off) stutters less. To close it we need, in this order:

1. The user's verdict on `takes/ab/speech_reference.wav` vs `speech_coherent.wav` (does
   the coherent-hop default, chosen on singing, fix it on speech?).
2. If not: a Record take of the user **speaking** through the Shure, hop 20, with the gate
   **off** for 20 s and on for 20 s. Gate off removes the gate from the suspect list in one
   take; `mic_conditioned` vs `mic_raw` shows whether the auto level's gain ramps are
   audible (`raw_input` points there).
3. Then a hop 40 take of the same: half the seams per second at the same compute (§8.2).

Items 2 and 3 are now one session: `vox bench take --matrix` (see §2). Its blind pairs
(`out/ab/matrix_*_pairs/pairs.md`) are the verdicts to collect: gate on vs off, auto level
on vs off, hop 20 vs 40, Shure vs Broadcast, each with everything else held. Do not read the
stutter's cause off the metric table; the pairs decide.

## 5. For typed speech (humanisation)

Typed clips do not stutter; what the user hears as "robotic" or "AI" is prosody. What is
already measured and switchable (§7.5): the loudness envelope (now default), F0 contour
widening (`contour_scale`), Kokoro voice blends, and the expressive layer (speed, pauses,
emphasis from punctuation and letter stretching: `docs/TTS_NOTES.md`). To improve it we need:

- **The texts** that sounded wrong and what they were meant to sound like (casual aside,
  excited, flat narration). The expressive layer keys on punctuation and spelling
  ("heyyy", "!!", "..."), so the exact string matters.
- A **reference** of the user saying the same line (a Record take with the gate off), so
  the TTS prosody can be compared to the real one: F0 range, syllable rate, pause lengths.
  `vox bench typed` prints F0 sd, loudness sd, voicing and HF/LF per variant.
- Which **TTS voice** (`vox say --list-voices`): the conversion keeps the TTS voice's
  prosody, so a monotone TTS voice cannot be fixed downstream. A voice already close to the
  target (female for a female model, pitch near 0) sounds best (`docs/TTS_NOTES.md`).
- Whether the model has a **calibrated register** (`target_f0_median_hz` in
  `profiles/default/model_overrides.json`): both `vox say` and auto pitch aim at it; without
  it they aim at 200 Hz.

## 6. For performance questions (dropouts, latency)

- The Performance screen's export (`Export report`: per-stage p50/p99, headroom, underruns,
  the 60 s history) or `vox run --stats-log file.jsonl`.
- What else was on the GPU: NVIDIA Broadcast, OBS, browsers and Discord are always
  resident on this machine (§8.2 measured them harmless at hop 20); a game is not measured.
- The hop and context in use (the take JSON): hop 10 cannot work on this GPU (§8.2).

## 7. What not to send

Descriptions of settings from memory (the JSON has them), screenshots of sliders, and
recordings made through Discord or OBS of the converted output (they add their own
processing; the take's `output.wav` is the engine's exact output).
