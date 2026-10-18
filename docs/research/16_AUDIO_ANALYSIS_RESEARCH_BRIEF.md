# 16 — Making the audio legible to an AI: status report and research brief

**Written:** 2026-10-17, after the raw-microphone matrix session.
**Branch:** `feature/modernize-p1` (commits 6e0ee97..a711dde).
**Audience:** a Claude session (or any capable model) picking up this project, and the user.
**What this asks for:** Part B is a research task. Part A is the context that research needs.

The problem in one sentence: the user hears a "stutter" in every live-microphone render of
their speech, the project has 16 carefully controlled recordings of it, and neither the
existing metrics nor an AI reading WAV headers and JSON sidecars can say where in the audio
the stutter is or what causes it. The user's ears are currently the only instrument that
detects it, and they cannot point at milliseconds. We need tools that turn audio into
something a language model can actually reason about.

---

# Part A — What exists and what was measured

## A1. The system in brief

rvc-vox / DEXAIV is a real-time RVC v2 voice changer on one Windows machine (RTX 5060 Ti,
see `TARGET.md`). A male talker (speech F0 ~100 Hz, floor ~67 Hz) speaks into a Shure MV6;
the engine converts in 20 ms hops (or 40/60) into a female target voice and writes it to a
Virtual Audio Cable that Discord uses as a microphone.

Per hop: 70 Hz high-pass, optional RNNoise, auto level (AGC to -20 dBFS), soft limiter, a
silence gate (energy + Silero VAD, 350 ms hangover), then on the GPU: resample to 16 kHz,
RMVPE pitch on a 310 ms tail, ContentVec features over a ~1.05 s window, optional k-NN index
blend, the RVC synthesizer (NSF-HiFi-GAN) decoding only the newest ~50 ms, resample, and a
SOLA crossfade (20 ms overlap, 10 ms search) against the previous hop. Every output sample
at hop 20 is a crossfade of two renders made from windows shifted by 20 ms. Full map:
`docs/ARCHITECTURE_AND_METHODS.md`.

Compute is not the problem: per-stage CUDA graphs put the hop at ~15-17 ms p50, and the
16 takes below had **zero underruns**.

## A2. The complaint

From the user's listening (docs/DEVIATIONS.md §8.8):

- Every mic render of speech "stutters BADLY", regardless of variant.
- Typed text (TTS -> the same model, full-file, no streaming) **never** stutters.
- With the input chain off (`raw_input`) it "stutters much less but still rough".
- On singing, the coherent-hop defaults (sine phase carry + rolled noise + 80 ms decoder
  context) were "the closest to perfection", with "a small small stutter" without the context.

Candidate causes, none confirmed: the silence gate opening/closing; the AGC's gain ramps;
seam density (50 crossfades per second at hop 20); the two-render blend when overlap ==
hop; and, newly raised by the modernization report, the NVIDIA Broadcast denoiser in front
of every previous take. Two facts constrain it: typed speech goes through the same model
without the streaming path, and it is clean.

## A3. Measurement tooling built in Phase 0

All under `benchmarks/` and reachable through `vox bench` (docs/DEVIATIONS.md §11):

| tool | what it measures |
|---|---|
| `quality.py` | WER (faster-whisper small.en) against a known script, with the raw mic's WER beside it; speaker cosine (ECAPA-TDNN) against target recordings or a labelled proxy; UTMOS (naturalness proxy); F0 median, IQR, octave flips per minute, voiced ratio; blind X/Y pair writer |
| `target_f0.py` | per-model target register (three estimators) |
| `matrix.py` | the raw-mic take protocol: input x gate x AGC x hop, recorded live, scored, paired |
| existing | R3 log-mel distance stream vs full-file, hop-seam "click ratio" (second-difference energy at seams vs mid-hop), HF/LF tilt, gate events, digital-zero cuts |

Every one of these is a whole-file number. None localizes an event in time, and none was
designed to detect "stutter".

## A4. The raw-microphone matrix (2026-10-17)

Folder: `recordings/matrix_20261017_192237/` (gitignored; local to the machine). Model Mom
(32 kHz), fixed pitch +12, extra 1000 ms, suppression off, gate threshold -40 dBFS with
auto level and -55 without. The user read `benchmarks/corpus/speech_script.txt` (150
words) once per cell, about a minute each. For each of the 16 cells there are three
sample-aligned float WAVs plus a JSON:

- `<cell>_mic_raw.wav`: what the device delivered
- `<cell>_mic_conditioned.wav`: after high-pass and auto level, what the engine converted
- `<cell>_output.wav`: what went to the cable; trails the mic by 30 ms (`stream_delay_ms`)
- `<cell>.json`: every parameter, the running plan, gate open/close events with hop index
  and time, underrun counts, `gate_open_per_hop` as a 0/1 string

Raw input levels: Shure takes -38 to -43 dBFS RMS with no exact zeros; Broadcast takes
-38 to -45 dBFS with 9-16% exact zeros (Broadcast gates its pauses to digital silence).
The first Broadcast attempt recorded silence because Broadcast's own source was set to the
virtual cable (a loop); it was re-recorded after fixing the source and enabling Broadcast's
noise removal.

Results (`matrix_report.md` in that folder). WER is the output's word error rate against
the script; the raw mic's own WER was 2.7% (Shure) and 13.3% (Broadcast). Speaker cosine is
against two `vox say` renders through the model ("proxy"): no recording of the real target
exists yet.

| cell | WER | spk cos | UTMOS | F0 med | flips/min | voiced | click | gate opens/min | speech lost ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shure gate off, agc off, hop 20 | 13.3% | 0.803 | 1.68 | 210 | 37.8 | 77% | 1.35 | 0 | 0 |
| shure gate off, agc on, hop 20 | 16.7% | 0.772 | 1.71 | 210 | 8.9 | 74% | 1.33 | 0 | 0 |
| shure gate on, agc off, hop 20 | 22.0% | 0.786 | 1.73 | 210 | 37.8 | 77% | 1.39 | 15.0 | 0 |
| shure gate on, agc on, hop 20 | 18.0% | 0.760 | 1.79 | 212 | 35.3 | 79% | 1.33 | 17.4 | 0 |
| shure gate off, agc off, hop 40 | 14.0% | 0.806 | 1.84 | 200 | 11.7 | 77% | 0.93 | 0 | 0 |
| shure gate off, agc on, hop 40 | 47.3% | 0.777 | 1.80 | 200 | 42.1 | 67% | 0.92 | 0 | 0 |
| shure gate on, agc off, hop 40 | 31.3% | 0.799 | 1.77 | 195 | 18.8 | 76% | 0.88 | 14.5 | 50 |
| shure gate on, agc on, hop 40 | 21.3% | 0.787 | 1.67 | 193 | 32.8 | 65% | 0.92 | 11.9 | 0 |
| broadcast gate off, agc off, hop 20 | 36.7% | 0.785 | 1.77 | 198 | 13.7 | 69% | 1.35 | 0 | 0 |
| broadcast gate off, agc on, hop 20 | 22.7% | 0.779 | 1.75 | 203 | 35.4 | 78% | 1.36 | 0 | 0 |
| broadcast gate on, agc off, hop 20 | 39.3% | 0.787 | 1.60 | 204 | 32.5 | 71% | 1.32 | 17.4 | 180 |
| broadcast gate on, agc on, hop 20 | 42.0% | 0.793 | 1.71 | 199 | 13.5 | 72% | 1.31 | 14.5 | 0 |
| broadcast gate off, agc off, hop 40 | 36.0% | 0.768 | 1.83 | 204 | 30.9 | 72% | 0.87 | 0 | 0 |
| broadcast gate off, agc on, hop 40 | 38.0% | 0.787 | 1.83 | 208 | 51.9 | 75% | 0.95 | 0 | 0 |
| broadcast gate on, agc off, hop 40 | 32.0% | 0.780 | 1.81 | 203 | 9.5 | 73% | 0.94 | 15.4 | 300 |
| broadcast gate on, agc on, hop 40 | 33.3% | 0.792 | 1.70 | 198 | 29.5 | 71% | 0.89 | 15.4 | 0 |

What can be said:

1. **No dropouts.** 0 underruns everywhere; the gate never closed inside a phrase.
2. **Broadcast costs intelligibility before RVC.** 13.3% vs 2.7% WER on the raw input; the
   converted Broadcast takes average ~35% WER vs ~23% for the Shure.
3. **Hop 40 has smoother seams in all 8 pairs** (click ratio ~1.33 -> ~0.91).
4. **Naturalness is uniformly low** (UTMOS 1.6-1.8) against ~3.5 for the same model on
   typed speech (§11.2).

What cannot be said:

- Anything about gate or AGC: the differences are smaller than the reading-to-reading
  spread (one Shure hop-40 cell reads 47% WER next to 14%; each cell is a single reading).
- Which cell stutters. The click ratio is the only seam metric and it disagreed with the
  user's ear twice before (§8.4, §8.8).
- Octave flips per minute swing 9-52 with no pattern; RMVPE on converted audio may simply
  be noisy at that rate.

Blind pairs awaiting the user: `out/ab/matrix_20261017_192237_pairs/pairs.md` (9 pairs:
gate, AGC, hop, everything-off vs everything-on per input, Shure vs Broadcast) and
`out/ab/p0_pairs/pairs.md` (3 pairs: f0_interp, 30 ms blend, index 0.35 on the old script).

## A5. Why the current approach is stuck

- **The metrics are file-level averages.** A stutter is a few-hundred-millisecond event
  that happens some number of times per minute. A one-minute average dilutes it below the
  noise of reading variation.
- **WER is too forgiving and too noisy.** Whisper's language model repairs smeared words;
  one reading against another differs by more than any setting.
- **An AI reading this repository sees numbers, never sound.** It can open a WAV only as
  samples and a JSON as events. It cannot hear "the second syllable of seashells doubles".
- **The user can hear it but cannot localize it.** Verdicts come per file ("stutters
  badly"), not per moment.
- **"Stutter" is undefined.** It could be a repeated or doubled syllable, a rhythmic
  amplitude flutter at the hop rate (50 Hz at hop 20, 25 Hz at hop 40), a pitch wobble, a
  brief phase cancellation in the crossfade, a gain pump from the AGC, a gate chop, or
  timing jitter between syllables. Each needs a different detector.

---

# Part B — Research brief

## B1. The goal

Find tools, models, services and methods that let an AI assistant (Claude, working through
a terminal on this Windows machine: Python 3.13, torch 2.7.1+cu128, CPU and one 16 GB GPU,
able to view PNG images and read text) **understand speech audio well enough to locate and
characterise artifacts in a voice-conversion output**, by comparing the converted output
against the aligned microphone input.

The deliverable is a way to answer, for a given take: *where* (timestamps, words) does the
output deviate from the input in a way a listener would call a stutter or glitch, *what kind*
of deviation is it, and *which engine event* (seam, gate transition, AGC ramp, pitch error)
coincides with it.

## B2. Questions to research

Please investigate each and report concrete, current options (name, licence, link, whether
it runs locally on Windows/Python 3.13 or needs an API, cost, and the evidence it works).
Verify every claim from a primary source; flag anything you could not verify. The
assistant's knowledge has a cutoff, so newer tools matter.

1. **Audio-capable language models as listeners.** Which multimodal models can take an
   audio clip and answer questions like "does this speech stutter or glitch, and where?" or
   "compare these two clips"? Candidates to check include Gemini's audio input, OpenAI's
   audio-input models, Qwen2-Audio / Qwen-Omni class open models, SALMONN, and anything
   newer (2025-2026). For each: can it give timestamps, can it compare two clips, how
   reliable is it on subtle artifacts (any benchmark on synthetic-speech artifact detection),
   and can a Claude session call it from Python? Is there a way for Claude itself to
   receive audio?
2. **Visual representations an AI can read.** Claude can look at images. What renderings make
   these artifacts visible: log-mel or linear spectrograms with a hop-seam grid overlaid,
   side-by-side input/output spectrograms aligned in time, F0 contours of both tracks on
   one plot, short-time energy envelopes, phase/group-delay plots, reassigned spectrograms?
   What resolution and zoom (whole minute vs 500 ms windows) works for a vision model?
   Any published practice of using vision-language models on spectrograms?
3. **Forced alignment and word-level timing.** WhisperX, Montreal Forced Aligner, Wav2Vec2
   CTC alignment, NeMo Forced Aligner, or newer: align the known script to both the mic
   and the output, then compare per-word and per-phone durations and onsets. A stutter
   should show as a lengthened, doubled or shifted segment. Which aligner is most robust on
   converted (non-natural) speech?
4. **Disfluency and stutter detection.** Models trained to find stuttering (repetitions,
   prolongations, blocks), e.g. work on SEP-28k / FluencyBank and newer. Do any transfer to
   synthetic glitches? Any detector for "robotic", "warble" or "chop" artifacts in TTS/VC
   output?
5. **Frame-level artifact and quality detectors.** No-reference or reference-based models
   that output a quality score *per frame or per segment* rather than per file (frame-level
   MOS predictors, deepfake/vocoder-artifact detectors that localize, glitch detectors used
   in streaming audio QA, packet-loss concealment evaluation tools). Which give a time
   series we can line up with hop seams and gate events?
6. **Voice-quality acoustics.** Praat/Parselmouth measures (jitter, shimmer, HNR, CPP),
   their frame-level versions, and whether they separate a natural voice from flutter at a
   fixed rate. Also: periodicity/aperiodicity estimators, pitch-tracker confidence, and
   modulation spectra (energy at 25 Hz / 50 Hz in the amplitude envelope would be a direct
   test for hop-rate flutter).
7. **Speaker identity and gender perception.** Better speaker-verification models than
   ECAPA-TDNN for cross-gender conversion (WavLM-large SV, ResNet/CAM++, newer), and any
   perceived-gender or vocal-femininity estimators, since a second complaint is "a man with
   a high voice".
8. **Online services and sites.** Hosted tools (APIs or web apps) for speech analysis,
   pronunciation/fluency scoring, voice-quality assessment, or audio QA that could be called
   on these files. Note privacy: these are the user's own voice recordings; prefer local
   tools and say when a file would leave the machine.
9. **Prior art in real-time voice conversion evaluation.** How do RVC forks, w-okada,
   Applio, Seed-VC, StreamVC, LLVC, RT-VC and the SVCC challenges evaluate *streaming*
   artifacts specifically (chunk-boundary artifacts, latency-quality trade-offs)? Any
   published "boundary artifact" metric better than ours?

## B3. What the result should enable

Rank the options by how directly they help this specific diagnosis, then propose a
concrete pipeline the assistant can run, for example:

1. Align the script to `mic_raw` and `output` (word and phone timings).
2. Per word: duration ratio output/mic, onset shift, energy envelope correlation, F0
   contour correlation, a local quality score.
3. Overlay engine events from the JSON: hop seams (every 20 or 40 ms after the 30 ms delay),
   gate opens/closes, and the AGC gain (derivable from `mic_conditioned / mic_raw`).
4. Flag outlier words and render, for each, a PNG panel (mic and output spectrograms,
   F0 of both, energy envelopes, seams and events marked) that the assistant can view.
5. Optionally ask an audio-capable model to describe each flagged clip.
6. Summarise per cell: number of flagged events per minute, their kind, and which engine
   event they coincide with. Compare cells (gate, AGC, hop, input) on those counts.

Say which of these steps you would build, with which libraries, roughly how much effort
each is, and what would make the result trustworthy. The user's listening verdicts on the
blind pairs are the ground truth to validate against: a detector is only useful if it
flags the cells the user says stutter.

## B4. Constraints

- Windows 11, Python 3.13 venv (`.venv\Scripts\python.exe`), torch 2.7.1+cu128. An
  Application Control policy blocks numba's DLLs (librosa's numba paths fail; soxr and the
  project's RMVPE work instead). Prefer tools that install with pip on Windows.
- The GPU is shared with the live engine; analysis should run on the CPU or one job at a
  time.
- Files are 48 kHz mono float32, ~50-75 s each, 16 cells x 3 tracks.
- Do not rank voice conversion on MCD, F0-RMSE, PESQ or NISQA-PSTN (doc 15: they do not
  track MOS for VC). Reporting them is fine; ranking on them is not.
- Do not change engine defaults on metric evidence alone (CLAUDE.md, doc 15).

## B5. Expected output

A markdown report in `docs/research/17_AUDIO_ANALYSIS_TOOLS.md` with: a ranked table of
options (what, local or API, licence, setup on this machine, evidence, fit for this
problem), the recommended pipeline (B3) with libraries and effort, the open risks, and a
short list of what to try first. Cite sources with links. Mark unverified claims.

---

# Part C — What the user can do meanwhile

1. Listen to the 9 pairs in `out\ab\matrix_20261017_192237_pairs\` and write in `pairs.md`
   which one stutters, if either, and **where** ("Y stutters on 'seashells'"). Moments
   with words are what any detector will be validated against.
2. If possible, mark two or three exact moments in one output file that stutter the worst
   (the sentence or a rough time). Even one localized example makes it possible to check
   whether a detector finds it.
3. Add 2-3 clips of each real target voice to `takes/target/<Model>/` for a real
   similarity number.
