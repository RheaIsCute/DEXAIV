# Recordings

Every recording made while tuning the live engine, so the audio behind the numbers in
`docs/DEVIATIONS.md` can be heard and re-analysed. WAV files are stored with **Git LFS**
(`git lfs pull` after cloning); JSON sidecars and this index are plain files. All audio is
48 kHz mono PCM-16 unless noted. Nothing here is committed to `recordings/` or `out/`,
which are gitignored working folders; this folder is the curated copy.

## `live/` — takes from the Record button

Three sample-aligned tracks per take plus a JSON with every setting, the gate events and
the session's underrun count: `mic_raw` (what the input device delivered),
`mic_conditioned` (after the high-pass, suppression, auto level and limiter: what the
engine converted), `output` (what went to the Virtual Audio Cable; trails the mic by the
SOLA delay noted in the JSON, plus 20 ms when suppression was on). The microphone is the
**NVIDIA Broadcast virtual mic** in every take so far: its own noise removal has already
run, and it outputs digital silence in pauses. The talker is the user (male speaking voice,
F0 median ~100 Hz; sung lines 140-280 Hz).

| take | model | length | what happens | settings | what it showed |
|---|---|---|---|---|---|
| `take_20261015_004247` | Mom (32 kHz) | 7.1 s | sings "You're my addiction, I believe in, I promise you" | pitch +11, index 0.85, protect 0.22, hop 60 (the JSON says 30 in `plan`: that bug is fixed, §8 S2), extra 2200, gate -45 | too high a transposition for a sung line (output 450-530 Hz); 222 underruns earlier in that session came from a 10 ms hop (§8.2) |
| `take_20261015_004329` | GoatMommy (48 kHz) | 113 s | a full song, two verses and choruses with pauses; the "Promise you" entrance at 22.0 s and "In you and me" at 72.8 s are the ones the external report cited | pitch +6, index 0.34, protect 0.14, hop 60, extra 2200, gate -45, trim -3 | the main singing test material (§8.3, §8.4): soft entrances wait ~140 ms for the VAD; seam pitch steps on held notes |
| `take_20261015_114008` | GoatMommy | 82 s | speaking and singing with **auto pitch** on (target 200 Hz), six typed clips played in between | pitch +5 (used only until the tracker warms), auto pitch, hop 60, extra 2200 | the first auto-pitch run; the session died right after on a hop change (fixed, §9) |
| `take_20261015_122550` | Mom | 86 s | speaking with **auto pitch** on and **suppression** on (mix 1.0) through the Broadcast mic | auto pitch target 200, suppression on (20 ms added delay), hop 40, extra 1900, index 0.31, protect 0.22, gate -45 | 27 gate openings in 86 s: the gate chatters on this take (closes 0.5-0.8 s into pauses and reopens); worth a look together with the "stutter" complaint |

## `mic/` — the raw microphone

| file | what |
|---|---|
| `obs_default_mic_2026-10-13.wav` | 49.7 s OBS capture of the **default mic at unity, no filters** (stereo): the user reading, with room tone and bleed from another voice in the pauses. Speech at -42 dBFS. The basis of the gate and auto-level work (§7) and of the suppression measurements (§8.5) |
| `mic_script_2026-10-13.wav` | the two speech passages of that capture cut together (25.75 s, mono): the standard test script of every speech benchmark (`vox bench ab`, `accept`, `underrun`) |

## `ab/` — A/B renders for listening

Renders of the test material through the engine with one setting changed at a time. The
user's verdicts on all of them are in `docs/DEVIATIONS.md` §8.8. "Mom +12 / GoatMommy +6"
= the model and transposition; "script" = `mic/mic_script_2026-10-13.wav`; "take 2" = the
first 13.5 s of `live/take_20261015_004329_mic_raw.wav`.

| files | input, model | what differs | section |
|---|---|---|---|
| `base_gate31`, `base_gate57`, `step1_gate31/40/57`, `fix1_sola`, `fix3_pitch`, `fix5_gate` | script, Mom +12 | the live-quality fixes of §7 applied one at a time, at three gate thresholds (`*_typed` = the same sentences typed, for comparison) | §7.2-7.3 |
| `reference`, `sola_blend3`, `overlap_10`, `raw_input`, `f0_interp`, `envelope_0.3`, `index_0.35`, `index_0.6`, `embed_pad`, `f0_bridge`, `decctx_80`, `noise_roll` | script, Mom +12 | the §7 A/B variants: SOLA geometry, input chain off, interpolated unvoiced F0, RMS envelope, index rate, ContentVec padding, F0 gap bridging, decoder context, noise rolling | §7.4 |
| `typed_plain`, `typed_envelope`, `typed_contour`, `typed_blend`, `typed_all` | typed text, Kokoro `af_heart` -> Mom | typed-speech prosody options; `typed_envelope` is now the default | §7.5, §8.8 |
| `sing_*` (hop 20), `hop60_sing_*`, `mom_sing_*` | take 2, GoatMommy +6 / Mom +6, Sing preset | sine phase carry, noise roll, 80 ms decoder context, RMVPE trim tail, pitch ceiling; `sing_phasecarry_noiseroll_decctx80` is the one the user chose and is now the default | §8.4, §8.8 |
| `s8_reference`, `s8_trimtail_2`, `s8_phase_carry` | script, Mom +12 | the §8 flags on speech | §8.4 |
| `speech_reference` / `speech_coherent`, `speech_goat_*` | script, Mom +12 / GoatMommy +6 | old defaults vs the coherent-hop defaults (phase carry + noise roll + decoder context) | §8.8 |
| `onset_promise_*`, `onset_inyou_*` (stereo: L = mic, R = output) | take 2 entrances at 22.0 s and 72.8 s, GoatMommy +6 | gate before/after the onset fixes, at hop 20 and 60; `margin_3db` = the A/B that opens the soft entrance in time | §8.3 |
| `suppress_mic_mix1`, `suppress_mic_mix0.5` | OBS raw mic | RNNoise at full and half mix (no conversion) | §8.5 |
| `suppress_sung_mix1`, `suppress_sung_mix0.5` | take 2 mic | RNNoise on the (Broadcast-processed) sung track | §8.5 |
| `autopitch_speech_*`, `autopitch_sing_*` | script / take 2, GoatMommy | fixed +6 vs auto pitch (target 200 Hz) vs auto with range 0.7 | §9 |

Regenerate any of them with `vox bench ab|typed|sing|onset|suppress` (see
`benchmarks/bench.py`); the exact commands and settings are in the sections cited.
