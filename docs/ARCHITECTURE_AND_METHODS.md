# How the voice conversion works: structure and methods

A map of the engine as built (2026-10-15), for reading the code or the measurements in
`docs/DEVIATIONS.md`. The research that motivated each choice is in `docs/research/`
(doc numbers below); this file says what runs, in what order, and why it is shaped that way.

## 1. What happens to one hop of microphone audio

The engine converts the microphone in **hops** of 20 ms (default; the GUI offers 20-100 in
20 ms steps). Every hop, the worker thread (`engine/worker.py`) does this, in order:

```
mic block (48 kHz, 960 samples)
  -> input chain (CPU, audio/conditioning.py)
       70 Hz high-pass -> [RNNoise suppression] -> auto level (AGC to -20 dBFS) + trim -> soft limiter
  -> silence gate (CPU, engine/gate.py): energy threshold + Silero VAD + hysteresis + 350 ms hangover
  -> upload to the GPU (pinned buffer, one copy)
  -> resample_in : 48 kHz -> 16 kHz, appended to a rolling 16 kHz ring (the analysis window)
  -> f0          : RMVPE on the newest 310 ms of the ring -> transposition -> pitch cache
  -> embed       : ContentVec (HuBERT, 768-d, 50 Hz) over the whole window (~1.1 s)
  -> index       : optional FAISS-style k-NN blend toward the training set, protect on unvoiced frames
  -> synth       : RVC v2 synthesizer: text encoder (features + coarse pitch) -> flow -> NSF-HiFiGAN decoder
                   with a sine excitation from the continuous pitch; only the tail is decoded
  -> post        : resample to the device rate; optional RMS envelope mix (volume_envelope)
  -> sola        : overlap-add against the previous hop's tail (20 ms crossfade, 10 ms search)
  -> download (one copy; this is where the hop synchronises) -> output ring -> device callback
```

The six GPU stages run as **per-stage CUDA graphs** replayed in sequence (`engine/graphs.py`;
~16 ms of GPU time per hop on the RTX 5060 Ti, against ~40 ms eager, doc 06 / §R5). That
imposes the engine's main discipline: every stage reads and writes only buffers allocated
once at start; hot parameters (pitch, index rate, protect, ceiling, auto-pitch target) are
values inside 0-dim device tensors, so moving a slider never changes the op sequence; the
decoder's random noise is drawn outside the graphs each hop; nothing in the hop calls
`.item()` or otherwise syncs; and nothing off the worker thread touches a device tensor
(§9 records what happens when something does).

Latency, end to end, is about 100-130 ms: hop (20) + compute (16-19) + output ring fill
(30-50) + SOLA overlap and search (30) + device buffers (11 exclusive / 33 shared), + 20 ms
when suppression is on.

## 2. The context window and the chunking law

RVC was trained on whole utterances, so each hop is converted **in context**: the model
sees the newest ~1.1 s of input (`extra` 1000 ms + crossfade 20 + search 10 + the hop), but
only the trailing `hop + overlap + search` of its output is kept and stitched. Everything
derived from those four durations (sample counts at 16 kHz, HuBERT frame counts, which
frames to decode, the pitch-cache shift) comes from one module, `engine/chunker.py` (P17),
so no two places can disagree about a size.

Consequences measured along the way:

- A streamed render is not log-mel-identical to a full-file render of the same audio (R3):
  this generator is sensitive to tiny input shifts, and every hop renders from a window
  shifted by one hop. The gap is 1.8-2.7x the model's own seed-to-seed noise floor and is
  not audible as such; what the ear catches is seams.
- Short context inflates loudness in pauses (ContentVec normalises over the window, R4),
  which is why `extra` went from 500 to 1000 ms and why the gate exists.
- The hop must be a whole number of 20 ms HuBERT frames (S3), or the same audio lands on
  a different 50 Hz feature grid from one hop to the next.
- Hop 10 is impossible on this GPU (16 ms of work per 10 ms); hop 20 runs with ~1-4 ms of
  headroom and 0 underruns in the paced and live tests (§8.2).

## 3. Making consecutive hops agree (the seam problem)

With a 20 ms hop every output sample is the crossfade of two renders of the same audio,
made from windows shifted by 20 ms. Three things made those renders differ beyond what
SOLA can hide, and all three are now on by default (§8.8, chosen by the user's ear):

| change | what it fixes |
|---|---|
| `noise_roll` | the decoder's latent and excitation noise used to be redrawn every hop; now it moves with the window, so a given 10 ms of audio is rendered with the same noise in every hop that contains it |
| `decoder_context_ms = 80` | the HiFi-GAN decoder has a ~±90 ms receptive field; decoding only the tail meant its first samples sat on zero padding. Eight frames are decoded ahead and dropped |
| `sine_phase_carry` | the NSF sine excitation restarted at phase 0 every hop; the phase at the next hop's start is now carried in a device scalar (reset at gate onsets) |

Plus the RMVPE right-edge guard (`f0_trim_tail = 2`): the newest two pitch frames of each
estimate are discarded (the BiGRU has the least right context there), which halved the
pitch step at seams on held notes (§8.4). The objective seam metric did not predict which
of these the user would prefer; see §8.4 and §8.8 before trusting it.

## 4. Pitch: extraction, transposition, auto pitch, ceiling

- **Extraction**: RMVPE (doc 05), tensor-native, on a 310 ms tail each hop; a rolling
  **pitch cache** keeps older frames and only the tail is re-estimated (the rvc-webui /
  Applio pattern). FCPE is the lighter alternative behind the same interface (P4).
- **Unvoiced frames** stay at 0 Hz (`f0_unvoiced = "zero"`, D8): that is what feeds the
  coarse pitch bucket 1 and what the `protect` blend keys on. Interpolating them (the
  rvc-webui default) sounds less raspy on this voice (Q15) but silently disables protect;
  it stays an A/B. Short dropouts inside vowels can be bridged (`f0_bridge_ms`, off).
- **Transposition** is a ratio on the continuous pitch before the coarse buckets
  (`f0_up_key` semitones). **Auto pitch** (§9) replaces the fixed ratio with one derived
  from a running mean of the talker's log F0 (2 s time constant in voiced frames) so the
  register lands on `target_f0_hz` (the model's own, as `vox say` already derives it)
  whatever register the talker is in; `pitch_range` scales the intonation around it.
- **Soft ceiling** (`f0_ceiling_hz`, Sing preset 360 Hz): a semitone-domain soft knee
  that bends notes above the model's training range down instead of letting the decoder
  crack; a safety, not a fix (notes 6 st over come out ~2 st flat).

## 5. The gate and the input chain

The engine's own cuts were the first complaint (§7): the gate closed mid-word. Now:
energy threshold (measured after the input chain, which lifts a quiet mic by up to
+18 dB), Silero VAD on every 32 ms chunk with the **max** probability over a hop's chunks,
6 dB level hysteresis and 0.15 VAD hysteresis once open, 350 ms hangover, and an
energy-only onset rule (12 dB rise to 6 dB over threshold) so a clear attack opens before
the VAD has answered. On close the SOLA tail is faded out once and the context ring keeps
advancing (so a reopen has true recent audio, no refill stall); on open the first hop
fades in over 5 ms, keeping the consonant decoded ahead of the block. Soft entrances
still wait for the VAD (~140 ms on the sung take, §8.3); a 3 dB onset margin fixes that
and is an A/B.

The input chain is the live equivalent of what the full-file converter does to its input:
high-pass, level, limiter; **RNNoise** (xiph, via the pyrnnoise DLL, 10 ms frames, 20 ms
delay) sits after the high-pass when suppression is on, with a wet/dry mix for sustained
notes. On the NVIDIA Broadcast mic, which is already denoised, it defaults off.

## 6. Typed speech

`vox say` and the Live screen's text bar: text -> TTS (Kokoro locally, or edge-tts) ->
the **full-file converter** (the same model, no streaming) -> the output. Pitch is measured
from the TTS render and transposed onto the model's register. The expressive layer
(`tts/expressive.py`) reads punctuation and spelling ("...", "!!", "heyyy", CAPS) into
per-piece speed, pauses and gain; the polish layer (`TypedPolish`) lets the output follow
the TTS loudness contour on 40 ms frames (now default, §8.8), widen the F0 contour, or
blend Kokoro voices. While a clip plays, the live voice is held (faded out, ring kept
current) and resumes like a gate onset (G6/G7).

## 7. Control surface

`api/control.py::LiveSession` builds the engine, rings, worker and streams and exposes hot
parameters, gate, conditioning and restart-class changes (hop / context / tracker: rebuilt
in place between hops, ~3 s mute). `api/controller.py::EngineController` wraps it for the
GUI: profile + user choices (`gui_state.json`, only what was changed), Speak/Sing presets,
device handling (all PortAudio calls on one thread, G5), typed clips, recording, and a
state / meters / perf event stream over local HTTP+SSE (`api/server.py`) to the pywebview
window (`ui/`). `vox run` drives the same `LiveSession` from the terminal.

## 8. Measuring

`benchmarks/bench.py` drives the real engine through the stream simulator (P13: no second
pipeline) in throughput or paced mode: `accept` (stream vs full-file, clicks, sync-free
hop, allocation stability, live-path cuts), `ab` / `sing` / `onset` / `typed` / `suppress`
(one change per variant -> `out/ab/*.wav` + metrics), `underrun` (hop x context, with the
Broadcast denoiser running), `perf`, `graph-probe`. Every number in `docs/DEVIATIONS.md` came
from these on the one target machine (`TARGET.md`); the user's verdicts on the wavs are in
§8.8. Tests: `tests/unit` and `tests/api` (CPU, ~400), `tests/engine` (real weights, GPU).

## 9. Where the research docs are wrong, in one list

`docs/DEVIATIONS.md` §1 (D1-D15), §4.1 (D16-D25), §5.1 (D26-D32), §7.1 (L1-L9), §8.1
(S1-S11), §9 (A1-A3) tabulate every departure with its reason; §2-§8 the measurements.
`docs/research/14_OPEN_QUESTIONS.md` tracks what is still undecided (Q7 seams, Q14
audible equivalence, Q15 interp).
