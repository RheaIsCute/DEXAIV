# 04 — Latency Analysis: Model, Budgets, Bottlenecks, Profiler, Optimizations

Companion to `03_REALTIME_AUDIO_ARCHITECTURE.md`. Same source base: w-okada
voice-changer (`/home/claude/voice-changer`), Applio (`/home/claude/Applio`), rvc-webui
(`/home/claude/rvc-webui`), Vonovox (`/home/claude/Vonovox`, compiled — README-level only).

---

## 1. The full theoretical latency model

End-to-end mouth-to-virtual-mic latency of a block/hop-based RVC changer:

```
T_total = T_audio_input          (ADC + driver + WASAPI capture period)
        + T_input_buffer         (block/hop accumulation: wait until a full hop exists)
        + T_queue_in             (hand-off callback → worker: 0..1 callback period + wake-up)
        + T_pre                  (mono-mix, gain, gate/VAD, resample 48k→16k)      [compute]
        + T_feature              (HuBERT/ContentVec forward over full context)     [compute]
        + T_f0                   (RMVPE/FCPE/crepe over new region)                [compute]
        + T_synth                (RVC decoder/vocoder over return_length)          [compute]
        + T_post                 (resample tgt→48k, NR, rms_mix, effects)          [compute]
        + T_sola                 (search + crossfade compute ~0, PLUS the overlap
                                  emission delay: overlap+search samples)          [algorithmic]
        + T_queue_out            (worker → output callback)                        
        + T_output_buffer        (output ring target fill / safety blocks)         [algorithmic]
        + T_audio_driver         (WASAPI render period + DAC or virtual-cable hop)
        + T_vac                  (virtual cable internal buffer + consumer app pull)
```

### Algorithmic vs. compute latency — the crucial distinction

- **Algorithmic latency** is built into the signal path *by design*: it exists even on an
  infinitely fast GPU. Sources:
  - **Block/hop accumulation** (`T_input_buffer`): you cannot convert sample *n* until
    the whole hop containing it has arrived → average ½ hop, worst-case 1 hop; in
    block-synchronous designs the output is emitted exactly one block after capture
    starts, so the term is a full `block_time`.
  - **Lookahead / context**: RVC as deployed here uses *past*-only context (`extra` is
    history, not future), so EXTRA adds **zero algorithmic latency** — only compute. (A
    design that waited for future context would add it directly; none of the four does.)
  - **SOLA overlap + search**: the stitcher emits audio `overlap+search` samples behind
    the newest synthesis. rvc-webui exposes this honestly in its GUI delay estimate,
    `realtime_gui.py:476-488`:
    ```python
    self.delay_time = self.stream.latency[-1] + block_time + crossfade_length + 0.01
    # +min(crossfade, 0.04) more if input noise-reduce is on   (:483-484)
    ```
    i.e. **delay = device output latency + block + crossfade + sola_search(10 ms) [+ NR]**.
  - **Resampler filter delay**: polyphase kaiser sinc group delay ≈ (filter_width/2)/rate;
    torchaudio default width 6 → ~0.2–0.4 ms per conversion. Real but negligible.
  - **Output safety buffering**: queue depth between worker and output callback. w-okada's
    output callback does a blocking `Queue.get()` (`ServerDevice.py::audioOutput_callback`)
    → effectively ≥1 block; Applio keeps queue depth ≤1 (latest-wins) → 0–1 block; our
    design makes it an explicit ring fill target (~1 hop + jitter margin).
  - **Noise suppression framing**: TorchGate STFT (n_fft = 4·zc = 40 ms) + the NR
    crossfade (rvc-webui adds `min(crossfade, 40 ms)`); RNNoise 10 ms frames.
- **Compute latency** is time spent calculating (`T_pre..T_post`). It delays the output
  only until the result is written; it must satisfy the **stability constraint**
  `T_compute(p99) < hop_interval`, otherwise the output ring underruns. Below that
  threshold, reducing compute does NOT reduce steady-state latency in a fixed-hop design
  — it only buys headroom (or lets you shrink the hop, which does).

### The "80 ms context vs 5 ms compute" example

Suppose the model consumes 80 ms of audio context per hop and the GPU computes a hop in
5 ms. Naive reading: "latency 85 ms." Wrong on both sides:

- If the 80 ms is **history** (RVC `extra`): algorithmic cost 0 — the context is already
  in the ring. Latency = hop accumulation + SOLA overlap + buffers + 5 ms compute. With a
  20 ms hop and 20 ms overlap, that's ~50–60 ms total even though the model "sees" 80 ms+.
- If 80 ms were **lookahead** (future samples required before output), it is a hard 80 ms
  algorithmic floor no GPU can remove.
- Conversely a 5 ms compute time does *not* allow a 5 ms hop unless p99 (not mean!)
  compute, including CUDA launch jitter, GC, and the GUI thread stealing the GIL, stays
  under 5 ms — this is why Applio's `automatic_block_frame`
  (`rvc/realtime/utils/chunks.py`) sets `block = max(100 ms, 1.5 × measured processing)`:
  a codified 50% headroom rule on exactly this constraint.

The practical corollary: **spend compute savings on shrinking the hop, not on bragging
rights** — halving T_compute lets you halve the hop, which removes *algorithmic* latency.

---

## 2. w-okada's `buf` and `res` mapped onto the model

From `tutorials/tutorial_rvc_en_latest.md`:

> "The lag time from speaking to conversion is `buf + res` seconds. When adjusting,
> please adjust the buffer time to be longer than the res time."
> **buf**: "The length of each chunk in milliseconds when capturing audio. Shortening
> CHUNK will decrease this number."
> **res**: "The time it takes to convert data with CHUNK and EXTRA added."

In code:

- **buf** = block duration = `CHUNK × 128 / 48000` seconds
  (`server/voice_changer/Local/ServerDevice.py:385`:
  `block_frame = int(chunkNum * 128 * (serverInputAudioSampleRate / 48000))`).
  This is pure **algorithmic** block-accumulation latency. Default CHUNK=256 → 683 ms(!);
  typical tuned values 48–192 → 128–512 ms.
- **res** = measured **compute** time of one conversion:
  `ServerDevice._processDataWithTime` wraps `on_request` in `Timer2("all_inference_time")`
  and emits `[all_inference_time] + times` to the UI (`ServerDevice.py:139-146`); inside,
  `VoiceChangerV2.on_request` stamps sub-stages with `t.record("generate_strength")`,
  `t.record("inference")` etc. In client mode the same perf list rides back on the
  Socket.IO `response` event (`sio/MMVC_Namespace.py:52-56`). res grows with EXTRA because
  the encoder reprocesses the whole `extra+block` window every hop (docs: "calculation
  time will increase by the square of this length" — attention cost).
- The tuning rule "buf > res" is exactly the stability constraint
  `T_compute < hop_interval`. And `total ≈ buf + res` matches the model: one block of
  accumulation + compute, with w-okada's additional queue/device terms (output queue ≥1
  block in server mode, browser/network in client mode) silently riding on top — which is
  why measured w-okada latency usually *exceeds* buf+res.

Mapping table:

| Model term | w-okada quantity | Type |
|---|---|---|
| T_input_buffer | **buf** (CHUNK×128/48k) | algorithmic |
| T_pre..T_post | **res** (Timer2 around on_request; resampy + pipeline) | compute |
| T_sola overlap | crossFadeOverlapSize (not shown in UI) | algorithmic |
| T_queue_out/T_output_buffer | outQueue blocking get (server mode), browser buffer (client mode) | algorithmic |
| T_audio_* / T_vac | not measured by app | device |

---

## 3. Realistic latency budgets

### 3.1 Why typical w-okada setups land at ~90–200 ms+ (often worse)

| Term | Typical tuned value | Source |
|---|---|---|
| buf (CHUNK 48–112 ×128/48k) | 128–300 ms | `ServerDevice.py:385`; default 256→683 ms |
| res (GPU mid-range, EXTRA ≈ 1–2.5 s) | 40–150 ms | docs; resampy CPU + full-context HuBERT each hop |
| output queue (server mode) | ~1 block = 128–300 ms worst case, ~½ avg | blocking `outQueue.get()` |
| crossfade overlap | ~85 ms default (`crossFadeOverlapSize`) | VoiceChangerV2 settings |
| WASAPI shared + VB-Cable | 20–40 ms | engine periods + cable buffer |
| **Total** | **~250–700 ms** as shipped; **~90–200 ms** aggressively tuned (CHUNK≈24–48, small EXTRA, server device mode) | matches community reports and the docs' buf+res framing |

Client (browser) mode adds Socket.IO framing + browser playback buffering — server device
mode exists precisely to cut that (`tutorial_device_mode_*`, `Local/ServerDevice.py`).

### 3.2 What Applio achieves

Defaults: block 512 ms @48k (`worker.py:260`), crossfade 100 ms, extra 0.5 s, process
worker with queue depth ≤1 each way. Auto mode floors block at 100 ms and 1.5× compute.
Budget at auto-min on a decent GPU (compute ~30–50 ms): 100 (block) + ~50 (avg ½ block
output latest-wins + queue) + 100 (crossfade overlap) + ~25 (devices/cable) ≈
**~250–300 ms defaults, ~150–200 ms tuned** (crossfade down, strong GPU). Its wins are
robustness (never glitches; drops stale blocks instead) rather than minimum latency.

### 3.3 What rvc-webui achieves

Inline-callback design: total ≈ `device_out + block + crossfade + 10 ms` (its own
formula, `realtime_gui.py:476-488`). With block 0.15 s, crossfade 0.05:
**~220 ms typical; ~80–120 ms** on a strong GPU at block 0.05–0.08 s with CUDA Graphs
(`tools/cuda_graph.py` + prewarm at `realtime_gui.py:686-740`) — but any inference spike
> block is an instant audible glitch, since there is no decoupling buffer.

### 3.4 Vonovox (claimed)

Not verifiable in source. README positions block size as the single knob ("lowest you can
get without choppy"), fixed 48 kHz, full-GPU pipeline, VAC recommended. Community-reported
usable range on mid GPUs corresponds to blocks of tens of ms — plausibly **~60–120 ms**
end-to-end on a 3060+ class GPU. Treat as target-setting anecdote, not data.

### 3.5 Aggressive design target for our app (from doc 03 architecture)

Assumptions: RTX 3060+ (user requirement: CUDA), WASAPI shared `latency="low"`, VB-Cable
pinned 48 kHz, hop decoupled from callback, CUDA-graphed pipeline, fp16.

| Stage | Conservative V1 | Aggressive | Type | Justification |
|---|---|---|---|---|
| T_audio_input (WASAPI capture, 10 ms period + callback block 10 ms) | 15 ms | 8 ms (exclusive, 5 ms block) | device | PortAudio WASAPI shared period ~10 ms; exclusive ~3 ms |
| T_input_buffer (hop accumulation) | 32 ms | 20 ms | algorithmic | hop must satisfy p99 compute < hop (§3.6) |
| T_queue_in | ~1 ms | ~1 ms | — | semaphore wake |
| T_compute total (pre+HuBERT+F0+synth+post) | 20 ms | 12 ms | compute | see §3.6; must be < hop |
| T_sola (overlap 32 ms + search 10 ms) | 42 ms | 30 ms (20+10) | algorithmic | quality floor ~20 ms overlap; below that audible roughness risk |
| T_output_buffer (ring target 1 hop + 10 ms jitter) | 42 ms | 25 ms | algorithmic | the knob to shave after profiling underruns |
| T_audio_driver (render 10 ms) | 10 ms | 5 ms | device | |
| T_vac (cable + consumer pull) | 15 ms | 8 ms (VAC tuned) | device | Vonovox README rationale for VAC |
| **Total** | **~175 ms** | **~110 ms** | | worst-case aligned; perceived average ~½ hop less |
| Stretch (hop 16 ms, overlap 16 ms, exclusive, compute <10 ms) | — | **~75–90 ms** | | parity with best community Vonovox/rvc-webui reports |

Numbers below ~70 ms require shrinking SOLA overlap below ~16 ms and hop below ~16 ms —
both run into RVC's 20 ms feature granularity (HuBERT hop 320 @16k) and quality collapse;
that regime belongs to different model families (streaming/causal VC), out of scope.

### 3.6 The stability inequality, concretely

`T_compute(p99) < hop`. T_compute per hop at extra=0.5 s (context ≈ 0.55 s, ≈27 HuBERT
frames of fresh audio but ~172 frames recomputed):

- HuBERT-base over 0.55 s, fp16, graphed: ~4–8 ms on a 3060, ~2–4 ms on a 4080.
- RMVPE over the fresh region (silence_front/pitch-cache pattern): ~2–5 ms (FCPE less).
- Synth (`return_length` ≈ hop+overlap+search only, thanks to skip_head): ~2–5 ms.
- Resamples + SOLA + copies: ~1–2 ms.
→ 10–20 ms typical, hence hop 20–32 ms with margin ≈1.5× (Applio's rule). Shrinking
EXTRA from 2.5 s (rvc-webui default) to 0.5 s is the single biggest compute cut, at a
modest quality cost (w-okada docs: superlinear in EXTRA).

---

## 4. Most likely bottlenecks (ranked)

1. **Block/hop accumulation** — dominant in every shipped config (w-okada default 683 ms;
   Applio 512 ms; rvc-webui 250 ms). Fix: decoupled small hop + worker (doc 03 §10).
2. **HuBERT full-context recompute** — forces hop ≥ compute; EXTRA length is quadratic-ish
   in cost (w-okada docs) and recomputed every hop in all four references. Fix: small
   EXTRA (0.5 s), fp16, CUDA Graphs; EXPERIMENTAL: feature caching.
3. **Output safety buffering** — hidden ≥1-block term in w-okada server mode (blocking
   queue get) and in any design that doesn't measure ring fill. Fix: explicit ring target
   + underrun counters + auto-tune.
4. **SOLA overlap** — 85–100 ms defaults (w-okada/Applio) are pure added latency; 20–40 ms
   suffices (rvc-webui caps its sola_buffer at 40 ms: `realtime_gui.py:607`).
5. Device/cable layer — 20–40 ms shared-mode + unpinned-sample-rate SRC; fix by pinning
   48 kHz and offering exclusive mode.
6. Python/GIL jitter — affects p99 not p50; mitigated by graphed GPU work (GIL released),
   callback minimalism, optional process isolation (Applio's escape hatch).

---

## 5. In-app profiler design

Requirements: per-stage, per-hop, negligible overhead, distinguishes GPU time from
wall-clock, and ties audio-health events (underruns) to timing history.

### 5.1 Timers

- **Host timers**: `time.perf_counter_ns()` at stage boundaries in the worker — measures
  *wall* time including GIL/queue effects. This is what the stability inequality needs.
  (References: Applio `worker.py:45-47` and `core.py::on_request` both use
  `perf_counter`; w-okada uses its `Timer2` context with `.record()` sub-marks —
  `VoiceChangerV2.py::on_request`.)
- **CUDA events** (`torch.cuda.Event(enable_timing=True)` pairs around each GPU stage):
  measure device time without forcing a sync — `elapsed_time()` is read **one hop later**
  (or on a 1 Hz stats flush) so profiling never inserts a `synchronize()` into the hot
  path. Caveat: inside a captured CUDA Graph you cannot add per-stage events; profile
  stage-wise in eager warmup, then only whole-graph events in steady state.
- Rule: *host timers for scheduling decisions, CUDA events for optimization decisions.*

### 5.2 Aggregation

- Fixed-size ring (e.g., 1024 hops) of per-stage float32 samples; p50/p95/p99 + max
  computed on the GUI thread at 1–2 Hz (numpy percentile over the ring — never in the
  worker loop).
- Counters: output-ring underruns (zeros emitted), input overruns (`status.input_overflow`
  from the sounddevice callback status arg — note all four references *ignore* `status`;
  we must not), dropped hops, SOLA offset histogram (persistently large offsets = phase
  instability), VAD transitions.
- Derived gauges: ring fill (samples) at output-callback entry → direct measurement of
  T_output_buffer; end-to-end estimate = hop + overlap + search + measured ring fill +
  `stream.latency` (PortAudio-reported, as rvc-webui uses at `realtime_gui.py:478`).
- Headroom gauge: `hop_ms / p99_compute_ms`; auto-tuner raises hop when < 1.2, may lower
  when > 2 (hysteresis).

### 5.3 Example profiler output (ILLUSTRATIVE / HYPOTHETICAL numbers)

```
=== rt-profiler | 60 s window | 1875 hops | hop=32.0 ms, overlap=32 ms, search=10 ms ===
stage                    p50     p95     p99     max   unit   notes
-------------------------------------------------------------------------
cb_in -> worker wake     0.4     0.9     1.6     4.2    ms
pre (gate+resample)      0.6     0.9     1.2     2.1    ms    GPU evt: 0.3
hubert (0.55s ctx)       5.8     6.9     8.1    11.0    ms    GPU evt: 5.1
f0 rmvpe (fresh)         2.9     3.6     4.4     7.3    ms    GPU evt: 2.5
synth (ret_len 74ms)     3.1     3.8     4.6     6.9    ms    GPU evt: 2.8
post (resample+rms)      0.8     1.1     1.5     2.4    ms
sola (+copy to ring)     0.7     1.0     1.4     2.0    ms    offset p50=3.1ms
TOTAL compute           14.6    17.4    20.9    28.4    ms    headroom p99 = 1.53x
-------------------------------------------------------------------------
output ring fill        41.2    47.8    51.0    58.3    ms (target 42)
est. end-to-end        ~128     —       —       —       ms (hop+overlap+search+fill+dev 13)
underruns: 0   input overflows: 0   dropped hops: 0   vad speech: 72%
```

---

## 6. Optimization catalog

Evidence tags: **PROVEN** = implemented and working in a reference repo; **LIKELY** =
standard technique, straightforward to validate; **EXPERIMENTAL** = research risk.

### LOW-RISK (do in V1)

| Optimization | Evidence | Where proven |
|---|---|---|
| `skip_head` / `return_length`: decode only hop+overlap+search, feed full context | PROVEN | rvc-webui `rtrvc.py::infer`, Applio `core.py::realloc` |
| Rolling pitch cache; F0 only on fresh region / `silence_front` skip | PROVEN | rvc-webui `rtrvc.py:290-296`; w-okada `RVCr2.py:237` + `Pipeline.extractPitch` |
| Incremental GPU resampling with overlap-discard | PROVEN | rvc-webui `realtime_gui.py:824-837` |
| fp16 inference on CUDA | PROVEN | all (config.is_half / Applio precision.py) |
| CUDA Graph capture of resample + RVC infer (+ SOLA) | PROVEN | rvc-webui `tools/cuda_graph.py` + prewarm; Applio `_graph_phase_output` |
| Silence gate skips embedder/F0/decoder, with tail-drain + onset-fade state protocol | PROVEN | Applio `core.py::inference/process_audio` |
| Bounded drop-oldest queues; output latest-wins; callback never blocks | PROVEN | Applio `audio.py::queue_latest`, `worker.py` |
| `latency="low"`, WASAPI settings via sounddevice; pin cable at 48 kHz | PROVEN | Applio `audio.py`; Vonovox README |
| EXTRA 0.5 s instead of 2.5 s | PROVEN (Applio default 0.5 s) | w-okada docs on EXTRA cost |
| Warmup passes before going live (avoid first-block stall) | PROVEN | Applio `core.py::warmup` (5 passes incl. SOLA/resample kernels), rvc-webui `prewarm_cuda_graph` |

### MEDIUM-RISK (V1.x, behind flags)

| Optimization | Evidence | Notes |
|---|---|---|
| Hop decoupled from callback block (5–10 ms callbacks, 20–32 ms hop) | LIKELY | none of the four does it; standard audio engineering; needs real ring + underrun telemetry |
| Auto hop/ring tuner from p99 compute + underruns | PROVEN (coarse) / LIKELY (fine) | Applio `automatic_block_frame` is the coarse version |
| WASAPI exclusive mode | PROVEN (option exists: rvc-webui `:746-752`, Applio) | device compatibility QA burden |
| Phase-vocoder crossfade in SOLA | PROVEN | Applio `phase_vocoder`; costs 2 rFFTs, improves overlap artifacts |
| SOLA overlap 20 ms + search 10 ms | LIKELY | rvc-webui caps at 40 ms already; listen-test below 20 ms |
| FCPE instead of RMVPE for F0 | PROVEN (option in rvc-webui/Vonovox) | faster; slightly less robust pitch |
| ONNX/TensorRT for HuBERT or full pipeline | PROVEN (w-okada ships ONNX paths, Applio `onnx_decoder.py`) | wins vary; fp16 TRT HuBERT is the big one |
| torch.compile on synthesizer | LIKELY | watch recompiles on shape change — fix shapes first |
| CUDA stream priority for VC vs. games sharing the GPU | LIKELY | Vonovox README says it's being attempted; `torch.cuda.Stream(priority=-1)` |
| Process-isolated worker if GIL jitter shows at p99 | PROVEN | Applio `worker.py` |

### EXPERIMENTAL

| Idea | Evidence | Risk |
|---|---|---|
| HuBERT feature caching across hops (recompute only right edge + small margin) | EXPERIMENTAL | convs are non-causal; edge frames differ from offline values → audible shimmer; needs margin study. Nobody ships it |
| Streaming/causal content encoder (distilled HuBERT) to cut context to <100 ms | EXPERIMENTAL | retraining project; breaks drop-in RVC model compatibility |
| Overlapped hops (hop < block of synthesis, OLA everywhere) for sub-16 ms hops | EXPERIMENTAL | multiplies compute by overlap factor |
| Custom kernel-mode virtual audio driver | EXPERIMENTAL | months + signing; saves ~5–15 ms vs VB-Cable/VAC; rejected in doc 03 |
| Speculative inference during silence / predictive pre-roll | EXPERIMENTAL | complexity vs. benefit unclear |

---

## Key takeaways

1. **T_total is dominated by algorithmic terms you choose, not by GPU speed**: hop
   accumulation + SOLA overlap + output safety buffer account for 70–90% of latency in
   every reference; compute matters only through the stability constraint
   `p99(T_compute) < hop`.
2. **w-okada's buf/res is a correct but partial model**: buf = CHUNK×128/48k block
   accumulation (`ServerDevice.py:385`), res = Timer2-measured compute
   (`ServerDevice._processDataWithTime`); the shipped UX hides ≥1 extra block of output
   queueing and the crossfade overlap, which is why real-world w-okada sits ~90–200 ms+
   even well-tuned and >500 ms at defaults.
3. **EXTRA is free latency-wise, expensive compute-wise** — it is pure history (0 ms
   algorithmic), but it's recomputed through HuBERT every hop with superlinear cost
   (w-okada docs). 0.5 s (Applio default) is the sweet spot; 2.5 s (rvc-webui default) is
   waste.
4. **Budgets**: references land ~150–300 ms tuned (defaults far worse). Our decoupled
   design targets **~175 ms conservative, ~110 ms aggressive, ~75–90 ms stretch**, floored
   by hop ≥ ~16–20 ms (p99 compute + RVC's 20 ms feature granularity) and overlap ≥
   ~16–20 ms (stitch quality).
5. **Profiler is a first-class feature**: host perf_counter for scheduling truth, CUDA
   events (read late, never synced inline) for GPU truth, p50/p95/p99 over a hop ring,
   underrun/overflow counters (the references all ignore the callback `status` flags —
   we must not), and a headroom gauge driving an auto hop tuner (generalizing Applio's
   `automatic_block_frame` 1.5× rule).
6. **Everything LOW-RISK is already proven in the references** — skip_head decoding,
   pitch caching, incremental GPU resampling, CUDA Graphs, gate-with-state. The only
   genuinely novel (MEDIUM) piece our design adds is hop/callback decoupling; the
   EXPERIMENTAL frontier (feature caching, causal encoders) is where sub-70 ms lives and
   is explicitly out of V1 scope.
