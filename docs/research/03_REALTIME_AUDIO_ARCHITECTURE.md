# 03 — Real-Time Audio Architecture for an Ultra-Low-Latency RVC Voice Changer

Research deliverable. Grounded in source inspection of four reference implementations
(local clone paths shown; note: Vonovox ships only Cython-compiled `.pyd` binaries, so
its analysis is based on its README/docs plus binary module layout):

| Repo | Local path | Realtime entry points inspected |
|---|---|---|
| w-okada/voice-changer | `/home/claude/voice-changer` | `server/voice_changer/VoiceChangerV2.py`, `server/voice_changer/RVC/RVCr2.py`, `server/voice_changer/RVC/pipeline/Pipeline.py`, `server/voice_changer/Local/ServerDevice.py`, `server/sio/MMVC_Namespace.py`, `tutorials/tutorial_rvc_en_latest.md` |
| Applio | `/home/claude/Applio` | `rvc/realtime/audio.py`, `worker.py`, `core.py`, `pipeline.py`, `client.py`, `utils/chunks.py`, `utils/vad.py`, `utils/torch.py` |
| RVC WebUI (RVC-Project) | `/home/claude/rvc-webui` | `realtime_gui.py` (this repo's current name for the file historically called `gui_v1.py`; identical architecture), `infer/rtrvc.py` |
| Vonovox | `/home/claude/Vonovox` | `README.md`; `core/audio/audio_core.pyd`, `core/audio/audio_processors.pyd`, `core/inference/*.pyd` (compiled, not readable) |

---

## 1. Callback block size vs. model context size — the core distinction

Every working real-time RVC implementation separates two sizes that beginners conflate:

1. **Audio callback block size (`block_frame`)** — how many samples the audio API hands
   you per callback. This sets the *hop* of the system and dominates round-trip latency.
   Typical: 10–350 ms.
2. **Model context size (`convert_size`)** — how many samples are actually fed to
   HuBERT/ContentVec + F0 + the synthesizer each hop. RVC quality collapses with short
   context, so every implementation feeds the model a much longer window than one block:

   ```
   convert_size = extra (history) + crossfade overlap + SOLA search + block
   ```

   Only the trailing `block + overlap + search` region of the model *output* is kept;
   everything produced for the `extra` region is discarded (it exists purely to give the
   encoder acoustic context and to stabilize F0).

Concrete evidence:

- **rvc-webui** `realtime_gui.py:619-626` allocates exactly this window:
  ```python
  self.input_wav = torch.zeros(
      self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame, ...)
  ```
  and `skip_head = extra_frame // zc`, `return_length = (block + sola_buffer + sola_search) // zc`
  (`realtime_gui.py:646-649`) tell the model which slice to synthesize/return.
- **Applio** `rvc/realtime/core.py` → `Realtime.realloc()`:
  ```python
  convert_size_16k = block_frame_16k + sola_search_frame_16k + extra_frame_16k + crossfade_frame_16k
  if (modulo := convert_size_16k % self.window_size) != 0:   # pad to hop multiple (160 @16k)
      convert_size_16k += self.window_size - modulo
  self.skip_head = extra_frame_16k // self.window_size
  self.return_length = self.convert_feature_size_16k - self.skip_head
  ```
- **w-okada** `server/voice_changer/RVC/RVCr2.py:139-143` (`generate_input`):
  ```python
  convertSize = inputSize + crossfadeSize + solaSearchFrame + extra_frame
  if convertSize % 160 != 0:
      convertSize = convertSize + (160 - (convertSize % 160))
  ```

The `% 160` padding in all three reflects RVC's 16 kHz feature hop (HuBERT hop = 320,
F0 hop = 160); the context window must be an integer number of feature frames or the
decoder output length drifts against the audio clock.

**Design rule for our app:** the callback block is a *transport* unit; the model input is
a *sliding window over history*. They must be independently configurable, with the window
assembled from a ring buffer, never by waiting for more callbacks.

---

## 2. How each project chunks

### 2.1 rvc-webui (`realtime_gui.py`)

Three user sliders, all rounded to multiples of `zc = samplerate // 100` (one 10 ms
"zero-crossing"/feature frame at device rate, 160 samples at 16 kHz):

| Parameter | GUI default | Code |
|---|---|---|
| `block_time` | 0.25 s (range 0.02–1.5) | `realtime_gui.py:53`, `block_frame = round(block_time*sr/zc)*zc` (`:586-595`) |
| `crossfade_time` | 0.05 s (range 0.01–0.15) | `:55`, `crossfade_frame` `:597-606` |
| `extra_time` | 2.5 s (range 0.05–5.0) | `:56`, `extra_frame` `:609-618` |
| `sola_buffer_frame` | `min(crossfade_frame, 4*zc)` → ≤40 ms | `:607` |
| `sola_search_frame` | `zc` = 10 ms | `:608` |

The `zc` alignment matters: `block_frame_16k = 160 * block_frame // zc` (`:596`) keeps the
16 kHz feature stream and the device-rate stream in exact integer ratio, so the pitch
cache can be shifted by `block_frame_16k // 160` frames per hop without fractional drift
(`infer/rtrvc.py:290-296`).

Per callback (`audio_callback`, `:772-937`): shift `input_wav` left by `block_frame`,
append new block, shift the parallel 16 kHz buffer `input_wav_res`, resample **only the
new block plus 2*zc of overlap** (`:831-837` — `resampler(input_wav[-indata-2*zc:])[160:]`,
i.e. incremental resampling with filter-edge discard), run `rvc.infer(...)` on the whole
window, SOLA-align, write exactly `block_frame` samples to `outdata`.

### 2.2 Applio (`rvc/realtime/*`)

- Device side runs at a fixed `AUDIO_SAMPLE_RATE = 48000` (`core.py:35`); default
  `block_frame = 24576` samples = **512 ms** (`audio.py: Audio.start`, `worker.py:260`) —
  conservative, but there is an *auto* mode: `utils/chunks.py::automatic_block_frame()`
  sets `block_ms = clamp(ceil(processing_ms*1.5/10)*10, 100, 1000)`, i.e. **block ≥ 1.5×
  measured inference time, min 100 ms** — an explicit codification of the
  "inference must finish inside one block" stability rule with 50% headroom.
- `cross_fade_overlap_size` default 0.10 s, `extra_convert_size` default 0.5 s,
  `sola_search_frame = AUDIO_SAMPLE_RATE // 100` = 10 ms (`core.py:418-421`, `worker.py:254-261`).
- Internally everything is converted to 16 kHz frame counts in `realloc()` and processed
  at the model rate; input resample 48k→16k and output resample tgt_sr→48k are
  `torchaudio.transforms.Resample` modules on the GPU (`core.py:183-188`).

### 2.3 w-okada voice-changer

- Chunking unit: **CHUNK (`serverReadChunkSize`) × 128 samples @48 kHz**;
  `ServerDevice.py:385`:
  ```python
  block_frame = int(self.currentInputChunkNum * 128 * (serverInputAudioSampleRate / 48000))
  ```
  Default `serverReadChunkSize = 256` → 32768 samples ≈ **683 ms** (!) at 48 kHz; users
  typically tune CHUNK down to 48–112 (128–300 ms). In *client device mode*, the browser
  slices mic audio into the same CHUNK×128 units and ships them over Socket.IO
  (`sio/MMVC_Namespace.py::on_request_message`), so block size also sets network message size.
- EXTRA (`extraConvertSize`, seconds-worth of samples at input rate) is pure history:
  `RVCr2.py:199` converts it to 16 kHz and `generate_input` prepends it.
- `sola_search_frame = int(0.012 * processing_sampling_rate)` = **12 ms**
  (`VoiceChangerV2.py:233`); `crossfade_frame = min(crossFadeOverlapSize, block_frame)` (`:235`).
- Docs (`tutorials/tutorial_rvc_en_latest.md`): "The lag time from speaking to conversion
  is `buf + res` seconds… adjust the buffer time to be longer than the res time";
  EXTRA "determines how much past audio to include… the longer, the better the accuracy,
  but the longer the res" (and the note that the transformer cost grows superlinearly
  with EXTRA).

### 2.4 Vonovox

Source is compiled (`core/audio/audio_core.cp312-win_amd64.pyd`,
`core/inference/cuvc.pyd`, etc. — Cython, per README "Compiled to C with Cython"), so
only black-box observations:

- README exposes **Block Size** as "the critical setting… the lowest you can get without
  audio being choppy. GPU dependent"; fixed 48 kHz output; WASAPI (shared/exclusive) or
  ASIO backends; resampling built in for mismatched device rates.
- Pipeline is "nearly every operation on the GPU" with FCPE or RMVPE F0, ContentVec via
  transformers, RNNoise input NR, Silero-VAD-based output gating with a **400 ms release
  window**, optional AP-BWE 48k upscaler. Module names (`sifigan`, `ringformer`,
  `bigvgan`) show swapped vocoders as a quality/latency lever.
- Recommends **VAC (Virtual Audio Cable)** over VB-Audio for its virtual line, 48 kHz
  mono 16-bit, and Windows "Listen to this device" for monitoring.

### 2.5 Chunk-strategy decision table

| Strategy | Used by | Hop latency | GPU efficiency | Notes |
|---|---|---|---|---|
| Block = hop = callback size, full-context recompute each hop | all four | 1 block | poor (recompute `extra` every hop) | Simple, state-light; the de facto RVC standard |
| Auto block = 1.5× measured inference | Applio `automatic_block_frame` | adaptive ≥100 ms | — | Great stability default; too coarse for our target |
| Small callback (≤10 ms) decoupled from model hop via rings + worker | **our proposal** | callback ≪ hop | same as above | Callback never blocks; hop can shrink independently of device buffer |
| Feature caching (reuse HuBERT frames for the unchanged history) | none of the four (all recompute) | — | best | EXPERIMENTAL: HuBERT convs are not causal; frame values near the right edge differ from offline. w-okada's `silenceFront` is a partial form (skips F0 only) |

---

## 3. Ring/circular buffer implementations found

| Repo | Implementation | Citation |
|---|---|---|
| rvc-webui | "Shift-left" pseudo-ring on GPU tensors: `input_wav[:-block] = input_wav[block:].clone(); input_wav[-n:] = new` — same for `input_wav_res`, `input_wav_denoise`, `output_buffer`, `rms_buffer`, pitch caches | `realtime_gui.py:795-808`, `infer/rtrvc.py:291-292` |
| Applio | `circular_write(new, target)` helper, identical shift-left tensor scheme, applied to `audio_buffer`, `convert_buffer`, `pitch_buffer`, `pitchf_buffer` | `rvc/realtime/utils/torch.py:7-11`, `core.py::inference` |
| w-okada | Grow-then-trim numpy buffers: `np.concatenate([buffer, new])` then `buffer = buffer[-convertSize:]`; zero-pad at startup when underfull | `RVCr2.py::generate_input` (lines 121-163) |
| Vonovox | unknown (compiled); block-based like the rest per README | — |

None of them use a true lock-free ring (power-of-two indices, single-producer
single-consumer). They get away with it because **the "ring" lives entirely inside the
inference thread/callback** — the shift-left is just the model-context window update, not
a cross-thread transport. Cross-thread transport is `queue.Queue` (w-okada ServerDevice,
Applio) with explicit drop-oldest policies:

- Applio `audio.py::queue_latest()` — `put_nowait`, on `Full` pop one and retry: bounded,
  never blocks the callback.
- Applio `worker.py::_worker_loop` — drains `output_q` before `put_nowait` so queue depth
  stays ≤1 (freshest-result-wins, stale blocks dropped).
- w-okada `ServerDevice.audioOutput_callback` — `outQueue.get()` then drains backlog
  (`while qsize() > 0: get()`): output callback **blocks** on an empty queue — a real-time
  violation it tolerates because block sizes are huge.

### What is recomputed each hop vs. reused

Per hop, all reference implementations **recompute the full model context**
(`extra + overlap + search + block` → HuBERT + F0 + synthesizer). What is *reused*:

- **Pitch cache**: rvc-webui and w-okada keep per-frame F0 in a rolling cache and only
  compute F0 for the new region. rvc-webui `infer/rtrvc.py:290-296` shifts
  `cache_pitch/cache_pitchf` by `block_frame_16k // 160` and writes the fresh frames;
  w-okada passes `silence_front = extraConvertSize seconds` into
  `Pipeline.extractPitch(..., silence_front=...)` (`RVCr2.py:237`,
  `RVC/pipeline/Pipeline.py:77-86`) so the pitch extractor skips the leading history.
- **SOLA buffer** (last `crossfade` samples of previous output) — reused for stitching.
- **Resampled 16 kHz stream**: rvc-webui resamples only the new block (incremental,
  `realtime_gui.py:831-837`); Applio/w-okada resample only the incoming block as well
  (their buffers already live at 16 kHz).
- **Not reused anywhere**: HuBERT features for the history region (recomputed every hop)
  and the synthesizer output for the history region (computed then thrown away —
  mitigated by `skip_head`: rvc-webui/Applio pass `skip_head` so the *decoder* only
  synthesizes the tail, which is the single biggest compute saving in the modern
  pipelines; w-okada's RVCr2 trims output with `outSize` after full-length decode).

---

## 4. SOLA / stitching — exact code and math

All four stitch consecutive hops with **SOLA** (synchronous overlap-add): pick the output
offset in a small search window that maximizes normalized cross-correlation with the tail
of the previous hop, then equal-power crossfade.

### 4.1 rvc-webui (`realtime_gui.py:901-926`) — the canonical version

```python
# SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
cor_den = torch.sqrt(F.conv1d(conv_input**2, self.sola_den_kernel) + 1e-8)
sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
infer_wav = infer_wav[sola_offset:]
infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
self.sola_buffer[:] = infer_wav[self.block_frame : self.block_frame + self.sola_buffer_frame]
```

Math: with previous tail `p` (length `L` = `sola_buffer_frame` ≤ 40 ms) and new output
`y`, offset `k ∈ [0, search]` (search = `zc` = 10 ms) maximizes

`NCC(k) = Σᵢ y[k+i]·p[i] / sqrt(Σᵢ y[k+i]² + ε)`

— the numerator is `conv1d` with the previous tail as kernel; the denominator normalizes
by the energy of the candidate segment only (the `p` energy is constant across `k`, so it
cancels in the argmax). Crossfade windows are **sin²/cos² (Hann-shaped, equal-power
complementary)**: `fade_in = sin(π/2·t)²`, `fade_out = 1 − fade_in`
(`realtime_gui.py:650-664`). Note the mac workaround at `:913-917` (torch.max vs argmax).

### 4.2 Applio (`rvc/realtime/core.py::process_audio`, `_phase_aligned_output`)

Identical correlation (`conv1d` numerator, `sqrt(conv1d(x², ones)+1e-8)` denominator,
`argmax`), with `crossfade_frame` default 100 ms and `sola_search_frame` 10 ms. Two
upgrades over the canonical version:

1. **Phase-vocoder crossfade** (`core.py:38-103`, `phase_vocoder(a, b, fade_out, fade_in)`):
   instead of a pure time-domain fade, it windows both segments with
   `window = sqrt(fade_out*fade_in)`, takes rFFTs, and resynthesizes
   `a·fade_out² + b·fade_in² + Σ |Fa|+|Fb| · cos(phase-interpolated grid)` with the delta
   phase wrapped to [−π, π] — phase-aligning the overlap to avoid comb-filter
   cancellation when SOLA's best offset still leaves residual phase error. This is the
   only "phase vocoder" use among the four Python-readable repos (the task brief's
   "Vonovox phase vocoder" could not be confirmed: Vonovox is compiled; Applio is where
   this code lives).
2. **CUDA-graph-captured SOLA** (`core.py::_graph_phase_output`): the alignment offset is
   kept **on-device** (`index_select` with a tensor offset) so the whole
   SOLA+phase-vocoder post-process replays as a captured graph without a host sync.

Applio also handles **onset after silence** (`core.py::process_audio`, time-domain
branch): when `sola_buffer` is all zeros it searches for the voice onset in 160-sample
hops and applies a sin² fade-in from the onset, and on entering silence it **drains the
overlap tail once** (`_tail_remaining` logic) instead of hard-cutting — both are direct
anti-click measures around the gate (see §5).

### 4.3 w-okada (`server/voice_changer/VoiceChangerV2.py:246-285`)

Numpy variant: `np.convolve(audio[:cf+search], np.flip(sola_buffer))` for the numerator
(convolution with flipped kernel = correlation), `np.sqrt(np.convolve(x², ones(cf)))`
denominator, `argmax`. Search window **12 ms** (`0.012 * processing_sr`). Crossfade
windows are **cos² "strengths" with configurable offset/end rates**
(`_generate_strength`, `:170-206`): `prev = cos(θ)²`, `cur = cos(π/2−θ)² = sin(θ)²`,
padded with ones/zeros so the user can crossfade over only part of the overlap. The next
`sola_buffer` is pre-multiplied by `np_prev_strength` at save time (`:278-284`) rather
than at mix time — same math, different factoring.

### 4.4 Stitching parameter summary

| | overlap (crossfade) | search window | window shape | domain |
|---|---|---|---|---|
| rvc-webui | min(10–150 ms, 40 ms cap via `sola_buffer_frame`) | 10 ms (`zc`) | sin²/cos² | GPU torch |
| Applio | 100 ms default | 10 ms | sin²/cos² + optional phase vocoder (`sqrt(hann)` analysis window) | GPU torch, CUDA-graphable |
| w-okada | `crossFadeOverlapSize` (≤ block) | 12 ms | cos² with offset/end rates | CPU numpy |
| Vonovox | unknown (compiled) | unknown | unknown | GPU (claimed) |

**Latency consequence (critical):** SOLA requires the model to produce
`block + overlap + search` samples each hop, and the stitched stream is emitted
`overlap + search` *behind* the newest synthesized audio — overlap length is pure
algorithmic latency. Keep overlap at 20–40 ms, not Applio's 100 ms.

---

## 5. Silence / VAD handling and click/state hazards

| Mechanism | Where | How | State hazards observed in code |
|---|---|---|---|
| RMS noise gate | rvc-webui `realtime_gui.py:781-794` | per-10 ms-frame RMS vs threshold on *input*; zeroes sub-threshold frames inside the block (frame-wise, with an `rms_buffer` carrying 40 ms of history so frame RMS is valid at block edges) | Zeroing frames mid-block feeds discontinuous audio into the model; relies on model+SOLA to smooth |
| Silent-block skip | w-okada `RVCr2.py:211-212` | `if vol < silentThreshold: return zeros(convertSize)` — skips the entire pipeline | Returns zeros but **buffers keep the stale speech**; crossfade state not drained → click/garbage on resume is possible; cheap though ("the burden will not be so large" per docs) |
| `silenceFront` | w-okada `RVCr2.py:237` → `Pipeline.extractPitch(..., silence_front)` | Skips **F0 extraction** over the leading `extra` region (history is assumed already-pitched/silent); disabled when RVC Quality padding is on (`Pipeline.py:165`) | Not an output gate — a compute optimization. If history actually contained speech the cached pitch must be valid, hence the rolling pitch caches |
| webrtcvad + periodicity | Applio `utils/vad.py::VADProcessor` | WebRTC VAD (10/20/30 ms frames, aggressiveness 0–3) OR a custom "sustained voice" test (autocorrelation periodicity ≥ 0.8 + 70–4000 Hz band energy ≥ 75%) to keep vowels/singing; **200 ms hangover**; adaptive noise-RMS floor | Extensive explicit state management in `core.py`: on silence, `convert_buffer/pitch buffers are zeroed` ("stale speech must not be replayed"), the SOLA tail is faded out over exactly one overlap (`_tail_remaining`), and on voice resume with a partially drained tail the `sola_buffer` is zeroed and an onset fade-in applied. This is the best-engineered gate of the four |
| Silero VAD + RNNoise | Vonovox (compiled) | README: RNNoise input NR "very minimum latency"; Silero-VAD output mute with **400 ms release window**; warns both are "not compatible with singing or whispering" | unknown internally |

**Lesson for our app:** skipping pipeline stages saves GPU but creates four distinct
state machines that must stay coherent — model context buffer, pitch cache, SOLA buffer,
and output gate. Applio's `process_audio` silence branch is the reference for doing this
without clicks: (a) zero context buffers so stale audio can't replay, (b) drain the SOLA
tail with a fade exactly once, (c) re-enter with an onset fade-in, (d) keep advancing the
context ring during silence.

---

## 6. Resampling paths found in code

| Repo | Input path | Output path | Device |
|---|---|---|---|
| rvc-webui | `torchaudio.transforms.Resample(sr→16000)` on GPU, **incremental** (new block + 2·zc overlap, first 160 output samples discarded as filter transient) — `realtime_gui.py:665-669, 824-837`; optionally wrapped in a CUDA Graph (`tools/cuda_graph.py` via `run_cuda_graph`) | `Resample(tgt_sr→device_sr)` GPU (`:670-677`) | GPU |
| Applio | `tat.Resample(48000→16000)` GPU (`core.py:183-185`) | `tat.Resample(tgt_sr→48000)` GPU (`core.py:186-188`) | GPU |
| w-okada | `resampy.resample(..., filter="kaiser_fast")` per block (`RVCr2.py:188-196`) | `resampy` again (`RVCr2.py:247-253`) | **CPU** |
| Vonovox | built-in resampling for arbitrary device rates (README); internals unknown | 48 kHz fixed out | GPU (claimed) |

Notes:
- `torchaudio.transforms.Resample` precomputes a polyphase kaiser-windowed sinc kernel;
  stateless calls on short blocks incur filter edge effects — rvc-webui's
  2·zc-overlap-and-discard trick is the correct streaming workaround and is the pattern
  to copy. Filter group delay is `width/2` taps (default `lowpass_filter_width=6` →
  ~0.4 ms at 16 kHz) — negligible but nonzero algorithmic latency.
- `resampy` on CPU per 683 ms block is fine; per 10 ms block it becomes a real cost and a
  GIL holder — avoid for our design.
- Alternatives not found in these repos: `soxr` (libsoxr bindings; excellent CPU
  streaming resampler with explicit streaming API), `scipy.signal.resample_poly`
  (stateless), `julius` (GPU torch sinc). librosa appears only for `to_mono` and RMS.

### Resampler decision table

| Option | Domain | Streaming state | Latency | Verdict |
|---|---|---|---|---|
| torchaudio Resample + overlap-discard | GPU | manual (copy rvc-webui) | <1 ms | **Choose** for 48k↔16k and tgt_sr→48k; CUDA-graphable, zero CPU |
| soxr (`soxr.ResampleStream`) | CPU | built-in | ~1 ms | Fallback / CPU-only path; good for mic-rate→48k normalization |
| resampy kaiser_fast | CPU | none | — | Reject (per-block cost, no streaming state) |
| scipy/librosa | CPU | none | — | Reject for the hot path |
| julius | GPU | none | — | Equivalent to torchaudio; no advantage |

Fixed-rate policy (what Vonovox and Applio do): run the device graph at **48 kHz always**
(WASAPI shared mixer rate on most Windows boxes), model at 16 kHz in, tgt_sr (32/40/48k)
out → 48 kHz. Exactly two GPU resamplers, both with static shapes.

---

## 7. Noise suppression options and latency cost

| Option | Found in | Latency cost | Notes |
|---|---|---|---|
| TorchGate spectral gating (noisereduce) | rvc-webui (`tools/torchgate`, `realtime_gui.py:680-682, 805-829`), Applio (`worker.py:112-133`, noisereduce TorchGate on output) | rvc-webui adds `min(crossfade, 40 ms)` to its own displayed delay when input NR is on (`realtime_gui.py:483-484, 508-516`) — the NR path crossfades over an extra `sola_buffer_frame`; STFT of `n_fft = 4*zc` (40 ms) implies ~20 ms inherent | GPU; rvc-webui explicitly notes bundled `torch.istft` is **not CUDA-graph capturable** (`realtime_gui.py:678-679`) |
| RNNoise | Vonovox (README) | ~10 ms frame (480 samples @48k), near-zero extra | C library, GRU-based; the right choice for a V1 "cheap NR" toggle; Python via `pyrnnoise`/ctypes |
| WebRTC VAD gate | Applio (`utils/vad.py`) | 10–30 ms frame granularity + 200 ms hangover (release, not added delay) | Gate, not denoiser |
| Silero VAD | Vonovox (README) | 32 ms windows @16k, ~1 ms compute (ONNX) | Better accuracy than WebRTC; 400 ms release in Vonovox |
| None (clean mic assumption) | — | 0 | Our V1 default for one known user with a decent mic |

Recommendation: ship V1 with **threshold gate + optional RNNoise pre-filter + Silero VAD
gating with hangover**, all off the audio callback. Spectral gating (TorchGate) as an
optional quality mode — it costs real algorithmic latency (crossfade) and blocks CUDA
Graph capture.

---

## 8. Audio backends on Windows

| Backend | Python access | Buffer control | Device selection | Latency floor (practical) | Notes |
|---|---|---|---|---|---|
| **WASAPI shared** | sounddevice/PortAudio (`sd.WasapiSettings`) | blocksize per stream; engine mixes at 10 ms periods (IAudioClient3 can reach ~2.7 ms but PortAudio doesn't use it) | by hostapi+index (rvc-webui `update_devices` filters per hostapi) | ~10–30 ms round trip added by the engine | Default; honors `auto_convert=True` for rate mismatch (Applio `audio.py:409-412`) |
| **WASAPI exclusive** | `sd.WasapiSettings(exclusive=True)` — used by rvc-webui (`realtime_gui.py:746-752`) and Applio (`audio.py:409-426`) | device-period buffers, ~3 ms possible | same | ~3–10 ms | Locks the device (mic exclusivity is fine; output to VAC exclusive can conflict with other apps); sample rate must match device exactly |
| **ASIO** | sounddevice built with ASIO (official PyPI wheels since PortAudio bundles it on Windows); `sd.AsioSettings(channel_selectors=...)` — Applio `audio.py:413-438` | driver-defined, 1.5–10 ms | per-channel selectors | 1–10 ms | Only with real ASIO hardware/driver; ASIO4ALL just wraps WDM. Applio re-inits PortAudio for clean ASIO state (`audio.py:284-293`) |
| PyAudio | PortAudio too, older API | same | clumsier | same | No WasapiSettings equivalents exposed as cleanly; reject |
| miniaudio (python bindings) | callback or generator API | yes | yes | similar to WASAPI | Nice single-file lib, but no exclusive-mode knob in the Python binding; smaller community for RT |
| RtAudio (rtaudio-python) | thin binding | yes | yes | similar | Bindings poorly maintained; reject |
| sd.Stream vs split streams | — | — | — | — | Full-duplex `sd.Stream` (one callback, in+out clocked together — rvc-webui) vs separate Input/OutputStream + queue (w-okada `runNoMonitorSeparate`, Applio when devices differ): split streams are mandatory when input and output are on **different clock domains** (mic vs VAC), at the cost of one queue hop |

**Backend decision: sounddevice (PortAudio) with WASAPI**, shared mode default,
exclusive-mode toggle, `latency="low"` (as Applio sets on every stream,
`audio.py:296-346`), separate InputStream (mic) and OutputStream (virtual cable) since
they are distinct devices/clocks. ASIO supported opportunistically via `AsioSettings`.
This matches all three Python references and is the only option giving us WASAPI
exclusive + ASIO + device enumeration from pip-installable wheels.

### Virtual audio cable strategy

- **V1: use an existing virtual cable driver — VB-Cable (VB-Audio) or VAC (Muzychenko).**
  Justification: a kernel-mode audio driver requires WDK development, WHQL/EV signing
  (or test-signing mode), and its own ring/clock logic — months of work orthogonal to
  voice conversion, with BSOD-class failure modes. Every reference app does exactly this:
  w-okada docs assume VB-Cable; Vonovox README prescribes VAC ("works much better with
  realtime applications… less glitches," set lines to 48000 Hz / 1ch / 16-bit).
- VB-Cable adds its own internal buffering (max latency ≈ its internal buffer, typically
  ~5–20 ms at 48 kHz depending on the consumer app's pull cadence); VAC lets you
  configure the ms-per-interrupt (down to ~2–5 ms) which is why Vonovox prefers it.
  Either way: **pin the cable to 48 kHz in Windows sound settings** so no hidden SRC runs
  in the audio engine, and document "Listen to this device" monitoring (Vonovox README)
  for the user.
- V2+ option: ship nothing and detect/install VB-Cable via its installer; writing our own
  driver is EXPERIMENTAL and not justified by the ~5–15 ms it could save.

---

## 9. Threading model

### The rules

1. **The audio callback never blocks**: no locks that can contend, no `Queue.get()`
   without timeout-0, no GPU sync, no allocation, no file/UI I/O. PortAudio callbacks run
   on a high-priority native thread; a late return = glitch for every app on the device.
2. **GIL reality**: the PortAudio callback acquires the GIL to run Python. Any other
   Python thread holding the GIL (e.g., numpy ops that don't release it, JSON parsing,
   logging) delays the callback. Heavy work must either release the GIL (torch CUDA ops
   mostly do), run in a **separate process** (Applio), or be short.
3. Cross-thread transport: bounded queues with drop-oldest (never grow, never block), or
   an SPSC ring buffer. `queue.Queue` is mutex-based but uncontended put_nowait/get_nowait
   is ~1 µs — acceptable; a numpy SPSC ring with atomic indices is the upgrade path.

### How each reference threads it

| Project | Input | Inference | Output | Verdict |
|---|---|---|---|---|
| **rvc-webui** | one full-duplex `sd.Stream` callback | **inline in the callback** (`audio_callback` runs resample→HuBERT→F0→synth→SOLA, `realtime_gui.py:772-937`) | same callback writes `outdata` | Simplest and lowest latency *when inference always fits in a block*: any GPU hiccup = immediate audible underrun; callback blocks on `.cpu()` sync; impossible to use blocks smaller than worst-case inference. GIL held by callback during Python portions |
| **w-okada server mode** | `sd.InputStream` callback → runs the whole pipeline **inline** then `outQueue.put` (`ServerDevice.py::audioInput_callback_outQueue`) | inline in input callback | separate `sd.OutputStream` callback does blocking `outQueue.get()` + drains backlog | Decouples in/out clocks (needed: mic and VAC are different devices) and absorbs jitter with ≥1 block of queue latency; inference still blocks the *input* callback; output callback blocking on get() is a wart |
| **w-okada client mode** | browser mic (AudioWorklet) → Socket.IO binary → server `on_request_message` (`sio/MMVC_Namespace.py:41-56`) | server thread | response → browser playback buffer | Adds network + browser queueing; the reason server-device mode exists and measures lower latency |
| **Applio** | `sd.InputStream` callback → `worker.submit()` (put_nowait, drops on Full) | **separate spawned process** (`worker.py::VoiceChangerWorker`, `mp.Queue` depth 2, stale-output dropping) | `sd.OutputStream` callback drains `out_queue` latest-wins (`audio.py::audio_queue`); `fill(0)` on empty | The right shape: callbacks are O(µs); GIL/CUDA completely isolated in another process. Costs: numpy arrays pickled through mp.Queue (~block-size copies), one block of pipeline latency, process spawn startup |
| **Vonovox** | unknown; README implies decoupled ("lowest block size without choppiness" tuning, CUDA stream prioritization work) | GPU pipeline | — | — |

### Language choice

**Recommendation: Python core with disciplined boundaries; no native rewrite for V1.**

- Everything latency-critical either runs on the GPU under torch (releases GIL, can be
  CUDA-graph captured — rvc-webui `tools/cuda_graph.py` usage and Applio's
  `_graph_phase_output` both prove RVC+SOLA are graphable) or is O(block) numpy that
  takes microseconds.
- The references demonstrate the full spectrum in pure Python: inline callback
  (rvc-webui) at 250 ms blocks down to Applio's process-isolated worker. None are
  bottlenecked by Python-the-language; they are bottlenecked by **block accumulation and
  model context** (see doc 04).
- Native (Rust/C++) earns its keep only for (a) a custom SPSC ring under a <5 ms
  callback, (b) our own virtual device, (c) shipping without a Python runtime. All V2+.
  Keep the boundary clean: audio I/O module, ring module, worker module with
  numpy-in/numpy-out interfaces so any one can be swapped for a native extension later.
- Within Python, prefer **threads over Applio's process**: torch CUDA work releases the
  GIL, avoiding the mp.Queue pickling copies and config-sync machinery
  (`worker.py::_apply_config` is ~200 lines of state mirroring). Keep the process
  escape hatch if profiling shows GIL stalls from the GUI.

---

## 10. Proposed real-time data-flow architecture (our app)

```
                 Windows / NVIDIA, 48 kHz everywhere on the device side
┌──────────┐  WASAPI shared/excl   ┌──────────────────────────────────────────┐
│ Physical │  InputStream callback │  INPUT RING  (mono f32 @48k, ~4 s)       │
│  mic     │ ───(≤10 ms blocks)──▶ │  write_idx advanced by callback only     │
└──────────┘   append + gate-RMS   │  [ ..........history.......... |new]     │
      callback work: memcpy +      └──────────────┬───────────────────────────┘
      peak/RMS meter only (~µs)                   │ hop event (every HOP=20–32 ms,
                                                  │  semaphore from callback when
                                                  ▼  ≥HOP new samples present)
                               ┌─────────────────────────────────────────────┐
                               │ INFERENCE WORKER (daemon thread, CUDA)      │
                               │ 1. slice window = extra(0.5–1.0s)+overlap   │
                               │    +search+hop from INPUT RING              │
                               │ 2. VAD/gate decision (Silero, hangover)     │
                               │    → silent: emit tail-fade, skip GPU       │
                               │ 3. GPU: resample 48k→16k (incremental,      │
                               │    overlap-discard) → HuBERT → F0 (cached   │
                               │    history, RMVPE/FCPE new frames only)     │
                               │    → synth with skip_head/return_length     │
                               │    → resample tgt_sr→48k                    │
                               │    (whole step CUDA-graph captured)         │
                               │ 4. SOLA state: argmax NCC over 10 ms search,│
                               │    sin² crossfade vs sola_buffer (20–40 ms),│
                               │    save new tail → sola_buffer              │
                               │ 5. rms_mix / effects (optional)             │
                               │ 6. write exactly HOP samples ──────────────┐│
                               └─────────────────────────────────────────────┘│
                                                  │ one .cpu() copy, pinned   │
                                                  ▼                           │
                               ┌─────────────────────────────────────────────┐
                               │ OUTPUT RING (f32 @48k, target fill:         │
                               │  1×HOP + jitter margin; underrun → emit     │
                               │  zeros + counter, overrun → drop oldest)    │
                               └──────────────┬──────────────────────────────┘
                                              │ OutputStream callback
┌──────────────┐   WASAPI (≤10 ms blocks)     │ reads blocksize, never blocks
│ VB-Cable /   │ ◀────────────────────────────┘
│ VAC input pin│ ──▶ Discord/OBS/game select "CABLE Output" as mic
└──────────────┘     (+ optional monitor OutputStream to headphones)
```

Buffer roles, precisely:

| Buffer | Owner (writer) | Reader | Size | Purpose |
|---|---|---|---|---|
| INPUT RING | input callback | worker | ≥ extra+overlap+search+hop (+1 hop slack) | model-context history; replaces rvc-webui's shift-left `input_wav` with a true ring so the callback does one memcpy |
| 16 kHz mirror (GPU) | worker | worker | same window at 16k | incremental resample target (rvc-webui `input_wav_res` pattern) |
| pitch cache (GPU) | worker | worker | window/160 frames | roll + fresh-frames-only F0 (rtrvc `cache_pitch` pattern) |
| `sola_buffer` (GPU) | worker | worker | overlap (20–40 ms) | stitching state; zeroed+tail-drained per Applio's gate protocol |
| OUTPUT RING | worker | output callback | 2–4 hops capacity, target fill ~1 hop + 10 ms | absorbs inference jitter; its *fill target* is the explicit, measurable "output safety buffer" latency term |

Key deltas vs. references: callback block (≈5–10 ms) is **decoupled from the model hop**
(20–32 ms), so device buffers are tiny and the hop is bounded by GPU speed, not by
callback granularity; inference in a thread (not inline like rvc-webui, not a process
like Applio); output underruns are counted and fed back into an auto-hop controller
(Applio's `automatic_block_frame` idea, but acting on hop and ring target, with tighter
bounds).

---

## Key takeaways

1. **Two sizes, one ring.** Callback block size (transport, 5–10 ms) and model context
   (quality, 0.5–1 s of history) are different things; all four references couple hop ==
   callback block, which is why their minimum latency is stuck at "block big enough for
   inference." Decouple them with input/output rings and a worker.
2. **The RVC chunking formula is universal**: `convert = extra + crossfade + sola_search + block`,
   padded to 160-sample multiples at 16 kHz, with `skip_head/return_length` so the
   decoder only synthesizes the tail (rvc-webui `realtime_gui.py:586-649`, Applio
   `core.py::realloc`, w-okada `RVCr2.py:139-143`). Copy it; don't reinvent it.
3. **SOLA is ~15 lines and settled**: conv1d normalized cross-correlation over a 10–12 ms
   search, sin²/cos² crossfade of 20–100 ms, tail saved as state. Applio adds a
   phase-vocoder crossfade and keeps the argmax on-device for CUDA Graph capture — adopt
   both; keep overlap ≤40 ms because it is pure algorithmic latency.
4. **Silence handling is a four-buffer state machine** (context, pitch cache, SOLA tail,
   gate). Applio's drain-the-tail-once + onset-fade + zero-stale-buffers protocol is the
   only click-free implementation among the four; w-okada's bare `return zeros` is the
   cautionary tale.
5. **Backend**: sounddevice/PortAudio + WASAPI (shared default, exclusive toggle,
   `latency="low"`), separate input/output streams for mic vs. cable clock domains, ASIO
   via `AsioSettings` when present. **VB-Cable or VAC for V1, never a custom kernel
   driver** — every reference does the same, and Vonovox's field experience says prefer
   VAC pinned at 48 kHz.
6. **Resampling**: GPU torchaudio `Resample` with rvc-webui's incremental
   overlap-and-discard trick; exactly two fixed-rate converters (48k→16k, tgt→48k);
   resampy/librosa-per-block (w-okada) is the anti-pattern.
7. **Python is sufficient** for V1 given: callbacks that only memcpy, a CUDA-graphed GPU
   pipeline (proven in rvc-webui and Applio), bounded drop-oldest queues/rings, and clean
   module boundaries for a later native swap of the ring/IO layer.
