# 01 — Reference Projects for a Real-Time RVC Voice Changer

**Scope.** Source-level survey of the four reference codebases cloned under `repos/`, plus other realtime-RVC projects found via web search. Target product: single-user, male→female, ultra-low-latency realtime RVC on Windows + NVIDIA CUDA, Python.

**Repos inspected (local paths / default branches):**

| Project | Local path | Branch | Link base |
|---|---|---|---|
| w-okada voice-changer | `/root/rvc-research/repos/voice-changer` | `master` | https://github.com/w-okada/voice-changer/blob/master/ |
| Applio | `/root/rvc-research/repos/Applio` | `main` | https://github.com/IAHispano/Applio/blob/main/ |
| Vonovox | `/root/rvc-research/repos/Vonovox` | `main` | https://github.com/dr87/Vonovox/blob/main/ |
| RVC-WebUI (canonical) | `/root/rvc-research/repos/rvc-webui` | `main` | https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/ |

Confidence labels: **PROVEN** (verified in source and widely deployed), **LIKELY** (verified in source, less battle-tested), **EXPERIMENTAL** (behind flags / new).

---

## 1. Summary comparison

| Dimension | RVC-WebUI `realtime_gui.py` | w-okada voice-changer | Applio `rvc/realtime/` | Vonovox |
|---|---|---|---|---|
| Architecture | Single process, FreeSimpleGUI + sounddevice duplex callback | Client(browser)/server (FastAPI+Socket.IO), modular model slots | FastAPI + WebSocket server, optional mp worker process, browser or `sounddevice` client | Monolithic desktop app (CustomTkinter), Cython-compiled `.pyd` |
| Audio I/O | `sounddevice.Stream` duplex callback, WASAPI exclusive option | Browser audio (client mode) or server-side `sounddevice` (`Local/ServerDevice.py`) | WS binary frames or local `sounddevice` (`rvc/realtime/audio.py`) | WASAPI/ASIO via its own audio core (compiled) |
| Chunking | block_time slider 0.02–1.5 s; ring buffer with extra context | client sends chunks; server keeps growing `audio_buffer` trimmed to convertSize | block_frame at fixed 48 kHz; auto block sizing from warmup timing | "Block Size" + "Lookahead buffer" (≈ extra_time) |
| Extra context | `extra_time` (default 2.5 s) prepended; model trims via `skip_head` | `extraConvertSize` prepended; output trimmed to `outSize` | `extra_convert_size`; `skip_head`/`return_length` into synthesizer | "Lookahead buffer", recommend 2.0 s |
| Silence handling | RMS gate per 10 ms sub-frame (threshold slider) | `silenceFront` skips F0 over the extra-context region; RMS `silentThreshold` gate | dB gate + WebRTC VAD w/ periodicity rescue + hangover; tail-drain of SOLA buffer | Silero VAD gate (400 ms release) + RNNoise denoise |
| Crossfade | SOLA + sin² windows on GPU | SOLA in NumPy on CPU, configurable offset/end rates | SOLA + **phase-vocoder** crossfade, CUDA-graph-captured | compiled; claims SOLA-equivalent (not auditable) |
| F0 | pm / rmvpe / fcpe, cached pitch ring buffer | dio/harvest/crepe(+onnx)/rmvpe(+onnx)/fcpe plugins | rmvpe / fcpe / swift-f0 / crepe, GPU-resident, graph-capturable | RMVPE / FCPE / SwiftF0 / torchfcpe (compiled) |
| GPU usage | CUDA graphs (opt-in `RVC_CUDA_GRAPH`), fp16 toggle, GPU buffers | autocast fp16, per-request eager torch; ONNX EPs (CUDA/DML) | FP32 default, opt-in CUDA graphs + fp16 HuBERT autocast + ONNX decoder | "nearly everything on GPU", TF32/fp16 (claims) |
| Index | FAISS `read_index` + `reconstruct_n`; k=8 weighted | FAISS, k=1 fast path; silent-front offset skip | `IndexWrapper` torch-tensor search (GPU), k=8 | unknown (compiled); README doesn't mention index |
| ONNX | no | full export + ONNX inferencers | experimental decoder-only ONNX | no evidence |
| Model mgmt | pick .pth + .index by file dialog | **model slots** with auto-detected metadata | logs-path resolution; version/vocoder auto-detect at load | drag in .pth (RVC v2 only) |

---

## 2. RVC-WebUI (canonical, RVC-Project) — `realtime_gui.py` + `infer/rtrvc.py`

The upstream project was restructured (2025): the old `gui_v1.py` + `tools/rvc_for_realtime.py` became `realtime_gui.py` (GUI/audio/SOLA) and `infer/rtrvc.py` (class `RVC`, the realtime inference core). This is the **reference implementation of the realtime math** everyone else copied.

### 2.1 Audio capture & buffering (trace)
- `realtime_gui.py` `GUI.start_stream()` opens a **duplex** `sd.Stream(callback=self.audio_callback, blocksize=self.block_frame, samplerate=…)`, with optional `sd.WasapiSettings(exclusive=True)` — [realtime_gui.py#L742-L761](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/realtime_gui.py). Why it matters: duplex callback keeps input→output clock-locked to one device pair; exclusive WASAPI shaves OS-mixer latency.
- `GUI.start_vc()` sizes everything in units of `self.zc = samplerate // 100` (10 ms "zero-crossing" frames, matching the model's 100 Hz feature rate):
  - `block_frame` (processing hop), `crossfade_frame`, `sola_buffer_frame = min(crossfade, 4*zc)`, `sola_search_frame = zc`, `extra_frame` (context).
  - Persistent GPU ring buffer `input_wav` of length `extra + crossfade + sola_search + block`; a parallel 16 kHz buffer `input_wav_res` of `160 * len/zc` samples.
  - `skip_head = extra_frame // zc`, `return_length = (block + sola_buffer + sola_search) // zc` — these are passed **into the synthesizer** so the decoder only vocodes the needed tail (see §2.4).
- `audio_callback` shifts the ring buffer left by `block_frame`, appends new mono input, resamples only the fresh tail (`tat.Resample` with 2*zc overlap, dropping the first 160 samples to hide filter edge effects) — [realtime_gui.py#L795-L837].

### 2.2 Silence handling
- Threshold gate: appends `rms_buffer`, computes `librosa.feature.rms` per `zc` hop, zeroes sub-blocks below `threhold` dB — [realtime_gui.py#L781-L794]. Cheap, per-10ms granularity, but still runs the whole pipeline (no inference skip). **PROVEN but wasteful** — Applio improved this.

### 2.3 F0 and feature caching (`infer/rtrvc.py`, class `RVC`)
- Keeps `cache_pitch`/`cache_pitchf` (1024-frame GPU ring). Each block extracts F0 only over the recent `block_frame_16k + 800` samples (RMVPE: rounded up to `5120*k - 160` because RMVPE's mel framing), then shifts the cache by `block/160` frames and writes the new frames **trimmed `[3:-1]`** (edge frames of the extractor are unreliable) — `RVC.infer` [infer/rtrvc.py#L281-L296]. Why it matters: F0 cost becomes O(block), not O(block+extra); the cache provides consistent pitch over the context region.
- HuBERT runs over the **whole** 16 kHz ring buffer each block (the expensive part), features duplicated at the end (`torch.cat((feats, feats[:,-1:,:]),1)`) then 2× interpolated.
- Index search only over `feats[0][skip_head//2:]` — context frames are not re-retrieved — [infer/rtrvc.py#L251-L266].
- Formant shift implemented as `f0_up_key - formant` at extraction + output resample by `factor = 2**(formant/12)` via cached `torchaudio.Resample(orig_freq=upp_res, new_freq=tgt_sr//100)` kernels — [infer/rtrvc.py#L279-L355].

### 2.4 `skip_head` / `return_length` inside the model
- `infer/module/models.py` `SynthesizerTrnMs*NSFsid.infer(..., skip_head, return_length, return_length2)` slices the latent `z`, mask and `nsff0` to `[head : head+return_length]` **before** the HiFiGAN-NSF decoder, and `dec(..., n_res=return_length2)` lets the decoder resize for formant resampling — [infer/module/models.py#L646-L686]. Why it matters: decoder cost (the heaviest op) is proportional to the output block, not the 2.5 s context. This is the key micro-optimization worth copying. **PROVEN**.

### 2.5 SOLA + crossfade (GPU)
- `audio_callback` computes normalized cross-correlation (`F.conv1d` numerator / sqrt-energy denominator) between new output head and `sola_buffer` over `sola_search_frame`, `argmax` → `sola_offset`, then sin²/cos² fade windows — [realtime_gui.py#L901-L926]. Comment credits DDSP-SVC. **PROVEN**.

### 2.6 CUDA graphs
- `tools/cuda_graph.py`: `run_cuda_graph(module, key, fn, *inputs)` caches captured graphs per key with static input tensors; `detect_cuda_graph_support` runs a capture/replay probe; opt-in via `RVC_CUDA_GRAPH=1`. Graph keys embed `skip_head/return_length` so shape changes recapture. `realtime_gui.py` `prewarm_cuda_graph()` pushes a sine probe through resampler + full RVC infer before the stream starts — [realtime_gui.py#L686-L740]. Note the comment that bundled `torch.istft` (TorchGate NR) is **not graph-capturable**. **LIKELY** (recent addition).

### 2.7 Strengths / weaknesses / verdict
- **Strengths:** cleanest minimal realtime loop; correct ring-buffer math; `skip_head`/`return_length` decoder trimming; algorithm-delay readout (`stream.latency[-1] + block + crossfade + 0.01`).
- **Weaknesses:** no VAD/inference skip on silence; GUI thread and audio callback share state loosely; heavy work inside the PortAudio callback (relies on block ≥ inference time, causes glitching when overloaded); index loads whole `reconstruct_n` into RAM; no multi-model management.
- **Do NOT copy:** doing full inference inside the audio callback; FreeSimpleGUI event loop coupling; zh-only strings.

---

## 3. w-okada voice-changer

Architecture: a Node/React **client** (browser) talks to a Python **server** (`server/MMVCServerSIO.py`) over Socket.IO/REST; audio can flow client-side (browser mic → server → browser out) or fully server-side ("server device mode", `server/voice_changer/Local/ServerDevice.py`). `VoiceChangerManager` routes to `VoiceChangerV2`, which hosts a pluggable `VoiceChangerModel` (RVC = `RVCr2`).

### 3.1 Chunk/extra-context handling (trace)
- `RVCr2.inference()` ([server/voice_changer/RVC/RVCr2.py#L182-L257]) resamples incoming audio (any client SR) to 16 kHz with `resampy` `kaiser_fast`, converts crossfade/sola/extra sizes to 16 kHz, then `generate_input()`:
  - appends to persistent `audio_buffer` (numpy, grows then trimmed), `convertSize = input + crossfade + solaSearch + extra`, rounded **up to a multiple of 160** ("モデルの出力のホップサイズで切り捨てが発生するので補う" — compensate hop truncation);
  - zero-pads front if the buffer hasn't filled yet (cold start);
  - `outSize = ((convertSize - extra) / 16000) * model_sr` — output trimming target;
  - RMS `vol` measured over just the output-relevant crop; below `silentThreshold` → return zeros without inference — [RVCr2.py#L109-L212].
- Also maintains `pitchf_buffer`/`feature_buffer` carried across calls so the pipeline can reuse past pitch and (in index path) past features.

### 3.2 silenceFront (w-okada's signature optimization)
- `RVCSettings.silenceFront: int = 1`. `RVCr2` passes `extraConvertSize/inputSampleRate` seconds as `silence_front` when enabled — [RVCr2.py#L237].
- `pitchExtractor/RMVPEPitchExtractor.extract()` skips F0 computation over the front region: `start_frame = silence_front*16000/160`, crops audio, computes F0 only on the tail, then splices into the persistent `pitchf` cache — [server/voice_changer/RVC/pitchExtractor/RMVPEPitchExtractor.py#L23-L35].
- `pipeline/Pipeline.exec` also uses `npyOffset = floor(silence_front*16000)//360` to skip index search over the front and, for ONNX inferencers, slices `feats` itself — [server/voice_changer/RVC/pipeline/Pipeline.py#L186-L250]. (The `//360` constant is a long-standing quirk ≈ 2× the 160-hop at feature rate; it survives because the spliced cache hides errors.)
- Why it matters: this is the same idea as rvc-webui's pitch cache but implemented as a parameter threaded through plugins. The naming ("silence front") reflects that on cold start the front is literally zeros.

### 3.3 SOLA & crossfade
- Done in `VoiceChangerV2.on_request()` **in NumPy on the CPU** over int16 audio: `sola_search_frame = int(0.012*model_sr)`, correlation via `np.convolve`, configurable `crossFadeOffsetRate/EndRate` shaping the cos² windows, and `sola_buffer` saved with prev-strength applied — [server/voice_changer/VoiceChangerV2.py#L215-L287]. First block returns zeros ("warming up... generating sola buffer").
- Weakness: CPU round-trip per block + int16 quantization before crossfade; also returns `np.zeros(4096)` sized blocks on warmup regardless of block size.

### 3.4 Pipeline (`RVC/pipeline/Pipeline.py`)
- Classic stages with **quality mode**: `repeat=1` ("RVC Quality") adds reflect padding `t_pad` on both sides and trims `t_pad_tgt` at output — reproducing offline-pipeline quality at ~2× cost; `repeat=0` disables padding and enables silenceFront/outSize — [Pipeline.py#L150-L176].
- Index: fast path `k=1` nearest-neighbor (`self.index.search(npy, 1)`), with the k=8 weighted variant disabled by `k == 1` — [Pipeline.py#L197-L206]. Protect blend only when `search_index` is true (differs from upstream which only requires f0 — a real behavioral divergence, noted in their comment linking PR #276).
- `infer()` multiplies output by 32767.5 to int16 on GPU and supports `out_size` trimming inside inferencers.

### 3.5 Model slots, client/server device mode, ONNX
- **Model slots:** `ModelSlotManager` + per-type `*ModelSlotGenerator`. `RVC/RVCModelSlotGenerator.py` auto-detects everything from the checkpoint (see doc 09 §6) and persists JSON per slot dir. **PROVEN** pattern for drag-and-drop model import.
- **Server device mode:** `Local/ServerDevice.py` enumerates devices (`AudioDeviceList.py`), validates sample rates against `SERVER_DEVICE_SAMPLE_RATES`, and supports three monitor topologies (`NoMonitorSeparate`, `WithMonitorStandard`, `WithMonitorAllSeparate` — see issue #620) with `serverReadChunkSize` controlling callback granularity and `exclusiveMode` for WASAPI. In client mode, the browser handles devices and ships audio chunks over Socket.IO (`sio/` handlers), which adds network+encode latency but allows remote GPU.
- **ONNX:** full export (`RVC/onnxExporter/export2onnx.py` with `SynthesizerTrnMs256/768NSFsid_ONNX` wrappers embedding JSON metadata into ONNX `custom_metadata_map`) and inference (`OnnxRVCInferencer` feeding `feats/p_len/pitch/pitchf/sid`, fp16 autodetected from input tensor type) — the most complete ONNX story of the four. Version tags "v1.1/v2.1/v2.2" indicate clip-inside-graph variants.

### 3.6 Strengths / weaknesses
- **Strengths:** model-slot metadata detection; pluggable embedder/pitch/inferencer managers; ONNX + DirectML reach; silenceFront; explicit perf telemetry; huge installed base (battle-tested defaults: chunk, extra 0.5–4 s).
- **Weaknesses:** heavyweight client/server split for a local app (Socket.IO serialization, int16 hops); CPU SOLA; `resampy` on hot path (slow vs torchaudio GPU); many model types inflate complexity; repo now archived/superseded by the author's closed Beatrice v2 work.
- **Do NOT copy:** Socket.IO audio transport for a single-machine app; int16 interchange through the pipeline; the `//360` offset constant; per-request `faiss` big_npy reconstruct at slot load without memory guard.

---

## 4. Applio `rvc/realtime/` (the most modern realtime stack)

Server: `rvc/realtime/client.py` — FastAPI with `/ws-audio` (binary float32 blocks in/out) + `/change-config` WS; optional separate **worker process** (`rvc/realtime/worker.py`) with input/output/config queues where the output queue is force-drained to depth 1 (drop-stale policy). Audio can also be driven locally by `rvc/realtime/audio.py` (sounddevice; `queue_latest()` non-blocking bounded playback queue; ASIO sample-rate quirk handled in `resolve_sample_rate`).

### 4.1 Core loop (`rvc/realtime/core.py`)
- Fixed external rate `AUDIO_SAMPLE_RATE = 48000`; internal 16 kHz. `Realtime.realloc()` computes 16 kHz frame sizes, rounds `convert_size_16k` up to hop multiple, derives `skip_head = extra_16k // 160` and `return_length = convert_feature_size - skip_head`, allocates persistent GPU `convert_buffer`, `pitch_buffer`, `pitchf_buffer` (+1 frame slack "to compensate for pitch extraction algorithm") — [rvc/realtime/core.py#L251-L305].
- `inference()`: resample 48 k→16 k on GPU, `circular_write` into buffers, RMS + optional **VAD** decide `is_silence`; if silent, zero buffers and **skip the whole pipeline** returning zeros — with the key comment "Advance context even while gated; stale speech must not be replayed" — [core.py#L326-L354].
- `VoiceChanger.process_audio()`: on silence, **drains the SOLA tail** (`_tail_remaining`) so already-generated overlap fades out exactly once rather than clicking off — [core.py#L579-L597]; on new utterance after a drain, zeroes the sola buffer to avoid replaying stale overlap; onset detection + sin² fade-in for the first voiced block — [core.py#L636-L666]. This is the most careful silence/gating state machine of the four. **PROVEN**.

### 4.2 Crossfade: SOLA + phase vocoder
- `phase_vocoder(a, b, fade_out, fade_in)` ([core.py#L38-L105]) blends overlap in the frequency domain (phase-aligned sinusoid resynthesis) on top of the equal-power crossfade — removes phase-cancellation artifacts SOLA alone can leave at small crossfades. `_phase_aligned_output` keeps the SOLA argmax **on-device** (`index_select` with offset tensor) so the whole output stage can be CUDA-graph captured (`_graph_phase_output`) — [core.py#L478-L535]. **LIKELY** (newer; costlier than plain crossfade, worth A/B).

### 4.3 Pipeline (`rvc/realtime/pipeline.py`)
- `RealtimeVoiceConverter`: loads checkpoint with `weights_only=True`, infers `tgt_sr`/`n_spk`/`use_f0`/`version`/`vocoder` from checkpoint, builds unified `Synthesizer` (supports HiFi-GAN / MRF HiFi-GAN / RefineGAN vocoders), `strip_parametrizations()` removes weight-norm for speed, frees `self.cpt` after load ("The network owns its weights now") — [pipeline.py#L29-L125]. FP32 by default (`dtype = torch.float32` with a comment that fp16 "may not be faster"); fp16 is opt-in per-submodel: `decoder_fp16` flag and `APPLIO_EXPERIMENTAL_HUBERT_FP16` autocast — **EXPERIMENTAL**.
- `Realtime_Pipeline._voice_conversion`: F0 only over `block_16k + 800` samples (same RMVPE 5120 rounding as rvc-webui; FCPE floor of 4800 samples with the comment "Chunk duration should control latency, not pitch quality"); pitch cache shift + `[3:-1]` interior trim; HuBERT over full buffer; v1 `final_proj`; feats last-frame duplication; GPU index retrieval; 2× interpolation; protect blend; `head = feats_len*(1 - return_length/p_len)` computed **without GPU .item() sync**; decoder via `net_g.infer(..., skip_head=head)` — [pipeline.py#L408-L573].
- **Index on GPU:** `IndexWrapper` (utils/torch.py) + `big_tsr` torch tensor; `_retrieve_speaker_embeddings` does k=8 inverse-square-distance weighting fully in torch over `feats[0][skip_head//2:]` — [pipeline.py#L702-L725]. `prepare_index()` loads/unloads retrieval tensors lazily and invalidates CUDA graphs when it changes. **PROVEN** improvement over faiss-on-CPU per block.
- **CUDA graphs:** whole `_voice_conversion` captured when `APPLIO_ENABLE_CUDA_GRAPHS=1`, NVIDIA-only, not ZLUDA/HIP, disabled for crepe/proposed-pitch/NR/pedalboard paths; cache key includes buffer `data_ptr()`s, model ids, TF32 flags; capture failure raises `CudaGraphCaptureError` → supervisor restarts process — [pipeline.py#L616-L700]. **EXPERIMENTAL but the right design** (graphs demand pointer-stable buffers; their `circular_write` in-place ring buffers provide exactly that).
- **ONNX decoder (experimental):** `onnx_decoder.py` exports **only the HiFi-GAN NSF decoder** ("encoder, flow and source stay in Torch"), keyed by checkpoint SHA-256; graph replay returns `(latent, source, speaker)` then runs the ONNX decoder outside the graph — [pipeline.py#L694-L699]. Clever split: the decoder is the dominant cost and is shape-static.

### 4.4 VAD (`rvc/realtime/utils/vad.py`)
- WebRTC VAD + custom `_is_sustained_voice` rescue (FFT band-energy 70–4000 Hz ≥75% + autocorrelation periodicity ≥0.8) because "conversational WebRTC VAD rejects periodic vowels/singing"; 200 ms hangover; adaptive noise-floor RMS. **PROVEN** and directly reusable.

### 4.5 Auto block size
- `utils/chunks.py` `automatic_block_frame(processing_ms)` = `ceil(processing*1.5/10)*10` ms clamped [100, 1000]; `client.py` warms up 5 blocks, takes max of last 3, and iteratively grows block_frame until it fits with 50% headroom — [client.py#L160-L190]. **PROVEN** idea; our target should aim lower than their 100 ms floor but keep the measure-then-size loop.

### 4.6 Strengths / weaknesses
- **Strengths:** best silence/gating state machine; GPU-resident everything (index, F0, RMS); CUDA-graph architecture with correct invalidation; phase-vocoder crossfade; multiprocess isolation of audio vs inference; multi-embedder (contentvec/spin/multilingual hubert); pedalboard post-FX; auto block sizing.
- **Weaknesses:** fixed 48 kHz external rate (extra resample if device differs); FP32-only default leaves speed on the table for RTX GPUs; heavy framework (full Applio app around it); WS adds a hop when used with the browser UI; config plumbing via positional `args[n]` indices in `voice_conversion` is fragile.
- **Do NOT copy:** positional-args config threading; fixed `AUDIO_SAMPLE_RATE` constant rather than negotiated device rate; pedalboard CPU round-trip inside the conversion path (breaks graphs, they know it).

---

## 5. Vonovox (dr87)

**Caveat:** all of `core/` and `gui/` ship only as Cython-compiled `cp312-win_amd64.pyd` binaries — no auditable Python. Findings below come from module names, `README.md`, and `docs/` only; treat implementation claims as **vendor-stated**.

- Target exactly our niche: NVIDIA-only, Windows, RVC v2-only, 48 kHz output, VAC-based routing — [README.md](https://github.com/dr87/Vonovox/blob/main/README.md).
- Module inventory (`core/inference/*.pyd`): `rmvpe`, `torchfcpe`, `swiftf0` (pitch); `models`, `models_ringformer`, `ringformer`, `sifigan/*` (SiFiGAN NSF generator, Snake activations, CheapTrick), `bigvgan/*` (anti-aliased activations incl. CUDA kernels) — i.e. it experiments with **alternative vocoders beyond stock NSF-HiFiGAN**; `cuvc` (presumably "CUDA voice conversion" core); `upscaler/` = modified **AP-BWE** bandwidth extension to 48 k in realtime.
- Audio/noise: **RNNoise** input denoise + **Silero VAD** output gate with 400 ms release (README "VAD Noise Reduction"); explicit warning that NR models break singing/whispering.
- Latency approach (README): block size as the critical knob; "Lookahead buffer" = extra context, recommended 2.0; flexible device SRs via built-in resampling; WASAPI/ASIO only.
- Embedder: Transformers port of ContentVec (`lengyue233/content-vec-best`) + optional **spin** embedder — same direction as rvc-webui's `infer/hubert.py` HF port.
- **Reusable concepts:** SwiftF0 as a cheap F0 option; AP-BWE 48 k upscaling for 32 k/40 k models; RNNoise (C, ~0 latency) instead of spectral-gate TorchGate; the product-level insight that VAC + "Listen to this device" is the most stable monitoring path on Windows.
- **Do NOT copy:** closed compiled distribution (can't verify DSP), premium-gated effects, Cython build pipeline (unnecessary for us).

---

## 6. Other projects worth knowing (web)

Web search in this area is dominated by SEO spam; the genuinely useful finds:

- **RVC-Project realtime VST** — the local clone contains `RVCRealtimeVST/` in rvc-webui, an upstream effort to host the same realtime core as a VST; worth watching for buffer-negotiation patterns.
- **Beatrice v2 (w-okada's successor)** — non-RVC paired VC model with far smaller compute; w-okada's `voice-changer` already special-cases it (`noCrossFade` path in `VoiceChangerV2.setModel`). Not RVC-compatible, so out of scope, but its existence explains why `voice-changer` RVC development stalled. **LIKELY**.
- **hf-rvc (esnya/hf-rvc)** — Transformers-native RVC reimplementation with ONNX export; useful as a reference for HF HuBERT-port equivalence (same approach rvc-webui now uses in `infer/hubert.py`).
- **OpenVoiceChanger (sioaeko)** — demonstrates the same rvc-webui realtime math in a different UI; nothing new algorithmically.
- **niel-blue/RVC-Realtime-GUI (HF)** — repackaged gui_v1; nothing new.
- No credible **TensorRT** realtime-RVC implementation surfaced; the dynamic-shape HuBERT and tiny batch make TRT gains marginal vs CUDA graphs. Recommendation: CUDA graphs first (two of four repos already do), TRT **EXPERIMENTAL** only for the decoder, mirroring Applio's decoder-only ONNX split.

Sources: [RVC-Project README](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/README.en.md), [RVC Realtime GUI](https://huggingface.co/niel-blue/RVC-Realtime-GUI), [hf-rvc](https://github.com/esnya/hf-rvc), [OpenVoiceChanger](https://github.com/sioaeko/OpenVoiceChanger), [Retrieval-based Voice Conversion (overview)](https://en.wikipedia.org/wiki/Retrieval-based_Voice_Conversion).

---

## 7. Latency decision table

| Technique | Source | Latency effect | Quality effect | Verdict |
|---|---|---|---|---|
| Small block + measured auto-sizing | Applio `utils/chunks.py`, `client.py` warmup loop | sets base latency ≈ block + crossfade + device | choppy if undersized | **PROVEN — adopt** |
| `skip_head`/`return_length` decoder trimming | rvc-webui `models.py#infer`, Applio `infer_latent` | big GPU-time cut (decoder ∝ block, not context) | none | **PROVEN — adopt** |
| F0 over tail only + pitch ring cache | rvc-webui `rtrvc.py`, Applio pipeline, w-okada silenceFront | F0 cost ∝ block | none (cache [3:-1] trim needed) | **PROVEN — adopt** |
| CUDA graph capture of full block | rvc-webui `tools/cuda_graph.py`, Applio `_graph_cache` | removes per-kernel launch overhead (dominant at small blocks) | none | **LIKELY — adopt with eager fallback + process-restart policy** |
| fp16 / autocast | w-okada (autocast), Applio (opt-in) | ~1.3–1.8× on RTX | occasional NaN on old GPUs (w-okada has fallback exceptions) | **LIKELY — fp16 with fp32 fallback** |
| GPU faiss-free index (torch topk) | Applio `IndexWrapper` | removes CPU hop per block | identical math | **PROVEN — adopt** |
| Phase-vocoder crossfade | Applio `core.py` | ~0 (GPU) | smoother joins at short crossfade | **LIKELY — A/B vs plain SOLA** |
| VAD inference skip + tail drain | Applio `core.py` | GPU idle while silent | must drain overlap (Applio shows how) | **PROVEN — adopt** |
| Decoder-only ONNX / TRT | Applio `onnx_decoder.py` | maybe 10–20% decoder | export drift risk | **EXPERIMENTAL** |
| Client/server split over WS/Socket.IO | w-okada, Applio | +5–20 ms + jitter | none | **reject for local app; keep process split via queues (Applio worker.py)** |
| RVC Quality mode (reflect pad `repeat=1`) | w-okada Pipeline | ~2× compute, +pad latency | slightly better edges | **reject for realtime** (use crossfade instead) |

---

## 8. Key takeaways for our implementation

1. **Base the realtime math on rvc-webui `realtime_gui.py` + `infer/rtrvc.py`** — it is the canonical, smallest-correct implementation: `zc = sr/100` unit sizing, GPU ring buffers, tail-only resample, pitch cache with `[3:-1]` trim, `skip_head/return_length` decoder slicing, GPU SOLA.
2. **Adopt Applio's control plane**: silence gate with VAD + sustained-voice rescue + hangover, SOLA tail-drain on gating, onset fade-in, warmup-then-auto-block-size, worker-process isolation with drop-stale output queue, GPU index retrieval, lazy index load/unload with graph invalidation.
3. **CUDA graphs are the single biggest small-block win** and both modern repos converged on the same requirements: pointer-stable static buffers (`circular_write` rings), shape-keyed caches, warmup replays, strict invalidation, and a hard-fail restart policy on capture errors. Design buffers graph-first from day one.
4. **Keep audio I/O out of the inference thread.** rvc-webui glitches because inference runs in the PortAudio callback; Applio's queue-based worker with latest-wins queues is the right Windows-friendly shape. Use WASAPI (exclusive optional); consider duplex single stream for clock lock like rvc-webui.
5. **Model management:** copy w-okada's slot generator metadata detection (doc 09) rather than file-dialog .pth picking.
6. **Don't copy:** network transports for local audio, CPU SOLA/int16 interchange (w-okada), inference-in-callback (rvc-webui), fixed 48 kHz constant (Applio), closed binary modules (Vonovox), RVC Quality reflect-padding in realtime.
7. **For male→female specifically:** formant control exists only in rvc-webui's realtime core (`formant_shift` → F0 offset + output resample trick, [infer/rtrvc.py#L279-L355]) — adopt it; plus Applio's `proposed_pitch` (auto pitch offset toward a target median F0, default threshold 155 Hz — exactly a M→F aid, [pipeline.py#L349-L379]).
8. Single-user focus lets us pre-bake: one model, one speaker id 0, fixed block/extra once measured — so one CUDA graph, captured at startup, replayed forever.
