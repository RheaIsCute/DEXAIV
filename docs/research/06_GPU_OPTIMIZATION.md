# 06 — GPU Optimization for Realtime RVC (CUDA / Windows)

Source-level research across four reference implementations (local clones):

| Repo | Path | Role in this study |
|---|---|---|
| rvc-webui | `/home/claude/rvc-webui` | Canonical RVC pipeline, plus its own realtime GUI and a CUDA-graph wrapper (`tools/cuda_graph.py`) |
| Applio | `/home/claude/Applio/rvc/realtime/` | Most complete open realtime path: CUDA Graphs, GPU retrieval, ONNX decoder with IO binding |
| w-okada voice-changer | `/home/claude/voice-changer/server/voice_changer/RVC/` | ONNX Runtime support, device/provider management |
| Vonovox | `/home/claude/Vonovox` | Closed-source (Cython `.pyd`), CUDA-only; limited but useful signals |

All line numbers refer to the current local checkouts.

---

## 1. CPU↔GPU boundary map: the canonical rvc-webui pipeline

File: `rvc-webui/infer/vc/pipeline.py`, class `Pipeline`. The offline path (`Pipeline.pipeline` → `Pipeline.vc`) crosses the PCIe bus and forces host synchronization at least **seven** times per segment. Each `.cpu()` / `.numpy()` on a CUDA tensor is an implicit `cudaStreamSynchronize`: the host blocks until every queued kernel finishes.

| # | Stage | Code (function, line) | Direction | Why it hurts |
|---|---|---|---|---|
| 1 | High-pass filter + padding | `pipeline()` L286–306: `signal.filtfilt(bh, ah, audio)`, `np.pad` | CPU only | scipy/numpy on host; audio arrives and stays on CPU |
| 2 | F0 extraction | `get_f0()` L79–123: parselmouth (pure CPU); RMVPE returns numpy (`infer_from_audio`); FCPE ends with `.squeeze().detach().cpu().numpy()` (L118–123) | GPU→CPU | F0 post-processing (`np.interp` fill of unvoiced, mel quantization L126–138) is all numpy; result re-uploaded as tensors L320–321 |
| 3 | Audio → feats upload | `vc()` L156–174: `torch.from_numpy(audio0)` … `feats.to(self.device)` | CPU→GPU | Pageable (non-pinned) copy every call |
| 4 | **FAISS retrieval round-trip** | `vc()` L182–195: `npy = feats[0].cpu().numpy()` → `index.search(npy, k=8)` → weighted blend in numpy → `torch.from_numpy(npy)...to(self.device)` | GPU→CPU→GPU | The signature RVC bottleneck: HuBERT output downloaded mid-pipeline, searched with CPU FAISS, blended in numpy, re-uploaded. Two full syncs sandwiching a CPU-bound search |
| 5 | Synthesizer output download | `vc()` L245: `audio1 = synthesized[0, 0].data.cpu().float().numpy()` | GPU→CPU | Necessary at the very end, but here it happens per segment, before RMS mixing |
| 6 | RMS envelope mix | `change_rms()` L23–42: `librosa.feature.rms` on numpy, interpolated with CPU torch, result multiplied back into numpy | CPU only | librosa on host over the full output |
| 7 | Resampling | `pipeline()` L398–401: `librosa.resample(audio_opt, orig_sr=tgt_sr, target_sr=resample_sr)` | CPU only | Host resampling of the final waveform |

Also: `torch.cuda.empty_cache()` at `vc()` L249 and `pipeline()` L409 — an expensive, synchronizing allocator flush per call (gated off only when its CUDA-graph mode is on).

### The realtime GUI (`rvc-webui/realtime_gui.py` + `infer/rtrvc.py`) keeps most of these

- `infer/rtrvc.py` `RVC.infer()` L252–263: same FAISS round-trip — `npy = feats[0][skip_head // 2:].cpu().numpy().astype("float32")` → CPU `index.search` → `torch.from_numpy(...).to(self.device)`.
- RMVPE path downloads audio: `x = x.cpu().numpy()` (rtrvc L168); FCPE result `.detach().cpu().numpy()` (L218).
- `realtime_gui.py` L874/L886: SOLA-adjacent RMS mix pulls both input and inferred audio to CPU (`.cpu().numpy()`) for `librosa.feature.rms`, then uploads the result back.
- SOLA offset: L914–917 — `sola_offset.item()` (device→host sync) in one branch; the other keeps `torch.argmax` as a tensor (sync deferred to indexing).
- Output: L927–931 — final `.cpu()` into the sounddevice `outdata` buffer (unavoidable, but it is the *only* justified download per block).
- Resampling, notably, is **already GPU torchaudio** here: `tat.Resample` wrapped in `run_cuda_graph` (L665–723).

**Net effect:** per realtime block the canonical pipeline performs ~4–6 device syncs plus a CPU FAISS search. On a 10–20 ms budget per block overhead, Python dispatch + syncs alone can exceed the GPU compute time of the models.

### w-okada does the same FAISS round-trip

`voice-changer/server/voice_changer/RVC/pipeline/Pipeline.py` L189–211: `npy = feats[0].cpu().numpy()` → CPU `self.index.search(npy, 1 or 8)` → numpy blend → `torch.from_numpy(npy)...to(self.device)`. Plus per-block `feats_buffer = feats.squeeze(0).detach().cpu()` (L259) — a cache download that syncs every block.

---

## 2. How Applio's realtime path fixes this

Directory: `Applio/rvc/realtime/`. The design goal visible in the code: **one upload per block (mic audio), one download per block (output audio), zero host decisions in between** — which is exactly what makes the whole block graph-capturable.

### 2.1 GPU-resident retrieval (kills boundary #4)

`rvc/realtime/utils/torch.py`, class `IndexWrapper`:
- `read_index_tensor()` (L227–252): FAISS index is read once, vectors reconstructed and moved to GPU as a contiguous tensor; `||b||²` norms precomputed; the numpy copy is then dropped.
- `search()` (L254–299): brute-force L2 via a single `torch.addmm` + `topk` on GPU. On OOM it falls back to CPU FAISS (`self.faiss_cpu = True`).
- `pipeline.py::_retrieve_speaker_embeddings` (L702–725) does the k=8 weighted blend entirely in torch on device — the dead numpy code is left as a comment (L710–712) showing exactly what was removed.

### 2.2 GPU F0 (kills boundary #2)

`pipeline.py::setup_f0` (L219–290) monkey-patches RMVPE so decoding is tensor-native: `_infer_from_audio` and `_to_local_average_cents` (L224–246) replace the numpy argmax/local-average decode with pure-torch equivalents, and `cents_mapping` is moved to GPU (L258–260). FCPE gets a `RealtimeMel` extractor and a torch `local_decoder`. The only remaining CPU excursions in `get_f0` are optional modes: `proposed_pitch` does `f0.cpu().numpy()` + `np.median/np.interp` (L349–375) and SWIFT returns numpy (L325) — and `voice_conversion()` (L624–631) explicitly **excludes those modes from graph capture**.

### 2.3 Persistent, preallocated buffers

`core.py::Realtime.realloc` (L251–305): `audio_buffer`, `convert_buffer`, `pitch_buffer`, `pitchf_buffer` allocated once on device; each block is written in with `circular_write` (`utils/torch.py` L7–11, an in-place shift+append). `pipeline.py` L183–184: `self._p_len_tensor` is a reused scalar tensor "to avoid per-block allocations" and, critically, to feed the synthesizer a *tensor* length instead of a Python int — `self._p_len_tensor.fill_(p_len)` (L544). The `head` trim is computed on host from static shapes (L539–543) explicitly "without a GPU `.item()` synchronization inside the synthesizer".

### 2.4 CUDA Graphs — what exactly is captured

`pipeline.py::voice_conversion` (L616–700):

- **Scope:** the *entire* conversion block — F0 model (RMVPE/FCPE), HuBERT, GPU retrieval, feature interpolation, synthesizer (and with the ONNX decoder active, everything up to decoder inputs via `return_decoder_inputs=True`). Comment at L620: "Graphs eliminate hundreds of host kernel submissions per block."
- **Static shapes:** guaranteed by design — `convert_buffer`/`pitch_buffer` have fixed sizes from `realloc`, and the cache `key` (L634–650) includes the input tensors' `data_ptr()`, all scalar args, model identities, and the global precision flags (`allow_tf32`, `cudnn.allow_tf32`, `cudnn.benchmark`). Any change invalidates and recaptures.
- **Variable chunk size** is handled by *not* varying it: the realtime block size is fixed per session; a settings change produces a new key → recapture (old graph released first, L652).
- **Warmup:** 3 eager iterations before capture (L657–665) — required by PyTorch to settle cuDNN autotuning/allocator state on a side stream; `torch.cuda.graph` supplies the capture side-stream itself. Mutable state (`pitch`/`pitchf` circular buffers) is snapshotted before warmup and restored after capture (L653–654, 687–689) so warmup doesn't corrupt the stream of real audio.
- **Eligibility gating** (L198–211): NVIDIA CUDA only, not ROCm/ZLUDA, opt-in via `APPLIO_ENABLE_CUDA_GRAPHS=1`, and only `fcpe`/`rmvpe` F0 (no CPU-touching modes, no pedalboard/noise-reduction effects — those call `.cpu().numpy()` inside the block, L592–598, which cannot be captured).
- **Failure policy:** capture failure raises `CudaGraphCaptureError` ("The service must restart...", L15–16) because a failed capture can leave the CUDA context poisoned; the `finally` block restores the caller's stream (L682–686).
- **Replay:** `graph.replay()` + reuse of the captured output tensor (L692–693). Inputs are "passed" by writing into the same preallocated buffers the graph captured (hence `data_ptr()` in the key).
- **SOLA is captured too:** `core.py::VoiceChanger._graph_phase_output` (L503–535) captures the SOLA correlation + phase-vocoder crossfade as a *second* graph, with the alignment offset kept on-device — `_phase_aligned_output` (L478–501) replaces the host-side `argmax().item()` with `torch.arange + offset` → `index_select`, so no sync is needed to apply the offset.
- **Model warmup at service level:** `core.py::VoiceChanger.warmup` (L684–710) runs 5 full silent blocks with VAD/silence gating disabled, so resampling, SOLA and both graph captures are warm before the first spoken block; buffers are zeroed afterwards.

### 2.5 Precision policy

- Baseline is deliberately **FP32**: `RealtimeVoiceConverter.__init__` L48: `self.dtype = torch.float32  # torch.float16 if config.is_half else torch.float32` — with the comment "it may not be faster". `core.py` L441 prints "Realtime computation device: ... (FP32)".
- **TF32 is an explicit opt-in** (`precision.py::enable_experimental_tf32`): env `APPLIO_EXPERIMENTAL_TF32=1`, NVIDIA CUDA only, SM ≥ 8.0 (Ampere+), set *before* warmup/capture because "it must not be toggled around asynchronous graph replay" (L18–19).
- **HuBERT FP16 is a scoped autocast opt-in** (`pipeline.py` L188–197, `APPLIO_EXPERIMENTAL_HUBERT_FP16=1`): only the embedder runs under `torch.autocast("cuda", dtype=torch.float16)` (L484–487); features are cast back to FP32 (`feats.to(self.dtype)`, L488) so retrieval and synthesis stay FP32 — "preserve FP32 retrieval/synthesis features while autocasting the large HuBERT network".
- **Decoder FP16** is a separate flag (`decoder_fp16`) threaded into `net_g.infer` and the NSF excitation source (`_voice_conversion` L556–561 wraps `decoder.source(f0, ...)` in autocast "Preserve the regular decoder's excitation precision policy").
- `rvc/configs/config.py` L84: `torch.backends.cudnn.benchmark = False` — deterministic plan selection; important because cudnn.benchmark re-autotunes on any new shape (a latency spike mid-stream) and interacts badly with graph capture.

### 2.6 Streams, pinned memory, non_blocking

- Applio's realtime path does **not** use pinned host memory or `non_blocking=True` anywhere (grep confirms); it doesn't need to, because after the single input upload everything stays on device, and CUDA graphs amortize launch overhead far more than async copies would.
- The ONNX decoder owns a dedicated non-default `torch.cuda.Stream` (`onnx_decoder.py` L344–347: "pointer 0 can be interpreted as an unspecified stream by providers") and fences it against the caller's stream with `wait_stream` both ways (L457–523) plus `record_stream` for allocator safety.
- rvc-webui's `tools/cuda_graph.py` *does* use `non_blocking=True` for the static-input copies at replay (`_CapturedCall.replay` L139) and a recorded event to serialize replays (L137–143).

### 2.7 Remaining syncs in Applio (deliberate, per-block)

`core.py` L331–332: `vol = vol_t.item()` — one scalar sync per block for the silence gate (runs *before* conversion, so it doesn't stall the model); VAD does `audio.cpu().numpy()` when enabled (L341); final output `.detach().cpu().numpy()` (L610/677). That's the irreducible set.

---

## 3. rvc-webui's own CUDA-graph wrapper (second PROVEN implementation)

`rvc-webui/tools/cuda_graph.py` is a generic per-callsite graph cache used in both offline `vc()` (synth only, L222–244 of `infer/vc/pipeline.py`) and the realtime GUI (input/output resamplers, noise gate, `rvc.infer`):

- Startup **probe** (`detect_cuda_graph_support`, L43–73): captures and replays a trivial graph and validates the numeric result before enabling (env `RVC_CUDA_GRAPH` override).
- `_CapturedCall` (L110–144): clones inputs into static tensors, 3 warmup iterations on a side stream, `torch.cuda.synchronize`, capture, then replay with `copy_(value, non_blocking=True)` and output cloning. LRU cache of 8 signatures (shape/stride/dtype/device), failures memoized to eager (L147–185).
- Unlike Applio it caches **multiple shape signatures** per callsite — a pragmatic answer to "variable chunk sizes": one graph per observed shape, evicting LRU.

Takeaway: two independent realtime RVC codebases converged on CUDA graphs with the same recipe (side-stream warmup ×3, static input buffers, shape-keyed cache, eager fallback).

---

## 4. Vonovox: what the code actually shows

Vonovox ships compiled Cython (`.pyd`, CPython 3.12, Windows x64) — `core/`, `gui/`, `utils/` are all binary; only `launcher.py` and `README.md` are readable. Verifiable facts:

- **CUDA-only by policy:** README: "FOR NVIDIA GPU's ONLY (GTX 900 or higher)... highly optimized pipeline for CUDA... Nearly every operation is processed on the GPU." PyTorch 2.7.0 + CUDA 12.8, Python 3.12.8.
- **Dedicated CUDA management module:** `utils/cuda_manager.cp312-win_amd64.pyd` — extracted strings include `allow_tf32`, `cudnn`, `empty_cache`, `synchronize`, `matmul`, `device`, `final_mem`, `freed_mem`: i.e. it toggles `torch.backends.cuda.matmul.allow_tf32` / cudnn TF32 and does memory accounting. No `CUDAGraph`/`capture` strings surfaced, so graphs cannot be confirmed.
- **`faiss-cpu==1.9.0.post1`** in `requirements.txt` — even the most CUDA-aggressive Windows implementation ships CPU FAISS (consistent with GPU FAISS being unavailable on Windows; see §9).
- **Vocoder choices:** `core/inference/bigvgan/` including `alias/cuda/load.pyd` — BigVGAN's anti-aliased activation with its optional fused CUDA extension — and `core/inference/sifigan/` (SiFiGAN, with `cheaptrick.pyd`). F0: compiled `rmvpe`, `torchfcpe`, `swiftf0` modules.
- README notes planned "CUDA stream prioritization" to fight contention with games — evidence that stream priority, not more graphs, is their next lever for the shared-GPU case.

Treat Vonovox as directional evidence (GPU-resident pipeline + TF32 + CPU faiss + optimized vocoder), not as citable mechanism.

---

## 5. Precision: FP16 vs BF16 vs TF32 per module

| Module | FP16 | BF16 | TF32 | Reference behavior |
|---|---|---|---|---|
| HuBERT / ContentVec | Safe under autocast; large matmuls, LayerNorm kept FP32 by autocast. Biggest win (it's the FLOPs-heaviest module) | Safe, slightly less precise mantissa but no overflow risk; ~same speed on Ampere+ | Safe | Applio: FP16 autocast opt-in (`APPLIO_EXPERIMENTAL_HUBERT_FP16`), features cast back to FP32. rvc-webui: whole pipeline `is_half` FP16 on capable GPUs (`configs/config.py` L43–65), with GPU blacklists (w-okada `DeviceManager.halfPrecisionAvailable` excludes GTX 16xx/10xx, P40, SM<7) |
| RMVPE / FCPE (F0) | **Risky**: salience decode (`to_local_average_cents`) computes weighted sums of small numbers; FP16 rounding shifts cents estimates and voiced/unvoiced thresholds (`thred=0.03`) sit near FP16 granularity | Safer than FP16 (FP32 exponent range) but mantissa loss still perturbs cent-level F0 | Fine | rvc-webui RMVPE accepts `is_half`; Applio realtime keeps F0 FP32 always (no autocast around `get_f0`) |
| Retrieval (index search + blend) | Risky: `1/score²` weights and L2 distances underflow/overflow; rvc-webui explicitly casts back to float32 before searching even when `is_half` (`pipeline.py` L183–184) | OK | OK | Applio `IndexWrapper` runs at pipeline dtype (FP32), clamps distances at `1e-8` |
| Flow + posterior encoder | FP16 generally OK | OK | OK | Applio ONNX path keeps "encoder, flow and source ... in Torch" (onnx_decoder.py module docstring) at pipeline dtype |
| NSF-HiFiGAN generator | Mostly OK, **except the NSF excitation source**: `SineGen` computes `cumsum` of phase (rad_values) — FP16 cumsum over thousands of samples accumulates phase error → audible pitch drift/metallic artifacts. Applio keeps the source generation under a *separate* autocast gate and computes it in Torch even with the ONNX decoder (`pipeline.py` L556–561, `onnx_decoder.py::forward` L423–425) | Worse mantissa than FP16 for cumsum; not used by any reference | Safe | Applio: `decoder_fp16` flag, default off; ONNX decoder defaults to FP16 **weights** on CUDA (`resolve_decoder_path` L212–214) but excitation stays Torch-side |
| SOLA / crossfade / resample | Keep FP32 — tiny cost, correlation peaks matter | — | — | Applio SOLA buffers are `torch.float32` (`core.py` L452–474) |

**Recommendation:** FP32 base + TF32 matmul/conv enabled (Ampere+) + FP16 autocast on HuBERT only; FP16 decoder as a measured option. BF16 buys nothing here: the risk modules (F0 decode, NSF cumsum) are precision- not range-limited, and FP16 has better mantissa than BF16.

### cudnn.benchmark, matmul flags, warmup

- `cudnn.benchmark=True` autotunes conv algorithms **per shape**; with strictly fixed block sizes it can help a few %, but every new shape triggers a multi-ms benchmark stall mid-stream. Applio sets it **False** (`rvc/configs/config.py` L84) and includes it in the graph cache key. Verdict: leave False; shapes are static anyway and CUDA graphs freeze the chosen kernels.
- `torch.backends.cuda.matmul.allow_tf32 = True` (+ `cudnn.allow_tf32`, default True) — set once at process start, before warmup/capture (precision.py comment). Equivalent modern API: `torch.set_float32_matmul_precision("high")`.
- **Kernel/model warmup is mandatory** regardless of graphs: first-call costs include cuDNN heuristics, lazy module init, allocator growth, and (if used) graph capture. Applio's `VoiceChanger.warmup` runs 5 full blocks and reports `max(timings[-3:])` as the realistic steady-state latency. Replicate this: warm up with the exact production block size, with silence gating disabled.

---

## 6. torch.compile for RVC — verdict

- With fixed realtime block sizes, shapes are static, so `torch.compile(mode="reduce-overhead")` (which uses CUDA graphs internally) is *theoretically* a drop-in for the launch-overhead problem.
- In practice for RVC modules: HuBERT (transformers `HubertModel`) compiles but with long cold-start (tens of seconds of Inductor compilation on Windows, where Triton support has only recently landed and remains the weak platform); the VITS synthesizer contains flow modules, `rand_like` noise, masking and python-side control flow that cause graph breaks; RMVPE/FCPE decode uses data-dependent indexing. Each break re-introduces eager dispatch between compiled regions, and any shape or scalar change triggers recompilation stalls — fatal mid-stream for audio.
- **No reference realtime RVC implementation uses torch.compile.** Both Applio and rvc-webui chose manual `torch.cuda.CUDAGraph` capture instead, which gives the same launch-overhead elimination with deterministic, controllable capture/fallback behavior.
- Verdict: **skip for V1.** Manual CUDA graphs dominate it for this workload (no compile stalls, no Triton-on-Windows dependency). Revisit `torch.compile` only for the HuBERT encoder in isolation (`fullgraph=False`, static shapes) as a measured experiment.

## 7. ONNX Runtime CUDA EP

**What w-okada ships:** full-model ONNX export (`RVC/onnxExporter/`) and `OnnxRVCInferencer` — but note its runtime pattern (`OnnxRVCInferencer.infer`, L49–77): `feats.cpu().numpy()` for every input and `torch.tensor(res)` for the output, i.e. **host round-trips every block, no IO binding**. CUDA EP selection is plain `["CUDAExecutionProvider"], [{"device_id": gpu}]` (`DeviceManager.getOnnxExecutionProvider`). ONNX there exists primarily for portability (DirectML for AMD/Intel via `DmlExecutionProvider`) and FP16 models, not for peak CUDA latency.

**What Applio ships (`rvc/realtime/onnx_decoder.py`)** — the state of the art among these repos:
- Exports **only the HiFi-GAN NSF decoder convolution stack** (`DecoderGraph`, L29–48); "encoder, flow and source stay in Torch" — the hard-to-export, precision-sensitive parts (NSF sine source with cumsum, flows with noise) never leave PyTorch.
- Export: opset 17, dynamic frame axes, optional FP16 conversion via `onnxruntime.transformers.float16.convert_float_to_float16`, sha256-manifested atomic cache (`resolve_decoder_path`).
- Runtime: CUDA EP with `user_compute_stream` (its own torch stream), `enable_cuda_graph=1`, `cudnn_conv1d_pad_to_nc1d=1`, `use_tf32` gated by precision (L350–364); `session.disable_cpu_ep_fallback` so a partially-CPU graph fails loudly instead of silently killing graph capture (L371–373).
- **IO binding + ORT CUDA graphs:** `_run_decoder` (L455–524) binds preallocated GPU buffers via `io_binding`, replays with `gpu_graph_id` run options, keeps ≤4 captured shape variants ("Uncommon additional sizes run without capture", L469–475), and fences everything with stream waits + `record_stream`.
- Integration with the Torch-side CUDA graph: `pipeline.py::voice_conversion` captures everything up to decoder inputs (`return_decoder_inputs=True`), then calls `run_decoder` outside the Torch graph (L694–699) — two graph domains stitched on one stream.

**Is ONNX actually faster for our workload?** For the decoder convs, CUDA EP with FP16 + cuDNN plans is typically on par to modestly faster (~10–30%) than PyTorch eager, and roughly a wash versus PyTorch-under-CUDA-graph; the win cases are FP16 tensor-core conv plans and avoidance of Python dispatch. w-okada-style whole-model ONNX **with host I/O is slower** than a graph-captured Torch pipeline — the `.cpu().numpy()` per block gives back everything ONNX gains. Applio labels its decoder "experimental" and gates it behind an env var — i.e. not a clear win even for them. **Verdict: not required for V1.** PyTorch + CUDA graphs reaches the same overhead floor; keep Applio's decoder design as the template if profiling later shows the decoder dominating.

## 8. TensorRT

- Feasibility: HuBERT exports to ONNX and builds in TRT (transformer plugins mature); NSF-HiFiGAN is harder — the sine source (cumsum, rand) is export-hostile (would follow Applio's split: excitation in Torch, conv stack in TRT), and the VITS flow/encoder has data-dependent ops.
- Engineering cost: per-GPU, per-shape engine builds (minutes at first run), engine cache management per voice model (RVC users swap `.pth` files constantly — every model needs its own decoder engine, as Applio's sha256-keyed ONNX cache illustrates), Windows DLL/version matrix (TRT × CUDA × ORT), FP16 accuracy validation per module.
- Expected gain over PyTorch-FP16-under-CUDA-graph: realistically 10–30% on the decoder convs — while the end-to-end block time is dominated by HuBERT + F0 + fixed overheads.
- **None of the four references uses TensorRT.** (w-okada historically experimented with ORT TensorRT EP; nothing in the current RVC provider code.)
- **Verdict: not worth it for V1.** Revisit for V2 only for the HuBERT encoder as a single static-shape engine, where shapes never change and the model never swaps.

## 9. GPU FAISS on Windows

- `faiss-gpu` has **no official Windows wheels or conda packages**; conda `pytorch` channel builds are Linux-only, and community Windows builds are fragile source compiles (see [faiss-wheels#54](https://github.com/kyamagu/faiss-wheels/issues/54), [faiss#821](https://github.com/facebookresearch/faiss/issues/821), [faiss#2273](https://github.com/facebookresearch/faiss/issues/2273), [install notes](https://github.com/xwild/faiss/blob/main/INSTALL.md), [overview](https://mljourney.com/how-to-install-faiss-for-gpu-on-windows/)). Vonovox — CUDA-maximalist, Windows-only — ships `faiss-cpu`.
- The correct alternative is already written: **Applio's `IndexWrapper`** — torch brute-force L2 (`addmm` + `topk`) over the reconstructed index tensor on GPU. RVC indexes are small (typically 10⁴–10⁵ × 768 vectors ≈ 30–300 MB FP32; FP16 storage halves it); exact search over the ~32–64 query rows per block is sub-millisecond and **graph-capturable**, which no FAISS call ever is. IVF is unnecessary at this scale; if an index is huge, cluster on CPU once and do a torch two-stage (centroid topk → cluster scan).
- Keep CPU FAISS only as the OOM fallback, exactly as `IndexWrapper.search` does.

## 10. Keeping audio on GPU end-to-end

| Stage | Where | Rationale / reference |
|---|---|---|
| Mic block ingest | CPU→GPU, once | `torch.as_tensor(..., device=...)` (Applio `core.py` L326–328). Optionally a pinned staging buffer + `non_blocking=True`; marginal at 48 kHz block sizes (tens of KB) |
| Input/output resampling | **GPU**, `torchaudio.transforms.Resample` precomputed and moved to device | Applio `core.py` L182–189; rvc-webui realtime_gui L665–723 (even graph-captures the resamplers). Never librosa per block |
| High-pass / noise gate | GPU (torch filtering / TorchGate) | Applio uses `noisereduce.torchgate` on device (L169–176) |
| F0, HuBERT, retrieval, synth | GPU, one capture domain | §2 |
| RMS envelope | GPU | `AudioProcessorTorch.change_rms` (utils/torch.py L97–151) vs rvc-webui's librosa round-trip |
| **SOLA + crossfade** | **GPU** — and avoid the `.item()` on the offset by `index_select` with a tensor offset (graph-capturable) | Applio `core.py::_phase_aligned_output` L478–501. CPU SOLA is *acceptable* (it's O(crossfade·search) on a few thousand samples) but then the inferred block must be downloaded before alignment — keeping it on GPU defers the single download to after crossfade |
| VAD / volume gate | CPU is fine — runs on the raw input before the GPU pipeline; one scalar `.item()` per block is the irreducible sync | Applio `core.py` L331–343 |
| Pedalboard-style FX | CPU unavoidable (C++ host library) — but then **exclude the block from graph capture**, as Applio does (pipeline.py L624–631) | Prefer torch-native FX on GPU where possible |
| Output to audio device | GPU→CPU, once, final `.cpu().numpy()` | The only justified download |

---

## 11. Explicit answers

**Can CUDA Graphs materially reduce realtime inference overhead?** Yes — this is the single highest-leverage optimization, with two independent proven implementations. One RVC block issues hundreds of small kernels (HuBERT ~100+ layers of ops, RMVPE CNN, synthesizer); at 5–20 µs Python+launch overhead each, eager dispatch costs multiple milliseconds per block — comparable to the GPU compute itself at small block sizes. Graph replay collapses that to one launch. Applio's comment is literal: "Graphs eliminate hundreds of host kernel submissions per block" (pipeline.py L620). Preconditions we must meet: fixed block size, preallocated I/O buffers, no `.item()`/CPU ops inside the captured region, warmup ×3 on a side stream, shape-keyed cache with eager fallback and a restart-on-capture-failure policy.

**Is ONNX actually faster for our workload?** Not materially, once PyTorch launch overhead is removed by graphs. w-okada's whole-model ONNX with host I/O per block is a portability feature, not a speed feature. Applio's decoder-only ONNX with IO binding + ORT CUDA graphs is well engineered but shipped as experimental/opt-in. Use ONNX in V1 only if we need DirectML portability; for NVIDIA-only V1, PyTorch eager + CUDA graphs is simpler and equal.

**Is TensorRT worth it for V1?** No. 10–30% on sub-dominant modules, against per-model engine builds, Windows packaging pain, and FP16 validation of precision-sensitive components. No reference implementation uses it. V2 candidate for the (static, never-swapped) HuBERT encoder only.

---

## 12. Prioritized optimization list

| # | Optimization | Tag | Evidence |
|---|---|---|---|
| 1 | Eliminate all mid-pipeline CPU round-trips: GPU F0 decode, GPU retrieval, GPU RMS, GPU resampling; one upload + one download per block | **PROVEN** | Applio `rvc/realtime/pipeline.py` + `utils/torch.py`; rvc-webui `realtime_gui.py` GPU resamplers |
| 2 | Persistent preallocated device buffers + circular writes; reused scalar tensors; no per-block allocation, no `empty_cache` | **PROVEN** | Applio `core.py::realloc`, `pipeline.py` L183–184; rvc-webui `rtrvc.py` `cache_pitch/cache_pitchf` |
| 3 | Torch GPU brute-force retrieval replacing CPU FAISS (keep CPU FAISS as OOM fallback) | **PROVEN** | Applio `IndexWrapper` (utils/torch.py L154–299) |
| 4 | Full-block CUDA graph capture (F0+HuBERT+retrieval+synth), 3-iter side-stream warmup, shape/flag-keyed cache, eager fallback, restart on capture failure | **PROVEN** | Applio `pipeline.py::voice_conversion` L616–700; rvc-webui `tools/cuda_graph.py` |
| 5 | Service-level warmup: ≥5 full silent blocks at production block size before going live (covers cuDNN, allocator, both graphs, resamplers, SOLA) | **PROVEN** | Applio `core.py::VoiceChanger.warmup` L684–710 |
| 6 | GPU SOLA with tensor-offset `index_select` (no `.item()`), captured as a second graph | **PROVEN** | Applio `core.py::_phase_aligned_output` / `_graph_phase_output` L478–535 |
| 7 | TF32 matmul/conv on Ampere+, set once before warmup/capture; `cudnn.benchmark=False` | **PROVEN** | Applio `precision.py`; `rvc/configs/config.py` L84; Vonovox `cuda_manager` strings (`allow_tf32`) |
| 8 | FP16 autocast on HuBERT only, features cast back to FP32 | **PROVEN** (opt-in) | Applio `pipeline.py` L188–197, 484–488; rvc-webui `is_half` with GPU blacklist (w-okada `DeviceManager.halfPrecisionAvailable`) |
| 9 | FP16 decoder (weights), NSF excitation source kept FP32 in Torch | **LIKELY** | Applio `decoder_fp16` + `onnx_decoder.py` FP16 default on CUDA; shipped experimental |
| 10 | Pinned-memory staging + `non_blocking=True` for the per-block input upload | **LIKELY** | rvc-webui `cuda_graph.py` uses non_blocking in replay; transfer sizes are small, gain is µs-level |
| 11 | High-priority CUDA stream for the conversion pipeline (contention with games/apps) | **EXPERIMENTAL** | Vonovox README names it as planned work; no reference ships it |
| 12 | ONNX Runtime CUDA decoder with IO binding + ORT cuda-graph replay | **EXPERIMENTAL** | Applio `onnx_decoder.py` (env-gated, "experimental") |
| 13 | torch.compile (HuBERT only, static shapes) | **EXPERIMENTAL** | No reference uses it; Windows/Triton + graph-break risk |
| 14 | TensorRT engine for HuBERT | **EXPERIMENTAL** | No reference uses TRT; defer to V2 |

### Precision decision table (V1)

| Module | Dtype | Notes |
|---|---|---|
| Global default | FP32 + TF32 (SM≥80) | Applio default; TF32 set once pre-capture |
| HuBERT | FP16 autocast | Cast output back to FP32 |
| RMVPE/FCPE | FP32 | Cent-level F0 accuracy; thresholds near FP16 granularity |
| Retrieval | FP32 (FP16 storage optional) | Clamp distances; compute in FP32 |
| Flow/encoder | FP32 | Small cost |
| NSF excitation (SineGen) | FP32 always | cumsum phase accumulation |
| HiFi-GAN conv stack | FP32 → measure FP16 | Behind a flag, A/B for artifacts |
| SOLA/resample/RMS | FP32 | Negligible cost |

### Inference-backend decision table

| Backend | V1 | Rationale |
|---|---|---|
| PyTorch eager + manual CUDA graphs | **Adopt** | Proven twice; lowest complexity for NVIDIA-only |
| torch.compile | Reject | Compile stalls, graph breaks, Windows Triton |
| ONNX Runtime CUDA (whole model) | Reject | Host I/O pattern loses the graph wins; portability-only feature |
| ONNX Runtime CUDA (decoder-only, IO binding) | Hold (V1.x flag) | Template exists in Applio; adopt only if decoder profiling demands it |
| TensorRT | Reject for V1 | Cost ≫ gain; V2 candidate for HuBERT |
| DirectML / DML EP | Out of scope | AMD support is a separate track (w-okada's approach) |

---

## Key takeaways for our implementation

1. **Architecture first, backend second.** The canonical pipeline loses more time to CPU↔GPU boundaries (FAISS round-trip, numpy F0, librosa RMS/resample, `.item()` syncs, `empty_cache`) than to slow kernels. Copy Applio's shape: one upload, one download, everything between is device-resident tensors in preallocated buffers.
2. **CUDA graphs are the proven overhead killer** — capture the whole block, warm up ×3 on a side stream, key the cache on shapes/pointers/precision flags, fall back to eager, and treat capture failure as a process-restart event. Capture SOLA separately with the offset kept on-device.
3. **Replace FAISS with torch `addmm`+`topk` on GPU** — it's faster, Windows-proof, and graph-capturable; keep CPU FAISS only as the OOM fallback.
4. **Precision policy: FP32 + TF32 base, FP16 only where proven** (HuBERT autocast); never FP16 the F0 decode or the NSF sine source. `cudnn.benchmark=False`; set all flags before warmup/capture and never toggle them during replay.
5. **Warm up like Applio**: 5 full silent blocks at the production block size, gating disabled, and report steady-state latency from the last runs.
6. **Say no to torch.compile, whole-model ONNX, and TensorRT for V1.** Keep Applio's decoder-only ONNX (IO binding + ORT cuda graph + split excitation) on file as the V1.x escape hatch if the decoder ever dominates the profile.
