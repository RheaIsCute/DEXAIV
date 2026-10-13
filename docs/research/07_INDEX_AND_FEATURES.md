# 07 — Content Features (HuBERT/ContentVec) and the FAISS Index

Scope: Part A — the content encoder (what it costs per block, whether it can be cached);
Part B — the `.index` retrieval stage (what it is, exact retrieval math, cost profile, strategy).
Source citations use `repos/` paths: `Applio`, `voice-changer` (w-okada), `rvc-webui`, `Vonovox` (closed-source `.pyd`; binary-string evidence tagged).

---

# Part A — Content features

## A.1 What the models are

| Embedder | Base | Dim used by RVC | Where |
|---|---|---|---|
| **ContentVec** (a.k.a. "hubert_base.pt" in RVC lore; HF `lengyue233/content-vec-best`) | HuBERT-Base (12-layer transformer, 768-d, conv feature extractor) retrained with speaker disentanglement ([arXiv:2204.09224](https://arxiv.org/abs/2204.09224)) | v1: layer 9 → `final_proj` → **256**; v2: layer 12 output → **768** | all four apps; Vonovox README: "Transformers implementation of ContentVec" |
| **SPIN / spin-v2** (Speaker-Invariant clustering, [arXiv:2305.11072](https://arxiv.org/abs/2305.11072)) | HuBERT-Base fine-tuned | 768 (v2-style) | Applio `rvc/lib/utils.py::load_embedding` lines 181-208: `contentvec, spin, spin-v2, chinese/japanese/korean-hubert-base, custom`; Vonovox GUI: "Select between contentvec or spin trained models" |
| HuBERT-Base variants (chinese/japanese/korean) | language-specific pretraining | 768 | Applio embedder list |

All are the **same architecture** (HuBERT-Base: 7-layer strided conv frontend ≈ 320× downsample, then 12 transformer layers, 768-d, ~94.7M params); the choice only changes weights, so every cost statement below applies to all of them. **The embedder must match the one the .pth was trained with** — ContentVec for almost all public models.

## A.2 The expected pipeline (canonical)

From `rvc-webui/infer/hubert.py::extract_hubert_features` (lines 68-99) — the clearest statement in any repo:

- Input: **16 kHz mono float**, shape `[1, n]` (padding mask all-False in realtime).
- v1: `final_proj(hidden_states[9])` → 256-d ("Transformers hidden_states[N] is numerically equivalent to the source checkpoint's output_layer=N"; RVC v1 = layer 9 + final_proj).
- v2: final (12th) encoder layer output → 768-d.
- Output frame rate: **50 Hz (20 ms)** — conv frontend stride 320. The pipeline then does `F.interpolate(scale_factor=2)` to reach the 100 Hz pitch grid, and duplicates the last frame first (`feats = torch.cat((feats, feats[:, -1:, :]), 1)`, Applio pipeline.py line 495 / rtrvc.py line ~248).

Applio realtime (`rvc/realtime/pipeline.py::_voice_conversion` lines 483-493) is equivalent: `self.hubert_model(feats)["last_hidden_state"]` (its `HubertModelWithFinalProj` checkpoint is 12-layer with the proj attached), then `final_proj` only for v1. w-okada routes through `Embedder.extractFeatures(feats, embOutputLayer, useFinalProj)` with `embOutputLayer` 9 or 12 (`voice-changer .../pipeline/Pipeline.py::extractFeatures`).

## A.3 Cost scaling with context — why w-okada's "extra" is expensive

Per block, every repo runs the embedder over the **entire conversion window** (block + crossfade + sola_search + extra), not just the new block:

- w-okada `Pipeline.exec`: `feats = audio_pad` (the whole padded window) → `extractFeatures` each chunk. The GUI's EXTRA setting (0.1–5 s) is prepended history; the embedder cost is therefore ~linear-plus in EXTRA even though only `out_size` worth of audio is emitted.
- Applio `_voice_conversion`: `feats = audio.view(1,-1)` where `audio` is the full `convert_buffer` (default extra = 0.5 s).
- rvc-webui `rtrvc.infer`: full `input_wav`.

Why super-linear-ish: the conv frontend is O(n) with a large constant (7 conv layers at waveform rate — at short windows it dominates), while the 12 transformer layers have **self-attention O(F²·d) in F = n/320 frames** plus O(F·d²) MLP/projection terms. At realtime window sizes (0.5–3 s → F = 25–150) the quadratic term is not yet dominant — the practical scaling is near-linear in window length until a few seconds, then bends upward. Either way: **2× extra ⇒ ≳2× embedder FLOPs for identical output audio**. The embedder is the single most expensive stage of the realtime pipeline (94M params vs ~36-40M for the F0 nets and the ~30M decoder which only renders `return_length`), so the "extra/lookahead" knob is primarily a HuBERT-cost knob. Vonovox's README says exactly this from the user side: "Lookahead buffer ... recommended 2.0 for best quality/latency ratio. The added latency of this setting is far less impactful than the block size" — impactful on GPU *occupancy*, not output delay.

Why extra context exists at all: HuBERT features of the first ~couple hundred ms of a window are boundary-degraded (conv padding + attention edge effects), and the decoder's `skip_head` mechanism (Applio `realloc`: `skip_head = extra_frame_16k // 160`; `net_g.infer(..., skip_head)`) discards the output of the context region — context frames stabilize the *kept* frames.

## A.4 Can features be cached / incrementally computed?

- **What the reference projects actually do: full recompute every block.** None of rvc-webui, Applio, or w-okada caches HuBERT states across blocks. w-okada returns `feats_buffer` from `Pipeline.exec` (line 257) but it is only used for the index "recover silent front" splice, not reused as encoder state.
- **KV-cache feasibility**: HuBERT's attention is **bidirectional (non-causal)** — a new frame changes the attention output of *all* previous frames in every layer, so a GPT-style KV cache is not valid; cached keys/values would be stale after layer 1 for any frame whose window membership changed. Options that do exist:
  - *Chunked/streaming attention with limited right context* (à la streaming conformers): requires retraining or accepting approximation error — EXPERIMENTAL, no RVC project does it.
  - *Conv-frontend caching*: the 7-layer conv stack is strictly causal-window-local (receptive field ≈ 400 samples, stride 320); its outputs for old samples are exact and could be cached, saving the waveform-rate convs (a real but minor win, ~10-20% of embedder cost). EXPERIMENTAL — not done anywhere in these repos.
  - *Sliding window + feature ring buffer* (mirror of the pitch cache): recompute the embedder only on a ~0.5 s tail and keep older 50 Hz feature frames in a ring. Approximation: boundary frames differ from full-window recompute. No reference project does this for features (they all do it for F0 only) — LIKELY workable with a small quality cost, but unproven; the proven lever is simply *keeping extra small* (Applio default 0.5 s vs w-okada's habit of 1-3 s).
- Practical v1 position: full recompute of a small window, like Applio.

## A.5 Precision / compilation compatibility (from source)

| Technique | Status in repos | Tag |
|---|---|---|
| FP16 weights end-to-end | rvc-webui offline (`is_half`, autocast in `Pipeline.extractFeatures` w-okada, with NaN→exception fallbacks `DeviceCannotSupportHalfPrecisionException`) | PROVEN offline, flaky on some devices |
| FP16 **autocast around HuBERT only**, FP32 elsewhere | Applio `pipeline.py` lines 186-197 + 483-487: `APPLIO_EXPERIMENTAL_HUBERT_FP16`, NVIDIA-CUDA-only, "preserve FP32 retrieval/synthesis features while autocasting the large HuBERT network" | PROVEN-experimental; the right shape for V1 (halves the dominant cost without touching retrieval/decoder numerics) |
| TF32 matmul | Applio `realtime/precision.py::enable_experimental_tf32` — opt-in env flag, SM≥8.0, set **before warmup/capture**, never toggled around graph replay | PROVEN-experimental |
| **CUDA Graphs** | rvc-webui `infer/hubert.py::run_cuda_graph` wraps HuBERT itself; Applio captures the *entire* `_voice_conversion` (F0+HuBERT+retrieval+synth) as one graph (`pipeline.py::voice_conversion`, cache keyed on data_ptrs/ids/backend flags, 3 warmup iters, restart-on-capture-failure semantics via `CudaGraphCaptureError`), plus a second graph for SOLA/phase-vocoder output (`core.py::_graph_phase_output`) | PROVEN — the flagship optimization; eliminates "hundreds of host kernel submissions per block" |
| ONNX | w-okada ships ONNX embedder/inferencer paths; Applio has an experimental ONNX **decoder** only (`realtime/onnx_decoder.py`, env-gated, checkpoint-hash-verified, composable with CUDA graphs via `infer_latent` + `run_decoder` split) | PROVEN for decoder; HuBERT-ONNX PROVEN in w-okada |
| TensorRT / torch.compile | not present in any of the four repos' realtime paths | EXPERIMENTAL for us |

---

# Part B — The FAISS index

## B.1 What the `.index` contains and how it's built

`rvc-webui/train/train_index.py` (canonical):

- Source data: per-utterance HuBERT features from training (`3_feature256` v1 / `3_feature768` v2) — the file is **a FAISS `IVF{nlist},Flat` index over every (optionally k-means-reduced) training feature frame**, L2 metric.
- If the dataset exceeds 200k frames it is compressed to 10k centroids with `MiniBatchKMeans(n_clusters=10000, batch_size=256*n_cpu)` (lines ~180-183).
- `n_ivf = max(1, min(int(16 * sqrt(N)), N // 39))` (line 195) — the famous nlist heuristic; `index_factory(dim, "IVF%s,Flat")`, `nprobe = 1` at build (name encodes it), `index.train(big_npy)` then `add` in 8192-row chunks (line 226). Two artifacts: `trained_*.index` (trained, empty — **invalid for inference**; rtrvc prints "必须使用added_xxxx.index" if vectors are missing, rtrvc.py line ~267) and `added_*.index` (with vectors). Applio auto-rewrites `trained→added` in the path (`create_pipeline`, pipeline.py line 750).

Because `IVF,Flat` stores raw vectors, `index.reconstruct_n(0, ntotal)` recovers the full feature matrix — every app does this at load: w-okada `Pipeline.__init__` line 58, rvc-webui `modules.py` (`reconstruct_n` at line 280), Applio `IndexWrapper.read_index`.

## B.2 Retrieval math at inference (exact code)

Canonical (`rvc-webui/infer/vc/pipeline.py::vc`, lines 186-195; byte-identical logic in `rtrvc.py::infer` and Applio `_retrieve_speaker_embeddings`):

```python
score, ix = index.search(npy, k=8)          # squared-L2 distances to 8 NN per frame
weight = np.square(1 / score)               # w_j = 1/d_j^2  (d is ALREADY squared L2 → w = d^-2)
weight /= weight.sum(axis=1, keepdims=True)
npy = np.sum(big_npy[ix] * weight[..., None], axis=1)   # weighted blend of neighbors
feats = npy * index_rate + (1 - index_rate) * feats     # linear blend with live features
```

- Done at the 50 Hz feature rate, **before** the 2× interpolation, and (realtime) only on frames past the context head: Applio `skip_offset = skip_head // 2` (`_retrieve_speaker_embeddings` line 714), rtrvc `feats[0][skip_head // 2:]` — context frames whose output is discarded are not searched. w-okada instead offsets by `silence_front` (`npyOffset = floor(silence_front*16000)//360`, Pipeline.py line 191 — note the odd `//360`) and even uses **k=1** by default (hardcoded `k = 1`, line 198, with the k=8 branch dead) then splices previous-block features back over the skipped head (line 209).
- The `protect` parameter interacts with retrieval: for frames with pitch (voiced), blended features are kept; `protect<0.5` restores a fraction of the pre-retrieval features on unvoiced/low-pitch frames to protect consonants (w-okada Pipeline.py lines 232-238; Applio lines 526-533).

## B.3 Why it helps some models and barely others

The blend drags live HuBERT frames toward the **convex hull of the training set's frames**. Effects:

- Small / noisy / accented datasets: retrieval suppresses out-of-distribution content (accent, background bleed, mic character — "timbre leakage" through content features), audibly stabilizing timbre. This is its raison d'être.
- Large clean datasets: the generator's own encoder already saw this distribution; retrieved ≈ live features, so `index_rate > 0` mostly adds compute + slight consonant smearing (hence `protect`). Benefit ~nil.
- Cross-domain input (your mic vs training data) is the other win case: retrieval acts as a learned denoiser of content features.
- Failure mode: k=8 averaging low-pass-filters fast phonetic transitions; at high `index_rate` speech gets "mushy". w-okada's k=1 choice trades that for quantization-style artifacts.

## B.4 Cost profile

- **CPU FAISS IVF search**: per block, N_frames(≈ window/20 ms, e.g. 25-150) queries of dim 256/768 against nlist·nprobe cells. Cheap on average (sub-ms to a few ms) but: (a) it forces a **GPU→CPU copy of feats and a CPU→GPU copy back** (`feats[0].cpu().numpy()` … `torch.from_numpy(npy).to(device)` — rvc-webui vc lines 186-195, rtrvc lines 252-266, w-okada lines 189-210), which stalls the stream and **breaks CUDA-graph capture**; (b) OpenMP thread spin-up / cache misses cause tail-latency spikes on loaded systems. In a 10-20 ms budget, millisecond-scale CPU spikes are audible dropouts.
- **Memory**: `reconstruct_n` materializes the full float32 matrix: N × 768 × 4 B — e.g. 1M frames ≈ 3 GB RAM (10k-kmeans-reduced: 30 MB). Applio additionally copies it to VRAM as `big_tensor` + precomputed `b_norms` (`IndexWrapper.read_index_tensor`), then frees the CPU copy.
- **faiss-gpu**: not used by any repo. Packaging on Windows is the blocker — `faiss-gpu` has no official Windows wheels (conda-only, CUDA-version-pinned); every Windows app ships `faiss-cpu`. Vonovox binary strings contain `index_cpu_to_gpus_list` (the faiss GPU-cloner API) — LIKELY it *attempts* GPU FAISS when available, but also contains "Failed to load index:" fallback and per-stage timing `"ms | Index: "`, i.e. it instruments index cost per block (closed source; EXPERIMENTAL evidence). Vonovox's public README conspicuously never mentions the index as a feature — the product de-emphasizes it.
- **Applio's solution — skip FAISS at query time entirely** (`rvc/realtime/utils/torch.py::IndexWrapper.search`): load the index once, reconstruct vectors to a GPU tensor, and do **brute-force squared-L2 via one `torch.addmm` (‖b‖² − 2a·bᵀ + ‖a‖²) + `topk(k=8)`** on-device. No CPU round-trip, graph-capturable, exact (better recall than IVF nprobe=12). Falls back to CPU FAISS on OOM (`faiss_cpu = True` path, lines 282-299). Cost: O(N_frames × N_index × d) GEMM — for a 10k-kmeans index this is trivial (~25×10k×768 ≈ 0.2 GFLOP); for a raw 1M-frame index it's a ~60 GFLOP GEMM per block plus 3 GB VRAM — why Applio lazily loads/unloads with `prepare_index(index_rate)` (pipeline.py lines 602-614) and drops the captured graph when retrieval toggles.

## B.5 Batching / caching options

- Queries are already naturally batched (all frames of the block in one `search`/GEMM).
- Caching retrieval results across blocks is pointless for new frames and unnecessary for old ones (Applio/rtrvc only search frames past `skip_head`). w-okada's `feature` splice (Pipeline.py line 209) is a crude cross-block feature cache used to re-fill the skipped silent head.
- PROVEN wins to adopt: search only non-context frames; keep everything on-GPU (Applio); lazy-load/unload on `index_rate` transitions; precompute `b_norms`.
- LIKELY win: offline-shrink any index > ~100-200k vectors with k-means (train_index already does ≥200k→10k) so GPU brute force is always cheap; or sub-sample `big_npy` at load.

## B.6 Index strategy decision table

| Strategy | Latency | Quality | VRAM/RAM | Complexity | Verdict for V1 |
|---|---|---|---|---|---|
| No index (`index_rate=0`) | best; enables single CUDA graph | fine for well-trained single-voice model | none | none | **Default** |
| GPU torch brute-force (Applio `IndexWrapper`) | +0.1–1 ms, graph-safe | exact k-NN | full matrix in VRAM (cap size!) | low (≈100 LoC) | **Ship as the only "on" mode** |
| CPU FAISS IVF (canonical) | ms-scale + transfer stalls + spikes; breaks graphs | approx (nprobe) | matrix in RAM | low | Fallback on VRAM OOM only (Applio already does) |
| faiss-gpu | fast | approx | VRAM + painful Windows packaging (no pip wheels) | high | Reject |
| k=1 lookup (w-okada) | marginally cheaper | rougher | same | low | Reject (k=8 weighted is canonical) |

**Auto-benchmark index on/off per model?** Yes — cheap and high-value. At model load: run the warmup block set (Applio `VoiceChanger.warmup` pattern: 5 silent blocks, take `max(timings[-3:])`) twice, `index_rate=0` and `=0.75`, report both block times and the VRAM delta, and auto-disable (with a UI note) if the indexed path exceeds the block budget or VRAM headroom. Since our product serves **one user / one model**, also offer a one-time offline A/B render of a fixed test utterance so the user decides by ear whether the index earns its cost; persist the choice per model. (Quality itself can't be auto-scored reliably — benchmark cost, let the ear judge benefit.)

---

## Key takeaways for our implementation

1. **Embedder = ContentVec first**, loaded HF-Transformers style with `final_proj` attached; support v2 (layer-12/768) as primary since modern models are v2; v1 = `final_proj(hidden_states[9])`. Expose SPIN/spin-v2 only as alternates for models trained on them (Applio's `load_embedding` list is the compatibility map).
2. **The embedder runs on the full conversion window every block — so the window, not the block, sets GPU cost.** Default extra context ≈ 0.5 s (Applio) rather than 2 s+ (w-okada habit); make it a quality slider and show measured per-block ms next to it. Do not promise feature caching: HuBERT is bidirectional, KV-caching is invalid (A.4); conv-frontend caching and tail-recompute feature rings are EXPERIMENTAL backlog items.
3. **Adopt Applio's precision ladder**: FP32 baseline → TF32 opt-in (set once, pre-capture) → FP16 autocast **around HuBERT only** (biggest single win, PROVEN-experimental) → whole-pipeline CUDA graph with restartable-process semantics and pointer-keyed graph cache; ONNX/TensorRT decoder as a later split (Applio `onnx_decoder.py` shows the `infer_latent`/`run_decoder` seam).
4. **Retrieval: default off; when on, GPU brute-force only** (addmm + topk, `w = 1/d²` k=8 blend at `skip_head//2` offset, exact canonical math from `rvc-webui/infer/vc/pipeline.py::vc`), with lazy load/unload tied to `index_rate` and CPU-FAISS only as OOM fallback. Never put faiss-gpu in the Windows dependency tree.
5. **Cap index size at load** (k-means or subsample > ~200k vectors) so brute-force cost and VRAM stay bounded, and **auto-benchmark on/off per model at load time**, persisting the user's choice — one user, one model makes this a one-time cost.
6. Respect the coupling rules Applio encodes: retrieval tensors are captured inside CUDA graphs, so toggling the index must invalidate the graph (`prepare_index` clears `_graph_cache`); `protect` needs a pre-retrieval copy of feats only when `protect < 0.5` and the model uses F0.
