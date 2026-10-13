# 09 — RVC Model Formats, Metadata Detection, and Model Folder Design

**Scope.** On-disk formats consumed by a realtime RVC engine: `.pth` synthesizer checkpoints, `.index` FAISS retrieval files, ONNX exports, embedder checkpoints — and an auto-detection algorithm for drag-and-drop import. Evidence: rvc-webui `infer/rtrvc.py` / `infer/vc/modules.py` / `infer/module/models.py`, Applio `rvc/realtime/pipeline.py` + `rvc/train/process/model_information.py`, w-okada `server/voice_changer/RVC/RVCModelSlotGenerator.py` + `onnxExporter/`.

---

## 1. The `.pth` inference checkpoint

An RVC *inference* checkpoint (as produced by "extract small model" at train end) is a plain pickled dict — **not** a training checkpoint (those have `model/optimizer/learning_rate/iteration`). Top-level keys, as consumed by every loader:

| Key | Type | Meaning | Consumers |
|---|---|---|---|
| `weight` | `dict[str, Tensor]` | synthesizer `state_dict` (half-precision floats in official releases) | `net_g.load_state_dict(cpt["weight"], strict=False)` everywhere |
| `config` | `list` (18 items for official models) | constructor args of `SynthesizerTrnMs*` (see §2) | `SynthesizerTrnMs768NSFsid(*cpt["config"])` ([infer/rtrvc.py#L34-L47]) |
| `f0` | `int` 0/1 | pitch-guided model? selects `*NSFsid` vs `*NSFsid_nono` | `cpt.get("f0", 1)` (default 1) |
| `version` | `"v1"` / `"v2"` | embedder dim/tap (see §3) | `cpt.get("version", "v1")` |
| `sr` | int or str (`"40k"` in some old dumps) | redundant with `config[-1]` | Applio `model_information`; loaders trust `config[-1]` |
| `info` | str | free text, typically `"{epoch}epoch"` or training notes | display only |
| `vocoder` | str, optional | Applio-fork extension: `"HiFi-GAN"` (default) / `"MRF HiFi-GAN"` / `"RefineGAN"` | Applio `setup_network`: `self.vocoder = self.cpt.get("vocoder", "HiFi-GAN")` ([rvc/realtime/pipeline.py#L92]) |
| `embedder_name`, `embedder_output_layer` | str/int, optional | non-official (ddPn08/rvc-webui-fork, VoRAS) models that trained on other embedders | w-okada `_setInfoByPytorch` DDPN branch ([RVCModelSlotGenerator.py#L87-L108]) |
| `speaker_info` | dict, optional | multi-speaker id→name map | w-okada (same file, L110-L112) |
| Applio extras: `epoch`, `step`, `author`, `model_hash`, `creation_date`, … | optional | provenance shown by `rvc/train/process/model_information.py` (loads with `weights_only=True`, tolerant-unpickler fallback for legacy community files) | display |

**Loading invariants (all three loaders do these):**
1. `cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]` — the stored spk count is unreliable; patch n_spk from the actual embedding table ([rtrvc.py#L35]; Applio pipeline L87; `infer/vc/modules.py#L176]).
2. `tgt_sr = cpt["config"][-1]`.
3. `del net_g.enc_q` (training-only posterior encoder) then `load_state_dict(..., strict=False)`; `remove_weight_norm()` / `strip_parametrizations()` before realtime use.
4. Prefer `torch.load(..., map_location="cpu", weights_only=True)` (Applio) — pickles from the internet are code-execution vectors; Applio's `_TolerantUnpickler` fallback exists only for metadata display, not for loading weights. **Adopt weights_only=True.**

---

## 2. The `config` tuple (18 fields, official v1/v2)

From the `SynthesizerTrnMs256NSFsid.__init__` signature ([infer/module/models.py#L681-L703], identical ordering in Applio `Synthesizer`):

| # | Name | Typical 40 k v2 value | Meaning |
|---|---|---|---|
| 0 | `spec_channels` | 1025 | filter_length/2+1 (training spectrogram; unused at inference) |
| 1 | `segment_size` | 32 | training segment (unused at inference) |
| 2 | `inter_channels` | 192 | latent z channels |
| 3 | `hidden_channels` | 192 | encoder width |
| 4 | `filter_channels` | 768 | FFN width in attention blocks |
| 5 | `n_heads` | 2 | attention heads |
| 6 | `n_layers` | 6 | encoder layers |
| 7 | `kernel_size` | 3 | FFN conv kernel |
| 8 | `p_dropout` | 0 | dropout (0 at inference) |
| 9 | `resblock` | "1" | HiFiGAN resblock type |
| 10 | `resblock_kernel_sizes` | [3,7,11] | |
| 11 | `resblock_dilation_sizes` | [[1,3,5]]×3 | |
| 12 | `upsample_rates` | 40 k: [10,10,2,2]; 48 k v2: [12,10,2,2]; 32 k v2: [10,8,2,2] | product = tgt_sr/100 = NSF `upp` |
| 13 | `upsample_initial_channel` | 512 | decoder entry width |
| 14 | `upsample_kernel_sizes` | 40 k: [16,16,4,4] | |
| 15 | `spk_embed_dim` | n_spk (patched from emb_g) | **config[-3]** |
| 16 | `gin_channels` | 256 | speaker-conditioning channels, **config[-2]** |
| 17 | `sr` | 32000/40000/48000 | **config[-1]**, target sample rate |

v1 vs v2 share this tuple; the 256/768 difference is **not** in `config` — it's the TextEncoder input width chosen by the class (`SynthesizerTrnMs768NSFsid` just rebuilds `enc_p` with 768, models.py L725-L735). That's why `version` must be read separately (or sniffed from weights, §5). Non-official forks (ddPn08) used longer configs where `config[17]` is `emb_channels` — w-okada detects that via `len(cpt["config"]) != 18` ([RVCModelSlotGenerator.py#L68-L96]).

### v1 vs v2 summary

| | v1 | v2 |
|---|---|---|
| embedding dim | 256 (`SynthesizerTrnMs256*`) | 768 (`SynthesizerTrnMs768*`) |
| embedder tap | HuBERT layer 9 → `final_proj` (768→256) | HuBERT layer 12 (last_hidden_state) |
| detector values (w-okada) | `embChannels=256, embOutputLayer=9, useFinalProj=True` | `768, 12, False` |
| index dim | 256 | 768 |
| typical SRs | 40 k (also 32/48 k) | 32/40/48 k |

---

## 3. `.index` files (FAISS)

- Built at train time from all training-set HuBERT features (`big_npy`): FAISS **IVF** (`IndexIVFFlat`, `n_ivf ≈ big_npy.shape[0]//39`, trained, then vectors added) → saved as `added_IVF{n}_Flat_nprobe_{p}_{name}_{version}.index`. A `trained_...index` (trained but with no vectors added) also exists and is **unusable** — rtrvc checks `(ix >= 0).all()` and warns "必须使用added_xxxx.index" ([infer/rtrvc.py#L253-L270]).
- Dim must equal embedding dim (256 v1 / 768 v2). Applio asserts compatibility at search time and drops the index on mismatch ("The index file structure is incompatible with the model", [rvc/realtime/pipeline.py#L506-L508]).
- At load, every project re-materializes the raw vectors: `big_npy = index.reconstruct_n(0, index.ntotal)` (rtrvc L86; w-okada Pipeline L58; Applio moves them to a GPU tensor via `IndexWrapper.read_index_tensor()`); search is k=8 + inverse-square-distance weighting (w-okada: k=1 fast path).
- Memory note: big_npy is `ntotal × dim × 4` bytes (a 2-hour-dataset v2 index ≈ hundreds of MB) — load lazily, only when `index_rate > 0` (Applio `prepare_index`).
- Filename is the only "metadata": parse `IVF(\d+)` and trailing `_v1/_v2` when present; otherwise probe `faiss.read_index(path).d` for the dim.

---

## 4. ONNX exports

- **w-okada full-model export** (`server/voice_changer/RVC/onnxExporter/export2onnx.py` + `SynthesizerTrnMs256/768NSFsid_ONNX.py`): whole synthesizer with inputs `feats[1,T,C] float32/16`, `p_len int64`, `pitch int64`, `pitchf float32`, `sid int64` (+ explicit `rnd` noise input in the wrappers so the latent is a feed), output `audio`. Crucially it writes a JSON blob into ONNX `custom_metadata_map["metadata"]`: `{modelType, samplingRate, f0, embChannels, embOutputLayer, useFinalProj, embedder, version}` — read back by `RVCModelSlotGenerator._setInfoByONNX` ([RVCModelSlotGenerator.py#L120-L182]); files without it are declared deprecated. fp16 detected from the first input tensor type (`OnnxRVCInferencer.loadModel`).
- **Applio decoder-only export** (`rvc/realtime/onnx_decoder.py`): exports just the HiFi-GAN NSF decoder graph (`DecoderGraph.forward(latent, source, speaker)`), cache keyed by checkpoint SHA-256 + `EXPORT_VERSION`; encoder/flow/SineGen stay in torch. **EXPERIMENTAL** but the right granularity if we ever do ONNX/TensorRT.
- HuBERT/ContentVec also exists as ONNX (w-okada `OnnxContentvec`), and RMVPE/CREPE have ONNX pitch extractors (`RMVPEOnnxPitchExtractor`, `onnxcrepe/`) — useful for a DirectML/AMD fallback story, irrelevant for CUDA-first.
- Recommendation: treat ONNX as optional import format; if we ever export, **copy w-okada's metadata-in-ONNX convention verbatim** for ecosystem compatibility.

---

## 5. Automatic metadata detection (algorithm for drag-and-drop)

How the repos do it: w-okada decodes everything from the checkpoint (§1/§2); Applio does the same inline in `setup_network`; neither needs the user to say v1/v2/sr/f0. Our version, hardened with weight-sniffing so it also survives checkpoints with missing `version`/`f0` keys:

```text
import_model(path):
  if path.endswith(".onnx"):
      meta = onnx custom_metadata_map["metadata"]      # w-okada convention
      if missing -> reject or mark legacy
      return {format:"onnx", **meta}

  cpt = torch.load(path, map_location="cpu", weights_only=True)   # never full pickle
  assert isinstance(cpt, dict) and "weight" in cpt and "config" in cpt
  w = cpt["weight"]

  # 1. sample rate
  sr = cpt["config"][-1]                      # int; map "32k/40k/48k" strings if str
  # cross-check: prod(config[12]) * 100 == sr  (upsample rates)

  # 2. f0 flag
  f0 = cpt.get("f0", None)
  if f0 is None:                              # sniff: NSF source module only in f0 models
      f0 = 1 if any(k.startswith("dec.m_source") or "cond" in k and "dec.noise_convs" in w for k in w) else 0
      # concretely: "dec.m_source.l_linear.weight" in w  -> f0 model

  # 3. version / embedding dim
  version = cpt.get("version", None)
  emb_dim = w["enc_p.emb_phone.weight"].shape[1]      # 256 -> v1, 768 -> v2  (authoritative)
  if version is None: version = "v2" if emb_dim == 768 else "v1"
  elif (version=="v2") != (emb_dim==768): trust emb_dim, warn

  # 4. speakers
  n_spk = w["emb_g.weight"].shape[0]; cpt["config"][-3] = n_spk

  # 5. embedder
  embedder = cpt.get("embedder_name", "contentvec")   # strip trailing "768" (w-okada)
  out_layer = cpt.get("embedder_output_layer", 9 if emb_dim==256 else 12)
  use_final_proj = (emb_dim == 256)

  # 6. vocoder (Applio forks)
  vocoder = cpt.get("vocoder", "HiFi-GAN")
  # sniff fallback: "dec.ups.0.weight" => HiFiGAN family; RefineGAN keys differ

  # 7. pair an index
  index = search same dir for *.index preferring "added_*"; verify faiss dim == emb_dim

  return {format:"pth", sr, f0, version, emb_dim, n_spk, embedder, out_layer,
          use_final_proj, vocoder, index_path, info: cpt.get("info",""),
          sha256: file hash}
```

Weight-key sniff references: `enc_p.emb_phone` input dim is the 256/768 discriminator (constructor evidence models.py L726-L735); `emb_g.weight` rows = n_spk (used by all loaders); `dec.m_source.*`/`dec.noise_convs.*` exist only in `GeneratorNSF` (models.py L403-L490). **PROVEN** signals.

Validation errors worth surfacing at import: training checkpoint (has `model`/`optimizer` keys) → tell user to extract; `trained_*.index` → ask for `added_*`; index dim ≠ model dim; config length ≠ 18 → mark "fork model", use w-okada's DDPN branch semantics.

---

## 6. Recommended `models/` layout

```
models/
  <ModelName>/
    model.pth              # renamed on import (original name kept in config.json)
    model.index            # optional, verified added_* index
    config.json            # detected + user metadata (below)
    icon.png               # optional
  _embedders/
    contentvec/{config.json, pytorch_model.bin}   # HF-format, like Applio rvc/models/embedders/
    spin/ ...
  _predictors/
    rmvpe.pt  fcpe.pt  swift_f0.onnx
```

Per-model `config.json` (superset of w-okada's `RVCModelSlot` + Applio kwargs):

```json
{
  "name": "MyVoiceF",
  "source_file": "MyVoiceF_e300_s42000.pth",
  "sha256": "…",
  "version": "v2", "sr": 48000, "f0": true, "n_spk": 1, "sid": 0,
  "embedder": "contentvec", "emb_dim": 768, "emb_output_layer": 12, "use_final_proj": false,
  "vocoder": "HiFi-GAN",
  "index": {"file": "model.index", "dim": 768, "ntotal": 123456},
  "defaults": {"f0_up_key": 12, "index_rate": 0.3, "protect": 0.4,
               "rms_mix_rate": 0.0, "formant": 0.1, "f0_method": "rmvpe"},
  "imported_at": "2026-10-13T…", "notes": ""
}
```

Rationale: w-okada proved per-slot persisted defaults (`defaultTune/defaultIndexRatio/defaultProtect` in `RVCModelSlot`, applied in `RVCr2.initialize`) are what makes model switching painless; Applio proved hashing the checkpoint (for ONNX cache + tamper check) is cheap and useful. Keep embedders/predictors **shared**, not per-model — they're model-independent downloads (Applio's `load_embedding` registry pattern, [rvc/lib/utils.py#L181-L227]).

---

## 7. Key takeaways for our implementation

1. Load with `weights_only=True`, patch `config[-3]` from `emb_g.weight`, read `tgt_sr = config[-1]`, drop `enc_q`, strip weight norm — the four-line invariant shared by all three loaders.
2. Detect `version` from `enc_p.emb_phone.weight.shape[1]`, `f0` from `dec.m_source` keys, not just from dict keys — community checkpoints lie or omit.
3. Support v2/40k+48k first (our M→F single-user case); v1 support is nearly free (final_proj + 256-d index) and worth keeping for the model ecosystem.
4. Index: accept only `added_*`-style usable indexes, verify `index.d == emb_dim`, load lazily to GPU tensor (Applio `IndexWrapper`), remember `reconstruct_n` RAM cost.
5. Persist per-model defaults (pitch +12 for M→F, index_rate, protect, formant) in `config.json`; one-click model switch should restore them (w-okada slot pattern).
6. If exporting ONNX, embed w-okada's metadata JSON for interop; if optimizing, follow Applio's decoder-only split rather than whole-graph export.
7. Checkpoint pickles are untrusted input — never `weights_only=False` on user drops; Applio's tolerant unpickler is for *metadata display only* and still riskier than we need.
