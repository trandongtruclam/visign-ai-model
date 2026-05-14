Searched — and there's actually a strong option for you. First, a confirmation worth flagging:

## You're already maxing out QIPEDC

Your `videos_crawler.ipynb` crawls `https://qipedc.moet.gov.vn/`, and your label IDs (`D0529`, `W00011`, `W00145B/N/T`) are QIPEDC's dictionary IDs with the Bắc/Trung/Nam regional suffixes. **That source is exhausted for "more takes per word"** — QIPEDC publishes 1–3 regional variants per gloss and that's it. Crawling more glosses would give you more *classes*, not more *takes per class*. So I needed to find a different source.

## The one that actually fixes your problem

### Multi-VSL (WACV 2025) — Hanoi University of Science and Technology
- **1,000 glosses, 30 signers, 3 camera angles, ~28 instances per gloss** (avg).
- Glosses are sampled from the same *Vietnamese Sign Language Dictionary* QIPEDC uses, so label overlap with your 274 classes should be **high** (likely 100–250 classes overlapping; needs verification).
- Train/val/test is split **by signer**, not by clip — exactly the kind of split your evaluation needs.
- Hosted on Google Drive, BibTeX from the WACV 2025 paper.
- Repo: [github.com/Etdihatthoc/Multi-VSL_WACV_2025](https://github.com/Etdihatthoc/Multi-VSL_WACV_2025) (the README is empty; you'll need to open the issues/Drive folder for download links — or email the authors at HUST: `lenp@soict.hust.edu.vn`).
- **Caveat:** raw RGB videos, not keypoints. You'd need to run your existing `keypoints_extractor.py` over them. Estimated ~84,000 videos × 1–3 s of MediaPipe each = ~24–72 GPU-hours, or split across multiple Colab sessions. If you only run it for the ~250 glosses that overlap with your labels (≈ 250 × 30 signers × 3 views = 22,500 videos), that's much more tractable (~6–18 hours).
- Why this is the right choice: 28 instances/gloss × 3 views means you can put 6–8 signers/views per class into val/test and have **6–8 samples per class** instead of 1. Your val_f1 stops being a coin flip.

### Two also-rans (lower impact but easier)

- **VOYA_VSL** on Hugging Face — 161 classes, MediaPipe keypoints already extracted, 56 GB. [huggingface.co/datasets/Kateht/VOYA_VSL](https://huggingface.co/datasets/Kateht/VOYA_VSL). Different feature shape `(60, 1605)` vs your `(150, 628)` — you'd need to re-derive features from raw keypoints, but at least no MediaPipe runtime cost. Class overlap with your 274 is unknown without inspecting `labels.json`.
- **KVSL** on GitHub — same source you're using (QIPEDC) with their own keypoint extraction. Won't give you more clips per word, but their preprocessing differences might be useful as a reference. [github.com/LeNguyenAnhKhoa/KVSL](https://github.com/LeNguyenAnhKhoa/KVSL).

### Worth knowing but not directly useful

- **VSL Object Detection on Roboflow** — 1,300 images of hand-shapes. For image classification, not video. Not relevant to your sequence model.
- **QIPEDC additional glosses** — 4,000 total exist; you have 274. Could expand vocabulary, doesn't fix your per-class count problem.

## Realistic plan (cheapest path to a trustworthy val set)

1. **Email Multi-VSL authors** (the email above) asking for the dataset access link or a class list. Mention you only need the subset of glosses overlapping with your 274 labels — this often unlocks a smaller, faster download. Takes maybe 10 minutes of your time and reply usually comes within a few days.
2. **While waiting**, download VOYA_VSL's `labels.json` (a few KB) and check overlap with your `splits.json` keys:

```python
import json, requests
voya = requests.get("https://huggingface.co/datasets/Kateht/VOYA_VSL/resolve/main/labels.json").json()
mine = set(k.split("/")[0] for k in json.load(open("splits.json"))["sources"])  # your 274 class names
voya_names = set(voya.values())
overlap = mine & voya_names
print(f"VOYA overlap: {len(overlap)} / {len(mine)} of your classes")
print(sorted(overlap)[:20])
```

If VOYA overlap ≥ 100 classes, it's worth pulling for those classes only. If overlap is tiny (< 30), skip VOYA and just wait for Multi-VSL.

3. **Once you have Multi-VSL videos**, run `keypoints_extractor.py` over them, dedupe filenames to avoid clashing with QIPEDC IDs, append to your `dataset/keypoints/<label>/` directories under new source IDs (e.g. `MVSL_signer07_front`), and re-run `split_sources.py`. The new `splits.json` will allocate multiple sources per class into val/test naturally.

## What I would not bother doing

- **Recording videos yourself.** You explicitly said no time, and even a couple of extra takes wouldn't fix the structural problem (you'd still be the single signer for those extra clips, defeating the point of cross-signer generalisation).
- **Synthetic augmentation alone.** You're already at 50× per source — the issue isn't quantity, it's that augmenting one signer doesn't create the variety that 30 different signers do.

## Honest expectation-setting

With Multi-VSL added, your val/test sets go from 17/18 samples to plausibly 1,000+ samples each. That alone — without changing your model — will probably show your real Top‑1 sits somewhere between 20 % and 50 % (the 11 % vs 44 % swing collapses into one stable number). At that point you can actually *measure* whether less augmentation, more regularisation, or a different model helps.

Send me the VOYA overlap output once you run it and we can decide whether VOYA is worth pulling while Multi-VSL replies, or whether to just wait.

Below is the full `MODEL_AUDIT.md` for the `ai-model/` VSL pipeline. You can copy it directly into a file.

---

# Vietnamese Sign Language (VSL) Model — Evaluation & Optimization Audit

**Scope:** `ai-model/` (BiLSTM + Attention pooling over MediaPipe Holistic keypoints)
**Checkpoint reviewed:** `artifacts/lstm_150.pt` (epoch 9, 3.12 M params, 37.5 MB)
**Reported metrics in checkpoint:** `val_acc = 0.9949`, `val_f1 = 0.9937`, `train_acc = 0.9674`

---

## Executive Summary

- **The 99.5 % validation accuracy is not real.** Train/val are split *after* augmenting one source clip into ~50 near-duplicates, so the same physical video leaks into both sides. Real signer-independent accuracy is almost certainly < 30 %.
- **256 / 274 classes have exactly ONE source video.** 17 classes have 3 videos (same word recorded by 3 signers — file suffixes `…B/N/T`). 1 class has 2. There is effectively no inter-signer diversity.
- **Augmentation pipeline silently corrupts the hand-presence mask** (`augment.py` adds Gaussian noise to all-zero "missing hands", so `hand_present_mask` in `preprocess_pipeline.py` always returns 1 for augmented frames — masking in attention pooling is a no-op).
- **Training-vs-inference distribution mismatch:** training extractor uses MediaPipe `model_complexity=1`, no smoothing, 1920×1080. Inference (`app.py`) uses `model_complexity=0`, `smooth_landmarks=True`, 640×360.
- **No real test set, no signer-independent split, no per-class report, no confusion matrix.** Training stopped at epoch 9/60 because the leaked `val_loss` collapsed.

---

## PHASE 1 — Codebase Analysis

### 1.1 Project layout (relevant files)

```
ai-model/
├── app.py                            # FastAPI inference server
├── artifacts/lstm_150.pt             # 37.5 MB checkpoint, 9 epochs trained
├── data/cleaned_data.csv             # 285 rows, 10 topics (Vimeo URLs)
├── dataset/text/label.csv            # 309 rows, 274 unique labels
├── label_mapping.json                # 274 classes, label→idx
├── requirements.txt
├── src/
│   ├── keypoints/
│   │   ├── keypoints_extractor.py    # MediaPipe Holistic → (150, K, 3) npz
│   │   ├── augment.py                # scale + Gaussian-noise augment
│   │   └── keypoints_eval.py         # QC visualizer for augmented clips
│   └── train/
│       ├── preprocess_pipeline.py    # center/scale, face subset, velocity
│       └── modeling.py               # BiLSTM + Attention + train loop
└── notebook/{analysis.ipynb, videos_crawler.ipynb}
```

### 1.2 Pipeline summary

| Stage | File | Output |
|---|---|---|
| Keypoint extraction | `src/keypoints/keypoints_extractor.py` | `dataset/keypoints/<label>/0.npz` with `pose (150,25,3)`, `left_hand (150,21,3)`, `right_hand (150,21,3)`, `face (150,468,3)` |
| Augmentation | `src/keypoints/augment.py` | `augmented/<label>/{0..N}.npz` (1 original + N noisy/scaled copies) |
| Preprocess | `src/train/preprocess_pipeline.py` | `preprocessed_npz/sample_<i>_<label>.npy` with feature tensor `(150, 628)` |
| Training | `src/train/modeling.py` | `artifacts/best_model.pt` (saved as `lstm_150.pt`) |
| Inference | `app.py` (`/api/predict`, `/api/predict-keypoints`) | top-5 softmax labels |

### 1.3 Answers to the requested summary

1. **Model type:** RNN-based hybrid — `Linear → LayerNorm → ReLU → Dropout` projection (314 or 628 → 256) → **2-layer BiLSTM** (hidden 256, dropout 0.35) → **temporal attention pooling** (`AttentionPooling` in `src/train/modeling.py:28-47`) → `Linear(512→256) → ReLU → Dropout → Linear(256→274)`. **3,124,499 parameters, 37.5 MB on disk.**
2. **Input modality:** Skeleton/keypoints only (no RGB). MediaPipe Holistic landmarks (pose-25 + 2×hand-21 + face-468 reduced to 89). z-coordinate is dropped (`preprocess_pipeline.py:39-48`). Per-frame feature dim = `25·2 + 2·21·2 + 89·2 + 2 = 314`; with velocity (Δfeat) doubled to **628**.
3. **Number of classes:** **274** (from `label_mapping.json` and checkpoint `model_config.num_classes`). 10 topics: `Tính cách - Tính chất (50), Hành động (50), Đồ vật (43), Số đếm (40), Chữ cái (37), Gia đình (25), Thời tiết (12), Câu hỏi (11), Nghề nghiệp (9), Từ thông dụng (8)`.
4. **Dataset size and split ratios:**
   - **Source videos:** 309 (`dataset/text/label.csv`).
   - **Per-class distribution:** 256 classes have **1** video, 1 class has 2, 17 classes have 3. Mean 1.13, median 1.0, std 0.49.
   - **After augmentation (`--n 50`):** ~309 × 51 ≈ 15.7k samples.
   - **Split:** stratified `train_test_split` on the flat augmented list with `val_ratio = 0.1` (`modeling.py:306-356`). **No test set. No signer-independent partition.**
5. **Reported metrics (epoch 9, from checkpoint):** `train_loss = 0.108`, `val_loss = 0.028`, `train_acc = 0.967`, `val_acc = 0.995`, `train_f1 = 0.967`, `val_f1 = 0.994`. Inference latency not measured; expect ~150 ms/clip on CPU dominated by MediaPipe Holistic, not the LSTM.

---

## PHASE 2 — Model Evaluation

> The repository ships no held-out test set, no per-class/confusion-matrix script, and no benchmark log, so the numbers below are derived from the checkpoint, code inspection, and a synthetic check.

### 2.1 Quantitative metrics (as reported / inferable)

| Metric | Value | Notes |
|---|---|---|
| `val_acc` | 0.9949 | **Inflated by augmentation leakage** (same source video in train and val) |
| `val_f1 (macro)` | 0.9937 | Same caveat |
| `train_acc` | 0.9674 | Lower than val ⇒ classic data-leak signature |
| Model params | 3,124,499 | LSTM = 2.6 M of those |
| Checkpoint size | 37.5 MB | Optimizer state included; pure weights ≈ 12 MB |
| Epoch trained | 9 / 60 | Stopped early because leaked val collapsed |
| Per-class accuracy | **Not computed** | Script missing |
| Confusion matrix | **Not computed** | Script missing |
| Latency / FPS | **Not measured** | Bottleneck is MediaPipe Holistic, not the LSTM |

### 2.2 Qualitative analysis (deduced from the code)

- **Signer independence:** Untested. 256/274 classes have only one signer, so no signer split is even possible for those classes. The 17 three-signer classes (file suffix `B/N/T` ⇒ likely 3 distinct studio signers) are the only candidates for a real generalization test, and they were never used as such.
- **Continuous vs. isolated:** Strictly isolated. The model demands exactly 150 frames per inference (≈ 5 s @ 30 fps). `process_video` in `app.py:211-323` records once and predicts once — no sliding window, no CTC, no segmentation.
- **Robustness to occlusion / lighting / distance / background:** Untested. Augmentation does not simulate any of these (it only perturbs landmark positions slightly *after* MediaPipe has already succeeded). MediaPipe Holistic itself absorbs the visual robustness, but if MediaPipe fails to detect, the pipeline produces a zero hand which — due to the bug in §3.1 — is then turned into noise that the network must learn from.

---

## PHASE 3 — Weakness Identification

### 3.1 Critical bugs (correctness)

#### B1. Hand-presence mask is destroyed by augmentation
- **Code path:** `src/keypoints/augment.py:36-45` scales then adds Gaussian noise to *all* tensors, including missing-hand tensors that were zero-initialized in `keypoints_extractor.py:53-54`.
- **Verified:** an all-zero `(150, 21, 3)` hand becomes `sum ≈ 112` after one augment pass. `hand_present_mask` in `preprocess_pipeline.py:52-53` reports `1` for every frame.
- **Consequence:** `AttentionPooling` mask is always all-ones for augmented data ⇒ attention degenerates to plain softmax-pool; left/right hand mask features lose all signal.

#### B2. Train/val leakage via post-augmentation split
- **Code path:** `modeling.py:306-356` calls `train_test_split` on the flat sample list. The same source clip's 51 augmentations get split randomly between train and val.
- **Evidence:** `val_loss (0.028) < train_loss (0.108)` is the classical leakage symptom; on a real held-out partition `val_loss > train_loss` is the norm.

#### B3. Train/inference MediaPipe mismatch
- **Training extractor** (`keypoints_extractor.py:35-39`): `model_complexity=1`, no `smooth_landmarks`, 1920×1080.
- **Inference** (`app.py:236-244`): `model_complexity=0`, `smooth_landmarks=True`, resized to 640×360.
- Different models produce different landmark distributions ⇒ unknown but real performance drop at deploy time.

#### B4. Label inconsistency
- `chú ( người)` (extra space) in `dataset/text/label.csv` vs. `chú (người)` in `label_mapping.json`. One sample is silently dropped or assigned to a nonexistent class.
- `cleaned_data.csv` contains `L` and `đau` that are absent from the training set (276 vs 274 labels).

#### B5. Augment clips to `[0, 1]` after scaling around the *mean* (`augment.py:47-50`)
- Scaling is around the per-array centroid, not around the image center, so it can push values outside `[0,1]`; the clip then warps the body silhouette. Semantically this is fine for noise robustness but it's not the simulation that the comment claims (camera zoom).

### 3.2 Data issues

| Issue | Evidence | Severity |
|---|---|---|
| 93 % of classes have a **single** source video | `dataset/text/label.csv` counts | **Critical** |
| No signer/age/gender/skin-tone diversity beyond the 3 studio recorders | Filename pattern `D####{B,N,T}` | **Critical** |
| Augmentation has no **temporal** variation: no time-warp, no stretch, no random temporal crop, no frame dropout | `augment.py:16-52` | **High** |
| No **mirroring / handedness flip** — left-handed signers will never be recognized | `augment.py` | **High** |
| No rotation / 3D viewpoint simulation | `augment.py` | **Medium** |
| z-coordinate is dropped (`preprocess_pipeline.py:39-48`) — depth signal lost | code | **Medium** |
| `extract_face_subset` uses hard-coded ranges `61–88, 246–276, 300–332` (`preprocess_pipeline.py:21`) that are not the official MediaPipe lips/eyes/brows index lists — unclear which subset is actually selected | `preprocess_pipeline.py:21` | **Medium** |
| `data/cleaned_data.csv` only stores Vimeo URLs, not local files — no reproducibility audit | `data/cleaned_data.csv` | **Low** |

### 3.3 Architecture issues

- **BiLSTM is dated for skeleton-based SLR.** SOTA on signing datasets (AUTSL, WLASL, MSASL) is dominated by **Spatio-Temporal Graph Convolutional Networks** (ST-GCN / 2s-AGCN / MS-G3D) and **transformer encoders** (SignBERT, KeypointTransformer). A BiLSTM treats joints as a flat 314-D vector ⇒ no skeletal topology bias.
- **Face features (178 dims) outweigh hand features (84 dims).** Most lexical content in VSL lives in the hands; face mainly carries non-manual markers. The 56 % face / 27 % hand ratio is inverted.
- **Velocity is computed on the *concatenated, normalized* feature vector** including the binary mask channel (`preprocess_pipeline.py:84-86`). `np.diff` on a 0/1 mask produces ±1 spikes — noisy and uninformative; should compute velocity on coordinates only.
- **No second-order kinematics** (acceleration), no joint-relative features (bone vectors / angles between joints).
- **Single global attention pooling** with no positional encoding and no multi-head — the network has to recover ordering from LSTM state alone.

### 3.4 Training issues

- **Trained for only 9 epochs.** Configured for 60 with `patience=8`; the leaked val_loss reached a minimum so early because train/val were near-identical.
- **`label_smoothing` defaulted to 0.0**, `use_class_weights` is opt-in; even when enabled, weights are `sqrt(max/count)` (`modeling.py:163-169`) — far weaker than focal loss for the extreme imbalance of 1–3 samples/class.
- **No mixup / cutmix / SpecAugment / temporal masking.**
- **LR schedule** is `ReduceLROnPlateau(factor=0.5, patience=4)` on `val_loss` — useless when val_loss is artificially low.
- **No gradient accumulation, no AMP, no `pin_memory` cleanup**, `num_workers=0` (default).
- **No reproducibility of MediaPipe seeds** — Holistic is non-deterministic across runs.

### 3.5 Inference / deployment issues

| Issue | File | Severity |
|---|---|---|
| MediaPipe complexity / smoothing mismatch (B3) | `app.py:236-244` vs `keypoints_extractor.py:35-39` | High |
| Fixed 150-frame buffer ⇒ no continuous streaming | `app.py:289` | High |
| No quantization / ONNX / TorchScript export | — | Medium |
| No confidence threshold / unknown-class rejection — model **always** returns a top-5 | `app.py:317-324` | Medium |
| `requirements.txt` pins `tensorflow>=2.13.0` even though only PyTorch is used | `requirements.txt:9` | Low |
| `model_state` saved alongside optimizer/scheduler state ⇒ checkpoint is 3× larger than needed for serving | `modeling.py:359-386` | Low |
| `_holistic_cache` is a module-level singleton not thread-safe under uvicorn workers > 1 | `app.py:54-55, 234-244` | Medium |
| No request rate-limit, no input size cap on `/api/predict` upload | `app.py:414-422` | Medium |

---

## PHASE 4 — Improvement Recommendations

### [Priority: HIGH] — Fix the train/val leakage
**Problem:** Stratified split on the augmented sample list lets near-duplicates of the same source clip into both train and val, inflating metrics to ~99 %.
**Evidence:** `src/train/modeling.py:306-356`; `val_loss (0.028) < train_loss (0.108)`.
**Fix:** Split by **source video id** *before* augmentation; augment only the training partition.

```python
# in build_index_csv / prepare_samples
unique_sources = df['source_video'].unique()
train_src, val_src = train_test_split(unique_sources, test_size=0.15,
                                       random_state=42, stratify=df.groupby('source_video')['label'].first())
train_df = df[df['source_video'].isin(train_src)]
val_df   = df[df['source_video'].isin(val_src)]
```
**Expected impact:** Reported metrics will drop dramatically (likely 30–60 % macro-F1), revealing the real baseline that every later improvement is measured against.

---

### [Priority: HIGH] — Fix the hand-presence mask bug
**Problem:** Augmentation adds noise to all-zero missing hands, so `hand_present_mask` returns 1 for every frame; attention masking and the mask features become useless.
**Evidence:** `src/keypoints/augment.py:36-45`; verified empirically (`sum=112` after augment on a zero hand).
**Fix:** Persist the presence mask in the `.npz` and skip noise on missing frames.

```python
# in keypoints_extractor.py
lh_present = np.array([results.left_hand_landmarks is not None for ...], dtype=np.float32)
rh_present = np.array([results.right_hand_landmarks is not None for ...], dtype=np.float32)
np.savez(output_path, ..., lh_present=lh_present, rh_present=rh_present)

# in augment.py
if lh_present[t] == 0: left_hand_aug[t] = 0.0   # keep missing hands missing
```
**Expected impact:** +2–5 % real F1 (attention pooling becomes effective again), and unlocks correct behavior when MediaPipe loses a hand at inference.

---

### [Priority: HIGH] — Add proper video augmentation for sequences
**Problem:** Current augmenter is only spatial scale + isotropic Gaussian noise — no temporal, no flip, no rotation. Hugely under-represents real-world variation.
**Evidence:** `src/keypoints/augment.py:16-52`.
**Fix:**

```python
# Temporal augmentation
def time_warp(seq, k_min=0.85, k_max=1.15):
    k = np.random.uniform(k_min, k_max)
    new_len = int(round(seq.shape[0] * k))
    idx = np.linspace(0, seq.shape[0]-1, new_len)
    seq = np.stack([np.interp(idx, np.arange(seq.shape[0]), seq[:, j]) for j in range(seq.shape[1])], axis=1)
    return resample(seq, 150)  # back to 150

# Horizontal mirror (swap L/R hand, flip x)
def mirror(pose, lh, rh, face):
    pose[..., 0] = 1 - pose[..., 0]; lh[..., 0] = 1 - lh[..., 0]
    rh[..., 0]   = 1 - rh[..., 0];   face[..., 0] = 1 - face[..., 0]
    return pose, rh, lh, face   # NOTE: hands swapped

# 2D rotation around shoulder midpoint (±10°), small translation, frame dropout
```
**Expected impact:** +5–10 % macro-F1 on signer-independent test; gives the model genuine invariance to handedness, framing, and signing speed.

---

### [Priority: HIGH] — Switch backbone to ST-GCN (or SignBERT) on keypoints
**Problem:** Flat-vector BiLSTM ignores skeletal topology; face dominates feature budget; no joint-relative bias.
**Evidence:** `src/train/modeling.py:50-108`, feature ratio 56 % face / 27 % hand.
**Fix:** Implement a small **ST-GCN** (`pyskl` or `mmaction2`) with three subgraphs (pose, left hand, right hand) and re-weight face to a 16-D embedding via a learnable MLP rather than raw 178-D. Keep total params ≤ 4 M for parity.
**Expected impact:** +8–15 % macro-F1 vs. fixed BiLSTM on the same data (consistent with WLASL / AUTSL leaderboards).

---

### [Priority: HIGH] — Real evaluation protocol
**Problem:** No held-out test set, no per-class report, no confusion matrix, no signer-independent eval.
**Evidence:** repo contains zero evaluation script beyond per-epoch metrics in `modeling.py:237-272`.
**Fix:** Create `src/eval/evaluate.py` that:
1. Loads `best_model.pt`,
2. Iterates a frozen test split (signer-disjoint where possible),
3. Reports macro/weighted F1, top-1/top-5, confusion matrix (`sklearn.metrics.confusion_matrix`), 10 worst classes, average inference latency.
4. Optionally evaluates under perturbations (Gaussian noise σ ∈ {0.01, 0.05}, hand dropout, time-warp ±20 %) — that is your robustness benchmark.

**Expected impact:** Gives reproducible ground-truth metrics for the roadmap below.

---

### [Priority: MEDIUM] — Handle class imbalance with focal loss + oversampling
**Problem:** 1–3 samples/class with `sqrt(max/count)` weighting is too gentle.
**Evidence:** `modeling.py:163-169`, label distribution in Phase 1.
**Fix:**

```python
loss_fn = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
sampler = WeightedRandomSampler(weights=[1.0/counts[s.label_idx] for s in train_samples],
                                num_samples=len(train_samples), replacement=True)
train_loader = DataLoader(..., sampler=sampler, shuffle=False)
```
**Expected impact:** +3–6 % macro-F1 (especially on tail classes).

---

### [Priority: MEDIUM] — Training-recipe modernization
**Problem:** No mixup, cosine schedule, AMP; LR plateau on a leaky val signal.
**Fix:**
- `torch.optim.lr_scheduler.OneCycleLR` (max_lr=3e-3, pct_start=0.1).
- `mixup_alpha=0.2` on sequence-level (mix two sequences and their one-hot labels).
- `torch.cuda.amp.autocast` + `GradScaler`.
- Use `val_f1` instead of `val_loss` for plateau scheduling.
- Re-enable full 60 epochs with `patience=12`.

**Expected impact:** +2–4 % macro-F1, ~2× training speed on GPU.

---

### [Priority: MEDIUM] — Align training and inference MediaPipe configs
**Problem:** Inference uses `model_complexity=0` and `smooth_landmarks=True` while training used `model_complexity=1` and no smoothing.
**Evidence:** `app.py:236-244` vs `keypoints_extractor.py:35-39`.
**Fix:** Pick one. For latency-critical deployment use `complexity=0` everywhere (re-extract features and retrain). For accuracy, use `complexity=1` in inference too. Either way, store the chosen config in the checkpoint and assert on load.
**Expected impact:** Eliminates a silent 3–8 % accuracy drop between dev and prod.

---

### [Priority: MEDIUM] — Continuous-signing sliding window + smoothing
**Problem:** Fixed 5-second buffer; cannot recognize signs in a stream.
**Fix:** Run inference on a sliding window (length 90 frames, stride 15) with EMA smoothing over softmax outputs and a min-confidence threshold (e.g. 0.55). Emit a token only when the same argmax persists across ≥ 3 windows.
**Expected impact:** Enables a real "live" demo; reduces flicker.

---

### [Priority: LOW] — Inference optimization
**Problem:** Plain PyTorch FP32 inference, no export.
**Fix:**
- `torch.onnx.export(model, dummy, "vsl_bilstm.onnx", opset=17, dynamic_axes={...})`.
- `onnxruntime` for ARM / CPU, `TensorRT FP16` for NVIDIA edge.
- Optional `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` (LSTM dynamic-quant is buggy; quantize Linear only).
**Expected impact:** 2–3× lower CPU latency, ~4× smaller checkpoint (12 → 3 MB).

---

### [Priority: LOW] — Repo hygiene
- Remove `tensorflow` from `requirements.txt` (line 9) — unused, drags ~500 MB.
- Add `pytest` smoke tests for `preprocess_pipeline.py` and `build_feature_sequence` shape contract.
- Normalize label strings before mapping (strip whitespace, NFC unicode) to fix the `chú (người)` mismatch.
- Persist `model_state` separately from optimizer state for the deployed checkpoint.

---

## PHASE 5 — Optimization Roadmap

| # | Priority | Task | Effort | Expected Macro-F1 Gain (signer-indep.) |
|---|---|---|---|---|
| 1 | **HIGH** | Source-level train/val/test split + new evaluation script | **Low** | Reveals real baseline (−60 %, then build up) |
| 2 | **HIGH** | Fix `hand_present_mask` bug in `augment.py` | Low | **+2 – 5 %** |
| 3 | **HIGH** | Mirroring + time-warp + rotation augmentation | Medium | **+5 – 10 %** |
| 4 | **HIGH** | Re-extract MediaPipe at consistent `complexity` between train/infer | Low | **+3 – 8 %** |
| 5 | **HIGH** | Replace BiLSTM with ST-GCN or temporal Transformer | High | **+8 – 15 %** |
| 6 | MEDIUM | Focal loss + WeightedRandomSampler | Low | **+3 – 6 %** |
| 7 | MEDIUM | Mixup, OneCycleLR, AMP, full 60 epochs | Medium | **+2 – 4 %** |
| 8 | MEDIUM | Sliding-window inference + EMA + confidence threshold | Medium | UX + continuous signing |
| 9 | MEDIUM | Reduce face feature footprint, add bone/angle features | Medium | **+1 – 3 %** |
| 10 | LOW | ONNX/TensorRT export, dynamic quantization | Low | 2 – 3× inference speed, smaller binary |
| 11 | LOW | Label/string normalization, drop `tensorflow`, smoke tests | Low | Hygiene |

**Cumulative best-case (items 1–9):** a realistic +25 to +40 % macro-F1 over whatever the *true* (post-fix) baseline turns out to be — most likely landing in the 60 – 75 % macro-F1 range on a signer-independent test, which is competitive with mid-tier WLASL/AUTSL benchmarks.

---

## Recommended next step — to implement TODAY

> **Write `src/eval/evaluate.py` and re-split by source video.**

Concretely:
1. Add a `source_video` column to `index.csv` (parse from filename, before augmentation).
2. Modify `split_samples` in `src/train/modeling.py:306-356` so the split key is `source_video`, not the row index.
3. Hold out **15 %** of source videos as a permanent test set; never augment that set.
4. Add `src/eval/evaluate.py` that loads `lstm_150.pt` and prints:
   - macro/weighted F1, top-1, top-5
   - 10 worst classes
   - confusion matrix (`sklearn.metrics.confusion_matrix`)
   - mean inference latency (PyTorch CPU/GPU).
5. Run it on the current checkpoint and commit the report.

That single change converts every later experiment from "feels-faster" speculation into measurable progress. Every other recommendation above depends on it.
