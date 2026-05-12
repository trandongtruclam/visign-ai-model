# VSL Model — Evaluation Report

- **Checkpoint:** `artifacts/lstm_150.pt`
- **Generated:** 2026-05-12 (initial run)
- **Script:** `python -m src.eval.evaluate`
- **Status:** **partial — latency + checkpoint metadata only.** Accuracy / F1 / confusion-matrix numbers below are reproducible once `index.csv` + `preprocessed_npz/` are regenerated (see [§ Reproducing the full evaluation](#reproducing-the-full-evaluation)).

## 1. Checkpoint metadata

| Field | Value |
|---|---|
| Architecture | BiLSTM (2 layers, hidden 256, bidirectional) + Attention pooling |
| `in_feat` | 628 (314 base + 314 velocity) |
| `proj_dim` | 256 |
| `num_classes` | 274 |
| `use_attention` | True |
| Total parameters | **3,124,499** |
| File size on disk | 37.5 MB (includes optimizer + scheduler state) |
| Epoch trained | **9 / 60** (early-stop fired) |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-4 |
| Scheduler | `ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6)` |
| Class weights | enabled (`sqrt(max/count)`) |
| Label smoothing | 0.0 |

### Training-time metrics persisted in the checkpoint

| Metric | Value | Reliability |
|---|---|---|
| `train_loss` | 0.1078 | trustworthy |
| `train_acc` | 0.9674 | trustworthy |
| `train_f1` (macro) | 0.9673 | trustworthy |
| `val_loss` | **0.0280** | **inflated by leakage** — `val_loss < train_loss` is the classical signal |
| `val_acc` | **0.9949** | **inflated** |
| `val_f1` (macro) | **0.9937** | **inflated** |

> **Why the val numbers are not real.** Augmentations of the same source video (`augment.py` produced ~50 noisy copies per clip) were randomly split between train and val by the old `split_samples`. Near-duplicates of the same physical recording therefore appeared in both partitions. The new source-level split (PR landing alongside this report) eliminates this leak; the next training run will produce honest numbers.

## 2. Accuracy / F1 on a held-out source-level test set

> **Not yet measured.** The augmented dataset (`augmented/`, `preprocessed_npz/`, `dataset/keypoints/`) is gitignored and not present in this checkout, so the script ran in latency-only mode. To populate this section:
>
> ```bash
> # 1. Re-extract keypoints with unique source filenames
> python src/keypoints/keypoints_extractor.py --process_dataset \
>     --videos_dir dataset/videos --labels_csv dataset/text/label.csv \
>     --output_dir dataset/keypoints
>
> # 2. Build the source-level split (15 % test, 10 % val, singletons stay in train)
> python -m src.keypoints.split_sources \
>     --keypoints-dir dataset/keypoints \
>     --output splits.json \
>     --val-ratio 0.10 --test-ratio 0.15 --seed 42
>
> # 3. Augment ONLY the train split (test/val stay pristine)
> python src/keypoints/augment.py dataset/keypoints augmented \
>     --n 50 --splits splits.json --augment-splits train
>
> # 4. Build index.csv (with split column) and preprocess to feature .npy
> python src/train/preprocess_pipeline.py \
>     --data-dir augmented --index-csv index.csv \
>     --feature-dir preprocessed_npz --splits-json splits.json
>
> # 5. Evaluate the current checkpoint on the held-out test split
> python -m src.eval.evaluate \
>     --checkpoint artifacts/lstm_150.pt \
>     --index-csv index.csv \
>     --feature-dir preprocessed_npz \
>     --split test \
>     --output-dir docs
> ```
>
> Step 5 fills in the table below, writes `docs/eval_confusion_matrix.npy`, `docs/eval_per_class.csv`, and `docs/eval_summary.json`.

| Metric | Value |
|---|---|
| Top-1 accuracy | _pending_ |
| Top-5 accuracy | _pending_ |
| Macro F1 | _pending_ |
| Weighted F1 | _pending_ |
| Classes with test support | _pending_ / 274 |

> **Expected range** (educated guess given the current data — 256/274 classes have only one source video):
> * On the **strict source-level test split (singleton classes excluded from test):** macro F1 in the **40 – 65 %** range. The model was never trained with augmentation diversity beyond Gaussian noise + scale, and the hand-presence mask bug effectively disabled its attention masking.
> * If `--strict-source-split` is used (test set may contain previously-unseen classes): macro F1 collapses to near-zero on those classes (zero-shot scenario for which this architecture has no mechanism).

## 3. Inference latency (PyTorch, FP32)

Measured with synthetic input shaped `(batch, 150, 628)` directly on the loaded checkpoint. The MediaPipe Holistic step inside `app.py` is **not** included here — it dominates real wall-clock by 10–50× per clip.

| Device | Batch | Seq | Iters | Mean (ms) | Median (ms) | P95 (ms) |
|---|---|---|---|---|---|---|
| CPU | 1 | 150 | 50 | **7.34** | 7.21 | 9.34 |
| CPU | 4 | 150 | 50 | 17.47 | 17.36 | 19.49 |
| CPU | 8 | 150 | 50 | 30.89 | 29.56 | 34.16 |
| CPU | 16 | 150 | 50 | 57.24 | 57.33 | 61.68 |
| CPU | 32 | 150 | 50 | 107.14 | 107.26 | 112.88 |

> The CPU run scales near-linearly with batch size, indicating the LSTM is single-core-bound. CUDA was **not available** on the machine that produced this report; rerun the script on the deployment box to populate GPU rows.

### Implications

- The classifier itself adds **~7 ms / clip** on CPU. That is **negligible** compared to MediaPipe Holistic at `model_complexity=0` (~30–60 ms/frame × 150 frames ≈ 4.5–9 s per clip in `app.py`).
- The deployment bottleneck is **keypoint extraction**, not the network. Quantization or ONNX export of the LSTM would shave at most a few ms; aligning MediaPipe across train/inference (and considering frame-rate sub-sampling) will move the needle far more.

## 4. Per-class metrics

_Pending — written to `docs/eval_per_class.csv` once step 5 above runs._

## 5. 10 worst classes (by F1, support > 0)

_Pending._

## 6. Top confusion pairs

_Pending — written to `docs/eval_confusion_matrix.npy` and surfaced in this section once step 5 runs._

## 7. Files produced by this script

| File | Purpose |
|---|---|
| `docs/EVAL_REPORT.md` | This report (overwritten on every run) |
| `docs/eval_summary.json` | Machine-readable summary (config + metrics + latency) |
| `docs/eval_per_class.csv` | Per-class precision / recall / F1 / support |
| `docs/eval_confusion_matrix.npy` | Raw `confusion_matrix` from sklearn |

## Reproducing the full evaluation

```bash
cd ai-model
python -m src.eval.evaluate \
    --checkpoint artifacts/lstm_150.pt \
    --index-csv index.csv \
    --feature-dir preprocessed_npz \
    --split test \
    --output-dir docs
```

For the latency benchmark alone (no dataset required):

```bash
python -m src.eval.evaluate --checkpoint artifacts/lstm_150.pt --device cpu
```

## Open issues blocking a meaningful test-set score

These are the items called out in `docs/notes.md` (the audit) that will *change the numbers* this report produces. Track them as follow-up PRs:

1. **`augment.py` hand-presence mask bug** — augmented "missing" hands become noisy, so the attention mask is always all-ones. Fix before retraining.
2. **Training extractor uses `model_complexity=1`, inference uses `0`** — different landmark distributions. Pick one and re-extract.
3. **256/274 classes have a single source video** — true signer-independent generalization is untestable for these. Either collect more recordings or restrict reporting to the 18 multi-source classes.
4. **Augmentation is too weak** — no time-warp, no mirroring, no rotation. Add these before training the next baseline.

These are independent of the evaluation framework itself; the framework is now ready to score every subsequent training run.
