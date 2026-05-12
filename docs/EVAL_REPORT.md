# VSL Model — Evaluation Report

- **Checkpoint:** `artifacts\lstm_150.pt`
- **Split evaluated:** `test`
- **Samples in split:** 18
- **Model config:** in_feat=628, proj_dim=256, hidden=256, layers=2, bidir=True, attention=True, num_classes=274

## Accuracy / F1

| Metric | Value |
|---|---|
| Top-1 accuracy | 0.4444 |
| Top-3 accuracy | 0.4444 |
| TOP5 accuracy | 0.4444 |
| Macro F1 (classes with test support, n=18) | **0.4444** |
| Macro F1 (all 274 classes, incl. zero-support) | 0.0292 |
| Weighted F1 | 0.4444 |
| Classes with support | 18 / 274 (6.6%) |
| Distinct classes predicted by the model on this split | 16 |

> **Read this:** `Macro F1 (all classes)` is dragged down by the 256 singleton-source classes that have zero test support — they all contribute F1 = 0 to the average. **`Macro F1 (classes with test support)` is the honest number** for the evaluable subset. `Weighted F1` averages F1 over supported classes weighted by support; with one sample per class it collapses to plain accuracy.

## 10 worst classes (by F1, support > 0)

| Rank | Label | Support | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1 | bao giờ_ | 1 | 0.000 | 0.000 | 0.000 |
| 2 | dấu sắc | 1 | 0.000 | 0.000 | 0.000 |
| 3 | g | 1 | 0.000 | 0.000 | 0.000 |
| 4 | h | 1 | 0.000 | 0.000 | 0.000 |
| 5 | p | 1 | 0.000 | 0.000 | 0.000 |
| 6 | t | 1 | 0.000 | 0.000 | 0.000 |
| 7 | â | 1 | 0.000 | 0.000 | 0.000 |
| 8 | ô | 1 | 0.000 | 0.000 | 0.000 |
| 9 | ă | 1 | 0.000 | 0.000 | 0.000 |
| 10 | đúng không_ | 1 | 0.000 | 0.000 | 0.000 |

## Top confusion pairs

| True → Predicted | Count |
|---|---|
| đúng không_ → cái túi | 1 |
| ă → 40 | 1 |
| ô → 40 | 1 |
| â → bảo đảm | 1 |
| t → e | 1 |
| p → l | 1 |
| h → 40 | 1 |
| g → con đẻ | 1 |
| dấu sắc → bố mẹ | 1 |
| bao giờ_ → cách ly | 1 |

## Inference latency

| Device | Batch | Seq | Iters | Mean (ms) | Median (ms) | P95 (ms) |
|---|---|---|---|---|---|---|
| cpu | 1 | 150 | 50 | 7.22 | 7.29 | 8.33 |
