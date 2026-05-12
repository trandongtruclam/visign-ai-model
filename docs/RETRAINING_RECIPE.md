# Retraining recipe — V2 (post-bug-fix baseline)

This is the next-cycle training command. It applies the two fixes that
landed alongside this doc:

1. **B1 bug fix in `augment.py`** — missing-hand frames are now re-zeroed
   after Gaussian noise, so `hand_present_mask` in
   `preprocess_pipeline.py` reports the truth and the attention-pool mask
   actually masks something.
2. **Mirror + time-warp augmentations** — exposed via `--mirror-prob` and
   `--time-warp-prob` in `src/keypoints/augment.py`.

The current honest baseline (measured on the source-disjoint test split,
18 samples, 18 classes) is **Top-1 = 0.4444, Macro F1 (supp) = 0.4444**
— see `docs/EVAL_REPORT.md`. Any change below should be judged against
those numbers, not the leaked `val_f1 = 0.9937` that the old pipeline
reported.

---

## Why retrain at all?

The B1 fix and the new augmentations only take effect on data that is
**re-augmented and re-preprocessed after this commit**. The model
checkpoint `artifacts/lstm_150.pt` was trained on the broken data; it
will not benefit until you retrain.

## Steps

> Run from `ai-model/`. Steps 1–4 produce a fresh dataset on disk; step 5
> retrains; step 6 regenerates `docs/EVAL_REPORT.md` with the new numbers.

```bash
# 0. Clean out stale data so nothing from the old pipeline leaks in.
#    (splits.json is regenerated in step 2, so it can go too.)
rm -rf augmented preprocessed_npz splits.json index.csv

# 1. Re-extract keypoints with unique-source filenames. Skip this step
#    if you already re-extracted after the source-split commit landed
#    (the on-disk filenames in dataset/keypoints/ should look like
#    `<label>/<videoid>.npz`, not `<label>/0.npz`).
python src/keypoints/keypoints_extractor.py --process_dataset \
    --videos_dir dataset/videos \
    --labels_csv dataset/text/label.csv \
    --output_dir dataset/keypoints

# 2. Build the source-level split (singletons stay in train by default).
python -m src.keypoints.split_sources \
    --keypoints-dir dataset/keypoints \
    --output splits.json \
    --val-ratio 0.10 --test-ratio 0.15 --seed 42

# 3. Augment ONLY the train split, with mirror + time-warp ON.
#    --augment-splits train keeps val/test pristine (copy-through only).
python src/keypoints/augment.py dataset/keypoints augmented \
    --n 50 \
    --splits splits.json --augment-splits train \
    --mirror-prob 0.5 \
    --time-warp-prob 0.7 --time-warp-min 0.85 --time-warp-max 1.15

# 4. Build index.csv (with `split` column) and preprocess to .npy features.
python src/train/preprocess_pipeline.py \
    --data-dir augmented \
    --index-csv index.csv \
    --feature-dir preprocessed_npz \
    --splits-json splits.json

# 5. Retrain. With the source-level split + bug fix + new augmentations,
#    val_loss should now be ABOVE train_loss (the leakage signal is gone),
#    and the model can run the full 60 epochs without early-stopping at
#    epoch 9 like before.
python src/train/modeling.py \
    --index-csv index.csv \
    --feature-dir preprocessed_npz \
    --output-dir artifacts \
    --epochs 60 \
    --batch-size 32 \
    --lr 1e-3 \
    --use-class-weights \
    --label-smoothing 0.05 \
    --patience 12

# 6. Evaluate on the fixed test split. Overwrites docs/EVAL_REPORT.md.
python -m src.eval.evaluate \
    --checkpoint artifacts/best_model.pt \
    --index-csv index.csv \
    --feature-dir preprocessed_npz \
    --split test \
    --output-dir docs
```

## Recommended augmentation flags

| Flag | Recommended | Default | Why |
|---|---|---|---|
| `--mirror-prob` | `0.5` | `0.0` | Half the augmented samples get mirrored. Forces the model to learn handedness-invariant features; left-handed signers should stop being a hard-mode class. |
| `--time-warp-prob` | `0.7` | `0.0` | Most augmented samples get a small speed perturbation. Targets the "short sign collapse" observed in the worst-classes list (single letters g/h/p/t etc.). |
| `--time-warp-min` | `0.85` | `0.85` | A 15 % faster sign duration. |
| `--time-warp-max` | `1.15` | `1.15` | A 15 % slower sign duration. Larger than ±15 % starts to break the 150-frame normalization assumption. |
| `--n` | `50` | `10` | Same as before; the new augmentations are independent, so each of the 50 copies is now meaningfully different (≈ 50 % × 70 % × scale × noise variety per sample). |

## What to look for in the new run

Beyond the headline Top-1 / Macro-F1 numbers:

1. **`val_loss > train_loss`** at convergence. If it's the other way
   round again, the leak is back; check that `index.csv` contains a
   `split` column and that `train_model` reported "Test samples (held
   out, untouched): 18" at startup.
2. **`distinct_predictions` in `docs/eval_summary.json`** should be
   close to 18 (one per test class). The old baseline already produced
   16 distinct predictions; if the new run drops below 12 the model is
   collapsing onto a few "comfortable" classes.
3. **Top-3 should now exceed Top-1.** In the broken baseline they were
   identical (0.444 / 0.444). When the model is genuinely uncertain,
   Top-3 carries useful information; if they stay equal after retraining,
   the network is still memorizing rather than ranking.
4. **The 10 worst classes** should change. If `g/h/p/t/â/ô/ă` still
   dominate the bottom of the list, mirroring is not helping — file a
   follow-up to inspect those specific test clips visually.

## When this recipe is not enough

Both fixes together typically buy ~5–15 pp of macro F1 on signer-disjoint
sign-language tasks (numbers from WLASL / AUTSL ablations with comparable
data scale). If after retraining you still see Top-1 ≤ 55 %, the
remaining gap is most likely:

* **Architectural** — flat-BiLSTM is a poor inductive prior for skeletal
  data. Next step is the ST-GCN swap in `docs/notes.md` item 5.
* **Data scarcity** — 256/274 classes have a single signer. Even a
  perfect model cannot generalize to new signers for those classes.
  Collect more recordings before chasing further accuracy.

## Reverting

`--mirror-prob 0.0 --time-warp-prob 0.0` reproduces the previous
augmentation behaviour exactly (except for the always-on B1 fix, which
is a pure bug fix and has no `--no-fix-presence` opt-out).
