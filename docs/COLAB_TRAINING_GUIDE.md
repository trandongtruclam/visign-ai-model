# Colab Training Guide — Visign AI Model

End-to-end recipe for training the Vietnamese Sign Language recogniser on
Google Colab when local hardware is too weak. Each section is independent;
do them in order the first time, then come back to whichever one you need
later.

---

## 0. What lives where

| Path | What it is |
|---|---|
| `src/keypoints/keypoints_extractor.py` | Runs MediaPipe Holistic over raw `.mp4` videos → writes `dataset/keypoints/<label>/<source>.npz`. **Slow; do this locally before going to Colab.** |
| `src/keypoints/augment.py` | Generates K augmented copies per source → writes `augmented/<label>/<source>__N.npz`. Slow-ish; do locally. |
| `src/keypoints/split_sources.py` | Assigns sources to train/val/test at the **source level** (no clip leakage) → writes `splits.json`. |
| `src/keypoints/voya_import.py` | **NEW.** Pulls VOYA_VSL keypoint samples from Hugging Face, converts them to our format, and appends them to `splits.json` as val/test sources. |
| `src/train/preprocess_pipeline.py` | Builds `index.csv` and `preprocessed_npz/sample_<i>_<label>.npy` (314 or 628 features per frame). Picks up `splits.json` and writes a `split` column. |
| `src/train/modeling.py` | The actual training loop. Reads `index.csv` + `preprocessed_npz/`, honours its `split` column, saves `artifacts/best_model.pt`. |
| `src/eval/evaluate.py` | Loads a checkpoint, runs it on a chosen split, writes `docs/EVAL_REPORT.md` + `docs/eval_summary.json`. |
| `splits.json` | The source-of-truth for which source goes to train / val / test. |
| `index.csv` | One row per training sample (augmentations included). Derived; rebuild with `preprocess_pipeline.py`. |
| `preprocessed_npz/` | One `.npy` per sample, named `sample_<row_idx>_<label>.npy`. Derived; ~14k files. |
| `artifacts/best_model.pt` | Best checkpoint by val F1, written during training. |
| `docs/EVAL_REPORT.md` | Human-readable accuracy / F1 / confusion report. |

---

## 1. One-time Colab session setup

Runs once per Colab session (they reset every ~12 h).

```python
# 1.1 Mount Drive (so we can read inputs and persist artifacts)
from google.colab import drive
drive.mount("/content/drive")

# 1.2 Clone the repo (or pull latest)
%cd /content
!git clone https://github.com/trandongtruclam/visign-ai-model.git || (cd visign-ai-model && git pull)
%cd visign-ai-model

# 1.3 Install Python deps for training only (skip OpenCV — not needed on Colab)
!pip install -q torch numpy pandas scikit-learn tqdm scipy huggingface_hub

# 1.4 Confirm GPU is attached: Runtime → Change runtime type → GPU (T4)
import torch; print("cuda:", torch.cuda.is_available(), "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
```

**Section purpose:** get Colab into a known state with code present and GPU
available. If `torch.cuda.is_available()` prints `False`, change the runtime
to GPU before continuing — otherwise training takes ~10× longer.

---

## 2. Get the features onto Colab

The repo on GitHub does **not** ship `preprocessed_npz/` (14 k `.npy` files,
gitignored) or `index.csv`. You must upload them yourself.

### 2.1 (Local, one time) — build the upload tarball

Run on your Windows machine, inside `D:/hub/sudocode/visign/ai-model`:

```powershell
# Use TAR, NOT zip — Compress-Archive mangles Vietnamese filenames.
tar -czf visign_features.tar.gz preprocessed_npz index.csv splits.json
```

Upload `visign_features.tar.gz` to Drive at e.g. `My Drive/visign/visign_features.tar.gz`.

### 2.2 (Colab) — extract into the repo root

```python
%cd /content/visign-ai-model
!tar -xzf "/content/drive/MyDrive/visign/visign_features.tar.gz" -C /content/visign-ai-model
```

### 2.3 Sanity-check before training

```python
import os, glob
print("cwd:", os.getcwd())
print("index.csv:", os.path.isfile("index.csv"))
print("preprocessed_npz/:", os.path.isdir("preprocessed_npz"))
print("npy count:", len(glob.glob("preprocessed_npz/*.npy")))
print("first 3:", sorted(glob.glob("preprocessed_npz/*.npy"))[:3])
```

**Expected:** `npy count: 14008` (or whatever your local count is), and the
file basenames must contain readable Vietnamese (e.g. `sample_0_0 _số không_.npy`).
If filenames look like `sample_0_0 _sс╗С kh├┤ng_.npy`, jump to
[§7 — Common failures](#7-common-failures--fixes).

**Section purpose:** put `preprocessed_npz/`, `index.csv`, `splits.json` at
the repo root, where `modeling.py` expects them.

---

## 3. Train the model

```python
%cd /content/visign-ai-model
!python src/train/modeling.py \
    --index-csv index.csv \
    --feature-dir preprocessed_npz \
    --output-dir artifacts \
    --epochs 60 --batch-size 32 --lr 1e-3 \
    --use-class-weights --label-smoothing 0.05 \
    --num-workers 2 --device cuda
```

**Each flag:**

| Flag | What it does |
|---|---|
| `--index-csv` | The CSV listing every sample, its label, and its train/val/test bucket. |
| `--feature-dir` | Where `sample_<i>_<label>.npy` files live. |
| `--output-dir` | Where to write `best_model.pt`, `training_history.json`, `splits.json`. |
| `--epochs` | Max training epochs; early-stopping (`--patience 8`, hard-coded default) cuts it short if val F1 stops improving. |
| `--batch-size 32` | Fits comfortably in T4 memory for 628-dim features × 150 frames. |
| `--lr 1e-3` | Standard for AdamW on this scale of data. |
| `--use-class-weights` | Re-weights cross-entropy by `sqrt(max_count / class_count)` to compensate for class imbalance. |
| `--label-smoothing 0.05` | Discourages over-confident wrong predictions. |
| `--num-workers 2` | Two DataLoader workers; set to `0` if you're debugging worker tracebacks. |
| `--device cuda` | Use the T4 GPU. Falls back to CPU if no GPU detected. |

**What you see while it runs (healthy pattern):**

```
Using device: cuda
Loaded ~14000 samples across 274 classes
Train samples: ~12200, Val samples: ~850, Test samples (held out, untouched): ~900
Persisted split manifest to artifacts/splits.json
Input dimension detected: 628
Epoch 001 | train_loss=... val_loss=... train_f1=... val_f1=... lr=0.001000
...
```

The line **`Test samples (held out, untouched): N`** is the confirmation that
the source-level split from `splits.json` is in effect. If you see
`Train samples: N, Val samples: M` with no test count, you're running an old
copy of `modeling.py` — pull latest.

**Section purpose:** produce `artifacts/best_model.pt` with the best
`val_f1`-selected checkpoint.

---

## 4. Persist artifacts back to Drive

Colab disconnects after ~90 min idle / ~12 h wall; copy out periodically:

```python
import shutil, os
drive_dir = "/content/drive/MyDrive/visign/artifacts"
os.makedirs(drive_dir, exist_ok=True)
for f in ["best_model.pt", "training_history.json", "splits.json"]:
    src = f"/content/visign-ai-model/artifacts/{f}"
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(drive_dir, f))
        print("copied", f)
```

Re-run this cell whenever a meaningful checkpoint is written. The training
loop overwrites `best_model.pt` only when val F1 improves, so it's safe to
overwrite on Drive each time.

**Section purpose:** save your work so you don't lose it on disconnect.

---

## 5. Evaluate on the test split

```python
%cd /content/visign-ai-model
!python src/eval/evaluate.py \
    --checkpoint artifacts/best_model.pt \
    --index-csv index.csv \
    --feature-dir preprocessed_npz \
    --split test \
    --output-dir docs \
    --top-k 5 \
    --device cuda \
    --skip-latency
```

**Each flag:**

| Flag | What it does |
|---|---|
| `--checkpoint` | Path to a saved `best_model.pt`. |
| `--split test` | Use the held-out test rows (must be `train`, `val`, or `test`). |
| `--output-dir docs` | Writes `docs/EVAL_REPORT.md` + `docs/eval_summary.json` + `docs/eval_per_class.csv` + `docs/eval_confusion_matrix.npy`. |
| `--top-k 5` | Reports Top-1, Top-3, Top-5 accuracy. **Gap between Top-1 and Top-5 is the most diagnostic number** — see §6. |
| `--skip-latency` | Skip the inference-time micro-benchmark (only useful when you actually care about latency). |

**Section purpose:** turn a checkpoint into a Markdown report you can read.

---

## 6. How to read the results

Three numbers, in order of importance:

1. **Top-5 minus Top-1.** Should be `≥ 10 percentage points`. If they're
   equal, the model is confidently wrong on its misses — usually means too
   much overfitting to augmentation style. Fix: less augmentation,
   stronger regularisation.
2. **`val_f1` curve in `training_history.json`.** Should be monotone-ish
   upward for at least 20 epochs. If it plateaus by epoch 5–10 and `train_f1`
   keeps climbing to 1.0, you're overfitting hard. Fix: raise `--dropout`,
   raise `--weight-decay`, lower `--hidden-size`.
3. **Macro F1 on classes with support** vs **Macro F1 on all classes**.
   The all-classes number is dragged down by classes that have **no**
   val/test samples (each contributes F1 = 0). The supported number is the
   honest one for the evaluable subset.

**Sanity floor:** random baseline on 274 classes is `1/274 ≈ 0.36 %`. Any
Top-1 above ~10 % means the model has learned something.

**When you can trust the number:** confidence interval scales with
`1 / sqrt(test_n)`. At test_n = 18 a Top-1 of "44 %" has 95 % CI
[22 %, 67 %]; you can't tell a "great" run from a "bad" one. At
test_n ≈ 3000 the CI on 30 % shrinks to ±1.6 %. **Until you have at least
~1000 test samples, treat single-run results as anecdotes.** See §8 for how
to grow the test set with VOYA.

---

## 7. Common failures + fixes

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: Missing preprocessed feature files. Examples: preprocessed_npz/sample_0_0 _số không_.npy` | Tarball uploaded with `Compress-Archive` (.zip) instead of tar; Vietnamese filenames got mojibake-encoded by `unzip`. | Re-make the archive with `tar -czf` (see §2.1). To salvage an already-broken extraction, see the rename loop in `docs/notes.md`. |
| `ValueError: cannot reshape array of size 54256 into shape (150,628)` | A `.npy` was truncated during upload or write. | Run the bad-file scanner in `docs/notes.md`; replace bad files with another sample from the same class or zeros. |
| `TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'` | Cloned an older `modeling.py` from before the PyTorch 2.4 fix. | `git pull` on Colab. If still old, push your local changes (`git push origin main`) — Colab pulls from `origin/main`. |
| `Train samples: 12607, Val samples: 1401` (no test row) | Running the old `prepare_samples` that returns 2-tuple and ignores the `split` column. | Same as above — pull/push the latest `modeling.py` (3-tuple version honouring `splits.json`). |
| `error: untracked working tree files would be overwritten by merge: index.csv splits.json` on `git pull` | Colab has the same filenames as untracked files (from the tarball). | `rm -f index.csv splits.json && git pull`. They'll come back identical from `main`. |
| `unterminated string literal (detected at line 17)` | Colab's `!python -c "..."` shell-interprets newlines. | Run plain Python in the notebook cell (no `!python -c`), or join the lines with `;`. |
| HF: `Warning: You are sending unauthenticated requests` | No HF_TOKEN set. Harmless for public datasets but rate-limited. | Optional: add `HF_TOKEN` to Colab secrets, or just ignore. |

---

## 8. Expanding the eval set with VOYA_VSL (optional but recommended)

Your bottleneck is **not enough val/test samples per class**. The script
`src/keypoints/voya_import.py` pulls public VOYA_VSL samples for any class
that overlaps with yours and assigns them to val/test in `splits.json`.

### 8.1 Check overlap before pulling anything

```python
import json, urllib.request
voya = json.loads(urllib.request.urlopen(
    "https://huggingface.co/datasets/Kateht/VOYA_VSL/resolve/main/labels.json").read())
voya_names = {v.strip().lower() for v in voya.values()}
mine = {k.split("/", 1)[0].split(" _")[0].strip().lower()
        for k in json.load(open("splits.json"))["sources"]}
overlap = mine & voya_names
print(f"exact overlap: {len(overlap)} / {len(mine)} of your classes")
```

Decision rule:
- `overlap ≥ 80 classes` → worth doing
- `30–80 classes` → marginal but probably worth it
- `< 30 classes` → skip

### 8.2 Pull and convert

```python
%cd /content/visign-ai-model
!python src/keypoints/voya_import.py \
    --splits-json splits.json \
    --index-csv index.csv \
    --feature-dir preprocessed_npz \
    --n-val 20 --n-test 20 \
    --seed 42
```

**What this does (end-to-end — no `preprocess_pipeline.py` rerun needed):**

1. Downloads each overlapping VOYA class file (~340 MB each) from Hugging Face, then deletes it after sampling — peak disk under 1 GB.
2. Samples 40 random clips per class (20 val + 20 test).
3. Converts VOYA's `(60, 1605)` features into our 25-pose-landmark intermediate format, resamples 60 → 150 frames.
4. Runs `preprocess_pipeline.preprocess_sample` on each clip → produces the same 628-D feature vector the trainer expects.
5. Writes `preprocessed_npz/sample_<row_idx>_<label>.npy` for each clip.
6. **Appends** rows to `index.csv` (does NOT overwrite the existing QIPEDC rows).
7. Updates `splits.json` with the new val/test source assignments.

About 30 % of VOYA's label IDs 404 (the `labels.json` lists more entries than
actual files). These are skipped automatically. Expect ~150–160 successful
class files, ~6000 new val + test samples total.

**Re-run safe:** sources already present in `index.csv` are skipped.

### 8.3 ⚠️ DO NOT rerun `preprocess_pipeline.py` after `voya_import.py`

`preprocess_pipeline.py` rebuilds `index.csv` from whatever is in `augmented/`. On Colab, `augmented/` only contains the new VOYA dumps (or nothing) — running it would **delete all 14k QIPEDC rows** from `index.csv` and leave you with only the few hundred VOYA samples for training. This is the failure mode that caused `Train samples: 0, Val samples: 80` on first attempt.

Just go straight from §8.2 to §8.4 — `voya_import.py` already produced everything the trainer needs.

### 8.4 Sanity-check the new splits

```python
import pandas as pd
df = pd.read_csv("index.csv")
print("rows:", len(df))
print(df["split"].value_counts())
print("classes per split:")
print(df.groupby("split")["label"].nunique())
```

Expected after VOYA import:

```
rows: ~20000
split
train    ~14000     ← all your original QIPEDC rows still here
val       ~3000
test      ~3000
classes per split:
train    274
val      ~155
test     ~155
```

If `train` is in the hundreds, you accidentally reran `preprocess_pipeline.py` after `voya_import.py`. Recovery: re-extract `visign_features.tar.gz` from Drive to restore `index.csv` + `preprocessed_npz/`, then rerun §8.2.

Then retrain (§3) and re-evaluate (§5).

**Section purpose:** turn val/test from a 17-sample coin flip into a
3000-sample real metric.

---

## 9. Typical full-run wall-clock budget

For one Colab T4 session, starting cold:

| Step | Time |
|---|---|
| §1 Setup + repo clone + pip install | ~2 min |
| §2 Tarball download from Drive + extract | ~3 min |
| §3 Training, 60 epochs | ~20–40 min |
| §4 Copy artifacts to Drive | <1 min |
| §5 Eval | ~1 min |
| §8 VOYA import (first time only) | ~15 min |
| **Total cold-start** | **~45–80 min** |

Well under Colab's 12-hour ceiling. The slow step is whichever you do first
in a session — the tarball extract or the VOYA import.

---

## 10. Where to put what you learn

After each significant run, write a short note in `docs/notes.md`
recording:

- The exact command run (or commit SHA + flags)
- Best `val_f1` and at which epoch
- Test Top-1 / Top-5 / Macro F1 (supported)
- Any anomalies in the curves

Future-you will thank past-you.
