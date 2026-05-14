"""
src/keypoints/voya_import.py

Pull VOYA_VSL samples for classes that overlap with our dataset, compute the
full 628-D feature vector that the trainer expects, append rows to index.csv
and write `sample_<row_idx>_<label>.npy` files directly into
preprocessed_npz/. Also assigns the new sources to val/test in splits.json.

This is self-sufficient: it does NOT require `augmented/` to exist on disk
and you do NOT need to re-run `preprocess_pipeline.py` after it. The trainer
can pick up the new rows immediately.

Usage:
    python src/keypoints/voya_import.py \
        --splits-json splits.json \
        --index-csv index.csv \
        --feature-dir preprocessed_npz \
        --n-val 20 --n-test 20 --seed 42
"""
from __future__ import annotations
import argparse, json, os, re, sys, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.interpolate import interp1d

# Make sibling modules importable when run as `python src/keypoints/voya_import.py`
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the exact same feature engineering the trainer was trained on.
from src.train.preprocess_pipeline import preprocess_sample  # noqa: E402
from src.keypoints.keypoints_extractor import UPPER_BODY_INDEXES  # noqa: E402


VOYA_REPO = "Kateht/VOYA_VSL"
VOYA_LABELS_URL = f"https://huggingface.co/datasets/{VOYA_REPO}/resolve/main/labels.json"

POSE_SLICE = slice(0, 99)            # 33 landmarks * 3
LH_SLICE   = slice(99, 162)          # 21 * 3
RH_SLICE   = slice(162, 225)         # 21 * 3
FACE_SLICE = slice(225, 1605)        # 460 * 3
FACE_PAD_TO = 468                    # preprocess_sample expects (T, 468, 3)

TARGET_FRAMES = 150                  # preprocess_sample expects 150-frame inputs


def normalise_name(s: str) -> str:
    """Strip regional-variant suffixes and our `_word_` annotation markers."""
    s = s.strip().lower()
    for suf in (" (bắc)", " (nam)", " (trung)"):
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
    s = re.sub(r"\s*_[^_]+_\s*", " ", s).strip()
    return s


def resample(arr: np.ndarray, target: int) -> np.ndarray:
    """Linear-interp from (T, *) to (target, *) along axis 0."""
    T = arr.shape[0]
    if T == target:
        return arr.astype(np.float32)
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target)
    flat = arr.reshape(T, -1)
    out = np.empty((target, flat.shape[1]), dtype=np.float32)
    for j in range(flat.shape[1]):
        out[:, j] = interp1d(x_old, flat[:, j], kind="linear",
                             bounds_error=False, fill_value="extrapolate")(x_new)
    return out.reshape(target, *arr.shape[1:])


def convert_one_sample(seq_1605: np.ndarray) -> dict:
    """VOYA (60, 1605) → our intermediate `.npz` layout, matching what
    `keypoints_extractor.py` writes for QIPEDC clips. Pose is filtered to
    25 upper-body landmarks so `preprocess_sample` consumes it unchanged."""
    pose_33 = seq_1605[:, POSE_SLICE].reshape(-1, 33, 3)
    lh   = seq_1605[:, LH_SLICE].reshape(-1, 21, 3)
    rh   = seq_1605[:, RH_SLICE].reshape(-1, 21, 3)
    face_460 = seq_1605[:, FACE_SLICE].reshape(-1, 460, 3)

    pose_25 = pose_33[:, UPPER_BODY_INDEXES, :]
    face = np.zeros((face_460.shape[0], FACE_PAD_TO, 3), dtype=np.float32)
    face[:, :460] = face_460

    return {
        "pose":       resample(pose_25, TARGET_FRAMES).astype(np.float32),
        "left_hand":  resample(lh,      TARGET_FRAMES).astype(np.float32),
        "right_hand": resample(rh,      TARGET_FRAMES).astype(np.float32),
        "face":       resample(face,    TARGET_FRAMES).astype(np.float32),
    }


def sanitize_filename(filename: str) -> str:
    """Same scheme as keypoints_extractor.sanitize_filename, kept local to
    avoid pulling MediaPipe just to do string replacement."""
    invalid = '<>:"/\\|?*()'
    for c in invalid:
        filename = filename.replace(c, "_")
    return filename.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-json", default="splits.json")
    ap.add_argument("--index-csv",   default="index.csv")
    ap.add_argument("--feature-dir", default="preprocessed_npz")
    ap.add_argument("--keypoints-dir", default=None,
                    help="If set, also dump the intermediate (pose/lh/rh/face) "
                         ".npz files here for debugging. Skipped otherwise.")
    ap.add_argument("--n-val",  type=int, default=20)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--workdir", default="voya_cache")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    voya_labels = json.loads(urllib.request.urlopen(VOYA_LABELS_URL).read())
    voya_norm = {cid: normalise_name(name) for cid, name in voya_labels.items()}

    with open(args.splits_json, "r", encoding="utf-8") as f:
        splits = json.load(f)
    sources = splits["sources"]
    my_classes = sorted({k.split("/", 1)[0] for k in sources})
    my_norm = {normalise_name(c.split(" _")[0]): c for c in my_classes}

    matches = [(cid, my_norm[n]) for cid, n in voya_norm.items() if n in my_norm]
    print(f"Will pull {len(matches)} VOYA class files for "
          f"{len(set(m[1] for m in matches))} user classes")

    # Load existing index.csv so we can append. Determine the next row_idx.
    if os.path.exists(args.index_csv):
        existing_df = pd.read_csv(args.index_csv)
        next_row_idx = len(existing_df)
        existing_sources = set(existing_df.get("source_video", pd.Series(dtype=str)).tolist())
        print(f"Existing index.csv has {next_row_idx} rows.")
    else:
        existing_df = pd.DataFrame(columns=["filepath", "label", "source_video", "split"])
        next_row_idx = 0
        existing_sources = set()
        print("No existing index.csv; creating a fresh one.")

    feat_dir = Path(args.feature_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)
    if args.keypoints_dir:
        Path(args.keypoints_dir).mkdir(parents=True, exist_ok=True)

    n_per_file = args.n_val + args.n_test
    new_rows = []
    new_assignments = {}
    written_features = 0

    for i, (cid, user_label) in enumerate(matches, 1):
        try:
            local = hf_hub_download(repo_id=VOYA_REPO, repo_type="dataset",
                                    filename=f"Merged/{cid}.npz",
                                    local_dir=args.workdir)
        except Exception as e:
            print(f"[{i}/{len(matches)}] skip {cid} ({user_label}): download failed ({e})")
            continue

        with np.load(local) as d:
            seqs = d["sequences"]              # (N, 60, 1605)
        N = seqs.shape[0]
        if N < n_per_file:
            print(f"  warn: {cid} only has {N} samples; using all")
            chosen = np.arange(N)
        else:
            chosen = rng.choice(N, size=n_per_file, replace=False)

        safe_label = sanitize_filename(user_label)

        for j, idx in enumerate(chosen):
            kp = convert_one_sample(seqs[idx])
            source_id = f"VOYA_{cid}_{int(idx):04d}"
            source_key = f"{user_label}/{source_id}"

            # Skip if this source is already in the index (re-run safety)
            if source_key in existing_sources or source_key in new_assignments:
                continue

            # Optionally dump intermediate .npz for debugging
            if args.keypoints_dir:
                npz_dir = Path(args.keypoints_dir) / safe_label
                npz_dir.mkdir(parents=True, exist_ok=True)
                np.savez(npz_dir / f"{source_id}.npz",
                         pose=kp["pose"], left_hand=kp["left_hand"],
                         right_hand=kp["right_hand"], face=kp["face"])

            # Compute features in-memory using the same logic the trainer expects.
            # preprocess_sample reads from a path, so write a small temp .npz.
            tmp_npz = feat_dir / f".__tmp_voya_{cid}_{idx}.npz"
            np.savez(tmp_npz,
                     pose=kp["pose"], left_hand=kp["left_hand"],
                     right_hand=kp["right_hand"], face=kp["face"])
            try:
                feat = preprocess_sample(str(tmp_npz))
            finally:
                try: os.remove(tmp_npz)
                except OSError: pass

            row_idx = next_row_idx
            next_row_idx += 1
            out_npy = feat_dir / f"sample_{row_idx}_{safe_label}.npy"
            np.save(out_npy, feat)
            written_features += 1

            assigned_split = "val" if j < args.n_val else "test"
            new_rows.append({
                "filepath": f"VOYA/{safe_label}/{source_id}.npz",
                "label": user_label,
                "source_video": source_key,
                "split": assigned_split,
            })
            new_assignments[source_key] = assigned_split

        print(f"[{i}/{len(matches)}] {user_label}: wrote {len(chosen)} samples "
              f"(features so far: {written_features})")

        try: os.remove(local)
        except OSError: pass

    # Persist updated index.csv (append new rows; preserve original order)
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        # Some old index.csv versions don't have a 'split' column; coerce.
        if "split" not in existing_df.columns:
            combined["split"] = combined["split"].fillna("train")
        combined.to_csv(args.index_csv, index=False)
        print(f"Appended {len(new_rows)} rows to {args.index_csv} "
              f"(total: {len(combined)})")
    else:
        print("No new rows to add (everything was already present).")

    sources.update(new_assignments)
    splits["voya_added"] = len(new_assignments) + splits.get("voya_added", 0)
    with open(args.splits_json, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    print(f"Updated {args.splits_json} with {len(new_assignments)} new source assignments.")


if __name__ == "__main__":
    main()
