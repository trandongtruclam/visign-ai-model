import json
import os
import glob
import re
import zipfile
import zlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# Errors np.load / npz access can raise on a corrupted or truncated file.
# Keep this tuple in sync with the `except` clauses below.
_NPZ_LOAD_ERRORS = (
    zipfile.BadZipFile,
    zlib.error,
    OSError,
    EOFError,
    ValueError,
    KeyError,
)

_LEGACY_NUMERIC_STEM = re.compile(r"^\d+$")


def _source_id_from_filename(filename):
    """Map an augmented .npz filename to its underlying source-video id.

    See ``src/keypoints/split_sources.py`` for the naming conventions.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    if "__" in stem:
        return stem.split("__", 1)[0]
    if _LEGACY_NUMERIC_STEM.match(stem):
        return "orig"
    return stem


# Build index CSV mapping filepath,label,source_video[,split]
def build_index_csv(data_dir, out_csv, splits_json=None):
    """Walk `data_dir` and emit an index of augmented samples.

    Columns:
        filepath      : absolute path to the .npz feature file
        label         : class label (folder name)
        source_video  : unique source-video id ``<label>/<source_id>``
        split         : "train" / "val" / "test" — only present when
                        `splits_json` is provided
    """
    splits = None
    if splits_json and os.path.exists(splits_json):
        with open(splits_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        splits = payload.get("sources", payload)

    rows = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for f in glob.glob(os.path.join(label_dir, '*.npz')):
            source_id = _source_id_from_filename(f)
            source_key = f"{label}/{source_id}"
            row = {"filepath": f, "label": label, "source_video": source_key}
            if splits is not None:
                # Default unassigned sources to "train" so the column is
                # always populated.
                row["split"] = splits.get(source_key, "train")
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved index to {out_csv}, total: {len(df)} samples.")
    if "split" in df.columns:
        print("Split counts:")
        print(df["split"].value_counts().to_string())

# Select face subset (lips + eyes + brows) or PCA-reduced
FACE_IDX_SUBSET = list(range(61, 88)) + list(range(246, 276)) + list(range(300, 332))  # lips + eyes + brows

def extract_face_subset(face_arr, use_pca=False, n_pca=30):
    if use_pca:
        flat = face_arr.reshape(-1, 468*3)
        pca = PCA(n_components=n_pca)
        reduced = pca.fit_transform(flat)
        return reduced.reshape(face_arr.shape[0], n_pca)
    subset = face_arr[:, FACE_IDX_SUBSET, :2]  # only x, y
    return subset.reshape(face_arr.shape[0], -1)

# Center and scale all keypoints per frame by shoulders
POSE_L_SH_IDX, POSE_R_SH_IDX = 11, 12

def center_and_scale(pose, left_hand, right_hand, face):
    left_sh = pose[:, POSE_L_SH_IDX, :2]
    right_sh = pose[:, POSE_R_SH_IDX, :2]
    center = 0.5 * (left_sh + right_sh)
    pose_centered = pose[:, :, :2] - center[:, None, :]
    lh_centered = left_hand[:, :, :2] - center[:, None, :]
    rh_centered = right_hand[:, :, :2] - center[:, None, :]
    face_centered = face[:, :, :2] - center[:, None, :]
    # mean scale by shoulder distance
    dist = np.linalg.norm(left_sh - right_sh, axis=1).mean()
    pose_norm = pose_centered / (dist + 1e-6)
    lh_norm = lh_centered / (dist + 1e-6)
    rh_norm = rh_centered / (dist + 1e-6)
    face_norm = face_centered / (dist + 1e-6)
    return pose_norm, lh_norm, rh_norm, face_norm

# Detect missing hand (returns mask: 1 if present, 0 if all-zeros)
def hand_present_mask(hand):
    return (hand[:, :, :2].sum(axis=(1,2)) != 0).astype(np.float32)

# Preprocess one npz sample to feature sequence
def preprocess_sample(npz_path, use_pca=False, n_pca=30, add_velocity=True):
    # Use a context manager so the underlying zipfile handle is closed even
    # if a per-key decompression fails partway through a 14k-file run.
    with np.load(npz_path) as d:
        pose = d['pose']           # (150, 25, 3)
        lh_raw = d['left_hand']   # (150, 21, 3)
        rh_raw = d['right_hand']  # (150, 21, 3)
        face = d['face']           # (150, 468, 3)

    # Check hand presence before normalization
    lh_mask = hand_present_mask(lh_raw)
    rh_mask = hand_present_mask(rh_raw)
    
    # Normalize keypoints
    pose, lh, rh, face = center_and_scale(pose, lh_raw, rh_raw, face)
    
    # Extract features
    face_feat = extract_face_subset(face, use_pca, n_pca)
    pose_feat = pose.reshape(pose.shape[0], -1)
    lh_feat = lh.reshape(lh.shape[0], -1)
    rh_feat = rh.reshape(rh.shape[0], -1)
    feat = np.concatenate([pose_feat, lh_feat, rh_feat, face_feat], axis=-1)
    
    # Add hand presence masks
    feat = np.concatenate([feat, lh_mask[:, None], rh_mask[:, None]], axis=-1)
    
    # Clip values
    feat = np.clip(feat, -1.5, 1.5)
    
    # Add velocity if requested
    if add_velocity:
        vel = np.diff(feat, axis=0, prepend=feat[[0], :])
        feat = np.concatenate([feat, vel], axis=-1)
    
    return feat  # shape: (frames, dims)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build index.csv and preprocess augmented keypoints to feature .npy files"
    )
    parser.add_argument("--data-dir", default="augmented",
                        help="Directory containing augmented .npz files (<label>/<src>__<i>.npz)")
    parser.add_argument("--index-csv", default="index.csv")
    parser.add_argument("--feature-dir", default="preprocessed_npz")
    parser.add_argument("--splits-json", default=None,
                        help="Optional splits.json from split_sources.py to add a `split` column")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Only build index.csv, skip feature extraction")
    parser.add_argument("--strict", action="store_true",
                        help="Abort on the first unreadable .npz instead of "
                             "skipping it (default: skip + log to "
                             "<index-csv>.failed.csv).")
    args = parser.parse_args()

    build_index_csv(args.data_dir, args.index_csv, splits_json=args.splits_json)

    if args.no_preprocess:
        raise SystemExit(0)

    df = pd.read_csv(args.index_csv)
    out_dir = args.feature_dir
    os.makedirs(out_dir, exist_ok=True)

    # We rewrite index.csv at the end so that its row order matches the
    # `sample_<i>_<label>.npy` filenames downstream (modeling.py joins on the
    # row index). A running counter `new_i` is used for the filename so that
    # skipped/corrupted rows do not leave gaps in the numbering.
    kept_rows = []
    failed = []
    total = len(df)
    for orig_i, row in df.iterrows():
        fpath = row['filepath']
        label = row['label']
        try:
            feat = preprocess_sample(fpath)
        except _NPZ_LOAD_ERRORS as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"  [SKIP] row={orig_i} {fpath} -> {msg}", flush=True)
            failed.append({"orig_row": orig_i, "filepath": fpath, "error": msg})
            if args.strict:
                raise
            continue

        new_i = len(kept_rows)
        out_path = os.path.join(out_dir, f"sample_{new_i}_{label}.npy")
        np.save(out_path, feat)
        kept_rows.append(row)
        if (new_i + 1) % 100 == 0:
            print(
                f"Processed {new_i + 1}/{total}"
                f" (failures so far: {len(failed)})",
                flush=True,
            )

    kept_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    kept_df.to_csv(args.index_csv, index=False)
    print(
        f"\nDone. Wrote features for {len(kept_df)}/{total} samples."
        f" Skipped {len(failed)} unreadable file(s)."
    )
    if "split" in kept_df.columns:
        print("Split counts (after skip):")
        print(kept_df["split"].value_counts().to_string())
    if failed:
        failed_csv = os.path.splitext(args.index_csv)[0] + ".failed.csv"
        pd.DataFrame(failed).to_csv(failed_csv, index=False)
        print(
            f"Failed files logged to {failed_csv}."
            " Re-run augment.py on the affected source(s) to recover them."
        )