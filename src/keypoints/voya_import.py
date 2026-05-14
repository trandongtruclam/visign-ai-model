"""
src/keypoints/voya_import.py

Pull VOYA_VSL samples for classes that overlap with our dataset,
convert to our native (150, K, 3) per-stream .npz format, and assign them
to val/test in splits.json.
"""
from __future__ import annotations
import argparse, json, os, re, sys, unicodedata
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from scipy.interpolate import interp1d
import urllib.request

VOYA_REPO = "Kateht/VOYA_VSL"
VOYA_LABELS_URL = f"https://huggingface.co/datasets/{VOYA_REPO}/resolve/main/labels.json"

POSE_SLICE = slice(0, 99)            # 33 landmarks * 3
LH_SLICE   = slice(99, 162)          # 21 * 3
RH_SLICE   = slice(162, 225)         # 21 * 3
FACE_SLICE = slice(225, 1605)        # 460 * 3
FACE_PAD_TO = 468                    # your pipeline expects 468

TARGET_FRAMES = 150                  # your pipeline expects 150


def normalise_name(s: str) -> str:
    """Strip regional variant suffixes and user-style annotations to compare names."""
    s = s.strip().lower()
    for suf in (" (bắc)", " (nam)", " (trung)"):
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
    # collapse any "_word_" annotation marker
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
    """Take one VOYA (60, 1605) sample and return our (150, *, 3) dict."""
    pose = seq_1605[:, POSE_SLICE].reshape(-1, 33, 3)
    lh   = seq_1605[:, LH_SLICE].reshape(-1, 21, 3)
    rh   = seq_1605[:, RH_SLICE].reshape(-1, 21, 3)
    face_460 = seq_1605[:, FACE_SLICE].reshape(-1, 460, 3)

    # pad face to 468 with zeros so existing FACE_IDX_SUBSET (max idx 331) works
    face = np.zeros((face_460.shape[0], FACE_PAD_TO, 3), dtype=np.float32)
    face[:, :460] = face_460

    return {
        "pose":       resample(pose, TARGET_FRAMES).astype(np.float32),
        "left_hand":  resample(lh,   TARGET_FRAMES).astype(np.float32),
        "right_hand": resample(rh,   TARGET_FRAMES).astype(np.float32),
        "face":       resample(face, TARGET_FRAMES).astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-json", default="splits.json")
    ap.add_argument("--keypoints-dir", default="dataset/keypoints")
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
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
    # If multiple VOYA classes map to the same user class (regional variants),
    # we keep all of them; samples from different variants all count.
    print(f"Will pull {len(matches)} VOYA class files for {len(set(m[1] for m in matches))} user classes")

    n_per_file = args.n_val + args.n_test
    out_root = Path(args.keypoints_dir)
    new_assignments = {}

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

        label_dir = out_root / user_label
        label_dir.mkdir(parents=True, exist_ok=True)

        for j, idx in enumerate(chosen):
            kp = convert_one_sample(seqs[idx])
            source_id = f"VOYA_{cid}_{int(idx):04d}"
            out_path = label_dir / f"{source_id}.npz"
            np.savez(out_path,
                     pose=kp["pose"], left_hand=kp["left_hand"],
                     right_hand=kp["right_hand"], face=kp["face"])
            source_key = f"{user_label}/{source_id}"
            new_assignments[source_key] = "val" if j < args.n_val else "test"

        print(f"[{i}/{len(matches)}] {user_label}: wrote {len(chosen)} samples")
        # Don't keep VOYA's bulky source npz around once we've extracted
        try:
            os.remove(local)
        except OSError:
            pass

    # Merge into splits.json
    sources.update(new_assignments)
    splits["voya_added"] = len(new_assignments)
    with open(args.splits_json, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    print(f"Added {len(new_assignments)} VOYA sources; updated {args.splits_json}")


if __name__ == "__main__":
    main()
