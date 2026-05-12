import json
import numpy as np
import os
import argparse
import random
import re

_LEGACY_NUMERIC_STEM = re.compile(r"^\d+$")


def _source_id_from_filename(filename: str) -> str:
    """Return the source-video id for a given .npz filename.

    Conventions:
        <src>.npz             -> "<src>"        (original from extractor)
        <src>__<aug_idx>.npz  -> "<src>"        (augmentation of <src>)
        <digits>.npz          -> "orig"         (legacy single-source layout)
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    if "__" in stem:
        return stem.split("__", 1)[0]
    if _LEGACY_NUMERIC_STEM.match(stem):
        return "orig"
    return stem


def _load_splits(splits_path):
    """Read splits.json (or return None when the path is not provided)."""
    if not splits_path:
        return None
    with open(splits_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("sources", payload)  # tolerate raw mapping

def add_noise(keypoints, sigma):
    """Add Gaussian noise to keypoints"""
    noise = np.random.normal(0, sigma, keypoints.shape)
    return keypoints + noise

def scale_keypoints(keypoints, scale_factor):
    """Scale keypoints around the center"""
    center = np.mean(keypoints, axis=0, keepdims=True)
    return (keypoints - center) * scale_factor + center


# ---------------------------------------------------------------------------
# Mirror augmentation
# ---------------------------------------------------------------------------
#
# After applying UPPER_BODY_INDEXES in the extractor the (T, 25, 3) pose
# array has the following per-keypoint layout (positions are array indices,
# NOT MediaPipe ids):
#
#   0:nose
#   1:l_eye_inner  2:l_eye  3:l_eye_outer
#   4:r_eye_inner  5:r_eye  6:r_eye_outer
#   7:l_ear        8:r_ear
#   9:mouth_l     10:mouth_r
#  11..16: left arm  (shoulder, elbow, wrist, pinky, index, thumb)
#  17..22: right arm (shoulder, elbow, wrist, pinky, index, thumb)
#  23:l_hip       24:r_hip
#
# On a horizontal mirror these index pairs must be swapped after flipping
# the x coordinate of every landmark.
POSE_MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6),                                            # eyes
    (7, 8),                                                            # ears
    (9, 10),                                                           # mouth
    (11, 17), (12, 18), (13, 19), (14, 20), (15, 21), (16, 22),        # arms
    (23, 24),                                                          # hips
]


def _flip_x(arr):
    """Flip the x coordinate in-place-safe (only operates on a copy)."""
    out = arr.copy()
    out[..., 0] = 1.0 - out[..., 0]
    return out


def _mirror_pose(pose):
    out = _flip_x(pose)
    for a, b in POSE_MIRROR_PAIRS:
        out[:, [a, b]] = out[:, [b, a]]
    return out


def _mirror_hands(left_hand, right_hand):
    """A horizontally-mirrored left hand becomes the new right hand, and
    vice versa. Within a hand the 21 anatomical landmark indices stay the
    same — they describe the same anatomical parts of what is now the
    opposite-side hand.
    """
    new_left = _flip_x(right_hand)
    new_right = _flip_x(left_hand)
    return new_left, new_right


def _mirror_face(face):
    """Approximate face mirror: flip x only. A fully anatomical mirror
    requires the MediaPipe FaceMesh symmetric-index table, which is large
    and undocumented for our face-subset. The face-subset features mostly
    encode non-manual markers (eyebrow / mouth state) that are largely
    left-right symmetric, so this approximation is acceptable.
    """
    return _flip_x(face)


def mirror_keypoints(pose, left_hand, right_hand, face):
    """Horizontal mirror of an entire clip (pose + both hands + face)."""
    pose_m = _mirror_pose(pose)
    lh_m, rh_m = _mirror_hands(left_hand, right_hand)
    face_m = _mirror_face(face)
    return pose_m, lh_m, rh_m, face_m


# ---------------------------------------------------------------------------
# Time-warp augmentation
# ---------------------------------------------------------------------------

def _hand_present_mask(hand):
    """Per-frame binary mask (1 = at least one landmark != 0). Operates on
    x/y coordinates only — z is ignored to stay consistent with
    preprocess_pipeline.hand_present_mask."""
    return (hand[:, :, :2].sum(axis=(1, 2)) != 0).astype(np.float32)


def _resample_xy(arr, new_idx, src_idx):
    """Linear interpolation along axis 0 for an array of shape (T, K, C)."""
    T_in, K, C = arr.shape
    out = np.empty((new_idx.shape[0], K, C), dtype=np.float32)
    for kk in range(K):
        for cc in range(C):
            out[:, kk, cc] = np.interp(new_idx, src_idx, arr[:, kk, cc])
    return out


def time_warp_keypoints(pose, left_hand, right_hand, face, k):
    """Stretch (k > 1) or compress (k < 1) the signing timeline by factor k.

    Output length matches the input length. Output frame t samples source
    frame t/k (clamped to [0, T-1]). Per-frame missing-hand status is
    propagated correctly: the source presence mask is linearly interpolated
    onto the new timeline and frames below 0.5 are forced back to zero.
    """
    T = pose.shape[0]
    src_idx = np.arange(T, dtype=np.float32)
    # output frame t reads from source position t / k
    new_idx = np.clip(src_idx / float(k), 0.0, T - 1.0).astype(np.float32)

    pose_w = _resample_xy(pose, new_idx, src_idx)
    face_w = _resample_xy(face, new_idx, src_idx)

    lh_present_src = _hand_present_mask(left_hand)
    rh_present_src = _hand_present_mask(right_hand)
    lh_present_w = np.interp(new_idx, src_idx, lh_present_src)
    rh_present_w = np.interp(new_idx, src_idx, rh_present_src)

    lh_w = _resample_xy(left_hand, new_idx, src_idx)
    rh_w = _resample_xy(right_hand, new_idx, src_idx)
    lh_w[lh_present_w < 0.5] = 0.0
    rh_w[rh_present_w < 0.5] = 0.0
    return pose_w, lh_w, rh_w, face_w


# ---------------------------------------------------------------------------
# Main augment_keypoints entry point
# ---------------------------------------------------------------------------

def augment_keypoints(pose, left_hand, right_hand, face,
                     k_min=0.8, k_max=1.2,
                     sigma_body=0.02, sigma_hand=0.015, sigma_face=0.01,
                     mirror_prob=0.0,
                     time_warp_prob=0.0, time_warp_min=0.85, time_warp_max=1.15):
    """Augment one keypoint clip.

    Pipeline (each step is independent and probabilistic):
        1. (prob `mirror_prob`)   horizontal mirror: x-flip + L/R swap
        2. (prob `time_warp_prob`) timeline rescale by factor k_t ~
           U[time_warp_min, time_warp_max]
        3. (always) global scale around the centroid by k_s ~ U[k_min, k_max]
        4. (always) per-channel Gaussian noise
        5. (always) clip x,y to [0,1]
        6. (always) restore originally-missing hand frames to zero so the
           downstream `hand_present_mask` keeps reporting the truth (this
           is the fix for the augmentation B1 bug — Gaussian noise on an
           all-zero "missing hand" otherwise turns it into a fictitious
           present hand and disables attention masking).

    Defaults preserve the previous behaviour (`mirror_prob=time_warp_prob=0`)
    except that step 6 is now always applied. Pass `mirror_prob=0.5
    time_warp_prob=0.7` for the recommended retraining recipe.
    """
    pose_aug = pose.copy().astype(np.float32)
    left_hand_aug = left_hand.copy().astype(np.float32)
    right_hand_aug = right_hand.copy().astype(np.float32)
    face_aug = face.copy().astype(np.float32)

    # 0. Record per-frame missing-hand masks BEFORE any transformation.
    lh_present = _hand_present_mask(left_hand_aug).astype(bool)
    rh_present = _hand_present_mask(right_hand_aug).astype(bool)

    # 1. Mirror
    if mirror_prob > 0 and random.random() < mirror_prob:
        pose_aug, left_hand_aug, right_hand_aug, face_aug = mirror_keypoints(
            pose_aug, left_hand_aug, right_hand_aug, face_aug
        )
        # The left/right semantics swapped — swap the masks accordingly.
        lh_present, rh_present = rh_present, lh_present

    # 2. Time-warp
    if time_warp_prob > 0 and random.random() < time_warp_prob:
        k_t = random.uniform(time_warp_min, time_warp_max)
        pose_aug, left_hand_aug, right_hand_aug, face_aug = time_warp_keypoints(
            pose_aug, left_hand_aug, right_hand_aug, face_aug, k_t
        )
        # The presence mask was already remapped inside time_warp_keypoints
        # (missing frames re-zeroed); recompute our boolean record from it.
        lh_present = _hand_present_mask(left_hand_aug).astype(bool)
        rh_present = _hand_present_mask(right_hand_aug).astype(bool)

    # 3. Global scale around centroid
    scale_factor = random.uniform(k_min, k_max)
    pose_aug = scale_keypoints(pose_aug, scale_factor)
    left_hand_aug = scale_keypoints(left_hand_aug, scale_factor)
    right_hand_aug = scale_keypoints(right_hand_aug, scale_factor)
    face_aug = scale_keypoints(face_aug, scale_factor)

    # 4. Per-channel Gaussian noise
    pose_aug = add_noise(pose_aug, sigma_body)
    left_hand_aug = add_noise(left_hand_aug, sigma_hand)
    right_hand_aug = add_noise(right_hand_aug, sigma_hand)
    face_aug = add_noise(face_aug, sigma_face)

    # 5. Clip x,y to [0,1] (z is left unclipped)
    pose_aug[..., :2] = np.clip(pose_aug[..., :2], 0.0, 1.0)
    left_hand_aug[..., :2] = np.clip(left_hand_aug[..., :2], 0.0, 1.0)
    right_hand_aug[..., :2] = np.clip(right_hand_aug[..., :2], 0.0, 1.0)
    face_aug[..., :2] = np.clip(face_aug[..., :2], 0.0, 1.0)

    # 6. Restore missing-hand frames to all-zero so hand_present_mask in
    #    preprocess_pipeline.py keeps reporting the truth.
    left_hand_aug[~lh_present] = 0.0
    right_hand_aug[~rh_present] = 0.0

    return pose_aug, left_hand_aug, right_hand_aug, face_aug

def augment_file(input_path, output_dir, n_augmentations=10,
                k_min=0.8, k_max=1.2, sigma_body=0.02, sigma_hand=0.015, sigma_face=0.01,
                source_id=None, skip_augmentation=False,
                mirror_prob=0.0,
                time_warp_prob=0.0, time_warp_min=0.85, time_warp_max=1.15):
    """
    Augment a .npz keypoint file and save augmented versions.

    Args:
        input_path: path to the .npz file
        output_dir: directory to save results
        n_augmentations: number of augmented files to generate
        k_min, k_max: scaling factor range
        sigma_*: noise standard deviation
        source_id: override the source-video identifier used in output
            filenames. Defaults to the basename of `input_path` without
            extension. The original is saved as ``<source_id>.npz`` and the
            i-th augmentation as ``<source_id>__<i>.npz``.
        skip_augmentation: when True, only the original is copied to
            `output_dir` (no perturbed samples are produced). Use this for
            val/test sources so the evaluation set is never augmented.
    """
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(input_path)
    pose = data['pose']
    left_hand = data['left_hand']
    right_hand = data['right_hand']
    face = data['face']

    if source_id is None:
        source_id = _source_id_from_filename(input_path)

    print(f"Augmenting {input_path} (source_id={source_id}, skip_aug={skip_augmentation})...")
    print(f"Original shapes: pose={pose.shape}, left_hand={left_hand.shape}, right_hand={right_hand.shape}, face={face.shape}")

    original_output = os.path.join(output_dir, f"{source_id}.npz")
    np.savez_compressed(original_output,
                        pose=pose,
                        left_hand=left_hand,
                        right_hand=right_hand,
                        face=face)
    print(f"  Saved original as {source_id}.npz")

    if skip_augmentation or n_augmentations <= 0:
        print(f"Done! Saved original only to {output_dir}")
        return

    for i in range(n_augmentations):
        random.seed(i)
        np.random.seed(i)

        pose_aug, left_hand_aug, right_hand_aug, face_aug = augment_keypoints(
            pose, left_hand, right_hand, face,
            k_min=k_min, k_max=k_max,
            sigma_body=sigma_body, sigma_hand=sigma_hand, sigma_face=sigma_face,
            mirror_prob=mirror_prob,
            time_warp_prob=time_warp_prob,
            time_warp_min=time_warp_min,
            time_warp_max=time_warp_max,
        )

        output_path = os.path.join(output_dir, f"{source_id}__{i+1}.npz")
        np.savez_compressed(output_path,
                            pose=pose_aug,
                            left_hand=left_hand_aug,
                            right_hand=right_hand_aug,
                            face=face_aug)
        if (i + 1) % 5 == 0:
            print(f"  Created {i + 1}/{n_augmentations} augmentations")

    print(f"Done! Saved 1 original + {n_augmentations} augmented files to {output_dir}")

def process_folder(input_folder, output_folder, n_augmentations=10,
                  k_min=0.8, k_max=1.2, sigma_body=0.02, sigma_hand=0.015, sigma_face=0.01,
                  splits=None, augment_splits=("train",),
                  mirror_prob=0.0,
                  time_warp_prob=0.0, time_warp_min=0.85, time_warp_max=1.15):
    """
    Process all .npz files in `input_folder` and augment them, preserving
    directory structure.

    When `splits` is provided (a mapping ``<label>/<source_id> -> "train"|
    "val"|"test"``) only sources whose split is listed in ``augment_splits``
    are augmented. Sources with any other split are *copied* (the original
    only) so that val/test sets remain pristine. Sources absent from the
    mapping are treated as ``train`` (i.e. augmented), so the function still
    works when no splits.json is available.
    """
    print(f"Processing folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    npz_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))

    print(f"Found {len(npz_files)} .npz files to augment")
    if splits is not None:
        print(f"Split-aware mode: augmenting sources in {augment_splits}")

    stats = {"augmented": 0, "copied_only": 0, "skipped_dupe": 0}
    seen_sources = set()
    for idx, npz_file in enumerate(npz_files):
        rel_path = os.path.relpath(npz_file, input_folder)
        rel_dir = os.path.dirname(rel_path)
        label = os.path.basename(rel_dir) if rel_dir else ""
        output_file_dir = os.path.join(output_folder, rel_dir)
        os.makedirs(output_file_dir, exist_ok=True)

        source_id = _source_id_from_filename(npz_file)
        source_key = f"{label}/{source_id}"

        # In the legacy layout a label folder contains 0.npz (original) and
        # 1..N.npz (augmentations of 0). All of them collapse to the same
        # source-id ("orig"), so we only process the folder once.
        if source_key in seen_sources:
            stats["skipped_dupe"] += 1
            continue
        seen_sources.add(source_key)

        split_name = splits.get(source_key, "train") if splits else "train"
        skip_aug = split_name not in augment_splits

        print(f"\n[{idx+1}/{len(npz_files)}] {rel_path}  split={split_name}")
        augment_file(
            npz_file,
            output_file_dir,
            n_augmentations,
            k_min, k_max, sigma_body, sigma_hand, sigma_face,
            source_id=source_id,
            skip_augmentation=skip_aug,
            mirror_prob=mirror_prob,
            time_warp_prob=time_warp_prob,
            time_warp_min=time_warp_min,
            time_warp_max=time_warp_max,
        )
        if skip_aug:
            stats["copied_only"] += 1
        else:
            stats["augmented"] += 1

    print(
        f"\nSummary: augmented={stats['augmented']}  copied_only={stats['copied_only']}  "
        f"legacy_dupe_skipped={stats['skipped_dupe']}"
    )

def main():
    parser = argparse.ArgumentParser(description='Augment keypoints data')
    parser.add_argument('input', help='Input .npz file path or folder')
    parser.add_argument('output', help='Output directory for augmented files')
    parser.add_argument('--n', type=int, default=10, help='Number of augmentations (default: 10)')
    parser.add_argument('--kmin', type=float, default=0.8, help='Min scaling factor (default: 0.8)')
    parser.add_argument('--kmax', type=float, default=1.2, help='Max scaling factor (default: 1.2)')
    parser.add_argument('--sigma_body', type=float, default=0.02, help='Body noise sigma (default: 0.02)')
    parser.add_argument('--sigma_hand', type=float, default=0.015, help='Hand noise sigma (default: 0.015)')
    parser.add_argument('--sigma_face', type=float, default=0.01, help='Face noise sigma (default: 0.01)')
    parser.add_argument('--splits', default=None,
                        help='Optional splits.json produced by split_sources.py. '
                             'Only sources in --augment-splits are augmented; the others '
                             'are copied as-is (original only).')
    parser.add_argument('--augment-splits', nargs='+', default=["train"],
                        help='Which splits to augment (default: train only).')
    parser.add_argument('--mirror-prob', type=float, default=0.0,
                        help='Probability of applying a horizontal mirror to each augmented '
                             'sample (default: 0.0 — off, for backward compatibility). '
                             'Recommended: 0.5.')
    parser.add_argument('--time-warp-prob', type=float, default=0.0,
                        help='Probability of applying a timeline rescale to each augmented '
                             'sample (default: 0.0). Recommended: 0.7.')
    parser.add_argument('--time-warp-min', type=float, default=0.85,
                        help='Lower bound of time-warp factor k (default: 0.85). '
                             'k<1 compresses the signing duration; k>1 stretches it.')
    parser.add_argument('--time-warp-max', type=float, default=1.15,
                        help='Upper bound of time-warp factor k (default: 1.15).')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input path {args.input} not found!")
        return

    splits = _load_splits(args.splits)

    aug_kwargs = dict(
        mirror_prob=args.mirror_prob,
        time_warp_prob=args.time_warp_prob,
        time_warp_min=args.time_warp_min,
        time_warp_max=args.time_warp_max,
    )

    if os.path.isfile(args.input):
        if not args.input.endswith('.npz'):
            print(f"Error: Input must be .npz file or folder!")
            return
        source_id = _source_id_from_filename(args.input)
        # For a single file we still honour --splits.
        label = os.path.basename(os.path.dirname(args.input))
        split_name = splits.get(f"{label}/{source_id}", "train") if splits else "train"
        skip_aug = split_name not in args.augment_splits
        augment_file(args.input, args.output, args.n,
                     args.kmin, args.kmax,
                     args.sigma_body, args.sigma_hand, args.sigma_face,
                     source_id=source_id, skip_augmentation=skip_aug,
                     **aug_kwargs)
    else:
        process_folder(args.input, args.output, args.n,
                       args.kmin, args.kmax,
                       args.sigma_body, args.sigma_hand, args.sigma_face,
                       splits=splits, augment_splits=tuple(args.augment_splits),
                       **aug_kwargs)

if __name__ == "__main__":
    main()
