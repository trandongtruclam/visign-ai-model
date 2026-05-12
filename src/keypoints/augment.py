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

def augment_keypoints(pose, left_hand, right_hand, face, 
                     k_min=0.8, k_max=1.2, 
                     sigma_body=0.02, sigma_hand=0.015, sigma_face=0.01):
    """
    Augment keypoints using vector-based scaling and Gaussian noise.

    Args:
        pose: (N, 25, 3) - pose keypoints
        left_hand: (N, 21, 3) - left hand keypoints  
        right_hand: (N, 21, 3) - right hand keypoints
        face: (N, 468, 3) - face keypoints
        k_min, k_max: range for scaling factor
        sigma_*: standard deviation for noise
    """
    pose_aug = pose.copy().astype(np.float32)
    left_hand_aug = left_hand.copy().astype(np.float32)
    right_hand_aug = right_hand.copy().astype(np.float32)
    face_aug = face.copy().astype(np.float32)

    scale_factor = random.uniform(k_min, k_max)

    pose_aug = scale_keypoints(pose_aug, scale_factor)
    left_hand_aug = scale_keypoints(left_hand_aug, scale_factor)
    right_hand_aug = scale_keypoints(right_hand_aug, scale_factor)
    face_aug = scale_keypoints(face_aug, scale_factor)

    pose_aug = add_noise(pose_aug, sigma_body)
    left_hand_aug = add_noise(left_hand_aug, sigma_hand)
    right_hand_aug = add_noise(right_hand_aug, sigma_hand)
    face_aug = add_noise(face_aug, sigma_face)

    pose_aug[..., :2] = np.clip(pose_aug[..., :2], 0.0, 1.0)
    left_hand_aug[..., :2] = np.clip(left_hand_aug[..., :2], 0.0, 1.0)
    right_hand_aug[..., :2] = np.clip(right_hand_aug[..., :2], 0.0, 1.0)
    face_aug[..., :2] = np.clip(face_aug[..., :2], 0.0, 1.0)

    return pose_aug, left_hand_aug, right_hand_aug, face_aug

def augment_file(input_path, output_dir, n_augmentations=10,
                k_min=0.8, k_max=1.2, sigma_body=0.02, sigma_hand=0.015, sigma_face=0.01,
                source_id=None, skip_augmentation=False):
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
            sigma_body=sigma_body, sigma_hand=sigma_hand, sigma_face=sigma_face
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
                  splits=None, augment_splits=("train",)):
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

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input path {args.input} not found!")
        return

    splits = _load_splits(args.splits)

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
                     source_id=source_id, skip_augmentation=skip_aug)
    else:
        process_folder(args.input, args.output, args.n,
                       args.kmin, args.kmax,
                       args.sigma_body, args.sigma_hand, args.sigma_face,
                       splits=splits, augment_splits=tuple(args.augment_splits))

if __name__ == "__main__":
    main()
