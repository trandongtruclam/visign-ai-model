import numpy as np
import os
import argparse
import random

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
                k_min=0.8, k_max=1.2, sigma_body=0.02, sigma_hand=0.015, sigma_face=0.01):
    """
    Augment a .npz keypoint file and save augmented versions.

    Args:
        input_path: path to the .npz file
        output_dir: directory to save results
        n_augmentations: number of augmented files to generate
        k_min, k_max: scaling factor range
        sigma_*: noise standard deviation
    """
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(input_path)
    pose = data['pose']
    left_hand = data['left_hand']
    right_hand = data['right_hand']
    face = data['face']

    print(f"Augmenting {input_path}...")
    print(f"Original shapes: pose={pose.shape}, left_hand={left_hand.shape}, right_hand={right_hand.shape}, face={face.shape}")

    original_output = os.path.join(output_dir, "0.npz")
    np.savez_compressed(original_output,
                        pose=pose,
                        left_hand=left_hand,
                        right_hand=right_hand,
                        face=face)
    print(f"  Saved original as 0.npz")

    for i in range(n_augmentations):
        random.seed(i)
        np.random.seed(i)

        pose_aug, left_hand_aug, right_hand_aug, face_aug = augment_keypoints(
            pose, left_hand, right_hand, face,
            k_min=k_min, k_max=k_max,
            sigma_body=sigma_body, sigma_hand=sigma_hand, sigma_face=sigma_face
        )

        output_path = os.path.join(output_dir, f"{i+1}.npz")
        np.savez_compressed(output_path,
                            pose=pose_aug,
                            left_hand=left_hand_aug,
                            right_hand=right_hand_aug,
                            face=face_aug)
        if (i + 1) % 5 == 0:
            print(f"  Created {i + 1}/{n_augmentations} augmentations")

    print(f"Done! Saved 1 original + {n_augmentations} augmented files to {output_dir}")

def process_folder(input_folder, output_folder, n_augmentations=10,
                  k_min=0.8, k_max=1.2, sigma_body=0.02, sigma_hand=0.015, sigma_face=0.01):
    """
    Process all .npz files in a folder (including subfolders) and augment them, preserving directory structure.
    """
    print(f"Processing folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    npz_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))

    print(f"Found {len(npz_files)} .npz files to augment")

    for idx, npz_file in enumerate(npz_files):
        rel_path = os.path.relpath(npz_file, input_folder)
        rel_dir = os.path.dirname(rel_path)
        output_file_dir = os.path.join(output_folder, rel_dir)
        os.makedirs(output_file_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(npz_file))[0]
        print(f"\n[{idx+1}/{len(npz_files)}] Processing: {rel_path}")

        augment_file(npz_file, output_file_dir, n_augmentations,
                     k_min, k_max, sigma_body, sigma_hand, sigma_face)

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

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input path {args.input} not found!")
        return

    if os.path.isfile(args.input):
        if not args.input.endswith('.npz'):
            print(f"Error: Input must be .npz file or folder!")
            return
        augment_file(args.input, args.output, args.n, 
                     args.kmin, args.kmax, 
                     args.sigma_body, args.sigma_hand, args.sigma_face)
    else:
        process_folder(args.input, args.output, args.n,
                       args.kmin, args.kmax,
                       args.sigma_body, args.sigma_hand, args.sigma_face)

if __name__ == "__main__":
    main()
