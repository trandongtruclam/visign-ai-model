import os
import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Build index CSV mapping filepath,label
def build_index_csv(data_dir, out_csv):
    rows = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for f in glob.glob(os.path.join(label_dir, '*.npz')):
            rows.append({'filepath': f, 'label': label})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved index to {out_csv}, total: {len(df)} samples.")

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
    d = np.load(npz_path)
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
    # Step 1: Build index CSV (run once)
    build_index_csv("augmented", "index.csv")

    # Step 2: Preprocess single sample demo
    # sample_path = "augmented/ai cho/0.npz"
    # feat = preprocess_sample(sample_path)
    # print(f"Shape: {feat.shape} (frames, features)")

    # Step 3: Batch process entire dataset
    df = pd.read_csv("index.csv")
    out_dir = "preprocessed_npz"
    os.makedirs(out_dir, exist_ok=True)
    for i, row in df.iterrows():
        fpath = row['filepath']
        label = row['label']
        feat = preprocess_sample(fpath)
        out_path = os.path.join(out_dir, f"sample_{i}_{label}.npy")
        np.save(out_path, feat)
        if i % 100 == 0:
            print(f"Processed {i}/{len(df)}")
    pass