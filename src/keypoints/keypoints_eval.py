import numpy as np
import cv2
import os
import argparse
import random
import pandas as pd

def load_keypoints(npz_path):
    """Load keypoints from .npz file"""
    data = np.load(npz_path)
    return {
        'pose': data['pose'],
        'left_hand': data['left_hand'],
        'right_hand': data['right_hand'],
        'face': data['face']
    }

def find_video_for_word(word_name):
    try:
        df = pd.read_csv('dataset/text/label.csv')
        video_row = df[df['LABEL'] == word_name]
        if len(video_row) > 0:
            video_file = video_row.iloc[0]['VIDEO']
            video_path = os.path.join('dataset/videos', video_file)
            if os.path.exists(video_path):
                return video_path
    except:
        pass
    return None

def draw_keypoints_on_video(frame, keypoints, frame_idx, color_offset=0):
    """
    Draw keypoints on a video frame with different colors for original/augmented.
    """
    h, w = frame.shape[:2]

    pose_color = (0, 255, 0) if color_offset == 0 else (0, 255, 255)
    left_color = (0, 0, 255) if color_offset == 0 else (0, 165, 255)
    right_color = (255, 0, 0) if color_offset == 0 else (255, 0, 255)

    pose = keypoints['pose'][frame_idx]
    for x, y, z in pose:
        if not (np.isnan(x) or np.isnan(y)):
            cv2.circle(frame, (int(x * w), int(y * h)), 4, pose_color, -1)
            cv2.circle(frame, (int(x * w), int(y * h)), 6, pose_color, 1)

    left_hand = keypoints['left_hand'][frame_idx]
    for x, y, z in left_hand:
        if not (np.isnan(x) or np.isnan(y)):
            cv2.circle(frame, (int(x * w), int(y * h)), 3, left_color, -1)
            cv2.circle(frame, (int(x * w), int(y * h)), 5, left_color, 1)

    right_hand = keypoints['right_hand'][frame_idx]
    for x, y, z in right_hand:
        if not (np.isnan(x) or np.isnan(y)):
            cv2.circle(frame, (int(x * w), int(y * h)), 3, right_color, -1)
            cv2.circle(frame, (int(x * w), int(y * h)), 5, right_color, 1)

def check_quality(keypoints):
    """
    Check quality of keypoints. Returns a list of issues if exists.
    """
    issues = []
    for key, data in keypoints.items():
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            issues.append(f"{key} has NaN/Inf")
    for key, data in keypoints.items():
        if np.any(data[..., :2] < 0) or np.any(data[..., :2] > 1):
            issues.append(f"{key} out of range [0,1]")
    left_x = np.mean(keypoints['left_hand'][..., 0])
    right_x = np.mean(keypoints['right_hand'][..., 0])
    if left_x > right_x:
        issues.append("Handedness flip detected")
    return issues

def show_video_comparison(original_kp, augmented_samples, video_path, word_name, n_samples=10):
    """
    Display video with keypoints overlays of original and augmented samples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    print("Controls:")
    print("  SPACE: pause/play")
    print("  'n': next augmented sample")
    print("  'q': quit")
    print("  's': save current frame")

    current_sample = 0
    paused = False
    frame_count = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break
            frame_count += 1

        frame_double = np.hstack([frame.copy(), frame.copy()])

        if frame_count < len(original_kp['pose']):
            draw_keypoints_on_video(frame_double[:, :width], original_kp, frame_count, 0)
            cv2.putText(frame_double[:, :width], "ORIGINAL", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if augmented_samples and frame_count < len(augmented_samples[current_sample]['pose']):
            draw_keypoints_on_video(frame_double[:, width:], augmented_samples[current_sample], frame_count, 1)
            cv2.putText(frame_double[:, width:], f"AUG {current_sample+1}/{len(augmented_samples)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame_double, f"Frame: {frame_count}/{total_frames}", (10, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_double, f"Sample: {current_sample+1}", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(f"Keypoints Evaluation - {word_name}", frame_double)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n'):
            current_sample = (current_sample + 1) % len(augmented_samples)
            print(f"Switched to sample {current_sample + 1}")
        elif key == ord('s'):
            cv2.imwrite(f"{word_name}_frame_{frame_count}_sample_{current_sample+1}.jpg", frame_double)
            print(f"Saved frame {frame_count}, sample {current_sample+1}")

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Evaluate augmented keypoints quality on video')
    parser.add_argument('word_name', help='Word name (e.g., "ai cho")')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of samples to display (default: 10)')

    args = parser.parse_args()

    word_name = args.word_name
    word_path = f"dataset/keypoints/{word_name}"
    augmented_path = f"augmented/{word_name}"

    print(f"=== VIDEO KEYPOINTS EVALUATION: {word_name} ===")

    video_path = find_video_for_word(word_name)
    if not video_path:
        print(f"Error: Video not found for word '{word_name}'")
        return
    print(f"Video: {os.path.basename(video_path)}")

    original_file = os.path.join(word_path, "0.npz")
    if not os.path.exists(original_file):
        print(f"Error: {original_file} not found!")
        return

    if not os.path.exists(augmented_path):
        print(f"Error: {augmented_path} not found!")
        print(f"Run: python src/keypoints/augment.py \"{word_path}\" \"{augmented_path}\" --n 10")
        return

    original_kp = load_keypoints(original_file)
    print(f"OK Original loaded: {original_kp['pose'].shape}")

    aug_files = [f for f in os.listdir(augmented_path) if f.endswith('.npz') and f != '0.npz']
    print(f"OK Found {len(aug_files)} augmented files")

    if len(aug_files) == 0:
        print("No augmented files found!")
        return

    augmented_samples = []
    issues_count = 0

    for aug_file in aug_files[:args.n_samples]:
        aug_path = os.path.join(augmented_path, aug_file)
        aug_kp = load_keypoints(aug_path)
        issues = check_quality(aug_kp)
        if issues:
            issues_count += 1
            print(f"BAD {aug_file}: {', '.join(issues)}")
        else:
            print(f"OK {aug_file}: OK")
        augmented_samples.append(aug_kp)

    print(f"\n=== SUMMARY ===")
    print(f"Word: {word_name}")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Augmented samples: {len(augmented_samples)}")
    print(f"Bad samples: {issues_count}")
    print(f"Bad rate: {issues_count/len(augmented_samples)*100:.1f}%")

    if issues_count / len(augmented_samples) > 0.1:
        print("WARNING: >10% samples have issues!")
    else:
        print("OK Quality looks good!")

    print(f"\nStarting video comparison...")
    show_video_comparison(original_kp, augmented_samples, video_path, word_name, args.n_samples)

if __name__ == "__main__":
    main()
