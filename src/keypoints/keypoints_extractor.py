import numpy as np
import mediapipe as mp
import cv2
import os
import argparse
import pandas as pd
from scipy.interpolate import interp1d
from typing import List, Tuple

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

UPPER_BODY_INDEXES = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 15, 17, 19, 21,
    12, 14, 16, 18, 20, 22,
    23, 24]

def landmark_to_array(landmark_list, expected_n=33):
    arr = np.zeros((expected_n, 3), dtype=np.float32)
    if not landmark_list:
        return arr
    for i, landmark in enumerate(landmark_list):
        if i >= expected_n: 
            break
        arr[i, 0] = landmark.x
        arr[i, 1] = landmark.y
        arr[i, 2] = landmark.z
    return arr

def extract_keypoints(video_path, output_path, show_video=True):
    cap = cv2.VideoCapture(video_path)
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False) as holistic:
        
        all_keypoints = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            
            frame_keypoints = {
                'pose': np.zeros((25, 3), dtype=np.float32),
                'left_hand': np.zeros((21, 3), dtype=np.float32),
                'right_hand': np.zeros((21, 3), dtype=np.float32),
                'face': np.zeros((468, 3), dtype=np.float32)
            }
            
            if results.pose_landmarks:
                pose_landmarks = landmark_to_array(results.pose_landmarks.landmark, 33)
                frame_keypoints['pose'] = pose_landmarks[UPPER_BODY_INDEXES]
            
            if results.left_hand_landmarks:
                frame_keypoints['left_hand'] = landmark_to_array(results.left_hand_landmarks.landmark, 21)
            
            if results.right_hand_landmarks:
                frame_keypoints['right_hand'] = landmark_to_array(results.right_hand_landmarks.landmark, 21)
            
            if results.face_landmarks:
                frame_keypoints['face'] = landmark_to_array(results.face_landmarks.landmark, 468)
            
            all_keypoints.append(frame_keypoints)
            
            if show_video:
                annotated_frame = frame.copy()
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS)
                
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS)
                
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                
                cv2.imshow('Sign Language Recognition', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not all_keypoints:
            print(f"Warning: No keypoints extracted from {video_path}")
            return
        
        # Resample to 150 frames
        print(f"Resampling from {len(all_keypoints)} frames to 150 frames")
        all_keypoints = resample_keypoints(all_keypoints, target_frames=150)
            
        pose_data = np.stack([kp['pose'] for kp in all_keypoints])
        left_hand_data = np.stack([kp['left_hand'] for kp in all_keypoints])
        right_hand_data = np.stack([kp['right_hand'] for kp in all_keypoints])
        face_data = np.stack([kp['face'] for kp in all_keypoints])
        
        np.savez(output_path, 
                pose=pose_data,
                left_hand=left_hand_data,
                right_hand=right_hand_data,
                face=face_data)
        
        print(f"Keypoints saved to {output_path}")
        print(f"Pose shape: {pose_data.shape} (150 frames, 25 keypoints, 3 coords)")
        print(f"Left hand shape: {left_hand_data.shape} (150 frames, 21 keypoints, 3 coords)")
        print(f"Right hand shape: {right_hand_data.shape} (150 frames, 21 keypoints, 3 coords)")
        print(f"Face shape: {face_data.shape} (150 frames, 468 keypoints, 3 coords)")

# RESAMPLING KEYPOINTS
def resample_keypoints(keypoints_data, target_frames=150):
    current_frames = len(keypoints_data)
    if current_frames == target_frames:
        return keypoints_data
    
    # Create time indices
    old_indices = np.linspace(0, current_frames - 1, current_frames)
    new_indices = np.linspace(0, current_frames - 1, target_frames)
    
    resampled_data = []
    
    # Process each key type separately
    for key in ['pose', 'left_hand', 'right_hand', 'face']:
        # Get all frames for this key type
        all_frames_data = [kp[key] for kp in keypoints_data]
        
        # Convert to numpy array with proper shape
        all_frames_data = np.stack(all_frames_data)  # Shape: (frames, keypoints, 3)
        
        # Interpolate each keypoint coordinate across time
        resampled_coords = np.zeros((target_frames, all_frames_data.shape[1], all_frames_data.shape[2]))
        for i in range(all_frames_data.shape[1]):  # For each keypoint
            for j in range(all_frames_data.shape[2]):  # For each coordinate (x, y, z)
                y_values = all_frames_data[:, i, j]  # All frames for this keypoint coordinate
                f = interp1d(old_indices, y_values, kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
                resampled_coords[:, i, j] = f(new_indices)
        
        # Create resampled keypoints
        for t in range(target_frames):
            if t < len(resampled_data):
                resampled_data[t][key] = resampled_coords[t].astype(np.float32)
            else:
                resampled_data.append({key: resampled_coords[t].astype(np.float32)})
    
    return resampled_data

def sanitize_filename(filename):
    invalid_chars = '<>:"/\\|?*()'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.strip()

def process_dataset(videos_dir, labels_csv, output_dir):
    df = pd.read_csv(labels_csv)
    video_to_label = dict(zip(df['VIDEO'], df['LABEL']))
    
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    
    for i, video_file in enumerate(video_files):
        if video_file in video_to_label:
            label = video_to_label[video_file]
            safe_label = sanitize_filename(label)
            label_dir = os.path.join(output_dir, safe_label)
            os.makedirs(label_dir, exist_ok=True)
            
            video_path = os.path.join(videos_dir, video_file)
            output_path = os.path.join(label_dir, '0.npz')
            
            print(f"Processing {i+1}/{len(video_files)}: {video_file} -> {safe_label}/0.npz")
            extract_keypoints(video_path, output_path, show_video=False)
        else:
            print(f"Warning: No label found for {video_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', nargs='?', help='Path to input video file')
    parser.add_argument('output_path', nargs='?', help='Path to save keypoints (.npz file)')
    parser.add_argument('--process_dataset', action='store_true')
    parser.add_argument('--videos_dir', default='dataset/videos')
    parser.add_argument('--labels_csv', default='dataset/text/label.csv')
    parser.add_argument('--output_dir', default='dataset/keypoints')
    parser.add_argument('--no-display', action='store_true')

    args = parser.parse_args()

    if args.process_dataset:
        process_dataset(args.videos_dir, args.labels_csv, args.output_dir)
    else:
        if not args.video_path or not args.output_path:
            print("video_path and output_path required.")
            return
        if not os.path.exists(args.video_path):
            print(f"{args.video_path} not found")
            return
        extract_keypoints(args.video_path, args.output_path, not args.no_display)

if __name__ == "__main__":
    main()