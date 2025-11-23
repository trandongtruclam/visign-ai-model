"""FastAPI web application for sign language video training."""

import csv
import io
import base64
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

import torch
import mediapipe as mp
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.train.modeling import LSTMClassifier
from src.train.preprocess_pipeline import (
    center_and_scale,
    extract_face_subset,
    hand_present_mask
)

app = FastAPI(title="sudo-visign Web App")

# Mount static files directory
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates directory
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# Constants
POSE_LEN = 25
HAND_LEN = 21
FACE_LEN = 468
TARGET_FRAMES = 150
UPPER_BODY_INDEXES = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 15, 17, 19, 21,
    12, 14, 16, 18, 20, 22,
    23, 24
]

# Global model bundle
model_bundle = None
# Cache MediaPipe holistic instance for reuse
_holistic_cache = None


def load_model_bundle(checkpoint_path: Path, json_path: Optional[Path] = None):
    """
    Load model from checkpoint with optional JSON label mapping.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt or .pth)
        json_path: Optional path to label_mapping.json (for validation/fallback)
    """
    global model_bundle
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("model_config", {})
    label2idx = checkpoint.get("label2idx", {})
    
    # Nếu checkpoint không có label2idx, thử load từ JSON
    if not label2idx and json_path and json_path.exists():
        print(f"Warning: Checkpoint không có label2idx, đang load từ JSON: {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                label2idx = json.load(f)
            print(f"Loaded {len(label2idx)} labels from JSON")
        except Exception as e:
            print(f"Error loading JSON: {e}")
            raise ValueError("Không thể load label2idx từ checkpoint hoặc JSON")
    elif not label2idx:
        raise ValueError("Checkpoint không chứa label2idx và không có JSON fallback")
    
    # Validate với JSON nếu có
    if json_path and json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_label2idx = json.load(f)
            
            if json_label2idx != label2idx:
                print(f"Warning: JSON mapping khác với checkpoint!")
                print(f"  Checkpoint: {len(label2idx)} labels")
                print(f"  JSON: {len(json_label2idx)} labels")
                print(f"  → Sử dụng mapping từ checkpoint (chính xác hơn)")
            else:
                print(f"✓ JSON mapping khớp với checkpoint ({len(label2idx)} labels)")
        except Exception as e:
            print(f"Warning: Không thể validate với JSON: {e}")
    
    model = LSTMClassifier(
        in_feat=config.get("in_feat"),
        proj_dim=config.get("proj_dim", 256),
        hidden_size=config.get("hidden_size", 256),
        num_layers=config.get("num_layers", 2),
        bidirectional=config.get("bidirectional", True),
        dropout=config.get("dropout", 0.35),
        num_classes=config.get("num_classes", len(label2idx) or 1),
        use_attention=config.get("use_attention", True),
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    idx_to_label = {idx: label for label, idx in label2idx.items()}
    input_dim = config.get("in_feat")
    
    model_bundle = {
        "model": model,
        "idx_to_label": idx_to_label,
        "label2idx": label2idx,  # Lưu thêm để dùng cho API
        "input_dim": input_dim,
        "device": device
    }
    
    print(f"Model loaded successfully. Device: {device}, Classes: {len(idx_to_label)}")


def landmark_to_array(landmark_list, expected_n: int) -> np.ndarray:
    """Convert MediaPipe landmarks to numpy array."""
    arr = np.zeros((expected_n, 3), dtype=np.float32)
    if not landmark_list:
        return arr
    for idx, landmark in enumerate(landmark_list):
        if idx >= expected_n:
            break
        arr[idx, 0] = landmark.x
        arr[idx, 1] = landmark.y
        arr[idx, 2] = landmark.z
    return arr


def resample_keypoints(sequence: List[Dict[str, np.ndarray]], target_frames: int = TARGET_FRAMES) -> List[Dict[str, np.ndarray]]:
    """Resample keypoint sequence to target number of frames."""
    num_frames = len(sequence)
    if num_frames == 0:
        raise ValueError("No frames captured for resampling.")
    if num_frames == target_frames:
        return sequence
    if num_frames == 1:
        single_frame = sequence[0]
        return [{k: v.copy() for k, v in single_frame.items()} for _ in range(target_frames)]
    
    old_idx = np.linspace(0.0, num_frames - 1, num_frames, dtype=np.float32)
    new_idx = np.linspace(0.0, num_frames - 1, target_frames, dtype=np.float32)
    
    keys = sequence[0].keys()
    stacked = {key: np.stack([frame[key] for frame in sequence], axis=0) for key in keys}
    resampled = {}
    
    for key, data in stacked.items():
        flat = data.reshape(num_frames, -1)
        interp = np.empty((target_frames, flat.shape[1]), dtype=np.float32)
        for col in range(flat.shape[1]):
            interp[:, col] = np.interp(new_idx, old_idx, flat[:, col])
        resampled[key] = interp.reshape(target_frames, data.shape[1], data.shape[2])
    
    output = []
    for t in range(target_frames):
        output.append({key: resampled[key][t].astype(np.float32) for key in keys})
    return output


def build_feature_sequence(
    pose: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    face: np.ndarray,
    add_velocity: bool = True,
) -> Dict[str, np.ndarray]:
    """Build feature sequence from keypoints."""
    lh_mask = hand_present_mask(left_hand)
    rh_mask = hand_present_mask(right_hand)
    
    pose_norm, lh_norm, rh_norm, face_norm = center_and_scale(pose, left_hand, right_hand, face)
    
    pose_feat = pose_norm.reshape(pose_norm.shape[0], -1)
    lh_feat = lh_norm.reshape(lh_norm.shape[0], -1)
    rh_feat = rh_norm.reshape(rh_norm.shape[0], -1)
    face_feat = extract_face_subset(face_norm, use_pca=False)
    
    feat = np.concatenate([pose_feat, lh_feat, rh_feat, face_feat], axis=-1)
    feat = np.concatenate([feat, lh_mask[:, None], rh_mask[:, None]], axis=-1)
    feat = np.clip(feat, -1.5, 1.5)
    
    if add_velocity:
        velocity = np.diff(feat, axis=0, prepend=feat[[0], :])
        feat = np.concatenate([feat, velocity], axis=-1)
    
    frame_mask = np.maximum(lh_mask, rh_mask).astype(np.float32)
    if frame_mask.sum() == 0:
        frame_mask[:] = 1.0
    
    return {"features": feat.astype(np.float32), "frame_mask": frame_mask}


def process_video(video_bytes: bytes) -> Dict:
    """Process video and return predictions."""
    global model_bundle
    
    if model_bundle is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Create temporary file for video
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    
    cap = cv2.VideoCapture(tmp_path)
    
    # Get video properties for optimization
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Optimize: resize frames to smaller size for faster processing
    TARGET_WIDTH = 640
    TARGET_HEIGHT = 360
    
    # Reuse cached holistic instance for better performance
    global _holistic_cache
    if _holistic_cache is None:
        _holistic_cache = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,  # Reduce complexity for speed (0=fastest, 2=most accurate)
            enable_segmentation=False,
            refine_face_landmarks=False,
            smooth_landmarks=True,
        )
    holistic = _holistic_cache
    
    keypoint_buffer = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        results = holistic.process(frame_rgb)
        
        kp = {
            "pose": landmark_to_array(
                results.pose_landmarks.landmark if results.pose_landmarks else [],
                expected_n=33
            )[UPPER_BODY_INDEXES],
            "left_hand": landmark_to_array(
                results.left_hand_landmarks.landmark if results.left_hand_landmarks else [],
                HAND_LEN
            ),
            "right_hand": landmark_to_array(
                results.right_hand_landmarks.landmark if results.right_hand_landmarks else [],
                HAND_LEN
            ),
            "face": landmark_to_array(
                results.face_landmarks.landmark if results.face_landmarks else [],
                FACE_LEN
            ),
        }
        keypoint_buffer.append(kp)
        frame_count += 1
    
    cap.release()
    # Don't close holistic - keep it cached for next request
    Path(tmp_path).unlink()  # Delete temp file
    
    if not keypoint_buffer:
        raise HTTPException(status_code=400, detail="No frames extracted from video")
    
    # Resample
    resampled = resample_keypoints(keypoint_buffer, TARGET_FRAMES)
    
    # Stack
    pose = np.stack([frame["pose"] for frame in resampled], axis=0)
    left_hand = np.stack([frame["left_hand"] for frame in resampled], axis=0)
    right_hand = np.stack([frame["right_hand"] for frame in resampled], axis=0)
    face = np.stack([frame["face"] for frame in resampled], axis=0)
    
    # Build features
    seq = build_feature_sequence(pose, left_hand, right_hand, face, add_velocity=True)
    features = seq["features"]
    frame_mask = seq["frame_mask"]
    
    if features.shape[1] != model_bundle["input_dim"]:
        raise HTTPException(
            status_code=400,
            detail=f"Feature dimension {features.shape[1]} does not match model expectation {model_bundle['input_dim']}"
        )
    
    # Predict
    inputs = torch.from_numpy(features).unsqueeze(0).to(model_bundle["device"])
    mask = torch.from_numpy(frame_mask).unsqueeze(0).to(model_bundle["device"])
    
    with torch.inference_mode():
        logits, _ = model_bundle["model"](inputs, mask)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    # Get top predictions
    top_indices = probs.argsort()[::-1][:5]
    predictions = []
    for idx in top_indices:
        label = model_bundle["idx_to_label"].get(int(idx), str(idx))
        prob = float(probs[idx] * 100.0)
        predictions.append({"label": label, "probability": prob})
    
    return {"predictions": predictions}


def load_video_data() -> List[Dict[str, str]]:
    """Load video data from CSV file."""
    csv_path = Path("data/cleaned_data.csv")
    videos = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            videos.append({
                "id": row["ID"],
                "topic": row["TOPIC"],
                "label": row["LABEL"],
                "video_url": row["VIDEO_URL"]
            })
    
    return videos


def extract_vimeo_id(video_url: str) -> str:
    """Extract Vimeo video ID from URL."""
    if "/video/" in video_url:
        video_id = video_url.split("/video/")[1].split("?")[0]
        return video_id
    return ""


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    # Tìm checkpoint ở nhiều vị trí có thể
    possible_checkpoints = [
        Path("lstm_150.pt"),  # Root directory
        Path("artifacts/lstm_150.pt"),  # Artifacts directory
        Path("checkpoints/lstm_150.pt"),  # Checkpoints directory
    ]
    
    checkpoint_path = None
    for cp in possible_checkpoints:
        if cp.exists():
            checkpoint_path = cp
            break
    
    if checkpoint_path is None:
        print("Warning: Không tìm thấy lstm_150.pt ở các vị trí:")
        for cp in possible_checkpoints:
            print(f"  - {cp}")
        print("Prediction will not be available until model is loaded.")
        return
    
    # Tìm JSON mapping
    json_path = Path("label_mapping.json")
    if not json_path.exists():
        print(f"Info: Không tìm thấy {json_path}, sẽ chỉ dùng mapping từ checkpoint")
        json_path = None
    
    try:
        load_model_bundle(checkpoint_path, json_path)
        print(f"✓ Model loaded từ: {checkpoint_path}")
        if json_path:
            print(f"✓ JSON mapping: {json_path}")
    except Exception as e:
        print(f"Error: Could not load model: {e}")
        print("Prediction will not be available until model is loaded.")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page."""
    videos = load_video_data()
    
    # Get unique topics
    topics = sorted(set(v["topic"] for v in videos))
    
    # Extract Vimeo IDs
    for video in videos:
        video["vimeo_id"] = extract_vimeo_id(video["video_url"])
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "videos": videos,
            "topics": topics
        }
    )


@app.post("/api/predict")
async def predict_video(file: UploadFile = File(...)):
    """Predict sign language from uploaded video."""
    try:
        video_bytes = await file.read()
        result = process_video(video_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class KeypointsRequest(BaseModel):
    keypoints: List[Dict[str, List]]


@app.post("/api/predict-keypoints")
async def predict_keypoints(keypoints_data: KeypointsRequest):
    """Predict sign language from keypoints array (real-time, much faster)."""
    global model_bundle
    
    if model_bundle is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        keypoint_buffer_raw = keypoints_data.keypoints
        
        if not keypoint_buffer_raw or len(keypoint_buffer_raw) < 10:
            return JSONResponse(content={"predictions": []})
        
        # Convert JSON format to numpy arrays
        keypoint_buffer = []
        for kp_frame in keypoint_buffer_raw:
            # Helper to convert list to numpy array with correct shape
            def to_array(data, expected_len):
                arr = np.zeros((expected_len, 3), dtype=np.float32)
                if isinstance(data, list) and len(data) > 0:
                    data_array = np.array(data, dtype=np.float32)
                    if data_array.ndim == 2 and data_array.shape[1] == 3:
                        copy_len = min(expected_len, data_array.shape[0])
                        arr[:copy_len] = data_array[:copy_len]
                return arr
            
            kp = {
                "pose": to_array(kp_frame.get("pose", []), POSE_LEN),
                "left_hand": to_array(kp_frame.get("left_hand", []), HAND_LEN),
                "right_hand": to_array(kp_frame.get("right_hand", []), HAND_LEN),
                "face": to_array(kp_frame.get("face", []), FACE_LEN),
            }
            keypoint_buffer.append(kp)
        
        # Resample to target frames
        resampled = resample_keypoints(keypoint_buffer, TARGET_FRAMES)
        
        # Stack
        pose = np.stack([frame["pose"] for frame in resampled], axis=0)
        left_hand = np.stack([frame["left_hand"] for frame in resampled], axis=0)
        right_hand = np.stack([frame["right_hand"] for frame in resampled], axis=0)
        face = np.stack([frame["face"] for frame in resampled], axis=0)
        
        # Build features
        seq = build_feature_sequence(pose, left_hand, right_hand, face, add_velocity=True)
        features = seq["features"]
        frame_mask = seq["frame_mask"]
        
        if features.shape[1] != model_bundle["input_dim"]:
            return JSONResponse(content={"predictions": []})
        
        # Predict
        inputs = torch.from_numpy(features).unsqueeze(0).to(model_bundle["device"])
        mask = torch.from_numpy(frame_mask).unsqueeze(0).to(model_bundle["device"])
        
        with torch.inference_mode():
            logits, _ = model_bundle["model"](inputs, mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Get top predictions
        top_indices = probs.argsort()[::-1][:5]
        predictions = []
        for idx in top_indices:
            label = model_bundle["idx_to_label"].get(int(idx), str(idx))
            prob = float(probs[idx] * 100.0)
            predictions.append({"label": label, "probability": prob})
        
        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/videos")
async def get_videos():
    """API endpoint to get all videos."""
    videos = load_video_data()
    for video in videos:
        video["vimeo_id"] = extract_vimeo_id(video["video_url"])
    return {"videos": videos}


@app.get("/api/topics")
async def get_topics():
    """API endpoint to get all topics."""
    videos = load_video_data()
    topics = sorted(set(v["topic"] for v in videos))
    return {"topics": topics}


@app.get("/api/labels")
async def get_labels():
    """API endpoint to get all available labels from model."""
    global model_bundle
    
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded alslksalknskdv")
    
    idx_to_label = model_bundle.get("idx_to_label", {})
    # Trả về danh sách labels sắp xếp theo index
    labels = [{"index": idx, "label": label} for idx, label in sorted(idx_to_label.items())]
    
    return {
        "total": len(labels),
        "labels": labels
    }


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
