# **visign - Vietnamese Sign Language Recognition**
---

## Keypoints - MediaPipe

**Video:**
    - Aspect Ratio: 16:9 (1920×1080)
    - FPS: 25–30 fps

**Keypoints Configuration**
- MediaPipe Holistic Configuration:
    * Hands Tracking: 21 x 2 hand landmarks
    * Pose Estimation: 33 landmarks
    * Face: 468 landmarks

- Frame: 150 frames (5 seconds @ 30 fps)
    * Resampling - Linear Interpolation

    *Input:* 114 frames
        Frame 0: hand at (x1, y1, z1)
        Frame 1: hand at (x2, y2, z2)
        ...
        Frame 113: hand at (x114, y114, z114)
    
    *Output:* 150 frames
        Frame 0: hand at (x1, y1, z1) - giữ nguyên
        Frame 1: hand at (x1.76, y1.76, z1.76) - interpolated
        Frame 2: hand at (x2.52, y2.52, z2.52) - interpolated
        ...
        Frame 149: hand at (x114, y114, z114) - giữ nguyên

UPPER_BODY_INDEXES = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 15, 17, 19, 21,
    12, 14, 16, 18, 20, 22,
    23, 24
]

**src/keypoints/keypoints_extractor.py**

Extract Keypoints
```ruby
python src/keypoints/keypoints_extractor.py --process_dataset
```

Demo with single video
```ruby
python src/keypoints/keypoints_extractor.py <video_path> output_keypoints.npz
```

**Data Augmentation**
- *Linear Scaling:* Randomize scale factor from k_min to k_max
- 

**src/keypoints/augment.py**
Augmentation for ONE WORD
```ruby
python src/keypoints/augment.py "dataset/keypoints/ai cho" "augmented/ai cho" --n 50
```

Augmentation for DATASET
```ruby
python src/keypoints/augment.py dataset/keypoints augmented --n 50
```

```ruby
python src/keypoints/augment.py dataset/keypoints augmented --n 100 --kmin 0.8 --kmax 1.2 --sigma_body 0.02
```

**src/keypoints/keypoints_eval.py**
Evaluate augmented keypoints on video
```ruby
python src/keypoints/keypoints_eval.py "ai cho" --n_samples 10
```

**Output Structure:**
```
augmented/
├── ai cho/
│   ├── 0.npz (original)
│   ├── 1.npz (augmented #1)
│   ├── 2.npz (augmented #2)
│   └── ...
```

## Preprocess
- Load data from /augmented
    ```ruby
    pose: (150, 25, 3)    # 150 frames, 25 keypoints, 3 coords (x,y,z)
    left_hand: (150, 21, 3)
    right_hand: (150, 21, 3)
    face: (150, 468, 3)
    ```
- Detect missing hands
    * Check missing hand
    * Create mask: lh_mask, rh_mask (1=present, 0=missing)
    * Shape: (150,) - 1 value for each frame

- Normalize keypoints (center_and_scale):
    * Center: Trừ keypoints theo shoulder midpoint
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        Trừ tất cả keypoints (pose, hands, face) theo center này
    * Scale: Chia theo shoulder distance (mean)
        scale = mean(||left_shoulder - right_shoulder||)
        Chia tất cả keypoints để đạt scale-invariant
    * Chỉ dùng x, y (bỏ z)                      

## Model Architecture
- **Feature Builder:** Sau chuẩn hoá, mỗi frame ghép `pose`, `left_hand`, `right_hand`, `face_subset`, kèm mặt nạ hiện diện bàn tay (`lh_mask`, `rh_mask`) và véc-tơ vận tốc. Chiều đặc trưng mặc định: `pose 25×2=50`, `hands 2×21×2=84`, `face_subset 89×2=178`, `masks=2` ⇒ 314 chiều; bật velocity → 628 chiều.
- **Projection Layer:** `nn.Linear(in_feat → 256)` + LayerNorm + ReLU + Dropout để gom đặc trưng khung hình về không gian chung.
- **BiLSTM Encoder:** 2 tầng LSTM hai chiều (hidden 256, dropout 0.35) giúp mô hình bắt tín hiệu thời gian thuận/ngược.
- **Attention Pooling:** Lớp `AttentionPooling` học trọng số theo thời gian (mask các frame bàn tay mất) → vector câu toàn cục.
- **Classifier Head:** `Linear(512→256) → ReLU → Dropout → Linear(256→#classes)` với label smoothing & class weight tuỳ chọn.
- **Checkpoint Artifacts:** `artifacts/best_model.pt` lưu `state_dict`, `label2idx`, cấu hình model; `training_history.json` ghi lại loss/acc/F1 từng epoch.

## Training Pipeline
1. **Sinh keypoints và augmentation**
   - Trích xuất keypoints: `python src/keypoints/keypoints_extractor.py --process_dataset`
   - Tăng cường dữ liệu: `python src/keypoints/augment.py dataset/keypoints augmented --n 50`

2. **Xây dựng chỉ mục & tiền xử lý**
   - Tạo `index.csv` & đặc trưng numpy: `python src/train/preprocess_pipeline.py` (mặc định đọc `augmented/`, sinh `index.csv` + `preprocessed_npz/sample_{i}_{label}.npy`). Nếu thư mục khác, sửa biến trong script hoặc chạy theo module có đối số tuỳ biến.

3. **Huấn luyện**
   ```bash
   python src/train/modeling.py \
     --index-csv index.csv \
     --feature-dir preprocessed_npz \
     --output-dir artifacts \
     --epochs 60 \
     --batch-size 32 \
     --lr 1e-3 \
     --use-class-weights \
     --label-smoothing 0.05
   ```
   - Tham số quan trọng: `--proj-dim`, `--hidden-size`, `--num-layers`, `--no-attention`, `--no-velocity`, `--val-ratio`, `--patience`, `--device`.
   - Trong training: in ra `train_loss/val_loss`, `train_f1/val_f1`, điều chỉnh LR bằng ReduceLROnPlateau, early stopping.

4. **Đánh giá**
   - Mỗi epoch in F1 macro & Accuracy trên `val_loader`.
   - Model tốt nhất (theo `val_f1`) được ghi đè `artifacts/best_model.pt` cùng metric snapshot.

## Inference & Demo

### Web Application (FastAPI)

Ứng dụng web đơn giản để học ngôn ngữ ký hiệu với video hướng dẫn và webcam.

**Cài đặt dependencies:**
```bash
pip install fastapi uvicorn jinja2 python-multipart
```

**Chạy ứng dụng:**
```bash
# Windows
python app.py

# Hoặc sử dụng uvicorn trực tiếp
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Truy cập:** Mở trình duyệt và vào `http://localhost:8000`

**Tính năng:**
- **Video hướng dẫn:** Hiển thị video Vimeo từ `data/cleaned_data.csv`
- **Webcam:** Stream camera của người dùng theo thời gian thực
- **Chọn từ vựng:** Dropdown để chọn theo chủ đề (Topic) và từ cụ thể (Label)
- **Phát lại video:** Nút để phát lại video hướng dẫn
- **Ghi 5 giây:** Nút "Bắt Đầu Ghi" để ghi lại video từ webcam trong 5 giây

**Lưu ý:** 
- Trình duyệt sẽ yêu cầu quyền truy cập camera khi mở trang
- Video được ghi sẽ được lưu dưới dạng Blob trong bộ nhớ trình duyệt

### Deploy lên Production

Xem file `DEPLOY.md` để biết hướng dẫn chi tiết deploy lên Railway, Render hoặc Vercel.

**Nhanh nhất với Railway:**
1. Đăng ký tại [railway.app](https://railway.app)
2. Connect GitHub repo
3. Railway tự động deploy
4. Done! 🎉