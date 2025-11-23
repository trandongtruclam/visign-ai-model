"""
Ví dụ cách sử dụng model checkpoint + label_mapping.json trong backend khác

LƯU Ý QUAN TRỌNG:
- Model checkpoint đã chứa label2idx, nên JSON KHÔNG BẮT BUỘC cho inference
- JSON chỉ hữu ích để:
  + Frontend biết danh sách labels có sẵn
  + Kiểm tra/backup mapping
  + Validate mapping giữa checkpoint và JSON
"""
import json
import torch
from pathlib import Path


def load_model_with_json(checkpoint_path: str, json_path: str = None):
    """
    Load model từ checkpoint và (tùy chọn) validate với JSON
    
    Args:
        checkpoint_path: Đường dẫn đến model checkpoint (.pth hoặc .pt)
        json_path: Đường dẫn đến label_mapping.json (tùy chọn, chỉ để validate)
    
    Returns:
        dict chứa model, idx_to_label, và các thông tin khác
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Lấy label2idx từ checkpoint (ĐÂY LÀ NGUỒN CHÍNH)
    label2idx = checkpoint.get("label2idx", {})
    
    if not label2idx:
        raise ValueError("Checkpoint không chứa label2idx! Model này không thể dùng được.")
    
    # Tạo idx_to_label để dùng khi inference
    idx_to_label = {idx: label for label, idx in label2idx.items()}
    
    # Nếu có JSON, validate xem có khớp không
    if json_path and Path(json_path).exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            json_label2idx = json.load(f)
        
        # Validate
        if json_label2idx != label2idx:
            print("⚠️  WARNING: JSON mapping khác với checkpoint!")
            print(f"   Checkpoint có {len(label2idx)} labels")
            print(f"   JSON có {len(json_label2idx)} labels")
            print("   → Nên dùng mapping từ checkpoint (chính xác hơn)")
        else:
            print("✅ JSON mapping khớp với checkpoint")
    
    return {
        "checkpoint": checkpoint,
        "label2idx": label2idx,  # Từ checkpoint
        "idx_to_label": idx_to_label,  # Để dùng khi inference
        "model_config": checkpoint.get("model_config", {}),
    }


def predict_example(model_bundle, features):
    """
    Ví dụ hàm predict sử dụng idx_to_label
    
    Args:
        model_bundle: Bundle từ load_model_with_json()
        features: Feature tensor từ video/keypoints
    """
    # Giả sử đã có model được load và features được xử lý
    # model = model_bundle["model"]
    # logits = model(features)
    # probs = torch.softmax(logits, dim=-1)
    
    # Ví dụ: có predicted index
    predicted_idx = 5  # Giả sử model predict ra index 5
    
    # Convert index -> label
    idx_to_label = model_bundle["idx_to_label"]
    predicted_label = idx_to_label.get(predicted_idx, f"Unknown_{predicted_idx}")
    
    print(f"Predicted index: {predicted_idx}")
    print(f"Predicted label: {predicted_label}")
    
    return predicted_label


def get_all_labels(model_bundle):
    """
    Lấy danh sách tất cả labels (hữu ích cho frontend)
    """
    idx_to_label = model_bundle["idx_to_label"]
    # Sắp xếp theo index
    all_labels = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    return all_labels


if __name__ == "__main__":
    # Ví dụ sử dụng
    checkpoint_path = "checkpoints/best_model.pth"  # Thay bằng đường dẫn thực tế
    json_path = "label_mapping.json"
    
    try:
        print("Loading model...")
        model_bundle = load_model_with_json(checkpoint_path, json_path)
        
        print(f"\n✅ Model loaded thành công!")
        print(f"   Số lượng classes: {len(model_bundle['idx_to_label'])}")
        
        # Lấy danh sách labels
        all_labels = get_all_labels(model_bundle)
        print(f"\n📋 Danh sách labels (10 đầu tiên):")
        for i, label in enumerate(all_labels[:10]):
            print(f"   {i}: {label}")
        
    except FileNotFoundError as e:
        print(f"❌ Không tìm thấy file: {e}")
        print("\n💡 Hướng dẫn:")
        print("   1. Đảm bảo có model checkpoint (.pth)")
        print("   2. Chạy: python scripts/csv_to_json.py để tạo label_mapping.json")
        print("   3. Model checkpoint đã có label2idx, JSON chỉ để tham khảo")

