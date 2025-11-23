# Hướng dẫn sử dụng label_mapping.json

## Tạo file JSON từ CSV

```bash
python scripts/csv_to_json.py --csv index.csv --output label_mapping.json
```

Hoặc đơn giản:
```bash
python scripts/csv_to_json.py
```

File `label_mapping.json` sẽ được tạo với format:
```json
{
    "0 _số không_": 0,
    "1": 1,
    "2": 2,
    ...
}
```

## Trả lời câu hỏi: Backend có hiểu được không?

### ✅ CÓ, nhưng cần hiểu rõ:

1. **Model checkpoint đã chứa `label2idx`**
   - Khi train model, code tự động lưu `label2idx` vào checkpoint
   - Khi load model ở backend khác, mapping này được load tự động
   - **→ JSON KHÔNG BẮT BUỘC cho inference**

2. **JSON hữu ích cho:**
   - ✅ Frontend biết danh sách labels có sẵn
   - ✅ Validate mapping giữa checkpoint và JSON
   - ✅ Backup mapping riêng
   - ✅ Dễ đọc/hiểu hơn CSV

3. **Quy trình deploy backend:**

```
Dự án mới:
├── model_checkpoint.pth  (CHỨA label2idx)
├── label_mapping.json    (tùy chọn, để tham khảo)
└── backend_code.py
```

**Code backend chỉ cần:**
```python
checkpoint = torch.load("model_checkpoint.pth")
label2idx = checkpoint["label2idx"]  # ← Lấy từ checkpoint
idx_to_label = {idx: label for label, idx in label2idx.items()}

# Khi predict:
predicted_idx = model(features).argmax()
predicted_label = idx_to_label[predicted_idx]  # ← Trả về label
```

## Khi nào cần retrain?

- ❌ **KHÔNG cần retrain** nếu:
  - Chỉ đổi format CSV → JSON
  - Labels và mapping giữ nguyên
  - Mapping trong checkpoint khớp với JSON

- ✅ **CẦN retrain** nếu:
  - Thêm/bớt/sửa labels
  - Thay đổi mapping label → index
  - Số lượng classes thay đổi

## Ví dụ sử dụng

Xem file `scripts/example_backend_usage.py` để biết cách:
- Load model với JSON validation
- Predict và convert index → label
- Lấy danh sách tất cả labels cho frontend

## Lưu ý quan trọng

⚠️ **Mapping trong checkpoint là nguồn chính xác nhất**
- Nếu JSON khác với checkpoint → dùng checkpoint
- JSON chỉ để tham khảo/validate, không dùng để inference

