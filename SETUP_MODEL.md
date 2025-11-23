# Hướng dẫn Setup Model cho Web App

## Yêu cầu

Để chạy web app, bạn cần có:

1. **Model checkpoint**: `lstm_150.pt`
2. **Label mapping** (tùy chọn): `label_mapping.json`

## Vị trí file

App sẽ tự động tìm model ở các vị trí sau (theo thứ tự ưu tiên):

1. `lstm_150.pt` (root directory)
2. `artifacts/lstm_150.pt`
3. `checkpoints/lstm_150.pt`

Label mapping sẽ được tìm ở:
- `label_mapping.json` (root directory)

## Cách tạo label_mapping.json

Nếu bạn chưa có file JSON, chạy lệnh sau:

```bash
python scripts/csv_to_json.py --csv index.csv --output label_mapping.json
```

## Cách hoạt động

1. **Khi app khởi động:**
   - Tự động tìm và load `lstm_150.pt`
   - Nếu có `label_mapping.json`, sẽ validate mapping với checkpoint
   - Nếu checkpoint không có `label2idx`, sẽ fallback sang JSON

2. **Priority:**
   - Checkpoint `label2idx` là nguồn chính (nếu có)
   - JSON chỉ dùng để validate hoặc fallback

3. **API mới:**
   - `GET /api/labels` - Lấy danh sách tất cả labels từ model

## Kiểm tra

Sau khi chạy app, kiểm tra console log để xem:
- ✓ Model loaded từ: [path]
- ✓ JSON mapping: [path] (nếu có)
- ✓ JSON mapping khớp với checkpoint (nếu validate thành công)

## Troubleshooting

**Lỗi: "Checkpoint not found"**
- Đảm bảo file `lstm_150.pt` ở một trong các vị trí trên
- Kiểm tra tên file chính xác (phân biệt hoa/thường)

**Lỗi: "Checkpoint không chứa label2idx"**
- Nếu có `label_mapping.json`, app sẽ tự động dùng JSON
- Nếu không có JSON, cần retrain model với code mới để lưu label2idx vào checkpoint

**Warning: "JSON mapping khác với checkpoint"**
- JSON và checkpoint có mapping khác nhau
- App sẽ dùng mapping từ checkpoint (chính xác hơn)
- Nên cập nhật JSON để khớp với checkpoint

