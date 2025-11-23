"""
Script để chuyển đổi index.csv sang label_mapping.json
Tạo mapping label -> index giống như cách model training tạo label2idx
"""
import pandas as pd
import json
from pathlib import Path
import argparse


def csv_to_json_mapping(csv_path: str, output_json: str = None, label_column: str = "label"):
    """
    Chuyển đổi CSV sang JSON mapping label -> index
    
    Args:
        csv_path: Đường dẫn đến file index.csv
        output_json: Đường dẫn output JSON (mặc định: label_mapping.json)
        label_column: Tên cột chứa label (mặc định: "label")
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File không tồn tại: {csv_path}")
    
    # Đọc CSV
    df = pd.read_csv(csv_path)
    if label_column not in df.columns:
        raise ValueError(f"Cột '{label_column}' không tồn tại trong CSV")
    
    # Tạo mapping giống như trong prepare_samples() của modeling.py
    # Sort unique labels để đảm bảo thứ tự giống với training
    unique_labels = sorted(df[label_column].unique())
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Tạo output path
    if output_json is None:
        output_json = csv_path.parent / "label_mapping.json"
    else:
        output_json = Path(output_json)
    
    # Lưu JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=4)
    
    print(f"[OK] Da tao file: {output_json}")
    print(f"[INFO] Tong so labels: {len(label2idx)}")
    print(f"\nVi du mot so labels dau tien:")
    for i, (label, idx) in enumerate(list(label2idx.items())[:5]):
        try:
            print(f"  {idx}: {label}")
        except UnicodeEncodeError:
            print(f"  {idx}: <label contains unicode>")
    
    return label2idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển đổi index.csv sang label_mapping.json")
    parser.add_argument(
        "--csv",
        type=str,
        default="index.csv",
        help="Đường dẫn đến file index.csv (mặc định: index.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn output JSON (mặc định: label_mapping.json cùng thư mục với CSV)"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Tên cột chứa label (mặc định: label)"
    )
    
    args = parser.parse_args()
    
    csv_to_json_mapping(args.csv, args.output, args.label_column)

