"""Evaluate a trained checkpoint on a held-out split.

Produces:
* Top-1 / top-5 accuracy, macro / weighted F1
* Per-class precision / recall / F1
* Confusion matrix (numpy `.npy` + top-K confusion pairs CSV)
* 10 worst-performing classes (lowest F1, ties broken by support)
* Mean inference latency on CPU and (if available) GPU
* A Markdown report at `<output-dir>/EVAL_REPORT.md`

Usage
-----
Run *after* `preprocess_pipeline.py --splits-json splits.json` has produced
both `index.csv` (with a `split` column) and the matching feature `.npy`
files::

    python -m src.eval.evaluate \\
        --checkpoint artifacts/lstm_150.pt \\
        --index-csv index.csv \\
        --feature-dir preprocessed_npz \\
        --split test \\
        --output-dir docs

If `--index-csv` is omitted the script only runs the latency benchmark on
synthetic input. This is useful as a sanity check on the deployed binary
when the original dataset is not available locally.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Make `python -m src.eval.evaluate` work both from repo root and when
# invoked as `python src/eval/evaluate.py`.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.modeling import (  # noqa: E402
    LSTMClassifier,
    SampleInfo,
    SignSequenceDataset,
    collate_batch,
)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[LSTMClassifier, Dict[str, int], Dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("model_config", {})
    label2idx: Dict[str, int] = ckpt.get("label2idx", {})
    if not label2idx:
        raise ValueError(
            "Checkpoint is missing `label2idx`; cannot evaluate without a label vocabulary."
        )
    model = LSTMClassifier(
        in_feat=cfg["in_feat"],
        proj_dim=cfg.get("proj_dim", 256),
        hidden_size=cfg.get("hidden_size", 256),
        num_layers=cfg.get("num_layers", 2),
        bidirectional=cfg.get("bidirectional", True),
        dropout=cfg.get("dropout", 0.35),
        num_classes=cfg.get("num_classes", len(label2idx)),
        use_attention=cfg.get("use_attention", True),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, label2idx, cfg


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_eval_samples(
    index_csv: Path,
    feature_dir: Path,
    split: str,
    label2idx: Dict[str, int],
) -> List[SampleInfo]:
    df = pd.read_csv(index_csv)
    if "split" in df.columns and split != "all":
        df = df[df["split"] == split].reset_index(drop=True)
    elif split != "all":
        raise ValueError(
            f"index.csv has no `split` column but --split={split!r} was requested. "
            "Either rebuild the index with --splits-json, or pass --split all."
        )

    if df.empty:
        raise RuntimeError(f"No rows match split={split!r} in {index_csv}")

    samples: List[SampleInfo] = []
    missing: List[str] = []
    skipped_unknown_label = 0
    for row_idx, row in df.iterrows():
        label = str(row["label"])
        if label not in label2idx:
            skipped_unknown_label += 1
            continue
        feature_path = feature_dir / f"sample_{row_idx}_{label}.npy"
        if not feature_path.exists():
            missing.append(str(feature_path))
            continue
        samples.append(
            SampleInfo(
                feature_path=feature_path,
                label_idx=label2idx[label],
                source_video=str(row.get("source_video", "")),
            )
        )

    if missing:
        print(f"[warn] {len(missing)} feature files missing (first 3): {missing[:3]}")
    if skipped_unknown_label:
        print(
            f"[warn] {skipped_unknown_label} rows skipped because their label is "
            "not in the checkpoint vocabulary."
        )
    return samples


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def collect_predictions(
    model: LSTMClassifier,
    loader: DataLoader,
    device: torch.device,
    top_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (top1_preds, top_k_preds, targets)."""
    top1s: List[np.ndarray] = []
    topks: List[np.ndarray] = []
    tgts: List[np.ndarray] = []
    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["labels"]
        logits, _ = model(inputs, mask)
        top1 = logits.argmax(dim=-1).cpu().numpy()
        k = min(top_k, logits.size(-1))
        topk = torch.topk(logits, k=k, dim=-1).indices.cpu().numpy()
        top1s.append(top1)
        topks.append(topk)
        tgts.append(labels.numpy())
    return (
        np.concatenate(top1s) if top1s else np.zeros(0, dtype=np.int64),
        np.concatenate(topks) if topks else np.zeros((0, top_k), dtype=np.int64),
        np.concatenate(tgts) if tgts else np.zeros(0, dtype=np.int64),
    )


def compute_metrics(
    top1: np.ndarray,
    topk: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    idx2label: Dict[int, str],
) -> Dict:
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    if targets.size == 0:
        raise RuntimeError("No predictions collected; the test split is empty.")

    top1_acc = float((top1 == targets).mean())
    topk_acc = float(np.mean([t in topk[i] for i, t in enumerate(targets)]))

    macro_f1 = float(f1_score(targets, top1, labels=list(range(num_classes)),
                              average="macro", zero_division=0))
    weighted_f1 = float(f1_score(targets, top1, labels=list(range(num_classes)),
                                 average="weighted", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        targets, top1, labels=list(range(num_classes)), zero_division=0
    )
    cm = confusion_matrix(targets, top1, labels=list(range(num_classes)))

    per_class = []
    for cls in range(num_classes):
        per_class.append(
            {
                "class_idx": cls,
                "label": idx2label.get(cls, str(cls)),
                "precision": float(precision[cls]),
                "recall": float(recall[cls]),
                "f1": float(f1[cls]),
                "support": int(support[cls]),
            }
        )

    # Worst 10 classes that actually have test support (support > 0)
    supported = [pc for pc in per_class if pc["support"] > 0]
    worst = sorted(supported, key=lambda d: (d["f1"], -d["support"]))[:10]

    # Top confusion pairs (excluding diagonal)
    cm_flat = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j or cm[i, j] == 0:
                continue
            cm_flat.append((cm[i, j], i, j))
    cm_flat.sort(reverse=True)
    top_confusions = [
        {
            "true_label": idx2label.get(i, str(i)),
            "pred_label": idx2label.get(j, str(j)),
            "count": int(c),
        }
        for c, i, j in cm_flat[:20]
    ]

    return {
        "top1_acc": top1_acc,
        f"top{topk.shape[1]}_acc": topk_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_samples": int(targets.size),
        "num_classes_with_support": int(np.sum(support > 0)),
        "per_class": per_class,
        "worst_classes": worst,
        "confusion_matrix": cm,
        "top_confusions": top_confusions,
    }


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def benchmark_latency(
    model: LSTMClassifier,
    in_feat: int,
    seq_len: int = 150,
    batch_size: int = 1,
    warmup: int = 5,
    iters: int = 50,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    dummy = torch.randn(batch_size, seq_len, in_feat, device=device)
    mask = torch.ones(batch_size, seq_len, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(dummy, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(dummy, mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)
    arr = np.array(timings) * 1000.0
    return {
        "device": str(device),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "iters": iters,
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_per_class_csv(metrics: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_idx", "label", "precision", "recall", "f1", "support"])
        for pc in metrics["per_class"]:
            writer.writerow(
                [pc["class_idx"], pc["label"], f"{pc['precision']:.4f}",
                 f"{pc['recall']:.4f}", f"{pc['f1']:.4f}", pc["support"]]
            )


def write_markdown_report(
    out_path: Path,
    checkpoint_path: Path,
    split: str,
    metrics: Optional[Dict],
    latency: List[Dict[str, float]],
    config: Dict,
    sample_count: int,
    notes: List[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# VSL Model — Evaluation Report")
    lines.append("")
    lines.append(f"- **Checkpoint:** `{checkpoint_path}`")
    lines.append(f"- **Split evaluated:** `{split}`")
    lines.append(f"- **Samples in split:** {sample_count}")
    lines.append(f"- **Model config:** "
                 f"in_feat={config.get('in_feat')}, "
                 f"proj_dim={config.get('proj_dim')}, "
                 f"hidden={config.get('hidden_size')}, "
                 f"layers={config.get('num_layers')}, "
                 f"bidir={config.get('bidirectional')}, "
                 f"attention={config.get('use_attention')}, "
                 f"num_classes={config.get('num_classes')}")
    lines.append("")

    if notes:
        lines.append("## Notes")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    if metrics is None:
        lines.append("## Accuracy / F1")
        lines.append("")
        lines.append("> _Not computed — no `--index-csv` and feature data was supplied._")
        lines.append("> Re-run with `--index-csv index.csv --feature-dir preprocessed_npz` "
                     "after generating `splits.json` to obtain the full metric set.")
        lines.append("")
    else:
        top_k_key = next((k for k in metrics if k.startswith("top") and k.endswith("_acc")
                          and k != "top1_acc"), None)
        lines.append("## Accuracy / F1")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Top-1 accuracy | {metrics['top1_acc']:.4f} |")
        if top_k_key:
            lines.append(f"| {top_k_key.replace('_acc','').upper()} accuracy | {metrics[top_k_key]:.4f} |")
        lines.append(f"| Macro F1 | {metrics['macro_f1']:.4f} |")
        lines.append(f"| Weighted F1 | {metrics['weighted_f1']:.4f} |")
        lines.append(f"| Classes with support | {metrics['num_classes_with_support']} / {config.get('num_classes')} |")
        lines.append("")

        lines.append("## 10 worst classes (by F1, support > 0)")
        lines.append("")
        lines.append("| Rank | Label | Support | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|---|---|")
        for i, pc in enumerate(metrics["worst_classes"], 1):
            lines.append(
                f"| {i} | {pc['label']} | {pc['support']} | "
                f"{pc['precision']:.3f} | {pc['recall']:.3f} | {pc['f1']:.3f} |"
            )
        lines.append("")

        lines.append("## Top confusion pairs")
        lines.append("")
        lines.append("| True → Predicted | Count |")
        lines.append("|---|---|")
        for c in metrics["top_confusions"]:
            lines.append(f"| {c['true_label']} → {c['pred_label']} | {c['count']} |")
        lines.append("")

    lines.append("## Inference latency")
    lines.append("")
    if not latency:
        lines.append("_No latency benchmark requested._")
    else:
        lines.append("| Device | Batch | Seq | Iters | Mean (ms) | Median (ms) | P95 (ms) |")
        lines.append("|---|---|---|---|---|---|---|")
        for L in latency:
            lines.append(
                f"| {L['device']} | {L['batch_size']} | {L['seq_len']} | {L['iters']} | "
                f"{L['mean_ms']:.2f} | {L['median_ms']:.2f} | {L['p95_ms']:.2f} |"
            )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a VSL classifier checkpoint")
    parser.add_argument("--checkpoint", default="artifacts/lstm_150.pt")
    parser.add_argument("--index-csv", default=None,
                        help="Path to index.csv with a `split` column (omit to skip metric eval)")
    parser.add_argument("--feature-dir", default="preprocessed_npz")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-velocity", action="store_true")
    parser.add_argument("--device", default=None,
                        help="Override device, e.g. 'cpu' or 'cuda'.")
    parser.add_argument("--output-dir", default="docs")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--latency-iters", type=int, default=50)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--latency-cpu-only", action="store_true",
                        help="Benchmark on CPU even if CUDA is available.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(selected_device)
    print(f"Loading checkpoint on device={device}")
    model, label2idx, config = load_model(Path(args.checkpoint), device)
    idx2label = {v: k for k, v in label2idx.items()}

    metrics: Optional[Dict] = None
    sample_count = 0
    notes: List[str] = []

    if args.index_csv:
        feature_dir = Path(args.feature_dir)
        samples = build_eval_samples(Path(args.index_csv), feature_dir,
                                     args.split, label2idx)
        sample_count = len(samples)
        if sample_count == 0:
            notes.append(
                f"No usable samples in split={args.split!r}. Skipping accuracy / F1 computation."
            )
        else:
            dataset = SignSequenceDataset(samples, has_velocity=not args.no_velocity)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                collate_fn=collate_batch,
            )
            print(f"Evaluating {len(samples)} samples on split={args.split!r}")
            top1, topk, targets = collect_predictions(model, loader, device, top_k=args.top_k)
            metrics = compute_metrics(top1, topk, targets, config["num_classes"], idx2label)

            # Save confusion matrix and per-class CSV alongside the report.
            cm_path = output_dir / "eval_confusion_matrix.npy"
            np.save(cm_path, metrics["confusion_matrix"])
            print(f"Confusion matrix saved to {cm_path}")
            write_per_class_csv(metrics, output_dir / "eval_per_class.csv")
            print(f"Per-class metrics saved to {output_dir / 'eval_per_class.csv'}")
    else:
        notes.append(
            "Ran without `--index-csv`; only the latency benchmark and checkpoint "
            "sanity check were executed. Provide an index with a `split` column to "
            "obtain accuracy / F1 / confusion matrix."
        )

    latency_results: List[Dict[str, float]] = []
    if not args.skip_latency:
        in_feat = config["in_feat"]
        print("Running latency benchmark on CPU...")
        latency_results.append(
            benchmark_latency(model, in_feat, iters=args.latency_iters,
                              device=torch.device("cpu"))
        )
        if torch.cuda.is_available() and not args.latency_cpu_only:
            print("Running latency benchmark on CUDA...")
            latency_results.append(
                benchmark_latency(model, in_feat, iters=args.latency_iters,
                                  device=torch.device("cuda"))
            )

    # Print summary to stdout (in addition to the markdown file)
    print("\n===== Summary =====")
    if metrics is not None:
        top_k_key = next((k for k in metrics if k.startswith("top") and k.endswith("_acc")
                          and k != "top1_acc"), None)
        print(f"Top-1     : {metrics['top1_acc']:.4f}")
        if top_k_key:
            print(f"{top_k_key:<10}: {metrics[top_k_key]:.4f}")
        print(f"Macro F1  : {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        print("\nWorst 10 classes:")
        for pc in metrics["worst_classes"]:
            print(f"  {pc['label']:<30} support={pc['support']:<3} f1={pc['f1']:.3f}")
    for L in latency_results:
        print(f"Latency on {L['device']}: mean={L['mean_ms']:.2f}ms p95={L['p95_ms']:.2f}ms "
              f"(batch={L['batch_size']}, seq={L['seq_len']}, iters={L['iters']})")

    write_markdown_report(
        out_path=output_dir / "EVAL_REPORT.md",
        checkpoint_path=Path(args.checkpoint),
        split=args.split,
        metrics=metrics,
        latency=latency_results,
        config=config,
        sample_count=sample_count,
        notes=notes,
    )

    # Persist the raw numeric summary too for programmatic consumption.
    summary_payload = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "sample_count": sample_count,
        "config": config,
        "latency": latency_results,
        "metrics": None if metrics is None else {
            k: v for k, v in metrics.items()
            if k not in {"confusion_matrix", "per_class"}
        },
    }
    with open(output_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2, default=float)


if __name__ == "__main__":
    main()
