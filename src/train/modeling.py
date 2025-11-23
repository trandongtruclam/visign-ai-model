"""Training script for the sign language classifier based on BiLSTM + attention."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AttentionPooling(nn.Module):
    """Learnable temporal attention pooling."""

    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D), mask: (B, T)
        scores = self.attn_fc(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        pooled = torch.sum(weights * x, dim=1)  # (B, D)
        return pooled, weights.squeeze(-1)


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        in_feat: int,
        proj_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.35,
        num_classes: int = 309,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_feat, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_size * self.num_directions
        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionPooling(lstm_out_dim, hidden=lstm_out_dim // 2, dropout=dropout)
            fc_in = lstm_out_dim
        else:
            fc_in = lstm_out_dim

        self.classifier = nn.Sequential(
            nn.Linear(fc_in, fc_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_in // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: (B, T, F), mask: (B, T)
        x = self.proj(x)
        out, _ = self.lstm(x)
        if self.use_attention:
            pooled, attn_weights = self.attn(out, mask)
        else:
            if mask is not None:
                mask = mask.unsqueeze(-1).float()
                pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            else:
                pooled = out.mean(dim=1)
            attn_weights = None
        logits = self.classifier(pooled)
        return logits, attn_weights


@dataclass
class SampleInfo:
    feature_path: Path
    label_idx: int


class SignSequenceDataset(Dataset):
    def __init__(
        self,
        samples: List[SampleInfo],
        has_velocity: bool = True,
    ) -> None:
        self.samples = samples
        self.has_velocity = has_velocity

    def __len__(self) -> int:
        return len(self.samples)

    def _split_dims(self, total_dim: int) -> int:
        if self.has_velocity:
            if total_dim % 2 != 0:
                raise ValueError(
                    "Expected even feature dimension when velocity features are enabled, got "
                    f"{total_dim}. Set has_velocity=False if velocity was disabled in preprocessing."
                )
            return total_dim // 2
        return total_dim

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        feat = np.load(sample.feature_path).astype(np.float32)
        feat_tensor = torch.from_numpy(feat)
        base_dim = self._split_dims(feat_tensor.size(-1))
        if base_dim < 2:
            raise ValueError("Feature dimension too small to extract hand masks.")
        # Extract left/right hand masks from the static feature slice (before velocity part)
        mask_slice = feat_tensor[:, base_dim - 2 : base_dim]
        frame_mask = (mask_slice.sum(dim=-1) > 0).float()
        return {
            "inputs": feat_tensor,
            "labels": torch.tensor(sample.label_idx, dtype=torch.long),
            "mask": frame_mask,
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    inputs = torch.stack([b["inputs"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    return {"inputs": inputs, "labels": labels, "mask": masks}


def compute_class_weights(labels: Iterable[int], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for idx in labels:
        counts[idx] += 1
    counts = counts.clamp(min=1.0)
    weights = torch.sqrt(counts.max() / counts)
    return weights


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (preds == targets).float().mean().item()


def macro_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = preds.cpu()
    targets = targets.cpu()
    f1_scores: List[float] = []
    for cls in range(num_classes):
        tp = torch.sum((preds == cls) & (targets == cls)).item()
        fp = torch.sum((preds == cls) & (targets != cls)).item()
        fn = torch.sum((preds != cls) & (targets == cls)).item()
        denom = 2 * tp + fp + fn
        if denom == 0:
            continue
        f1_scores.append((2 * tp) / denom)
    if not f1_scores:
        return 0.0
    return float(sum(f1_scores) / len(f1_scores))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for batch in loader:
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)

        logits, _ = model(inputs, mask)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.detach().cpu())
        all_targets.append(labels.detach().cpu())

    preds_cat = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": accuracy(preds_cat, targets_cat),
        "f1": macro_f1(preds_cat, targets_cat, model.classifier[-1].out_features),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for batch in loader:
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        mask = batch.get("mask")
        mask = mask.to(device) if mask is not None else None

        logits, _ = model(inputs, mask)
        loss = loss_fn(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())

    preds_cat = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": accuracy(preds_cat, targets_cat),
        "f1": macro_f1(preds_cat, targets_cat, model.classifier[-1].out_features),
    }


def prepare_samples(
    index_csv: Path,
    feature_dir: Path,
    label_column: str = "label",
) -> Tuple[List[SampleInfo], Dict[str, int]]:
    df = pd.read_csv(index_csv)
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' missing from {index_csv}")

    df = df.reset_index(drop=True)
    unique_labels = sorted(df[label_column].unique())
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}

    samples: List[SampleInfo] = []
    missing_files: List[str] = []
    for row_idx, row in df.iterrows():
        label = row[label_column]
        feature_path = feature_dir / f"sample_{row_idx}_{label}.npy"
        if not feature_path.exists():
            missing_files.append(str(feature_path))
            continue
        samples.append(SampleInfo(feature_path=feature_path, label_idx=label2idx[label]))

    if missing_files:
        raise FileNotFoundError(
            "Missing preprocessed feature files. Examples:\n" + "\n".join(missing_files[:5])
        )

    return samples, label2idx


def split_samples(
    samples: List[SampleInfo],
    val_ratio: float,
    seed: int,
) -> Tuple[List[SampleInfo], List[SampleInfo]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    labels = [s.label_idx for s in samples]
    num_classes = max(labels) + 1
    indices = np.arange(len(samples))

    # Stratified split
    from sklearn.model_selection import train_test_split

    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            stratify=labels,
        )
    except ValueError as exc:
        print(
            "[warn] Stratified split failed (likely due to rare classes in val split). "
            "Falling back to unstratified split."
        )
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            stratify=None,
        )
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    # Ensure every class present in train split by moving rare examples from val if needed
    present = {s.label_idx for s in train_samples}
    missing = set(range(num_classes)) - present
    if missing:
        val_list = list(val_samples)
        for cls in sorted(missing):
            candidate = next((s for s in val_list if s.label_idx == cls), None)
            if candidate is None:
                raise RuntimeError(
                    f"Unable to ensure presence of class {cls} in training split. "
                    "Consider reducing val_ratio or augmenting data."
                )
            train_samples.append(candidate)
            val_list.remove(candidate)
        val_samples = val_list
    return train_samples, val_samples


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    epoch: int,
    metrics: Dict[str, float],
    label2idx: Dict[str, int],
    out_path: Path,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
        "label2idx": label2idx,
        "model_config": {
            "in_feat": model.proj[0].in_features,
            "proj_dim": model.proj[0].out_features,
            "hidden_size": model.hidden_size,
            "num_layers": model.lstm.num_layers,
            "bidirectional": model.num_directions == 2,
            "dropout": getattr(model.lstm, "dropout", 0.0),
            "num_classes": model.classifier[-1].out_features,
            "use_attention": model.use_attention,
        },
    }
    torch.save(checkpoint, out_path)


def train_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    index_csv = Path(args.index_csv)
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, label2idx = prepare_samples(index_csv, feature_dir)
    num_classes = len(label2idx)
    print(f"Loaded {len(samples)} samples across {num_classes} classes")

    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    train_dataset = SignSequenceDataset(train_samples, has_velocity=not args.no_velocity)
    val_dataset = SignSequenceDataset(val_samples, has_velocity=not args.no_velocity)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    example_batch = next(iter(train_loader))
    input_dim = example_batch["inputs"].shape[-1]
    print(f"Input dimension detected: {input_dim}")

    model = LSTMClassifier(
        in_feat=input_dim,
        proj_dim=args.proj_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=not args.no_bidirectional,
        dropout=args.dropout,
        num_classes=num_classes,
        use_attention=not args.no_attention,
    ).to(device)

    if args.use_class_weights:
        class_weights = compute_class_weights([s.label_idx for s in train_samples], num_classes).to(device)
    else:
        class_weights = None

    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.min_lr,
    )

    history: List[Dict[str, float]] = []
    best_metric = -math.inf
    epochs_without_improve = 0
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            grad_clip=args.grad_clip,
        )

        val_metrics = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_metrics["loss"])

        metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(metrics)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={metrics['train_loss']:.4f} val_loss={metrics['val_loss']:.4f} "
            f"train_f1={metrics['train_f1']:.4f} val_f1={metrics['val_f1']:.4f} lr={metrics['lr']:.6f}"
        )

        if metrics["val_f1"] > best_metric + args.improve_delta:
            best_metric = metrics["val_f1"]
            epochs_without_improve = 0
            save_checkpoint(model, optimizer, scheduler, epoch, metrics, label2idx, best_path)
            print(f"Saved new best model to {best_path} (val_f1={best_metric:.4f})")
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= args.patience:
            print("Early stopping triggered")
            break

    # Persist training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LSTM-based sign language classifier")
    parser.add_argument("--index-csv", type=str, default="index.csv", dest="index_csv")
    parser.add_argument("--feature-dir", type=str, default="preprocessed_npz", dest="feature_dir")
    parser.add_argument("--output-dir", type=str, default="artifacts", dest="output_dir")
    parser.add_argument("--val-ratio", type=float, default=0.1, dest="val_ratio")
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--proj-dim", type=int, default=256, dest="proj_dim")
    parser.add_argument("--hidden-size", type=int, default=256, dest="hidden_size")
    parser.add_argument("--num-layers", type=int, default=2, dest="num_layers")
    parser.add_argument("--grad-clip", type=float, default=5.0, dest="grad_clip")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--improve-delta", type=float, default=1e-3, dest="improve_delta")
    parser.add_argument("--label-smoothing", type=float, default=0.0, dest="label_smoothing")
    parser.add_argument("--use-class-weights", action="store_true", dest="use_class_weights")
    parser.add_argument("--lr-factor", type=float, default=0.5, dest="lr_factor")
    parser.add_argument("--lr-patience", type=int, default=4, dest="lr_patience")
    parser.add_argument("--min-lr", type=float, default=1e-6, dest="min_lr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0, dest="num_workers")
    parser.add_argument("--no-attention", action="store_true", dest="no_attention")
    parser.add_argument("--no-bidirectional", action="store_true", dest="no_bidirectional")
    parser.add_argument("--no-velocity", action="store_true", dest="no_velocity")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()