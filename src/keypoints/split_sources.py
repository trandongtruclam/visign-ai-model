"""Build a source-level train/val/test split.

Conventions for source-video identifiers
----------------------------------------
A source video lives at::

    <keypoints_root>/<label>/<source_id>.npz

where `source_id` is the original video's basename (no extension). After
augmentation, additional files are emitted::

    <augmented_root>/<label>/<source_id>__<aug_idx>.npz

Both the original (`<source_id>.npz`) and its augmentations share the same
source-video identifier ``<label>/<source_id>``.

The legacy layout used numeric filenames (`0.npz, 1.npz, ...`) where `0.npz`
was the original and `1..N.npz` were augmentations of `0.npz`. We map every
purely-numeric stem in a folder to the same source ``<label>/orig`` so that
legacy data continues to work without re-extraction.

Splitting strategy
------------------
* For every label, gather its unique source videos.
* For labels with ``len(sources) == 1``: keep that source in *train* (with
  ``--strict`` you can override and let the random partition decide; default
  is to preserve singleton classes in train so the model still has *some*
  example to learn from).
* For labels with ``len(sources) >= 2``: stratify-split the sources between
  train / val / test using ``--val-ratio`` and ``--test-ratio``. At least one
  source per multi-source label is forced to remain in train.

Output
------
``splits.json`` with shape::

    {
        "seed": 42,
        "val_ratio": 0.10,
        "test_ratio": 0.15,
        "strict": false,
        "sources": {
            "<label>/<source_id>": "train" | "val" | "test",
            ...
        }
    }
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


_LEGACY_NUMERIC_STEM = re.compile(r"^\d+$")


def parse_source_video(filepath: Path) -> str:
    """Return the unique source-video identifier for a sample file.

    See module docstring for the naming conventions.
    """
    label = filepath.parent.name
    stem = filepath.stem
    if "__" in stem:
        return f"{label}/{stem.split('__', 1)[0]}"
    if _LEGACY_NUMERIC_STEM.match(stem):
        return f"{label}/orig"
    return f"{label}/{stem}"


def enumerate_sources(keypoints_root: Path) -> Dict[str, List[str]]:
    """Map ``label -> [source_id, ...]`` by walking the extractor output."""
    label_to_sources: Dict[str, set] = defaultdict(set)
    for npz_path in keypoints_root.rglob("*.npz"):
        label = npz_path.parent.name
        stem = npz_path.stem
        if "__" in stem:
            source_id = stem.split("__", 1)[0]
        elif _LEGACY_NUMERIC_STEM.match(stem):
            source_id = "orig"
        else:
            source_id = stem
        label_to_sources[label].add(source_id)
    return {k: sorted(v) for k, v in label_to_sources.items()}


def build_split(
    label_to_sources: Dict[str, List[str]],
    val_ratio: float = 0.10,
    test_ratio: float = 0.15,
    seed: int = 42,
    strict: bool = False,
) -> Dict[str, str]:
    """Return ``{"<label>/<source>": split_name}`` for every source video.

    `strict=False` (default) keeps every singleton-class source in train so
    the model retains coverage of the full vocabulary. `strict=True` ignores
    that safeguard and lets singleton sources land in val/test, which means
    those classes will have zero training examples — useful only when you
    explicitly want zero-shot evaluation.
    """
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio and test_ratio must be >=0 and sum to <1")

    rng = random.Random(seed)
    assignment: Dict[str, str] = {}

    for label in sorted(label_to_sources.keys()):
        sources = sorted(label_to_sources[label])
        n = len(sources)
        if n == 0:
            continue

        if n == 1 and not strict:
            assignment[f"{label}/{sources[0]}"] = "train"
            continue

        shuffled = sources.copy()
        rng.shuffle(shuffled)

        n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 else 0
        n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
        # Always keep at least one source in train when not strict.
        max_holdout = n - 1 if not strict else n
        if n_test + n_val > max_holdout:
            # Reduce val first, then test.
            over = n_test + n_val - max_holdout
            shrink_val = min(over, n_val)
            n_val -= shrink_val
            over -= shrink_val
            n_test = max(0, n_test - over)

        test_sources = shuffled[:n_test]
        val_sources = shuffled[n_test : n_test + n_val]
        train_sources = shuffled[n_test + n_val :]

        for src in train_sources:
            assignment[f"{label}/{src}"] = "train"
        for src in val_sources:
            assignment[f"{label}/{src}"] = "val"
        for src in test_sources:
            assignment[f"{label}/{src}"] = "test"

    return assignment


def summarize(assignment: Dict[str, str]) -> Dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for split in assignment.values():
        counts[split] = counts.get(split, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source-level data splits")
    parser.add_argument(
        "--keypoints-dir",
        default="dataset/keypoints",
        help="Directory containing <label>/<source>.npz files from the extractor",
    )
    parser.add_argument("--output", default="splits.json", help="Where to write splits.json")
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Allow singleton-class sources to be assigned to val/test. "
            "Off by default to preserve training coverage of every class."
        ),
    )
    args = parser.parse_args()

    keypoints_root = Path(args.keypoints_dir)
    if not keypoints_root.exists():
        raise FileNotFoundError(f"keypoints directory not found: {keypoints_root}")

    label_to_sources = enumerate_sources(keypoints_root)
    if not label_to_sources:
        raise RuntimeError(f"No .npz files found under {keypoints_root}")

    assignment = build_split(
        label_to_sources,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        strict=args.strict,
    )

    payload = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "strict": bool(args.strict),
        "sources": assignment,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    counts = summarize(assignment)
    n_total = sum(counts.values())
    n_labels = len(label_to_sources)
    n_multi = sum(1 for v in label_to_sources.values() if len(v) > 1)
    print(f"Wrote {args.output}")
    print(f"  labels      : {n_labels}  (multi-source: {n_multi})")
    print(f"  sources     : {n_total}")
    for split in ("train", "val", "test"):
        print(f"  {split:<10}: {counts.get(split, 0)}")


if __name__ == "__main__":
    main()
