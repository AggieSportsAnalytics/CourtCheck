"""
Build a TCN training manifest from a Supabase swing_labels export.

Maps annotator labels (forehand/backhand/serve/volley) onto the TCN classes
defined in backend/models/stroke_classifier_tcn.py (Forehand/Backhand/Serve/Overhead/Slice).
Volley has no direct TCN class and is dropped.

Usage:
    python -m backend.tools.build_courtcheck_manifest \
        --labels-csv data/annotation/labeled_export_2026-05-11.csv \
        --clips-dir data/annotation/clips \
        --output backend/training/data/courtcheck_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Annotator label → TCN class name. Volley intentionally omitted.
LABEL_MAP = {
    "forehand": "Forehand",
    "backhand": "Backhand",
    "serve": "Serve/Overhead",
}


def _load_labels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df = df[df["label"].isin(LABEL_MAP)].copy()
    df["mapped_label"] = df["label"].map(LABEL_MAP)
    return df.reset_index(drop=True)


def _build_entries(df: pd.DataFrame, clips_dir: Path) -> tuple[list[dict], int, int]:
    """Return (entries, missing_clip_files, already_extracted_keypoints)."""
    entries: list[dict] = []
    missing = 0
    already_extracted = 0
    for _, row in df.iterrows():
        clip_id = row["clip_id"]
        clip_path = clips_dir / f"{clip_id}.mp4"
        if not clip_path.exists():
            missing += 1
            continue
        npy_path = clip_path.with_suffix(".npy")
        has_kp = npy_path.exists()
        if has_kp:
            already_extracted += 1
        entries.append({
            "clip_path": str(clip_path),
            "original_label": row["label"],
            "mapped_label": row["mapped_label"],
            "split": "train",
            "keypoints_path": str(npy_path) if has_kp else None,
            "valid": True if has_kp else None,
            "clip_id": clip_id,
            "source_video": row["source_video"],
            "annotator": row["annotator"],
        })
    return entries, missing, already_extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TCN training manifest from labels CSV")
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--clips-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    df = _load_labels(args.labels_csv)
    print(f"Loaded {len(df):,} labeled rows (forehand/backhand/serve)")

    entries, missing, already = _build_entries(df, args.clips_dir)
    print(f"Manifest entries with clip files present: {len(entries):,}")
    if already:
        print(f"  - already have keypoints (.npy on disk): {already:,}")
        print(f"  - pending extraction: {len(entries) - already:,}")
    if missing:
        print(f"Skipped {missing:,} labeled rows — clip file not found locally")

    by_class: dict[str, int] = {}
    for e in entries:
        by_class[e["mapped_label"]] = by_class.get(e["mapped_label"], 0) + 1
    print("By class:")
    for cls, n in sorted(by_class.items()):
        print(f"  {cls}: {n}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote manifest → {args.output}")
    print("\nNext: extract keypoints")
    print(f"  python -m backend.training.data.extract_keypoints --manifest {args.output} --device mps")


if __name__ == "__main__":
    main()
