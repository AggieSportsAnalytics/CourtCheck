"""
Filter and sample the raw manifest into a clean annotation queue.

Applies a minimum wrist velocity threshold, shuffles, then writes a
sampled queue manifest ready for label_swings.py.

Usage:
    python -m backend.tools.prepare_annotation_queue \
        --manifest data/annotation/manifest.csv \
        --output  data/annotation/queue.csv

    # Custom threshold and sample size:
    python -m backend.tools.prepare_annotation_queue \
        --manifest data/annotation/manifest.csv \
        --output   data/annotation/queue.csv \
        --min-velocity 40 \
        --sample 8000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_MIN_VELOCITY = 35.0
DEFAULT_SAMPLE = 8000


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare annotation queue from manifest")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-velocity", type=float, default=DEFAULT_MIN_VELOCITY,
                        help=f"Minimum wrist velocity (default: {DEFAULT_MIN_VELOCITY})")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help=f"Max clips to include (default: {DEFAULT_SAMPLE}, 0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest, dtype=str).fillna("")
    total = len(df)

    df["wrist_velocity"] = pd.to_numeric(df["wrist_velocity"], errors="coerce")
    filtered = df[df["wrist_velocity"] >= args.min_velocity].copy()
    print(f"Total clips:      {total:,}")
    print(f"After v >= {args.min_velocity:.0f}:   {len(filtered):,}  ({len(filtered)/total*100:.0f}%)")

    # Only include unlabeled clips
    unlabeled = filtered[filtered["label"] == ""]
    print(f"Unlabeled:        {len(unlabeled):,}")

    if args.sample and len(unlabeled) > args.sample:
        queue = unlabeled.sample(n=args.sample, random_state=args.seed)
        print(f"Sampled:          {len(queue):,}  (seed={args.seed})")
    else:
        queue = unlabeled
        print(f"Using all:        {len(queue):,}")

    queue = queue.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.output, index=False)
    print(f"\nQueue saved → {args.output}")

    # Per-video breakdown
    print("\nBreakdown by video:")
    for src, count in queue.groupby(queue["source_video"].str.split("/").str[-1]).size().items():
        print(f"  {src}: {count:,}")

    print(f"\nNext step:")
    print(f"  streamlit run backend/tools/label_swings.py -- --manifest {args.output} --annotator <name>")


if __name__ == "__main__":
    main()
