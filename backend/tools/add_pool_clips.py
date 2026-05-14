"""
Add a shared pool of unassigned clips to the annotation queue.

Pool clips have empty `assigned_to` — any annotator can label them via the
updated label_swings UI (first-come-first-serve).

Stratified by source video + velocity bucket to keep diversity high.
Existing rows in queue.csv are preserved.

Usage:
    python -m backend.tools.add_pool_clips \
        --manifest data/annotation/manifest.csv \
        --queue    data/annotation/queue.csv \
        --pool-size 1000 \
        --min-velocity 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.tools._swing_io import MANIFEST_COLUMNS, read_manifest, write_manifest

VELOCITY_BUCKETS = [0, 50, 75, 100, 1e9]
SEED = 42


def _bucket_label(v: float) -> int:
    for i in range(len(VELOCITY_BUCKETS) - 1):
        if VELOCITY_BUCKETS[i] <= v < VELOCITY_BUCKETS[i + 1]:
            return i
    return len(VELOCITY_BUCKETS) - 2


def _load_pool(manifest_path: Path, min_velocity: float, exclude_clip_ids: set[str]) -> pd.DataFrame:
    df = read_manifest(manifest_path)
    df["wrist_velocity_num"] = pd.to_numeric(df["wrist_velocity"], errors="coerce").fillna(0.0)
    df = df[df["wrist_velocity_num"] >= min_velocity]
    df = df[df["label"].astype(str) == ""]
    df = df[~df["clip_id"].isin(exclude_clip_ids)]
    df["bucket"] = df["wrist_velocity_num"].apply(_bucket_label)
    return df.reset_index(drop=True)


def _stratified_sample(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Pick n clips stratified across source_video and velocity bucket."""
    courts = sorted(pool["source_video"].unique())
    if not courts:
        return pool.head(0)

    base, rem = divmod(n, len(courts))
    targets = {c: base + (1 if i < rem else 0) for i, c in enumerate(courts)}

    rng = pd.Series(range(len(pool))).sample(frac=1, random_state=seed).tolist()
    pool = pool.iloc[rng].reset_index(drop=True)

    picked: list[int] = []
    for court in courts:
        court_pool = pool[pool["source_video"] == court]
        target = targets[court]

        bucket_groups: dict[int, list[int]] = {}
        for _, row in court_pool.iterrows():
            bucket_groups.setdefault(int(row["bucket"]), []).append(row.name)

        ordered: list[int] = []
        bucket_keys = sorted(bucket_groups.keys())
        cursors = {b: 0 for b in bucket_keys}
        while True:
            advanced = False
            for b in bucket_keys:
                if cursors[b] < len(bucket_groups[b]):
                    ordered.append(bucket_groups[b][cursors[b]])
                    cursors[b] += 1
                    advanced = True
            if not advanced:
                break

        picked.extend(ordered[:target])

    return pool.loc[picked].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Add unassigned pool clips to the annotation queue")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--pool-size", type=int, required=True)
    parser.add_argument("--min-velocity", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.queue.exists():
        existing = pd.DataFrame(columns=MANIFEST_COLUMNS)
    else:
        existing = read_manifest(args.queue)
        if "assigned_to" not in existing.columns:
            existing["assigned_to"] = ""

    pool = _load_pool(
        manifest_path=args.manifest,
        min_velocity=args.min_velocity,
        exclude_clip_ids=set(existing["clip_id"]),
    )
    print(f"Eligible pool size at velocity >= {args.min_velocity}: {len(pool):,}")
    print("Eligible by court:")
    print(pool["source_video"].apply(lambda s: Path(str(s)).name).value_counts().to_string())

    sample = _stratified_sample(pool, args.pool_size, args.seed)
    sample["assigned_to"] = ""  # pool

    print(f"\nSampled {len(sample):,} pool clips")
    print("Sampled by court:")
    print(sample["source_video"].apply(lambda s: Path(str(s)).name).value_counts().to_string())
    print("Sampled by velocity bucket:")
    print(sample["bucket"].value_counts().sort_index().to_string())

    sample = sample.drop(columns=[c for c in ("wrist_velocity_num", "bucket") if c in sample.columns])
    for col in MANIFEST_COLUMNS:
        if col not in sample.columns:
            sample[col] = ""
    sample = sample[MANIFEST_COLUMNS]

    out = pd.concat([existing, sample], ignore_index=True)

    if args.dry_run:
        print("\nDry run — not writing.")
        return

    args.queue.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(args.queue, out)
    print(f"\nQueue written → {args.queue} ({len(out):,} total rows)")
    print("\nNext steps:")
    print(f"  1. Extract new clips:")
    print(f"     python -m backend.tools.extract_by_manifest --queue {args.queue} \\")
    print(f"         --videos-dir data/raw_videos --clips-dir data/annotation/clips")
    print(f"  2. Upload new clips:")
    print(f"     python -m backend.tools.upload_clips --manifest {args.queue} --clips-dir data/annotation/clips")
    print(f"  3. Seed DB: python -m backend.tools.seed_swing_labels --csv {args.queue}")


if __name__ == "__main__":
    main()
