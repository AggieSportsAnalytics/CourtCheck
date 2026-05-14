"""
Assign annotation clips to teammates.

Builds (or extends) data/annotation/queue.csv so that each annotator gets
exactly --per-person clips, stratified across source videos and velocity
buckets so no one is stuck with one match's footage.

Existing labeled rows are preserved (label != ''); their `assigned_to` is
back-filled to whoever labeled them so the count is honest.

Usage:
    python -m backend.tools.assign_clips \
        --manifest data/annotation/manifest.csv \
        --queue    data/annotation/queue.csv \
        --annotators kenny anik joseph michelle sienna \
        --per-person 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.tools._swing_io import MANIFEST_COLUMNS, read_manifest, write_manifest

DEFAULT_MIN_VELOCITY = 35.0
VELOCITY_BUCKETS = [0, 50, 75, 100, 1e9]  # 4 buckets: slow / medium / fast / very fast
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
    df = df[df["label"].astype(str) == ""]  # drop already-labeled clips (prior runs)
    df = df[~df["clip_id"].isin(exclude_clip_ids)]
    df["bucket"] = df["wrist_velocity_num"].apply(_bucket_label)
    return df.reset_index(drop=True)


def _balanced_assign(
    pool: pd.DataFrame,
    annotators: list[str],
    per_person: int,
    seed: int,
) -> pd.DataFrame:
    """Assign clips so each annotator gets per_person clips, equally split across courts.

    Strategy:
    1. Target per (annotator, court) = per_person / num_courts (rounded fairly across courts).
    2. Within each court, shuffle clips and round-robin to annotators (advancing past
       full annotators), with velocity-bucket interleaving for diversity inside the court.
    """
    courts = sorted(pool["source_video"].unique())
    if not courts:
        return pd.DataFrame(columns=list(pool.columns) + ["assigned_to"])

    # Distribute per-person count across courts as evenly as possible (e.g. 500 / 3 = 167+167+166)
    base, rem = divmod(per_person, len(courts))
    targets_per_court = {
        court: base + (1 if i < rem else 0)
        for i, court in enumerate(courts)
    }

    # Shuffle within each priority class so high-priority (reusable) rows still come first
    rng = pd.Series(range(len(pool))).sample(frac=1, random_state=seed).tolist()
    pool = pool.iloc[rng].reset_index(drop=True)
    if "priority" in pool.columns:
        pool = pool.sort_values("priority", kind="stable").reset_index(drop=True)

    per_annotator_court: dict[tuple[str, str], list[int]] = {
        (a, c): [] for a in annotators for c in courts
    }

    for court in courts:
        court_pool = pool[pool["source_video"] == court]
        target = targets_per_court[court]

        # Interleave by velocity bucket so each annotator gets a velocity mix within the court
        bucket_groups: dict[int, list[int]] = {}
        for _, row in court_pool.iterrows():
            bucket_groups.setdefault(int(row["bucket"]), []).append(row.name)

        # Round-robin across buckets to build a balanced order
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

        # Assign to annotators round-robin, skipping any annotator already at their court target
        ann_cursor = 0
        for row_idx in ordered:
            placed = False
            for _ in range(len(annotators)):
                a = annotators[ann_cursor % len(annotators)]
                ann_cursor += 1
                if len(per_annotator_court[(a, court)]) < target:
                    per_annotator_court[(a, court)].append(row_idx)
                    placed = True
                    break
            if not placed:
                # Every annotator is full for this court — stop
                break

    # Materialize assignments
    out_chunks: list[pd.DataFrame] = []
    for a in annotators:
        for c in courts:
            indices = per_annotator_court[(a, c)]
            if not indices:
                continue
            chunk = pool.loc[indices].copy()
            chunk["assigned_to"] = a
            out_chunks.append(chunk)
    if not out_chunks:
        return pd.DataFrame(columns=list(pool.columns) + ["assigned_to"])
    return pd.concat(out_chunks, ignore_index=True)


# Back-compat alias
_stratified_round_robin = _balanced_assign


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign annotation clips to annotators")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--annotators", required=True, nargs="+",
                        help="Annotator names, e.g. kenny anik joseph michelle sienna")
    parser.add_argument("--per-person", type=int, default=500)
    parser.add_argument("--min-velocity", type=float, default=DEFAULT_MIN_VELOCITY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without writing the queue file")
    args = parser.parse_args()

    annotators = [a.strip().lower() for a in args.annotators]

    # Load existing queue (may be empty / pre-assigned_to schema)
    if args.queue.exists():
        existing = read_manifest(args.queue)
        if "assigned_to" not in existing.columns:
            existing["assigned_to"] = ""
    else:
        existing = pd.DataFrame(columns=MANIFEST_COLUMNS)

    labeled_mask = existing["label"].astype(str) != ""

    # Already-labeled rows are out of scope for assignment — preserve as-is for training
    untouched_labeled = existing[labeled_mask].copy()

    # Reusable: queued rows that aren't labeled (already in Supabase, no upload needed)
    reusable = existing[~labeled_mask].copy()

    # Fresh pool: manifest rows not in queue and not labeled
    fresh = _load_pool(
        manifest_path=args.manifest,
        min_velocity=args.min_velocity,
        exclude_clip_ids=set(existing["clip_id"]),
    )

    # Combine into a single pool, with `priority=0` for reusable (preferred) and `1` for fresh
    if not reusable.empty:
        reusable["wrist_velocity_num"] = pd.to_numeric(
            reusable["wrist_velocity"], errors="coerce"
        ).fillna(0.0)
        reusable["bucket"] = reusable["wrist_velocity_num"].apply(_bucket_label)
    reusable["priority"] = 0
    fresh["priority"] = 1

    pool = pd.concat([reusable, fresh], ignore_index=True)
    pool = pool.sort_values(["priority"], kind="stable").reset_index(drop=True)
    print(f"Pool size: {len(pool):,} (reusable {len(reusable):,} + fresh {len(fresh):,})")

    print("\nPool by court:")
    print(pool["source_video"].apply(lambda s: Path(str(s)).name).value_counts().to_string())

    # Assign with per-court balance per person
    assigned = _balanced_assign(
        pool=pool,
        annotators=annotators,
        per_person=args.per_person,
        seed=args.seed,
    )

    # Build the new queue.csv: untouched labeled rows + balanced assignments
    out = pd.concat(
        [
            untouched_labeled,
            assigned.drop(
                columns=[c for c in ("wrist_velocity_num", "bucket", "priority") if c in assigned.columns],
                errors="ignore",
            ),
        ],
        ignore_index=True,
    )

    for col in MANIFEST_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[MANIFEST_COLUMNS]

    # Final summary
    print("\n=== Final assignment ===")
    for a in annotators:
        rows = out[out["assigned_to"] == a]
        per_court = rows["source_video"].apply(lambda s: Path(str(s)).name).value_counts().to_dict()
        court_str = ", ".join(f"{c}={n}" for c, n in sorted(per_court.items()))
        print(f"  {a}: {len(rows)} clips  ({court_str})")
    print(f"Total assigned rows: {(out['assigned_to'].isin(annotators)).sum()}")
    print(f"Plus {len(untouched_labeled)} pre-labeled (untouched, training data)")

    if args.dry_run:
        print("\nDry run — not writing.")
        return

    args.queue.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(args.queue, out)
    print(f"\nQueue written → {args.queue}")
    print("\nNext steps:")
    print(f"  1. Run schema migration: supabase/migrations/20260502_add_assigned_to.sql")
    print(f"  2. Extract any clips not yet on disk:")
    print(f"     python -m backend.tools.extract_by_manifest --queue {args.queue} \\")
    print(f"         --videos-dir data/raw_videos --clips-dir data/annotation/clips")
    print(f"  3. Upload new clips: python -m backend.tools.upload_clips --manifest {args.queue} --clips-dir data/annotation/clips")
    print(f"  4. Seed DB: python -m backend.tools.seed_swing_labels --csv {args.queue}")


if __name__ == "__main__":
    main()
