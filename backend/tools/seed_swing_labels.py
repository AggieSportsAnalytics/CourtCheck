"""
Seed the Supabase swing_labels table from a manifest CSV.

Upserts all rows — safe to run multiple times (idempotent on clip_id).

Usage:
    python -m backend.tools.seed_swing_labels \
        --csv data/annotation/queue.csv

    # Dry run (print row count, no DB writes):
    python -m backend.tools.seed_swing_labels \
        --csv data/annotation/queue.csv \
        --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.tools._swing_io import db_seed_from_csv, get_supabase_client, read_manifest

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed swing_labels table from CSV")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = read_manifest(args.csv)
    print(f"Rows in CSV: {len(df):,}")

    if args.dry_run:
        print("Dry run — no DB writes.")
        return

    client = get_supabase_client()
    inserted = db_seed_from_csv(args.csv, client=client)
    print(f"Upserted {inserted:,} rows into swing_labels.")


if __name__ == "__main__":
    main()
