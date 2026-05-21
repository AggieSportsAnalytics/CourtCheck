"""Sweep raw uploads for processed recordings past the retention window.

Backend default: raws stay in storage for 7 days after processing so coaches
can hit Reprocess without re-uploading. This script is the cleanup pass that
removes them after that window expires. Intended to run daily (Modal scheduled
function or cron).

Safety:
  - Dry-run by default (`--apply` required to actually delete).
  - Only touches rows where `status='done'` AND `created_at` is older than the
    retention window. Rows mid-flight or failed are skipped.
  - Verifies the processed video is reachable in `results` before deleting the
    raw, so a corrupted row doesn't lose the only copy.
  - Prints every action; pipe to a log if you want a paper trail.

Usage:
    # Default 7-day window, dry run:
    python -m backend.tools.cleanup_old_raws

    # Apply with default window:
    python -m backend.tools.cleanup_old_raws --apply

    # Shorter window (e.g. when storage is tight):
    python -m backend.tools.cleanup_old_raws --min-age-days 3 --apply

    # Backfill / wipe everything regardless of age:
    python -m backend.tools.cleanup_old_raws --min-age-days 0 --apply

    # Just look at one match id (ignores age):
    python -m backend.tools.cleanup_old_raws --match-id <uuid> --apply
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Iterable

from supabase import create_client


def _get_supabase():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print(
            "ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in the env."
        )
        sys.exit(1)
    return create_client(url, key)


def _processed_exists(supabase, results_path: str) -> bool:
    """Confirm the processed video is actually in `results` before nuking raw."""
    if not results_path:
        return False
    # Cheapest check: list the directory and look for the file. `info` would be
    # nicer but isn't on all supabase-py versions.
    try:
        prefix, _, filename = results_path.rpartition("/")
        listing = supabase.storage.from_("results").list(prefix or "")
        names = {item.get("name") for item in listing or []}
        return filename in names
    except Exception as e:
        print(f"  ! could not verify processed video at {results_path}: {e}")
        return False


def cleanup(match_ids: Iterable[str] | None, apply: bool, min_age_days: int) -> None:
    supabase = _get_supabase()

    query = (
        supabase.table("matches")
        .select("id, name, input_path, results_path, status, created_at")
        .eq("status", "done")
        .not_.is_("input_path", "null")
    )
    if match_ids:
        ids = list(match_ids)
        query = query.in_("id", ids)
    elif min_age_days > 0:
        # Only sweep rows older than the retention window. Single-match runs
        # ignore the window (operator already specified the row).
        cutoff = datetime.now(timezone.utc) - timedelta(days=min_age_days)
        query = query.lt("created_at", cutoff.isoformat())
    rows = query.execute().data or []

    if not rows:
        window_note = (
            f" older than {min_age_days} days" if min_age_days > 0 and not match_ids else ""
        )
        print(f"No eligible rows found (no status=done rows{window_note}).")
        return

    print(
        f"Found {len(rows)} candidate rows "
        f"(retention: {min_age_days} days). Mode: {'APPLY' if apply else 'DRY RUN'}\n"
    )

    removed = 0
    skipped = 0
    failed = 0
    for row in rows:
        match_id = row["id"]
        raw_key = row["input_path"]
        results_path = row.get("results_path")
        name = row.get("name") or "(unnamed)"
        print(f"- {match_id} {name!r}")
        print(f"    raw: {raw_key}")
        print(f"    processed: {results_path}")

        if not raw_key:
            print("    skip: no raw key")
            skipped += 1
            continue
        if not results_path:
            print("    skip: no processed video — won't risk losing the raw")
            skipped += 1
            continue
        if not _processed_exists(supabase, results_path):
            print("    skip: processed video not found in results bucket")
            skipped += 1
            continue

        if not apply:
            print("    would delete raw (dry run)")
            removed += 1
            continue

        try:
            supabase.storage.from_("raw-videos").remove([raw_key])
            print("    deleted raw")
            removed += 1
        except Exception as e:
            print(f"    ! failed to delete raw: {e}")
            failed += 1

    print()
    verb = "Deleted" if apply else "Would delete"
    print(f"{verb} {removed} raws · skipped {skipped} · failed {failed}")
    if not apply:
        print("Re-run with --apply to actually perform the deletes.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete. Without this flag the script does a dry run.",
    )
    parser.add_argument(
        "--match-id",
        action="append",
        dest="match_ids",
        help="Restrict the run to one or more match ids. Repeatable. Ignores --min-age-days.",
    )
    parser.add_argument(
        "--min-age-days",
        type=int,
        default=7,
        help="Minimum age (in days, by created_at) before a raw is eligible for cleanup. "
             "Default: 7 (matches backend retention window). Set to 0 to sweep everything.",
    )
    args = parser.parse_args()
    cleanup(args.match_ids, apply=args.apply, min_age_days=args.min_age_days)


if __name__ == "__main__":
    main()
