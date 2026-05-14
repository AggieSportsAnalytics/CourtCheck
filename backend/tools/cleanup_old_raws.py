"""One-shot cleanup of raw uploads for already-processed recordings.

The pipeline only deletes the raw video for runs that succeed *after* the
DELETE_RAW_AFTER_PROCESS feature shipped. Existing rows still have their raw
sitting in the `raw-videos` bucket — this script reclaims that storage in one
pass.

Safety:
  - Dry-run by default (`--apply` required to actually delete).
  - Only touches rows where `status='done'` and a processed video lives in
    `results_path`; rows mid-flight or failed are skipped.
  - Verifies the processed video is reachable in `results` before deleting the
    raw, so a corrupted row doesn't lose the only copy.
  - Prints every action; pipe to a log if you want a paper trail.

Usage:
    # Dry run (default):
    python -m backend.tools.cleanup_old_raws

    # Actually delete:
    python -m backend.tools.cleanup_old_raws --apply

    # Just look at one match id:
    python -m backend.tools.cleanup_old_raws --match-id <uuid> --apply
"""

from __future__ import annotations

import argparse
import os
import sys
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


def cleanup(match_ids: Iterable[str] | None, apply: bool) -> None:
    supabase = _get_supabase()

    query = (
        supabase.table("matches")
        .select("id, name, input_path, results_path, status")
        .eq("status", "done")
        .not_.is_("input_path", "null")
    )
    if match_ids:
        ids = list(match_ids)
        query = query.in_("id", ids)
    rows = query.execute().data or []

    if not rows:
        print("No eligible rows found (no status=done rows with a non-null input_path).")
        return

    print(f"Found {len(rows)} candidate rows. Mode: {'APPLY' if apply else 'DRY RUN'}\n")

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
        help="Restrict the run to one or more match ids. Repeatable.",
    )
    args = parser.parse_args()
    cleanup(args.match_ids, apply=args.apply)


if __name__ == "__main__":
    main()
