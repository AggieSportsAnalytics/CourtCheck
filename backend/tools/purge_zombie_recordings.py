"""Purge zombie recordings — rows in `matches` that are marked `status='done'`
but have no playable processed video in the `results` bucket.

These pile up from early pipeline runs that wrote a DB row + raw upload but
never persisted the processed mp4. They cost raw-storage and clutter the UI
without being viewable.

Behaviour:
  - Identifies rows where (status='done' AND results_path IS NULL) OR
    (status='done' AND processed file not present in results/<match_id>/).
  - Best-effort delete of the raw upload from `raw-videos`.
  - Best-effort delete of any orphaned heatmap PNGs in results/<match_id>/.
  - Deletes the DB row.

Safety:
  - Dry-run by default. `--apply` required to actually delete.
  - Will NEVER touch a row whose processed video IS in storage. Use
    cleanup_old_raws.py for those instead.

Usage:
    python -m backend.tools.purge_zombie_recordings        # dry run
    python -m backend.tools.purge_zombie_recordings --apply
"""

from __future__ import annotations

import argparse
import os
import sys

from supabase import create_client


def _get_supabase():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.")
        sys.exit(1)
    return create_client(url, key)


def _processed_present(supabase, results_path: str | None) -> bool:
    if not results_path:
        return False
    prefix, _, filename = results_path.rpartition("/")
    try:
        listing = supabase.storage.from_("results").list(prefix or "")
        names = {item.get("name") for item in (listing or [])}
        return filename in names
    except Exception:
        # If we can't list, assume present (the safer default for "don't delete").
        return True


def _list_results_folder(supabase, match_id: str) -> list[str]:
    try:
        listing = supabase.storage.from_("results").list(match_id) or []
        return [f"{match_id}/{item.get('name')}" for item in listing if item.get("name")]
    except Exception:
        return []


def purge(apply: bool) -> None:
    supabase = _get_supabase()

    rows = (
        supabase.table("matches")
        .select("id, name, input_path, results_path, created_at")
        .eq("status", "done")
        .execute()
        .data
        or []
    )

    zombies = []
    for row in rows:
        if not _processed_present(supabase, row.get("results_path")):
            zombies.append(row)

    if not zombies:
        print("No zombie rows found. Nothing to purge.")
        return

    print(f"Found {len(zombies)} zombie rows. Mode: {'APPLY' if apply else 'DRY RUN'}\n")

    rows_deleted = 0
    raws_deleted = 0
    results_objs_deleted = 0
    failures = 0

    for row in zombies:
        mid = row["id"]
        name = row.get("name") or "(unnamed)"
        raw_key = row.get("input_path")
        results_path = row.get("results_path")
        print(f"- {mid} {name!r}")
        print(f"    raw: {raw_key}")
        print(f"    results_path: {results_path}")

        # Find any leftover files in results/<match_id>/ (heatmaps from a half
        # run, etc.) so we can sweep them too.
        leftover_results = _list_results_folder(supabase, mid)
        if leftover_results:
            print(f"    leftover results objects: {leftover_results}")

        if not apply:
            print("    would delete raw + leftover results + DB row (dry run)")
            continue

        # 1. Delete raw upload (best-effort)
        if raw_key:
            try:
                supabase.storage.from_("raw-videos").remove([raw_key])
                raws_deleted += 1
                print("    deleted raw")
            except Exception as e:
                print(f"    ! failed to delete raw: {e}")
                failures += 1

        # 2. Delete leftover heatmaps/etc in results bucket (best-effort)
        if leftover_results:
            try:
                supabase.storage.from_("results").remove(leftover_results)
                results_objs_deleted += len(leftover_results)
                print(f"    deleted {len(leftover_results)} leftover results obj(s)")
            except Exception as e:
                print(f"    ! failed to delete results leftovers: {e}")
                failures += 1

        # 3. Delete DB row
        try:
            supabase.table("matches").delete().eq("id", mid).execute()
            rows_deleted += 1
            print("    deleted DB row")
        except Exception as e:
            print(f"    ! failed to delete DB row: {e}")
            failures += 1

    print()
    if apply:
        print(
            f"Deleted: {rows_deleted} DB rows · {raws_deleted} raw uploads · "
            f"{results_objs_deleted} leftover results objects · "
            f"{failures} failures"
        )
    else:
        print(f"Would delete {len(zombies)} DB rows + their raws + any leftover results objects.")
        print("Re-run with --apply to do it.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete. Default is dry run.",
    )
    args = parser.parse_args()
    purge(apply=args.apply)


if __name__ == "__main__":
    main()
