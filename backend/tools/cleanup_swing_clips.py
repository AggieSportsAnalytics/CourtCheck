"""
Delete labeled swing clip files from Supabase storage.

Labels in the `swing_labels` table are tiny and durable. The clip files in the
`swing-clips` bucket are bulky and only needed during active labeling. Once
labels are done and extracted to a local manifest, the clips can be reclaimed.

Default behavior: deletes ALL clips listed in the bucket (Brian confirmed
labels are done + extracted on 2026-05-13). Use --only-labeled to instead
delete clips that have a non-empty `label` field (safer if some labeling is
still in progress).

The label rows are preserved. supabase_path is cleared on deleted rows so
nothing thinks the storage still has the file.

Usage:
    python -m backend.tools.cleanup_swing_clips --dry-run
    python -m backend.tools.cleanup_swing_clips
    python -m backend.tools.cleanup_swing_clips --only-labeled
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from backend.tools._swing_io import (
    SWING_CLIPS_BUCKET,
    SWING_LABELS_TABLE,
    get_supabase_client,
)


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be deleted, do not delete.",
    )
    parser.add_argument(
        "--only-labeled",
        action="store_true",
        help="Only delete clips whose swing_labels row has a non-empty label.",
    )
    parser.add_argument(
        "--keep-rows",
        action="store_true",
        help="Skip clearing supabase_path on label rows (default: clear it).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=100,
        help="Storage delete batch size (Supabase caps at ~100 per call).",
    )
    args = parser.parse_args()

    client = get_supabase_client()
    storage = client.storage.from_(SWING_CLIPS_BUCKET)

    # 1. List all files in the bucket. Paginate; Supabase caps list at 100/call.
    print(f"Listing files in bucket '{SWING_CLIPS_BUCKET}'...")
    all_files = []
    offset = 0
    page_size = 100
    while True:
        batch = storage.list(
            path="",
            options={"limit": page_size, "offset": offset, "sortBy": {"column": "name", "order": "asc"}},
        )
        if not batch:
            break
        all_files.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    if not all_files:
        print("Bucket is empty. Nothing to delete.")
        return 0

    total_bytes = sum(int(f.get("metadata", {}).get("size") or 0) for f in all_files)
    print(f"Found {len(all_files)} files, total {_format_bytes(total_bytes)}.")

    # 2. Optionally filter to clips with a label set.
    paths_to_delete: list[str] = [f["name"] for f in all_files]

    if args.only_labeled:
        print("Filtering to labeled clips...")
        rows = (
            client.table(SWING_LABELS_TABLE)
            .select("clip_id, supabase_path, label")
            .neq("label", "")
            .execute()
            .data
            or []
        )
        labeled_paths = {r["supabase_path"] for r in rows if r.get("supabase_path")}
        before = len(paths_to_delete)
        paths_to_delete = [p for p in paths_to_delete if p in labeled_paths]
        print(f"  {before} files in bucket -> {len(paths_to_delete)} labeled.")

    if not paths_to_delete:
        print("Nothing matches the filter. Exiting.")
        return 0

    # 3. Dry-run summary
    if args.dry_run:
        print(f"\nDRY RUN: would delete {len(paths_to_delete)} files.")
        print(f"First 5: {paths_to_delete[:5]}")
        return 0

    # 4. Confirm
    confirm = input(
        f"\nDelete {len(paths_to_delete)} files from '{SWING_CLIPS_BUCKET}'? "
        f"(~{_format_bytes(total_bytes)}) [yes/N]: "
    ).strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return 1

    # 5. Delete in batches
    deleted = 0
    errors: list[str] = []
    for i in range(0, len(paths_to_delete), args.batch):
        chunk = paths_to_delete[i : i + args.batch]
        try:
            storage.remove(chunk)
            deleted += len(chunk)
            print(f"  Deleted {deleted}/{len(paths_to_delete)}...")
        except Exception as e:
            errors.append(f"chunk starting {chunk[0]}: {e}")

    print(f"\nDeleted {deleted} files.")
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:5]:
            print(f"  {e}")

    # 6. Clear supabase_path on labels so nothing thinks storage still has them
    if not args.keep_rows and deleted:
        print("Clearing supabase_path on affected label rows...")
        chunk_size = 500
        cleared = 0
        for i in range(0, len(paths_to_delete), chunk_size):
            chunk = paths_to_delete[i : i + chunk_size]
            client.table(SWING_LABELS_TABLE).update({"supabase_path": ""}).in_(
                "supabase_path", chunk
            ).execute()
            cleared += len(chunk)
        print(f"  Cleared {cleared} rows.")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
