"""
Upload swing clips to Supabase and patch supabase_path in the manifest.

Uses a thread pool for parallel uploads — typically 20-30x faster than sequential.

Usage:
    python -m backend.tools.upload_clips \
        --manifest data/annotation/queue.csv \
        --clips-dir data/annotation/clips

    # Dry run (no uploads):
    python -m backend.tools.upload_clips \
        --manifest data/annotation/queue.csv \
        --clips-dir data/annotation/clips \
        --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.tools._swing_io import SWING_CLIPS_BUCKET, read_manifest, write_manifest

load_dotenv()

SAVE_INTERVAL = 200
MAX_RETRIES = 3


def _upload_one(clip_id: str, local_path: Path, url_base: str, key: str) -> tuple[str, str | None, str | None]:
    """Upload one clip. Returns (clip_id, supabase_path or None, error or None)."""
    import subprocess
    dest = f"{clip_id}.mp4"
    url = f"{url_base}/storage/v1/object/{SWING_CLIPS_BUCKET}/{dest}"
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            result = subprocess.run([
                "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                "-X", "POST", url,
                "-H", f"Authorization: Bearer {key}",
                "-H", "Content-Type: video/mp4",
                "-H", "x-upsert: true",
                "--data-binary", f"@{local_path}",
            ], capture_output=True, text=True, timeout=120)
            code = int(result.stdout.strip())
            if code not in (200, 201):
                raise RuntimeError(f"HTTP {code}")
            return (clip_id, dest, None)
        except Exception as e:
            last_err = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return (clip_id, None, last_err)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload swing clips to Supabase")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--clips-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16,
                        help="Concurrent upload workers (default: 16)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = read_manifest(args.manifest)
    pending = df[df["supabase_path"] == ""].copy()
    print(f"Clips to upload: {len(pending):,} / {len(df):,} total")

    if args.dry_run:
        print("Dry run — no uploads performed.")
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    from dotenv import dotenv_values
    env_vals = dotenv_values(Path(__file__).resolve().parents[2] / ".env")
    url_base = env_vals["SUPABASE_URL"].rstrip("/")
    key = env_vals["SUPABASE_SERVICE_ROLE_KEY"]

    tasks: list[tuple[str, Path]] = []
    skipped = 0
    for _, row in pending.iterrows():
        clip_id = row["clip_id"]
        local_path = args.clips_dir / f"{clip_id}.mp4"
        if not local_path.exists():
            skipped += 1
            continue
        tasks.append((clip_id, local_path))

    uploaded = failed = 0
    df_lock = Lock()
    save_lock = Lock()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_upload_one, cid, lp, url_base, key): cid
            for cid, lp in tasks
        }
        with tqdm(total=len(tasks), unit="clip") as bar:
            for fut in as_completed(futures):
                clip_id, dest, err = fut.result()
                with df_lock:
                    if dest:
                        df.loc[df["clip_id"] == clip_id, "supabase_path"] = dest
                        uploaded += 1
                    else:
                        failed += 1
                        bar.write(f"[WARN] {clip_id}: {err}")
                completed += 1
                bar.update(1)
                if completed % SAVE_INTERVAL == 0:
                    with save_lock, df_lock:
                        write_manifest(args.manifest, df)

    write_manifest(args.manifest, df)
    print(f"\nDone. Uploaded: {uploaded} | Skipped (missing): {skipped} | Failed: {failed}")
    print(f"Manifest updated: {args.manifest}")


if __name__ == "__main__":
    main()
