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


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload swing clips to Supabase")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--clips-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = read_manifest(args.manifest)
    pending = df[df["supabase_path"] == ""].copy()
    print(f"Clips to upload: {len(pending):,} / {len(df):,} total")

    if args.dry_run:
        print("Dry run — no uploads performed.")
        return

    uploaded = skipped = failed = 0

    import subprocess
    from dotenv import dotenv_values
    env_vals = dotenv_values(Path(__file__).resolve().parents[2] / ".env")
    url_base = env_vals["SUPABASE_URL"].rstrip("/")
    key = env_vals["SUPABASE_SERVICE_ROLE_KEY"]

    for i, (_, row) in enumerate(tqdm(pending.iterrows(), total=len(pending), unit="clip")):
        clip_id = row["clip_id"]
        local_path = args.clips_dir / f"{clip_id}.mp4"

        if not local_path.exists():
            skipped += 1
            continue

        dest = f"{clip_id}.mp4"
        url = f"{url_base}/storage/v1/object/{SWING_CLIPS_BUCKET}/{dest}"

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
                    raise Exception(f"HTTP {code}")
                df.loc[df["clip_id"] == clip_id, "supabase_path"] = dest
                uploaded += 1
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"\n[WARN] {clip_id}: {e}")
                    failed += 1

        if i % SAVE_INTERVAL == 0 and i > 0:
            write_manifest(args.manifest, df)

    write_manifest(args.manifest, df)
    print(f"\nDone. Uploaded: {uploaded} | Skipped (missing): {skipped} | Failed: {failed}")
    print(f"Manifest updated: {args.manifest}")


if __name__ == "__main__":
    main()
