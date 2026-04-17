"""
Re-encode existing clips from MPEG-4 Part 2 → H.264 and re-upload to Supabase.

Only processes clips listed in the manifest. Transcodes in parallel using a
thread pool (CPU-bound ffmpeg), then re-uploads via curl (same pattern as
upload_clips.py to avoid Python SSL issues on miniforge arm64).

Usage:
    python -m backend.tools.transcode_and_reupload \
        --manifest data/annotation/queue.csv \
        --clips-dir data/annotation/clips

    # Dry run:
    python -m backend.tools.transcode_and_reupload \
        --manifest data/annotation/queue.csv \
        --clips-dir data/annotation/clips \
        --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import dotenv_values
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.tools._swing_io import SWING_CLIPS_BUCKET, read_manifest

MAX_WORKERS = 6
MAX_RETRIES = 3


def _transcode(src: Path, dst: Path) -> bool:
    """Re-encode src (mp4v) → dst (H.264). Returns True on success."""
    result = subprocess.run([
        "ffmpeg", "-y", "-i", str(src),
        "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
        "-movflags", "+faststart",
        str(dst),
    ], capture_output=True)
    return result.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def _upload(local_path: Path, dest: str, url_base: str, key: str) -> bool:
    """Upload via curl (avoids miniforge arm64 SSL issues). Returns True on success."""
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
            if code in (200, 201):
                return True
        except Exception:
            pass
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)
    return False


def _process_clip(clip_id: str, clips_dir: Path, url_base: str, key: str) -> str:
    """Transcode + upload one clip. Returns 'ok', 'missing', or 'failed'."""
    src = clips_dir / f"{clip_id}.mp4"
    if not src.exists():
        return "missing"

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        if not _transcode(src, tmp_path):
            return "failed"
        if not _upload(tmp_path, f"{clip_id}.mp4", url_base, key):
            return "failed"
        return "ok"
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcode clips to H.264 and re-upload")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--clips-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    df = read_manifest(args.manifest)
    clip_ids = df["clip_id"].tolist()
    print(f"Clips to transcode + re-upload: {len(clip_ids):,}")

    if args.dry_run:
        print("Dry run — nothing processed.")
        return

    env_file = Path(__file__).resolve().parents[2] / ".env"
    env_vals = dotenv_values(env_file)
    url_base = env_vals["SUPABASE_URL"].rstrip("/")
    key = env_vals["SUPABASE_SERVICE_ROLE_KEY"]

    ok = missing = failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_clip, cid, args.clips_dir, url_base, key): cid
            for cid in clip_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), unit="clip"):
            status = future.result()
            if status == "ok":
                ok += 1
            elif status == "missing":
                missing += 1
            else:
                failed += 1

    print(f"\nDone. OK: {ok:,} | Missing: {missing:,} | Failed: {failed:,}")


if __name__ == "__main__":
    main()
