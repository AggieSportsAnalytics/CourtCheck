"""
Extract clips listed in a queue.csv by reading window_start/end from the source video.

Skips clips already on disk. Verifies output codec is H.264 (browsers
won't play mpeg4). Re-runs ffmpeg if the codec is wrong.

Usage:
    python -m backend.tools.extract_by_manifest \
        --queue data/annotation/queue.csv \
        --videos-dir data/raw_videos \
        --clips-dir data/annotation/clips
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.tools._swing_io import read_manifest


def _verify_h264(clip_path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_streams", str(clip_path)],
        capture_output=True, text=True,
    )
    return "codec_name=h264" in result.stdout


def _extract_one(
    video_path: Path,
    out_path: Path,
    start: int,
    end: int,
) -> bool:
    """Slice frames [start, end] from `video_path`, write H.264 mp4 to `out_path`."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start))
        writer = cv2.VideoWriter(
            str(tmp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        frames_written = 0
        for _ in range(end - start + 1):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frames_written += 1
        writer.release()
        cap.release()

        if frames_written == 0:
            return False

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(tmp_path),
                "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
                "-movflags", "+faststart",
                str(out_path),
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            return False

        return _verify_h264(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract queue.csv clips from raw videos")
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--videos-dir", required=True, type=Path)
    parser.add_argument("--clips-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0,
                        help="Max clips to extract this run (0 = all)")
    args = parser.parse_args()

    args.clips_dir.mkdir(parents=True, exist_ok=True)
    df = read_manifest(args.queue)

    # Cast numeric fields
    for col in ("window_start", "window_end"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    pending = []
    already = 0
    skipped_uploaded = 0
    for _, row in df.iterrows():
        # Already in Supabase? No need to extract a local copy.
        if str(row.get("supabase_path", "")).strip():
            skipped_uploaded += 1
            continue
        clip_path = args.clips_dir / f"{row['clip_id']}.mp4"
        if clip_path.exists() and _verify_h264(clip_path):
            already += 1
            continue
        pending.append(row)

    if skipped_uploaded:
        print(f"Already in Supabase (skipped): {skipped_uploaded:,}")

    if args.limit:
        pending = pending[: args.limit]

    print(f"Already on disk (H.264): {already:,}")
    print(f"To extract: {len(pending):,}")
    if not pending:
        return

    extracted = failed = 0
    for row in tqdm(pending, unit="clip"):
        video_filename = Path(str(row["source_video"])).name  # e.g. StMarys_Court2.mp4
        video_path = args.videos_dir / video_filename
        if not video_path.exists():
            failed += 1
            continue
        out_path = args.clips_dir / f"{row['clip_id']}.mp4"
        if _extract_one(video_path, out_path, int(row["window_start"]), int(row["window_end"])):
            extracted += 1
        else:
            failed += 1
            out_path.unlink(missing_ok=True)

    print(f"\nExtracted: {extracted} | Failed: {failed}")


if __name__ == "__main__":
    main()
