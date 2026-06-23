"""
Cut 2-minute training clips from a raw match video for bounce annotation.

Picks evenly spaced segments across the source video and re-encodes them to
H.264 (browsers + OpenCV-friendly, matches the upload-path transcode).

Usage:
    python -m backend.training.cut_bounce_clips \
        --source data/raw_videos/StMarys_Court2.mp4 \
        --segments 3 \
        --out-dir data/bounce_train/clips

The default produces filenames like `court2_seg01.mp4` that
extract_ball_trajectory.py's auto-detect can map to a calibrated camera id.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


_CAMERA_TOKEN = re.compile(r"court\d[_-]?zoom|court\d", re.I)


def _camera_token(source: Path) -> str:
    m = _CAMERA_TOKEN.search(source.stem)
    return (m.group(0).lower().replace("-", "_") if m else source.stem.lower())


def _probe_duration(source: Path) -> float:
    """Return source duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(source),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


def _cut(source: Path, start_s: float, dur_s: float, out: Path) -> None:
    """ffmpeg cut + H.264 re-encode (browser + OpenCV friendly)."""
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-i", str(source),
        "-t", f"{dur_s:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        str(out),
    ]
    print(f"[cut] {' '.join(cmd[:8])} ... {out.name}")
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cut 2-min training clips for bounce annotation")
    ap.add_argument("--source", required=True, help="raw match video, e.g. data/raw_videos/StMarys_Court2.mp4")
    ap.add_argument("--segments", type=int, default=3, help="number of segments to cut (default 3)")
    ap.add_argument("--duration", type=float, default=120.0, help="seconds per segment (default 120)")
    ap.add_argument("--lead", type=float, default=300.0, help="skip seconds at the start of the source (default 300)")
    ap.add_argument("--out-dir", default="data/bounce_train/clips")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ffmpeg + ffprobe must be on PATH", file=sys.stderr)
        return 1

    source = Path(args.source).resolve()
    if not source.exists():
        print(f"source not found: {source}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = _probe_duration(source)
    usable = max(0.0, duration - args.lead - args.duration)
    if usable <= 0 or args.segments < 1:
        print(f"source too short for {args.segments}x{args.duration}s with {args.lead}s lead", file=sys.stderr)
        return 1

    step = usable / max(1, args.segments - 1) if args.segments > 1 else 0.0
    cam = _camera_token(source)

    for i in range(args.segments):
        start = args.lead + i * step
        out = out_dir / f"{cam}_seg{i + 1:02d}.mp4"
        if out.exists():
            print(f"[cut] skip existing {out.name}")
            continue
        _cut(source, start, args.duration, out)
    print(f"[cut] done -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
