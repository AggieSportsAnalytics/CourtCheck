"""
Extract a per-frame ball trajectory from a clip using BallDetector
(main TrackNet pass + far-court ROI backfill, matching the live pipeline).

Output is `<clip>.ball.npz` next to the clip with arrays:
    x_ball   float32 (T,)   NaN where TrackNet missed the ball
    y_ball   float32 (T,)
    fps      float
    total    int
    camera_id  str (or empty)

Decoupled from backend/pipeline/run.py so retrain workflows don't have to spin
up the full pipeline (court detector, player tracker, stroke classifier, etc.).
The 12-feature CatBoost input only needs (x_ball, y_ball).

Usage:
    python -m backend.training.extract_ball_trajectory \
        --clip data/bounce_train/clips/court2_seg01.mp4 \
        --camera-id uc_davis_court2

    # auto-detect camera_id from filename (matches uc_davis_<id> tokens):
    python -m backend.training.extract_ball_trajectory \
        --clip data/bounce_train/clips/court2_seg01.mp4

    # batch mode — glob:
    python -m backend.training.extract_ball_trajectory \
        --glob 'data/bounce_train/clips/*.mp4'
"""
from __future__ import annotations

import argparse
import glob as glob_mod
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.models import BallDetector
from backend.vision.calibration import load_calibration


_DEFAULT_CALIBRATION = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "calibration_frames", "court_calibration.json")
)
_DEFAULT_WEIGHTS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "weights", "tracknet_v2_official.pt")
)

_CAMERA_PATTERNS = [
    ("uc_davis_court1_zoomed", re.compile(r"court1[_-]?zoom", re.I)),
    ("uc_davis_court1",        re.compile(r"court1", re.I)),
    ("uc_davis_court2",        re.compile(r"court2", re.I)),
    ("uc_davis_court4",        re.compile(r"court4", re.I)),
    ("uc_davis_court6",        re.compile(r"court6", re.I)),
]


def detect_camera_id(clip_path: Path) -> str | None:
    name = clip_path.name
    for cam_id, pat in _CAMERA_PATTERNS:
        if pat.search(name):
            return cam_id
    return None


def extract_trajectory(
    clip_path: Path,
    camera_id: str | None,
    weights_path: str = _DEFAULT_WEIGHTS,
    calibration_path: str = _DEFAULT_CALIBRATION,
    device: str | None = None,
) -> Path:
    if not clip_path.exists():
        raise FileNotFoundError(f"clip not found: {clip_path}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    H_frame = None
    if camera_id:
        try:
            _, H_frame, _ = load_calibration(calibration_path, camera_id)
            print(f"[extract] loaded calibration for {camera_id}")
        except Exception as exc:
            print(f"[extract] WARN no calibration for {camera_id}: {exc} — main-pass only")

    detector = BallDetector(path_model=weights_path, device=device)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {clip_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    xs = np.full(total, np.nan, dtype=np.float32)
    ys = np.full(total, np.nan, dtype=np.float32)

    frame_idx = 0
    backfilled = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if H_frame is not None:
            x_pred, y_pred = detector.infer_with_far_roi(frame, H_frame)
        else:
            x_pred, y_pred = detector.infer_single(frame)

        if x_pred is not None and y_pred is not None:
            xs[frame_idx] = float(x_pred)
            ys[frame_idx] = float(y_pred)

        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"[extract] {frame_idx}/{total}")

    cap.release()
    backfilled = int(detector.roi_backfill_count)
    detected = int(np.sum(~np.isnan(xs)))
    print(f"[extract] done — {detected}/{total} detected ({backfilled} ROI backfills)")

    out_path = clip_path.with_suffix(clip_path.suffix + ".ball.npz")
    np.savez_compressed(
        out_path,
        x_ball=xs,
        y_ball=ys,
        fps=float(fps),
        total=int(total),
        camera_id=str(camera_id or ""),
    )
    print(f"[extract] wrote {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract per-frame ball trajectory for bounce training")
    ap.add_argument("--clip", help="path to single clip .mp4")
    ap.add_argument("--glob", help="glob for multiple clips, e.g. 'data/bounce_train/clips/*.mp4'")
    ap.add_argument("--camera-id", help="override camera id (otherwise auto-detect from filename)")
    ap.add_argument("--weights", default=_DEFAULT_WEIGHTS)
    ap.add_argument("--calibration", default=_DEFAULT_CALIBRATION)
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available else cpu)")
    ap.add_argument("--skip-existing", action="store_true", help="skip clips that already have .ball.npz")
    args = ap.parse_args()

    if not args.clip and not args.glob:
        ap.error("--clip or --glob required")

    clips: list[Path] = []
    if args.clip:
        clips.append(Path(args.clip).resolve())
    if args.glob:
        clips.extend(sorted(Path(p).resolve() for p in glob_mod.glob(args.glob)))

    if not clips:
        print("no clips matched", file=sys.stderr)
        return 1

    for clip in clips:
        out = clip.with_suffix(clip.suffix + ".ball.npz")
        if args.skip_existing and out.exists():
            print(f"[extract] skip (exists): {out}")
            continue
        cam = args.camera_id or detect_camera_id(clip)
        if not cam:
            print(f"[extract] WARN could not infer camera_id from {clip.name} — running without ROI pass")
        try:
            extract_trajectory(
                clip_path=clip,
                camera_id=cam,
                weights_path=args.weights,
                calibration_path=args.calibration,
                device=args.device,
            )
        except Exception as exc:
            print(f"[extract] FAIL {clip.name}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
