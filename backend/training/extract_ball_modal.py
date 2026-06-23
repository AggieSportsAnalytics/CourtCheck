"""
One-shot Modal function for ball trajectory extraction on A10G GPU.

Local CPU TrackNet runs at ~0.3 fps (3-5 hr per 2-min clip); A10G runs at
~60 fps (~1 min per clip). For the bounce-retrain workflow this is the
difference between "extract overnight" and "extract while making coffee."

Run locally (does not deploy the live tennis-modal app):
    modal run backend/training/extract_ball_modal.py \\
        --glob 'data/bounce_train/clips/*.mp4'

Writes <clip>.ball.npz next to each input clip — same format as
extract_ball_trajectory.py.

App ("tennis-bounce-extract") is ephemeral. `modal run` brings it up, dispatches
the work, writes results, stops. Nothing stays running.
"""
from __future__ import annotations

import glob as glob_mod
import io
import os
import re
import sys
from pathlib import Path

import modal


app = modal.App("tennis-bounce-extract")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg", "libgomp1")
    .env({"PYTHONUNBUFFERED": "1"})
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("backend")
    .add_local_dir("backend/weights", remote_path="/root/backend/weights")
    .add_local_dir("backend/calibration_frames", remote_path="/root/backend/calibration_frames")
)


_CAMERA_PATTERNS = [
    ("uc_davis_court1_zoomed", re.compile(r"court1[_-]?zoom", re.I)),
    ("uc_davis_court1",        re.compile(r"court1", re.I)),
    ("uc_davis_court2",        re.compile(r"court2", re.I)),
    ("uc_davis_court4",        re.compile(r"court4", re.I)),
    ("uc_davis_court6",        re.compile(r"court6", re.I)),
]


def _detect_camera_id(name: str) -> str | None:
    for cam, pat in _CAMERA_PATTERNS:
        if pat.search(name):
            return cam
    return None


@app.function(image=image, gpu="A10G", timeout=900)
def extract_one(clip_bytes: bytes, filename: str, camera_id: str | None) -> bytes:
    """TrackNet (+ ROI when camera_id set) on a clip — returns .npz bytes."""
    import tempfile
    import cv2
    import numpy as np
    import torch
    from backend.models import BallDetector
    from backend.vision.calibration import load_calibration

    print(f"[modal-extract] {filename} (camera_id={camera_id})  gpu={torch.cuda.is_available()}")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(clip_bytes)
        tmp_path = tmp.name

    H_frame = None
    if camera_id:
        try:
            _, H_frame, _ = load_calibration(
                "/root/backend/calibration_frames/court_calibration.json",
                camera_id,
            )
            print(f"[modal-extract] loaded calibration {camera_id}")
        except Exception as exc:
            print(f"[modal-extract] WARN no calibration: {exc}")

    detector = BallDetector(
        path_model="/root/backend/weights/tracknet_v2_official.pt",
        device="cuda",
    )

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {filename}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    xs = np.full(total, np.nan, dtype=np.float32)
    ys = np.full(total, np.nan, dtype=np.float32)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if H_frame is not None:
            xp, yp = detector.infer_with_far_roi(frame, H_frame)
        else:
            xp, yp = detector.infer_single(frame)
        if xp is not None and yp is not None:
            xs[idx] = float(xp)
            ys[idx] = float(yp)
        idx += 1
        if idx % 600 == 0:
            print(f"[modal-extract] {filename} {idx}/{total}")
    cap.release()

    detected = int(np.sum(~np.isnan(xs)))
    backfilled = int(getattr(detector, "roi_backfill_count", 0))
    print(f"[modal-extract] {filename} done — {detected}/{total} detected ({backfilled} ROI backfills)")

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        x_ball=xs,
        y_ball=ys,
        fps=float(fps),
        total=int(total),
        camera_id=str(camera_id or ""),
    )
    os.unlink(tmp_path)
    return buf.getvalue()


@app.local_entrypoint()
def main(clip: str = "", glob: str = "", skip_existing: bool = True):
    if not clip and not glob:
        print("--clip or --glob required", file=sys.stderr)
        sys.exit(1)

    clips: list[Path] = []
    if clip:
        clips.append(Path(clip).resolve())
    if glob:
        clips.extend(sorted(Path(p).resolve() for p in glob_mod.glob(glob)))

    if not clips:
        print("no clips matched", file=sys.stderr)
        sys.exit(1)

    work: list[tuple[Path, bytes, str | None]] = []
    for c in clips:
        out = c.with_suffix(c.suffix + ".ball.npz")
        if skip_existing and out.exists():
            print(f"[local] skip existing {out.name}")
            continue
        cam = _detect_camera_id(c.name)
        if cam is None:
            print(f"[local] WARN no camera id for {c.name} — running main pass only")
        work.append((c, c.read_bytes(), cam))

    if not work:
        print("[local] nothing to do")
        return

    print(f"[local] dispatching {len(work)} clip(s) to Modal (spawn + collect)…")

    # spawn() + per-clip handle.get() isolates failures — one bad clip can't
    # cancel siblings the way starmap's first-error propagation does.
    handles = [(path, extract_one.spawn(blob, path.name, cam))
               for path, blob, cam in work]

    ok, fail = 0, []
    for path, handle in handles:
        try:
            npz_bytes = handle.get(timeout=900)
        except Exception as exc:
            print(f"[local] FAIL {path.name}: {type(exc).__name__}: {exc}")
            fail.append(path.name)
            continue
        out = path.with_suffix(path.suffix + ".ball.npz")
        out.write_bytes(npz_bytes)
        print(f"[local] wrote {out.name} ({len(npz_bytes)/1024:.1f} KB)")
        ok += 1

    print(f"[local] done — {ok}/{len(work)} succeeded"
          + (f" | failed: {fail}" if fail else ""))
