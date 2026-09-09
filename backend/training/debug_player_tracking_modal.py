"""One-shot Modal function: run player tracking on a clip and print the
[Player] diagnostic line (including near-fallback count) so we know
whether the strict-track_id-fix actually fired.

Usage:
    modal run backend/training/debug_player_tracking_modal.py \\
        --clip data/bounce_train/clips/court2_pick01.mp4 \\
        --camera-id uc_davis_court2
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

app = modal.App("tennis-debug-player")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg", "libgomp1")
    .env({"PYTHONUNBUFFERED": "1"})
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("backend")
    .add_local_dir("backend/weights", remote_path="/root/backend/weights")
    .add_local_dir("backend/calibration_frames", remote_path="/root/backend/calibration_frames")
)


@app.function(image=image, gpu="A10G", timeout=600)
def track(clip_bytes: bytes, filename: str, camera_id: str) -> str:
    import tempfile
    import cv2
    from backend.models.player_tracker import PlayerTracker
    from backend.vision.calibration import load_calibration
    from backend.vision.court_reference import CourtReference

    print(f"[debug] tracking {filename} (camera_id={camera_id})")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(clip_bytes)
        path = tmp.name

    H_ref, H_frame, _ = load_calibration(
        "/root/backend/calibration_frames/court_calibration.json",
        camera_id,
    )
    court_ref = CourtReference()

    tracker = PlayerTracker(
        model_path="yolov8m-pose.pt",
        device="cuda",
        imgsz=1280,
        conf=0.05,
    )

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[debug] {total} frames @ {fps:.1f} fps")

    player_detections = []
    pose_keypoints_per_frame = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pd, kp = tracker.detect_frame_with_far_roi(frame, H_frame)
        player_detections.append(pd)
        pose_keypoints_per_frame.append(kp)
        i += 1
        if i % 600 == 0:
            print(f"[debug] {i}/{total}")
    cap.release()

    print(f"[debug] running choose_and_filter_players")
    filtered_players, filtered_poses = tracker.choose_and_filter_players(
        H_ref=H_ref,
        player_detections=player_detections,
        pose_keypoints_per_frame=pose_keypoints_per_frame,
        court_ref=court_ref,
    )

    # near slot is stored under near_id; far is stored under the FAR offset (-1000).
    # "near present" = the frame has a key that is NOT the far offset.
    FAR = PlayerTracker._FAR_ROI_ID_OFFSET

    def _near_present(f):
        return any(tid != FAR for tid in f)

    def _far_present(f):
        return FAR in f

    # Compute per-5-sec near-present timeline
    bin_size = int(5 * (fps or 30.0))
    timeline = []
    for start in range(0, total, bin_size):
        end = min(start + bin_size, total)
        present = sum(1 for f in filtered_players[start:end] if _near_present(f))
        n = end - start
        pct = 100 * present / n if n else 0
        timeline.append(f"  t={start/(fps or 30.0):5.1f}-{end/(fps or 30.0):5.1f}s  near={present}/{n} ({pct:.0f}%)")

    near_total = sum(1 for f in filtered_players if _near_present(f))
    far_total = sum(1 for f in filtered_players if _far_present(f))
    print(f"\n[debug] FINAL: near_present={near_total}/{total} ({100*near_total/total:.1f}%), far_present={far_total}/{total} ({100*far_total/total:.1f}%)")
    print("[debug] per-5sec near-present (TRACKED):")
    for line in timeline:
        print(line)

    # --- Second pass: RAW detection (predict-only, no tracking) ---
    # Isolates detection from tracking. If P1 is present in raw YOLO output but
    # absent from the tracked timeline after the walk-off, the failure is
    # BoT-SORT track confirmation, not detection.
    from ultralytics import YOLO
    from backend.models.player_tracker import _project_foot

    net_y = court_ref.net[0][1]
    left_x = court_ref.left_court_line[0][0]
    right_x = court_ref.right_court_line[0][0]
    x_margin = 500
    NEAR_MIN_H = 80  # a real near player is large; filters tiny far/noise dets

    det = YOLO("yolov8m-pose.pt")
    det.to("cuda")

    cap = cv2.VideoCapture(path)
    raw_near_present = [False] * total
    j = 0
    while True:
        ret, frame = cap.read()
        if not ret or j >= total:
            break
        res = det.predict(frame, verbose=False, conf=0.05, imgsz=1280, half=True)[0]
        found = False
        if res.boxes is not None:
            names = res.names
            for box in res.boxes:
                if names[int(box.cls.item())] != "person":
                    continue
                bb = box.xyxy[0].tolist()
                if (bb[3] - bb[1]) < NEAR_MIN_H:
                    continue
                proj = _project_foot(bb, H_ref)
                if proj is None:
                    continue
                cx, cy = proj
                if not (left_x - x_margin <= cx <= right_x + x_margin):
                    continue
                if cy > net_y:  # near side
                    found = True
                    break
        raw_near_present[j] = found
        j += 1
        if j % 600 == 0:
            print(f"[debug] raw-pass {j}/{total}")
    cap.release()

    print("[debug] per-5sec near-present (RAW predict, h>=80px, near side):")
    for start in range(0, total, bin_size):
        end = min(start + bin_size, total)
        present = sum(1 for v in raw_near_present[start:end] if v)
        n = end - start
        pct = 100 * present / n if n else 0
        print(f"  t={start/(fps or 30.0):5.1f}-{end/(fps or 30.0):5.1f}s  raw-near={present}/{n} ({pct:.0f}%)")

    raw_total = sum(1 for v in raw_near_present if v)
    print(f"[debug] RAW near total: {raw_total}/{total} ({100*raw_total/total:.1f}%)")

    os.unlink(path)
    return "OK"


@app.local_entrypoint()
def main(clip: str, camera_id: str = "uc_davis_court2"):
    p = Path(clip).resolve()
    if not p.exists():
        print(f"not found: {p}", file=sys.stderr)
        sys.exit(1)
    result = track.remote(p.read_bytes(), p.name, camera_id)
    print(f"[local] result: {result}")
