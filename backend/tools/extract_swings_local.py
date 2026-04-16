"""
Local swing clip extractor — runs on your Mac, no Modal required.

Uses YOLOv8-Pose wrist velocity only (no TrackNet). Designed for
long 4-hour raw recordings. False positives are fine — annotators
label them as 'skip'. Optimized for Apple Silicon MPS.

Leave running overnight for a 4-hour video (~35-40 min on M-series).

Usage:
    python -m backend.tools.extract_swings_local \\
        --videos-dir data/raw_videos \\
        --output-dir data/annotation

    # Single file:
    python -m backend.tools.extract_swings_local \\
        --video-file data/raw_videos/StMarys_Court2.mp4 \\
        --output-dir data/annotation
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.vision.swing_detector import SwingDetector
from backend.tools._swing_io import MANIFEST_COLUMNS, append_rows

DETECTION_INTERVAL = 5   # run YOLO every Nth frame
INFERENCE_SIZE = 640     # lower res = 2x faster than 1280, still detects players fine


def _pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _extract_clip(cap: cv2.VideoCapture, out_path: Path, start: int, end: int, fps: float, size: tuple) -> bool:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(end - start + 1):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    writer.release()
    return out_path.exists() and out_path.stat().st_size > 0


def process_video(video_path: Path, clips_dir: Path, model: YOLO, device: str) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open {video_path.name}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_min = total_frames / fps / 60
    print(f"\n{video_path.name} — {total_frames:,} frames ({duration_min:.0f} min), device={device}")

    pose_kps: list[dict] = []
    player_dets: list[dict] = []
    last_kps: dict = {}
    last_bboxes: dict = {}

    print("  Pass 1: pose detection ...")
    for frame_idx in tqdm(range(total_frames), unit="fr", leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % DETECTION_INTERVAL == 0:
            results = model.track(
                frame, imgsz=INFERENCE_SIZE, conf=0.05,
                persist=True, verbose=False,
                # MPS requires half=False
                half=(device == "cuda"),
            )
            last_kps, last_bboxes = {}, {}
            if results and results[0].keypoints is not None:
                kps_data = results[0].keypoints.data.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                raw_ids = results[0].boxes.id
                ids = raw_ids.cpu().numpy().astype(int) if raw_ids is not None else range(len(boxes))
                for i, tid in enumerate(ids):
                    last_kps[int(tid)] = kps_data[i]
                    last_bboxes[int(tid)] = boxes[i].tolist()

        pose_kps.append(dict(last_kps))
        player_dets.append(dict(last_bboxes))

    cap.release()

    # Dummy ball track — wrist velocity is the sole trigger
    ball_track = [None] * len(pose_kps)
    events = SwingDetector(velocity_threshold=15.0, ball_proximity=9999.0).detect(
        pose_kps, ball_track, player_dets
    )
    print(f"  Found {len(events)} swing events")

    if not events:
        return []

    print("  Pass 2: extracting clips ...")
    cap = cv2.VideoCapture(str(video_path))
    rows = []
    video_stem = video_path.stem

    for event in tqdm(events, unit="clip", leave=False):
        peak, start, end = event["peak_frame"], event["window_start"], event["window_end"]
        player_idx, velocity = event["track_id"], event["wrist_velocity"]

        clip_id = f"{video_stem}_{peak:06d}_p{player_idx}"
        clip_path = clips_dir / f"{clip_id}.mp4"

        ok = _extract_clip(cap, clip_path, start, end, fps, (width, height))
        if not ok:
            continue

        rows.append({
            "clip_id": clip_id,
            "supabase_path": "",          # not uploaded — local only
            "source_video": str(video_path),
            "peak_frame": peak,
            "window_start": start,
            "window_end": end,
            "player_idx": player_idx,
            "wrist_velocity": f"{velocity:.1f}",
            "label": "",
            "annotator": "",
            "labeled_at": "",
        })

    cap.release()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Local swing clip extractor")
    parser.add_argument("--videos-dir", type=Path, help="Directory of raw .mp4 files")
    parser.add_argument("--video-file", type=Path, help="Single video file to process")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.videos_dir and not args.video_file:
        parser.error("Provide --videos-dir or --video-file")

    if args.video_file:
        video_files = [args.video_file]
    else:
        video_files = sorted(args.videos_dir.glob("*.mp4"))
        if not video_files:
            print(f"No .mp4 files in {args.videos_dir}")
            return

    clips_dir = args.output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"

    device = _pick_device()
    print(f"Device: {device}")
    model = YOLO("yolov8m-pose.pt")
    if device in ("cuda", "mps"):
        model.to(device)

    all_rows: list[dict] = []
    for video_path in video_files:
        rows = process_video(video_path, clips_dir, model, device)
        all_rows.extend(rows)
        print(f"  +{len(rows)} clips (total: {len(all_rows)})")

    if all_rows:
        append_rows(manifest_path, all_rows)
        print(f"\nDone. {len(all_rows)} clips → {manifest_path}")
        print(f"Next: streamlit run backend/tools/label_swings.py -- --manifest {manifest_path} --annotator <name>")
    else:
        print("\nNo swing events found.")


if __name__ == "__main__":
    main()
