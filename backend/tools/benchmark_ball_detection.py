"""
A/B benchmark for TrackNet ball detection weights.

Runs both weight files on a video clip and reports:
  - Detection rate (non-None frames / total frames)
  - Consecutive miss streak stats (max, mean)
  - Per-second detection rates

Usage:
    python -m backend.tools.benchmark_ball_detection --video data/raw_videos/StMarys_Court2.mp4 --max-frames 500
"""
import argparse
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.models.ball_tracker import BallDetector

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'weights')

WEIGHT_FILES = {
    'original':  'tracknet_weights.pt',
    'official':  'tracknet_v2_official.pt',
}


def benchmark(video_path: str, weights_name: str, weights_file: str, max_frames: int, device: str) -> dict:
    path = os.path.join(WEIGHTS_DIR, weights_file)
    if not os.path.exists(path):
        return {'error': f'weights not found: {path}'}

    detector = BallDetector(path_model=path, device=device)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    results = []
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        x, y = detector.infer_single(frame)
        results.append(x is not None)
        frame_idx += 1

    cap.release()

    total = len(results)
    detected = sum(results)
    detection_rate = detected / total if total > 0 else 0.0

    # Consecutive miss streaks
    streaks, current = [], 0
    for hit in results:
        if not hit:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)

    # Per-second detection rate
    frames_per_sec = int(fps)
    sec_rates = []
    for i in range(0, total, frames_per_sec):
        window = results[i:i + frames_per_sec]
        if window:
            sec_rates.append(sum(window) / len(window))

    return {
        'weights': weights_name,
        'total_frames': total,
        'detected': detected,
        'detection_rate': detection_rate,
        'max_miss_streak': max(streaks) if streaks else 0,
        'mean_miss_streak': np.mean(streaks) if streaks else 0.0,
        'num_miss_streaks': len(streaks),
        'min_sec_rate': min(sec_rates) if sec_rates else 0.0,
        'mean_sec_rate': np.mean(sec_rates) if sec_rates else 0.0,
    }


def print_result(r: dict):
    if 'error' in r:
        print(f"  [{r['weights']}] ERROR: {r['error']}")
        return
    print(f"  [{r['weights']:10s}]  "
          f"detection={r['detection_rate']:.1%}  "
          f"({r['detected']}/{r['total_frames']} frames)  "
          f"miss_streaks={r['num_miss_streaks']}  "
          f"max_streak={r['max_miss_streak']}  "
          f"mean_streak={r['mean_miss_streak']:.1f}  "
          f"worst_sec={r['min_sec_rate']:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--max-frames', type=int, default=500)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    print(f"\nBenchmarking ball detection on: {args.video}")
    print(f"Frames: {args.max_frames} | Device: {args.device}\n")

    results = {}
    for name, file in WEIGHT_FILES.items():
        print(f"Running {name} ({file})...")
        results[name] = benchmark(args.video, name, file, args.max_frames, args.device)
        print_result(results[name])

    # Delta summary
    if 'error' not in results.get('original', {}) and 'error' not in results.get('official', {}):
        delta = results['official']['detection_rate'] - results['original']['detection_rate']
        print(f"\n  Delta: official vs original = {delta:+.1%}")
        if delta > 0:
            print("  => Official weights WIN")
        elif delta < 0:
            print("  => Original weights win (unexpected — check manually)")
        else:
            print("  => No difference")


if __name__ == '__main__':
    main()
