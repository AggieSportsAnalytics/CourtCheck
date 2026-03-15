#!/usr/bin/env python3
"""
One-time court calibration tool for fixed camera setups.

Usage:
    # From a video (extracts frame 30 by default):
    python -m backend.tools.calibrate_court \\
        --video /path/to/match.mp4 \\
        --camera-id uc_davis_court1

    # From a saved frame image:
    python -m backend.tools.calibrate_court \\
        --image /path/to/frame.jpg \\
        --camera-id uc_davis_court1

    # Skip interactive correction (auto-save detected keypoints):
    python -m backend.tools.calibrate_court \\
        --video /path/to/match.mp4 \\
        --camera-id uc_davis_court1 \\
        --no-interactive

Interactive controls:
    Left-click near a keypoint dot  → select it (highlighted in cyan)
    Left-click elsewhere            → move selected keypoint to that position
    r                               → reset all keypoints to auto-detected positions
    s                               → save calibration and exit
    q                               → quit without saving
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.models.court_line_detector import CourtLineDetector
from backend.vision.homography import HomographyEstimator
from backend.vision.calibration import save_calibration, visualize_keypoints


TARGET_W, TARGET_H = 1280, 720


def extract_frame(video_path: str, frame_number: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_number = min(frame_number, max(0, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_number} from {video_path}")
    if w != TARGET_W or h != TARGET_H:
        print(f"Resizing frame from {w}x{h} → {TARGET_W}x{TARGET_H} (matches pipeline)")
        frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    print(f"Extracted frame {frame_number} / {total} from {video_path}")
    return frame


def interactive_correction(frame: np.ndarray, keypoints: list) -> list | None:
    """
    Open an OpenCV window to let the user verify and correct keypoint positions.

    Returns the corrected keypoints, or None if the user pressed 'q'.
    """
    original_kps = list(keypoints)
    kps = list(keypoints)
    selected_idx = None
    window = "Court Calibration | click=select/move | s=save | r=reset | q=quit"

    def redraw():
        vis = visualize_keypoints(frame, kps)
        if selected_idx is not None and kps[selected_idx] is not None:
            x, y = int(kps[selected_idx][0]), int(kps[selected_idx][1])
            cv2.circle(vis, (x, y), 14, (0, 255, 255), 3)
        cv2.imshow(window, vis)

    def on_mouse(event, x, y, _flags, _param):
        nonlocal selected_idx
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if selected_idx is None:
            best_dist, best_i = 50.0, None
            for i, kp in enumerate(kps):
                if kp is None:
                    continue
                d = ((kp[0] - x) ** 2 + (kp[1] - y) ** 2) ** 0.5
                if d < best_dist:
                    best_dist, best_i = d, i
            selected_idx = best_i
        else:
            kps[selected_idx] = (float(x), float(y))
            selected_idx = None
        redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s'):
            cv2.destroyAllWindows()
            return kps
        elif key == ord('r'):
            kps[:] = original_kps
            selected_idx = None
            redraw()
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None


def main():
    parser = argparse.ArgumentParser(
        description="One-time court calibration for fixed camera angles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to input video file")
    group.add_argument("--image", help="Path to a single frame image")

    parser.add_argument(
        "--frame", type=int, default=30,
        help="Frame number to extract from video (default: 30)",
    )
    parser.add_argument(
        "--camera-id", required=True,
        help="Unique camera identifier, e.g. 'uc_davis_court1'",
    )
    parser.add_argument(
        "--calibration",
        default="backend/calibration_frames/court_calibration.json",
        help="Path to the calibration JSON (default: backend/calibration_frames/court_calibration.json)",
    )
    parser.add_argument(
        "--weights-dir", default="backend/weights",
        help="Directory containing model weights (default: backend/weights)",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Inference device (default: cpu)",
    )
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="Skip the correction window and auto-save detected keypoints",
    )
    args = parser.parse_args()

    # Load frame
    if args.video:
        frame = extract_frame(args.video, args.frame)
    else:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: Cannot read image: {args.image}")
            sys.exit(1)
        h, w = frame.shape[:2]
        if w != TARGET_W or h != TARGET_H:
            print(f"Resizing image from {w}x{h} → {TARGET_W}x{TARGET_H} (matches pipeline)")
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    # Run court keypoint detector
    model_path = Path(args.weights_dir) / "keypoints_model.pth"
    if not model_path.exists():
        print(f"ERROR: Keypoints model not found: {model_path}")
        sys.exit(1)

    print(f"Detecting court keypoints on {args.device}...")
    detector = CourtLineDetector(model_path=str(model_path), device=args.device)
    keypoints = detector.infer_single(frame)
    n_detected = sum(1 for k in keypoints if k is not None)
    print(f"Detected {n_detected} / {len(keypoints)} keypoints")

    # Interactive correction (unless --no-interactive)
    if args.no_interactive:
        final_kps = keypoints
    else:
        print("\nOpening correction window...")
        print("  Click near a dot to select it, then click again to move it")
        print("  Press 's' to save, 'r' to reset, 'q' to cancel\n")
        final_kps = interactive_correction(frame, keypoints)
        if final_kps is None:
            print("Calibration cancelled — nothing saved.")
            sys.exit(0)

    # Compute homography
    estimator = HomographyEstimator()
    H_ref, H_frame = estimator.estimate(final_kps)
    if H_ref is None or H_frame is None:
        print("ERROR: Could not compute homography from current keypoints.")
        print("Ensure at least 4 keypoints are correctly placed and try again.")
        sys.exit(1)

    # Save
    save_calibration(
        calibration_path=args.calibration,
        camera_id=args.camera_id,
        H_frame=H_frame,
        H_ref=H_ref,
        keypoints=final_kps,
    )

    print(f"\nCalibration saved successfully!")
    print(f"  Camera ID : {args.camera_id}")
    print(f"  File      : {args.calibration}")
    print(f"\nTo use in the pipeline:")
    print(f"  config.calibration_path = '{args.calibration}'")
    print(f"  config.camera_id        = '{args.camera_id}'")


if __name__ == "__main__":
    main()
