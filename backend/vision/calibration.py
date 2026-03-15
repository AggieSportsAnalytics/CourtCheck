"""
Court calibration: save/load/visualize per-camera homography matrices.

Workflow:
    1. Run `python -m backend.tools.calibrate_court` once per fixed camera angle.
    2. The tool saves a verified homography to court_calibration.json.
    3. In the pipeline, set config.calibration_path and config.camera_id.
    4. The pipeline skips per-frame court detection and uses the saved matrix.
"""
import json
import cv2
import numpy as np
from pathlib import Path


KEYPOINT_COLORS = [
    (255,   0,   0), (  0, 255,   0), (  0,   0, 255), (255, 255,   0),
    (255,   0, 255), (  0, 255, 255), (128, 255,   0), (255, 128,   0),
    (  0, 128, 255), (128,   0, 255), (255,   0, 128), (  0, 255, 128),
    (200, 200,   0), (  0, 200, 200),
]

KEYPOINT_NAMES = [
    "Top-left baseline", "Top-right baseline",
    "Bottom-left baseline", "Bottom-right baseline",
    "Left singles top", "Right singles top",
    "Left singles bottom", "Right singles bottom",
    "Net left", "Net right",
    "Top service left", "Top service right",
    "Bottom service left", "Bottom service right",
]


def save_calibration(
    calibration_path: str,
    camera_id: str,
    H_frame: np.ndarray,
    H_ref: np.ndarray,
    keypoints: list | None = None,
) -> None:
    """
    Save homography matrices for a camera to a JSON calibration file.

    Args:
        calibration_path: Path to the JSON file (created if it doesn't exist).
        camera_id: Unique string identifier for this camera angle.
        H_frame: 3×3 ndarray — court reference → image frame.
        H_ref:   3×3 ndarray — image frame → court reference.
        keypoints: Optional list of 14 (x, y) tuples or None values.
    """
    path = Path(calibration_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict = {}
    if path.exists():
        with open(path) as f:
            data = json.load(f)

    entry: dict = {
        "H_frame": H_frame.tolist(),
        "H_ref": H_ref.tolist(),
    }
    if keypoints is not None:
        entry["keypoints"] = [
            [float(kp[0]), float(kp[1])] if kp is not None else None
            for kp in keypoints
        ]

    data[camera_id] = entry
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[Calibration] Saved '{camera_id}' → {path}")


def load_calibration(
    calibration_path: str,
    camera_id: str,
) -> tuple[np.ndarray | None, np.ndarray | None, list | None]:
    """
    Load homography matrices and keypoints for a camera from the calibration JSON.

    Returns:
        (H_ref, H_frame, keypoints)
        H_ref:     frame → court reference homography (3×3)
        H_frame:   court reference → frame homography (3×3)
        keypoints: list of 14 (x, y) tuples or None values (or None if not saved)
        All three are None if the entry is missing.
    """
    path = Path(calibration_path)
    if not path.exists():
        print(f"[Calibration] File not found: {path}")
        return None, None, None

    with open(path) as f:
        data = json.load(f)

    if camera_id not in data:
        print(f"[Calibration] camera_id='{camera_id}' not in {path}")
        return None, None, None

    entry = data[camera_id]
    H_frame = np.array(entry["H_frame"], dtype=np.float32)
    H_ref = np.array(entry["H_ref"], dtype=np.float32)

    raw_kps = entry.get("keypoints")
    keypoints = None
    if raw_kps is not None:
        keypoints = [tuple(kp) if kp is not None else None for kp in raw_kps]

    print(f"[Calibration] Loaded '{camera_id}' from {path}")
    return H_ref, H_frame, keypoints


def visualize_keypoints(frame: np.ndarray, keypoints: list) -> np.ndarray:
    """
    Draw the 14 court keypoints on a frame copy for visual inspection.

    Args:
        frame: BGR image as a numpy array.
        keypoints: List of 14 (x, y) tuples or None values.

    Returns:
        Annotated copy of the frame.
    """
    vis = frame.copy()
    for idx, kp in enumerate(keypoints):
        if kp is None:
            continue
        x, y = int(kp[0]), int(kp[1])
        color = KEYPOINT_COLORS[idx % len(KEYPOINT_COLORS)]
        cv2.circle(vis, (x, y), 8, color, -1)
        cv2.circle(vis, (x, y), 8, (255, 255, 255), 1)
        name = KEYPOINT_NAMES[idx] if idx < len(KEYPOINT_NAMES) else "?"
        cv2.putText(
            vis, f"{idx}: {name}", (x + 10, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )
    return vis
