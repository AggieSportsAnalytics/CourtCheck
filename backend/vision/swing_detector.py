"""
Swing trigger detector based on wrist keypoint velocity.

Instead of running the stroke classifier on every frame, this module monitors
wrist velocity from YOLOv8-Pose keypoints and only triggers the classifier
when a player's wrist moves fast enough AND the ball is nearby.

COCO keypoint indices (YOLOv8-Pose):
    0: nose         5: left shoulder    6: right shoulder
    7: left elbow   8: right elbow
    9: left wrist  10: right wrist
   11: left hip    12: right hip
   15: left ankle  16: right ankle
"""
from __future__ import annotations

import numpy as np


WRIST_LEFT_IDX = 9
WRIST_RIGHT_IDX = 10

# Frames to collect before + after wrist velocity peak
WINDOW_BEFORE = 15
WINDOW_AFTER = 30


class SwingDetector:
    """
    Detects tennis swing events from per-frame pose keypoints.

    Usage:
        detector = SwingDetector(velocity_threshold=15.0, ball_proximity=300.0)
        swing_events = detector.detect(pose_keypoints_per_frame, ball_track, player_ids)
    """

    def __init__(
        self,
        velocity_threshold: float = 15.0,
        ball_proximity: float = 300.0,
        min_frames_between_swings: int = 15,
    ):
        """
        Args:
            velocity_threshold: Wrist speed (px/frame) to count as swing start.
            ball_proximity: Ball must be within this many pixels of the player bbox.
            min_frames_between_swings: Minimum gap to avoid double-counting one swing.
        """
        self.velocity_threshold = velocity_threshold
        self.ball_proximity = ball_proximity
        self.min_frames_between_swings = min_frames_between_swings

    def detect(
        self,
        pose_keypoints_per_frame: list[dict[int, np.ndarray | None]],
        ball_track: list[tuple[float, float] | None],
        player_detections: list[dict[int, list[float]]],
    ) -> list[dict]:
        """
        Detect swing events across all frames.

        Args:
            pose_keypoints_per_frame: List (one per frame) of
                {track_id: keypoints_array (shape [17, 3]) or None}.
                The third coordinate is the confidence score.
            ball_track: List of (x, y) or None per frame.
            player_detections: List of {track_id: [x1, y1, x2, y2]} per frame.

        Returns:
            List of swing event dicts:
                {
                    "peak_frame": int,          # frame of maximum wrist velocity
                    "window_start": int,         # first frame of the clip
                    "window_end": int,           # last frame of the clip (inclusive)
                    "track_id": int,
                    "wrist_velocity": float,     # peak velocity in px/frame
                }
        """
        n_frames = len(pose_keypoints_per_frame)
        if n_frames == 0:
            return []

        # Collect all track IDs that appear in any frame
        all_track_ids: set[int] = set()
        for frame_kps in pose_keypoints_per_frame:
            all_track_ids.update(frame_kps.keys())

        events: list[dict] = []

        for track_id in all_track_ids:
            wrist_velocities = _compute_wrist_velocities(
                pose_keypoints_per_frame, track_id, n_frames
            )
            swing_peaks = _find_velocity_peaks(
                wrist_velocities,
                threshold=self.velocity_threshold,
                min_gap=self.min_frames_between_swings,
            )

            for peak_frame in swing_peaks:
                if not _ball_near_player(
                    peak_frame, ball_track, player_detections, track_id,
                    proximity=self.ball_proximity,
                ):
                    continue

                window_start = max(0, peak_frame - WINDOW_BEFORE)
                window_end = min(n_frames - 1, peak_frame + WINDOW_AFTER)
                events.append({
                    "peak_frame": peak_frame,
                    "window_start": window_start,
                    "window_end": window_end,
                    "track_id": track_id,
                    "wrist_velocity": float(wrist_velocities[peak_frame]),
                })

        events.sort(key=lambda e: e["peak_frame"])
        return events


def extract_pose_sequence(
    pose_keypoints_per_frame: list[dict[int, np.ndarray | None]],
    track_id: int,
    window_start: int,
    window_end: int,
    target_len: int = 45,
) -> np.ndarray:
    """
    Extract a fixed-length pose keypoint sequence for one player in a window.

    Returns:
        Float32 array of shape (target_len, 34) — 17 keypoints × (x, y).
        Missing frames are forward/back filled; if no data at all, returns zeros.
    """
    raw: list[np.ndarray | None] = []
    for i in range(window_start, window_end + 1):
        kps = pose_keypoints_per_frame[i].get(track_id) if i < len(pose_keypoints_per_frame) else None
        if kps is not None:
            xy = kps[:, :2].flatten().astype(np.float32)  # (34,)
        else:
            xy = None
        raw.append(xy)

    # Forward fill
    last = None
    for i in range(len(raw)):
        if raw[i] is not None:
            last = raw[i]
        elif last is not None:
            raw[i] = last

    # Backward fill
    last = None
    for i in range(len(raw) - 1, -1, -1):
        if raw[i] is not None:
            last = raw[i]
        elif last is not None:
            raw[i] = last

    # Replace remaining None with zeros
    zero = np.zeros(34, dtype=np.float32)
    filled = [v if v is not None else zero for v in raw]

    if not filled:
        return np.zeros((target_len, 34), dtype=np.float32)

    seq = np.stack(filled, axis=0)  # (T, 34)

    # Resize to target_len via linear interpolation
    if seq.shape[0] == target_len:
        return seq
    indices = np.linspace(0, seq.shape[0] - 1, target_len)
    resampled = np.stack([
        np.interp(indices, np.arange(seq.shape[0]), seq[:, d])
        for d in range(34)
    ], axis=1).astype(np.float32)
    return resampled


# ── helpers ──────────────────────────────────────────────────────────────────

def _compute_wrist_velocities(
    pose_keypoints_per_frame: list[dict[int, np.ndarray | None]],
    track_id: int,
    n_frames: int,
) -> np.ndarray:
    """Compute per-frame wrist speed (max of left/right wrist) for one player."""
    velocities = np.zeros(n_frames, dtype=np.float32)
    prev_left = prev_right = None

    for i in range(n_frames):
        kps = pose_keypoints_per_frame[i].get(track_id)
        if kps is None or kps.shape[0] < 11:
            prev_left = prev_right = None
            continue

        left_xy = kps[WRIST_LEFT_IDX, :2]
        right_xy = kps[WRIST_RIGHT_IDX, :2]

        v_left = float(np.linalg.norm(left_xy - prev_left)) if prev_left is not None else 0.0
        v_right = float(np.linalg.norm(right_xy - prev_right)) if prev_right is not None else 0.0
        velocities[i] = max(v_left, v_right)

        prev_left = left_xy.copy()
        prev_right = right_xy.copy()

    return velocities


def _find_velocity_peaks(
    velocities: np.ndarray,
    threshold: float,
    min_gap: int,
) -> list[int]:
    """Return frame indices where wrist velocity exceeds threshold (local maxima)."""
    peaks: list[int] = []
    n = len(velocities)
    last_peak = -min_gap - 1

    for i in range(1, n - 1):
        if velocities[i] < threshold:
            continue
        if velocities[i] < velocities[i - 1] or velocities[i] < velocities[i + 1]:
            continue
        if i - last_peak < min_gap:
            continue
        peaks.append(i)
        last_peak = i

    return peaks


def _ball_near_player(
    frame_idx: int,
    ball_track: list,
    player_detections: list,
    track_id: int,
    proximity: float,
) -> bool:
    """Return True if the ball is within `proximity` pixels of the player bbox."""
    if frame_idx >= len(ball_track) or frame_idx >= len(player_detections):
        return True  # can't check — don't filter out

    ball_pos = ball_track[frame_idx]
    if ball_pos is None or ball_pos[0] is None or ball_pos[1] is None:
        return True  # ball not detected — don't filter

    bbox = player_detections[frame_idx].get(track_id)
    if bbox is None:
        return False

    bx, by = float(ball_pos[0]), float(ball_pos[1])
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
    return dist <= proximity
