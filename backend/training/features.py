"""
Feature engineering for stroke classification.

Two output paths:
  - TCN path: normalize_keypoints() -> (seq_len, 34) array matching extract_pose_sequence()
  - CatBoost path: extract_clip_features() -> flat (N_FEATURES,) vector

COCO-17 keypoint indices:
    0  nose           1  left_eye        2  right_eye
    3  left_ear       4  right_ear       5  left_shoulder
    6  right_shoulder 7  left_elbow      8  right_elbow
    9  left_wrist    10  right_wrist    11  left_hip
   12  right_hip     13  left_knee      14  right_knee
   15  left_ankle    16  right_ankle
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# COCO-17 keypoint index constants
# ---------------------------------------------------------------------------
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Return the angle at vertex b (in degrees) given 3 2D points a, b, c.

    Args:
        a: 2D point shape (2,)
        b: vertex 2D point shape (2,)
        c: 2D point shape (2,)

    Returns:
        Angle in degrees [0, 180].
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _resample_to_len(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a (T, D) array to (target_len, D) via linear interpolation.
    Matches the interpolation logic in extract_pose_sequence().
    """
    t = seq.shape[0]
    if t == target_len:
        return seq
    indices = np.linspace(0, t - 1, target_len)
    return np.stack([
        np.interp(indices, np.arange(t), seq[:, d])
        for d in range(seq.shape[1])
    ], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# TCN path
# ---------------------------------------------------------------------------

def hip_center_normalize(seq: np.ndarray) -> np.ndarray:
    """
    Normalize a (T, 34) pose sequence to be invariant to player position and scale.

    This is the key preprocessing step that allows the TCN to generalize
    across different camera resolutions and player positions. Without it,
    THETIS-trained weights (~640x480 pixel coords) cannot be applied to
    CourtCheck footage (1920x1080), causing 0% accuracy.

    Steps:
      1. Hip-center: subtract the hip midpoint from every (x, y) per frame
         so the hip is always at the origin.
      2. Scale-normalize: divide all coordinates by the median torso height
         (hip-to-shoulder distance), putting values in roughly [-3, 3] range.

    COCO-17 indices in the flattened (T, 34) layout:
        kp5  left_shoulder   x=seq[:,10]  y=seq[:,11]
        kp6  right_shoulder  x=seq[:,12]  y=seq[:,13]
        kp11 left_hip        x=seq[:,22]  y=seq[:,23]
        kp12 right_hip       x=seq[:,24]  y=seq[:,25]

    Args:
        seq: (T, 34) float32 array of raw pixel coordinates.

    Returns:
        (T, 34) float32 array in torso-height units, hip-centered.
        Falls back to hip-centering only if scale is degenerate (<1px).
    """
    seq = seq.copy()
    T = seq.shape[0]

    # Hip midpoint per frame
    hip_x = (seq[:, 22] + seq[:, 24]) / 2.0   # (T,)
    hip_y = (seq[:, 23] + seq[:, 25]) / 2.0   # (T,)

    # Translate: hip to origin
    seq[:, 0::2] -= hip_x.reshape(T, 1)
    seq[:, 1::2] -= hip_y.reshape(T, 1)

    # Torso height = distance from hip midpoint to shoulder midpoint (after centering)
    shoulder_x = (seq[:, 10] + seq[:, 12]) / 2.0
    shoulder_y = (seq[:, 11] + seq[:, 13]) / 2.0
    torso_heights = np.sqrt(shoulder_x ** 2 + shoulder_y ** 2)   # (T,)
    scale = float(np.median(torso_heights))

    if scale < 1.0:
        # Degenerate — hip/shoulder keypoints missing or collapsed; return centered only
        return np.clip(seq, -5.0, 5.0)

    return np.clip(seq / scale, -5.0, 5.0)


# COCO-17 left/right keypoint pairs. Used by mirror_pose_for_lefty so the
# classifier sees a right-handed canonical pose for left-handed players.
#
# After we flip x-axis, the body is physically mirrored — but the keypoint
# *labels* still say "left_shoulder" for what's now on the player's right
# side. Without swapping these pairs the classifier sees a "right-handed
# pose with left/right labels swapped" and fails. With the swap, the result
# is bit-for-bit equivalent to a real right-handed player.
#
# Indices reference the flattened (T, 34) layout: keypoint k occupies
# columns 2k (x) and 2k+1 (y).
_COCO_LR_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),    # eyes
    (3, 4),    # ears
    (5, 6),    # shoulders
    (7, 8),    # elbows
    (9, 10),   # wrists
    (11, 12),  # hips
    (13, 14),  # knees
    (15, 16),  # ankles
)


def mirror_pose_for_lefty(seq: np.ndarray) -> np.ndarray:
    """Convert a left-handed pose sequence to its right-handed canonical form.

    Operates on hip-centered, torso-normalized output from hip_center_normalize:
    the hip midpoint sits at x=0, so mirroring is simply negating the x
    column. We then swap each (left, right) keypoint pair so the channel
    that used to encode "the shoulder on the player's left side" now encodes
    "the shoulder on the player's right side" — matching how the TCN was
    trained on righty data.

    HANDOFF_2026-05-13 documents why horizontal flip was disabled during
    training: a flipped Forehand visually IS a Backhand. This function
    re-introduces the flip selectively at INFERENCE time for lefties only,
    so the model still gets canonical right-handed input.

    Args:
        seq: (T, 34) float32 array from hip_center_normalize. Must be
            normalized — passing raw pixel coords here would not flip
            around the player's hip but around x=0 in image space.

    Returns:
        (T, 34) float32 array — same player, mirrored as if right-handed.
    """
    mirrored = seq.copy()
    # Negate every x column (even indices in the flattened layout).
    mirrored[:, 0::2] = -mirrored[:, 0::2]
    # Swap (left, right) keypoint pairs so semantics survive the mirror.
    for l_idx, r_idx in _COCO_LR_PAIRS:
        lx, ly = l_idx * 2, l_idx * 2 + 1
        rx, ry = r_idx * 2, r_idx * 2 + 1
        # Atomic swap via fancy indexing — single allocation, no temps leaking.
        mirrored[:, [lx, ly, rx, ry]] = mirrored[:, [rx, ry, lx, ly]]
    return mirrored


def temporal_derivatives(seq: np.ndarray, orders: int = 1) -> np.ndarray:
    """Append temporal derivatives to a (T, D) pose feature sequence.

    Velocity (1st derivative) carries direction-of-motion signal that raw
    positions do not. For tennis stroke classification, wrist velocity
    direction is the primary discriminator between Forehand and Backhand
    when posture alone is ambiguous.

    Args:
        seq: (T, D) feature sequence.
        orders: number of derivative orders to append (1 = velocity,
            2 = velocity + acceleration).

    Returns:
        (T, D * (orders + 1)) sequence with positions + derivatives stacked.
    """
    feats = [seq]
    current = seq
    for _ in range(orders):
        d = np.zeros_like(current)
        d[1:] = current[1:] - current[:-1]
        feats.append(d)
        current = d
    return np.concatenate(feats, axis=1)


def normalize_keypoints(keypoints: np.ndarray, seq_len: int = 45) -> np.ndarray:
    """
    Normalize raw COCO-17 keypoints for TCN input.

    Steps:
      1. Drop confidence channel: keep only x, y columns -> (T, 17, 2)
      2. Flatten to (T, 34)
      3. Resample to seq_len via linear interpolation
      4. Hip-center + torso-height scale normalization

    Step 4 makes the sequence invariant to absolute pixel coordinates and
    player scale, allowing the TCN to generalize across resolutions.
    extract_pose_sequence() in swing_detector.py applies the same steps.

    Args:
        keypoints: (T, 17, 3) raw COCO-17 keypoints with (x, y, confidence).
        seq_len: Target sequence length (must match StrokeTCN SEQ_LEN = 45).

    Returns:
        (seq_len, 34) float32 normalized array ready for StrokeTCN input.
    """
    if keypoints.ndim != 3 or keypoints.shape[1] != 17 or keypoints.shape[2] < 2:
        raise ValueError(
            f"Expected (T, 17, 3) keypoints array, got {keypoints.shape}"
        )

    t = keypoints.shape[0]

    # Drop confidence, flatten to (T, 34)
    xy = keypoints[:, :, :2]
    seq = xy.reshape(t, -1).astype(np.float32)   # (T, 34)

    # Resample then normalize
    seq = _resample_to_len(seq, seq_len)          # (seq_len, 34)
    return hip_center_normalize(seq)              # invariant to position and scale


# ---------------------------------------------------------------------------
# CatBoost path
# ---------------------------------------------------------------------------

def extract_clip_features(keypoints: np.ndarray) -> np.ndarray:
    """
    Extract a flat feature vector from a (T, 17, 3) keypoint sequence.

    Feature groups (~45 total):
      - Wrist velocity stats (6): max/mean/std for left and right wrist
      - Elbow angle at 5 key frames x 2 sides (10)
      - Shoulder-wrist vertical offset at swing peak (2): left and right
      - Shoulder rotation (3): max, min, range of kp5 x-displacement rel to kp6
      - Arm extension ratio at swing peak (2): left and right
      - Hip tilt stats (2): mean and std of hip angle over time
      Total: 25 features

    Args:
        keypoints: (T, 17, 3) raw COCO-17 keypoints.

    Returns:
        (N_FEATURES,) float32 feature vector.
    """
    if keypoints.ndim != 3 or keypoints.shape[1] != 17:
        raise ValueError(
            f"Expected (T, 17, 3) keypoints array, got {keypoints.shape}"
        )

    t = keypoints.shape[0]
    xy = keypoints[:, :, :2].astype(np.float32)  # (T, 17, 2)

    features: list[float] = []

    # ------------------------------------------------------------------
    # 1. Wrist velocity: max, mean, std for left (kp9) and right (kp10)
    # ------------------------------------------------------------------
    for wrist_idx in (KP_LEFT_WRIST, KP_RIGHT_WRIST):
        wrist_pos = xy[:, wrist_idx, :]         # (T, 2)
        diffs = np.diff(wrist_pos, axis=0)       # (T-1, 2)
        speeds = np.linalg.norm(diffs, axis=1)   # (T-1,)
        features.extend([
            float(np.max(speeds)) if len(speeds) > 0 else 0.0,
            float(np.mean(speeds)) if len(speeds) > 0 else 0.0,
            float(np.std(speeds)) if len(speeds) > 0 else 0.0,
        ])
    # 6 features so far

    # ------------------------------------------------------------------
    # 2. Elbow angle at 5 key frames for each side
    #    Left:  shoulder(5) -> elbow(7) -> wrist(9)
    #    Right: shoulder(6) -> elbow(8) -> wrist(10)
    # ------------------------------------------------------------------
    key_frames = [0, t // 4, t // 2, 3 * t // 4, t - 1]
    for frame_idx in key_frames:
        # Left elbow angle
        a = xy[frame_idx, KP_LEFT_SHOULDER]
        b = xy[frame_idx, KP_LEFT_ELBOW]
        c = xy[frame_idx, KP_LEFT_WRIST]
        features.append(compute_angle(a, b, c))

        # Right elbow angle
        a = xy[frame_idx, KP_RIGHT_SHOULDER]
        b = xy[frame_idx, KP_RIGHT_ELBOW]
        c = xy[frame_idx, KP_RIGHT_WRIST]
        features.append(compute_angle(a, b, c))
    # 10 more = 16 total

    # ------------------------------------------------------------------
    # 3. Shoulder-wrist vertical offset at swing peak
    #    Swing peak = frame with minimum wrist y (highest in image = lowest y)
    # ------------------------------------------------------------------
    left_wrist_y = xy[:, KP_LEFT_WRIST, 1]
    right_wrist_y = xy[:, KP_RIGHT_WRIST, 1]
    combined_min_y = (left_wrist_y + right_wrist_y) / 2.0
    peak_frame = int(np.argmin(combined_min_y))

    left_shoulder_y = float(xy[peak_frame, KP_LEFT_SHOULDER, 1])
    right_shoulder_y = float(xy[peak_frame, KP_RIGHT_SHOULDER, 1])
    features.append(left_shoulder_y - float(left_wrist_y[peak_frame]))
    features.append(right_shoulder_y - float(right_wrist_y[peak_frame]))
    # 2 more = 18 total

    # ------------------------------------------------------------------
    # 4. Shoulder rotation: x-displacement of kp5 relative to kp6 over time
    #    max, min, range
    # ------------------------------------------------------------------
    shoulder_x_disp = xy[:, KP_LEFT_SHOULDER, 0] - xy[:, KP_RIGHT_SHOULDER, 0]
    features.append(float(np.max(shoulder_x_disp)))
    features.append(float(np.min(shoulder_x_disp)))
    features.append(float(np.max(shoulder_x_disp) - np.min(shoulder_x_disp)))
    # 3 more = 21 total

    # ------------------------------------------------------------------
    # 5. Arm extension ratio at swing peak
    #    ratio = wrist-to-shoulder / (upper_arm + forearm) length
    # ------------------------------------------------------------------
    for shoulder_idx, elbow_idx, wrist_idx in (
        (KP_LEFT_SHOULDER, KP_LEFT_ELBOW, KP_LEFT_WRIST),
        (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW, KP_RIGHT_WRIST),
    ):
        shoulder = xy[peak_frame, shoulder_idx]
        elbow = xy[peak_frame, elbow_idx]
        wrist = xy[peak_frame, wrist_idx]

        upper_arm = float(np.linalg.norm(elbow - shoulder))
        forearm = float(np.linalg.norm(wrist - elbow))
        wrist_to_shoulder = float(np.linalg.norm(wrist - shoulder))
        total_arm = upper_arm + forearm

        ratio = wrist_to_shoulder / total_arm if total_arm > 1e-6 else 0.0
        features.append(ratio)
    # 2 more = 23 total

    # ------------------------------------------------------------------
    # 6. Hip tilt: angle between hip keypoints over time (mean, std)
    #    Angle defined as atan2(y_diff, x_diff) for left_hip - right_hip
    # ------------------------------------------------------------------
    left_hip = xy[:, KP_LEFT_HIP, :]   # (T, 2)
    right_hip = xy[:, KP_RIGHT_HIP, :]  # (T, 2)
    hip_diff = left_hip - right_hip      # (T, 2)
    hip_angles = np.degrees(np.arctan2(hip_diff[:, 1], hip_diff[:, 0]))
    features.append(float(np.mean(hip_angles)))
    features.append(float(np.std(hip_angles)))
    # 2 more = 25 total

    return np.array(features, dtype=np.float32)


# N_FEATURES constant for downstream reference
N_FEATURES: int = 25
