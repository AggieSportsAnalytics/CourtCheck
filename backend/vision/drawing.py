# vision/drawing.py

import cv2
import numpy as np
from backend.vision.court_reference import CourtReference

keypoint_names = [
    "BTL",
    "BTR",
    "BBL",
    "BBR",
    "BTLI",
    "BBLI",
    "BTRI",
    "BBRI",
    "ITL",
    "ITR",
    "IBL",
    "IBR",
    "ITM",
    "IBM",
]

court_lines = [
    ("BTL", "BTLI"),
    ("BTLI", "BTRI"),
    ("BTRI", "BTR"),
    ("BTL", "BBL"),
    ("BTR", "BBR"),
    ("BBL", "BBLI"),
    ("BBLI", "BBRI"),
    ("BBLI", "IBL"),
    ("BBRI", "IBR"),
    ("BBRI", "BBR"),
    ("BTLI", "ITL"),
    ("BTRI", "ITR"),
    ("ITL", "ITM"),
    ("ITM", "IBM"),
    ("ITL", "IBL"),
    ("ITR", "IBR"),
    ("IBL", "IBM"),
    ("IBM", "IBR"),
    ("ITM", "ITR"),
]

# -----------------------------
# Ball drawing
# -----------------------------

def draw_ball_trace(
    frame,
    ball_track,
    frame_idx,
    trace_length=7,
    base_color=(255, 255, 0),  # light_blue from Colab
):
    """
    Draw a fading ball trajectory on the main frame.

    Parameters
    ----------
    frame : np.ndarray
        Current video frame (modified in-place)
    ball_track : list[(x, y) | None]
        Ball positions for all frames
    frame_idx : int
        Current frame index
    trace_length : int
        Number of previous frames to draw
    base_color : tuple
        Base BGR color of the trace
    """

    start_idx = max(0, frame_idx - trace_length + 1)

    for idx in range(start_idx, frame_idx + 1):
        if (
            idx >= len(ball_track)
            or ball_track[idx] is None
            or ball_track[idx][0] is None
        ):
            continue

        x, y = ball_track[idx]

        # Alpha fade: newest = brightest
        alpha = 1.0 - ((frame_idx - idx) / trace_length)
        color = tuple(int(c * alpha) for c in base_color)

        cv2.circle(
            frame,
            (int(x), int(y)),
            3,
            color,
            -1,
        )

    return frame


# -----------------------------
# Court drawing
# -----------------------------

def draw_court_keypoints_and_lines(frame, kps):
    """
    Draw tennis court lines (green) and keypoints (red) on 'frame'.
    """

    if kps is None:
        return frame

    for start_name, end_name in court_lines:
        try:
            s_idx = keypoint_names.index(start_name)
            e_idx = keypoint_names.index(end_name)
            if kps[s_idx] is None or kps[e_idx] is None:
                continue

            x1, y1 = map(int, kps[s_idx])
            x2, y2 = map(int, kps[e_idx])

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except (ValueError, IndexError):
            continue

    # Keypoints + labels
    for i, pt in enumerate(kps):
        if pt is None:
            continue
        x, y = map(int, pt)
        
        # kp
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # label
        label = keypoint_names[i]
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(
            frame, (x - 5, y - th - 5), (x - 5 + tw, y - 5), (255, 255, 255), -1
        )
        cv2.putText(
            frame, label, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
        )

    return frame

# -----------------------------
# Player drawing
# -----------------------------

def draw_players(frame, players):
    """
    players: list of (bbox, center, label)
    """
    for bbox, center, label in players:
        if bbox is None:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if label:
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
    return frame


def draw_player_bboxes(frame, player_dict, color=(0, 255, 255)):
    """
    Draw player bounding boxes on a single frame

    Args:
        frame: Video frame (numpy array)
        player_dict: {track_id: [x1, y1, x2, y2]}
        color: BGR color tuple for bbox and label (default: yellow)

    Returns:
        frame: Frame with player bboxes drawn
    """
    if player_dict is None:
        return frame

    for track_id, bbox in player_dict.items():
        x1, y1, x2, y2 = map(int, bbox)
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Draw label
        cv2.putText(
            frame,
            f"Player {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )
    return frame


# -----------------------------
# Minimap drawing
# -----------------------------

def draw_minimap(
    frame,
    court_img,
    homography_inv,
    ball_xy,
    players,
    bounce=False,
    size=(166, 350),
):
    if homography_inv is None:
        return frame

    minimap = court_img.copy()

    # Draw ball on minimap
    if ball_xy and ball_xy[0] is not None:
        pt = np.array([[ball_xy]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, homography_inv)
        x, y = int(mapped[0, 0, 0]), int(mapped[0, 0, 1])
        color = (0, 255, 255) if bounce else (0, 255, 0)
        cv2.circle(minimap, (x, y), 6, color, -1)

    # Draw players on minimap
    for bbox, center, _ in players:
        if center is None:
            continue
        pt = np.array([[center]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, homography_inv)
        x, y = int(mapped[0, 0, 0]), int(mapped[0, 0, 1])
        cv2.circle(minimap, (x, y), 6, (255, 0, 0), -1)

    # Resize + paste
    minimap = cv2.resize(minimap, size)
    h, w = frame.shape[:2]
    mh, mw = size[1], size[0]
    frame[30 : 30 + mh, w - 30 - mw : w - 30] = minimap

    return frame

def draw_minimap_ball_and_bounces(
    minimap,
    homography_inv,
    ball_track,
    frame_idx,
    bounces,
    trace_length=7,
    trace_color=(255, 255, 0),
    bounce_color=(0, 255, 255),
):
    """
    Draw ball trace and accumulated bounces on the minimap.

    minimap: reference court image (will be modified in-place)
    homography_inv: frame -> reference court homography
    ball_track: list of (x, y)
    frame_idx: current frame index
    bounces: set of frame indices where bounces occurred
    """

    if homography_inv is None:
        return minimap

    # ---- 1. Ball trace ----
    start_idx = max(0, frame_idx - trace_length + 1)

    for idx in range(start_idx, frame_idx + 1):
        if (
            idx >= len(ball_track)
            or ball_track[idx] is None
            or ball_track[idx][0] is None
        ):
            continue

        x, y = ball_track[idx]
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)

        try:
            mapped = cv2.perspectiveTransform(pt, homography_inv)
        except cv2.error:
            continue

        mx, my = int(mapped[0, 0, 0]), int(mapped[0, 0, 1])

        alpha = 1.0 - ((frame_idx - idx) / trace_length)
        color = tuple(int(c * alpha) for c in trace_color)

        cv2.circle(
            minimap,
            (mx, my),
            3,
            color,
            20,  # OG thick trace
        )

    # ---- 2. Accumulated bounces ----
    for bounce_idx in bounces:
        if bounce_idx > frame_idx or bounce_idx >= len(ball_track):
            continue

        bx, by = ball_track[bounce_idx] if ball_track[bounce_idx] else (None, None)
        if bx is None or by is None:
            continue

        pt = np.array([[[float(bx), float(by)]]], dtype=np.float32)

        try:
            mapped = cv2.perspectiveTransform(pt, homography_inv)
        except cv2.error:
            continue

        mx, my = int(mapped[0, 0, 0]), int(mapped[0, 0, 1])

        if 0 <= mx < minimap.shape[1] and 0 <= my < minimap.shape[0]:
            cv2.circle(
                minimap,
                (mx, my),
                10,
                bounce_color,
                40,
            )

    return minimap


def draw_minimap_players(
    minimap,
    homography_inv,
    player_dict,
    color=(0, 0, 255),
):
    """
    Draw player positions on the minimap using foot position projected through homography.

    minimap: reference court image (modified in-place)
    homography_inv: frame -> reference court homography
    player_dict: {track_id: [x1, y1, x2, y2]}
    color: BGR color for player dots
    """
    if homography_inv is None or player_dict is None:
        return minimap

    for track_id, bbox in player_dict.items():
        x1, y1, x2, y2 = bbox
        # Foot position = bottom center of bbox
        foot_x = (x1 + x2) / 2
        foot_y = float(y2)

        pt = np.array([[[foot_x, foot_y]]], dtype=np.float32)

        try:
            mapped = cv2.perspectiveTransform(pt, homography_inv)
        except cv2.error:
            continue

        mx, my = int(mapped[0, 0, 0]), int(mapped[0, 0, 1])

        if 0 <= mx < minimap.shape[1] and 0 <= my < minimap.shape[0]:
            cv2.circle(minimap, (mx, my), 30, color, -1)

    return minimap
