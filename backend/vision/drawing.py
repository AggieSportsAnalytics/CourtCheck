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
    Draw tennis court lines as a semi-transparent white overlay.
    Keypoint dots and debug labels are intentionally omitted.
    """
    if kps is None:
        return frame

    overlay = frame.copy()
    for start_name, end_name in court_lines:
        try:
            s_idx = keypoint_names.index(start_name)
            e_idx = keypoint_names.index(end_name)
            if kps[s_idx] is None or kps[e_idx] is None:
                continue
            x1, y1 = map(int, kps[s_idx])
            x2, y2 = map(int, kps[e_idx])
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
        except (ValueError, IndexError):
            continue

    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    return frame

# -----------------------------
# Player drawing
# -----------------------------

def draw_stroke_labels(frame, player_dict, frame_stroke_labels, frame_idx):
    """
    Draw stroke classification label above each player's bbox when a swing
    is active at this frame.

    Args:
        frame: Video frame (numpy array, BGR).
        player_dict: {track_id: [x1, y1, x2, y2]} for this frame.
        frame_stroke_labels: {frame_idx: {track_id: label}} built from swing events.
        frame_idx: Current frame index.

    Returns:
        Annotated frame.
    """
    labels_at_frame = frame_stroke_labels.get(frame_idx, {})
    for track_id, label in labels_at_frame.items():
        bbox = player_dict.get(track_id) if player_dict else None
        if bbox is None:
            continue

        x1, y1, x2, _y2 = map(int, bbox)
        color = STROKE_COLORS_BGR.get(label, (255, 255, 255))

        # Pill background
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.65, 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        pad = 6
        rx1, ry1 = x1, y1 - th - pad * 2 - 28
        rx2, ry2 = x1 + tw + pad * 2, y1 - 28
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, -1)
        cv2.putText(frame, label, (rx1 + pad, ry2 - pad), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return frame


# Per-player colors (BGR) — up to 4 tracked players
_PLAYER_COLORS = [
    (255, 180,  40),   # player 1 — amber
    ( 80, 200, 255),   # player 2 — sky blue
    (100, 255, 120),   # player 3 — mint
    (200,  80, 255),   # player 4 — violet
]


def _player_color(track_id: int):
    return _PLAYER_COLORS[(track_id - 1) % len(_PLAYER_COLORS)]


def draw_player_bboxes(frame, player_dict, color=None):
    """
    Draw clean player bounding boxes with a small pill label.

    color is ignored when player_dict has multiple players — each gets
    its own color from _PLAYER_COLORS for easy differentiation.
    """
    if player_dict is None:
        return frame

    # Sort so rendering order is deterministic
    for track_id, bbox in sorted(player_dict.items()):
        x1, y1, x2, y2 = map(int, bbox)
        c = color if color is not None else _player_color(track_id)

        # Thin box with anti-aliasing
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 1, cv2.LINE_AA)

        # Small pill label at top-left corner of box
        label = f"P{track_id}"
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        pad = 4
        lx1, ly1 = x1, y1 - th - pad * 2
        lx2, ly2 = x1 + tw + pad * 2, y1

        # Clamp label above frame edge
        if ly1 < 0:
            ly1, ly2 = y1, y1 + th + pad * 2

        # Semi-transparent pill background
        overlay = frame.copy()
        cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), c, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, label, (lx1 + pad, ly2 - pad), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return frame


# -----------------------------
# Minimap drawing
# -----------------------------

STROKE_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "Forehand":       (0, 220, 80),    # green
    "Backhand":       (220, 100, 0),   # blue
    "Serve/Overhead": (0, 210, 255),   # yellow
    "Slice":          (160, 60, 220),  # purple
}
_STROKE_BOUNCE_DEFAULT = (0, 220, 200)  # teal — unknown / None


def draw_minimap_ball_and_bounces(
    minimap,
    homography_inv,
    ball_track,
    frame_idx,
    bounces,
    trace_length=7,
    trace_color=(255, 255, 0),
    bounce_color=(0, 255, 255),
    frame_stroke_labels=None,
):
    """
    Draw ball trace and accumulated bounces on the minimap.

    minimap: reference court image (will be modified in-place)
    homography_inv: frame -> reference court homography
    ball_track: list of (x, y)
    frame_idx: current frame index
    bounces: set of frame indices where bounces occurred
    frame_stroke_labels: dict[int, dict[int, str]] mapping frame_idx ->
        {track_id -> stroke label}.  When provided, each bounce dot is
        colored by the stroke type that produced it instead of the
        uniform bounce_color.
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
            # Determine color from stroke label, falling back to bounce_color
            dot_color = bounce_color
            if frame_stroke_labels is not None:
                labels_at_bounce = frame_stroke_labels.get(bounce_idx, {})
                # Use the first available player's label at the bounce frame
                if labels_at_bounce:
                    label = next(iter(labels_at_bounce.values()))
                    dot_color = STROKE_COLORS_BGR.get(label, _STROKE_BOUNCE_DEFAULT)

            cv2.circle(
                minimap,
                (mx, my),
                40,
                dot_color,
                -1,
            )
            # White outline ring for contrast
            cv2.circle(
                minimap,
                (mx, my),
                40,
                (255, 255, 255),
                3,
            )

    return minimap


def draw_minimap_players(
    minimap,
    homography_inv,
    player_dict,
    color=None,
):
    """
    Draw player positions on the minimap using foot position projected through homography.
    Each player gets a filled dot in their assigned color with a white outline ring.

    minimap: reference court image (modified in-place)
    homography_inv: frame -> reference court homography
    player_dict: {track_id: [x1, y1, x2, y2]}
    color: override BGR color (if None, uses per-player color)
    """
    if homography_inv is None or player_dict is None:
        return minimap

    for track_id, bbox in sorted(player_dict.items()):
        x1, y1, x2, y2 = bbox
        foot_x = (x1 + x2) / 2
        foot_y = float(y2)

        pt = np.array([[[foot_x, foot_y]]], dtype=np.float32)
        try:
            mapped = cv2.perspectiveTransform(pt, homography_inv)
        except cv2.error:
            continue

        mx, my = int(mapped[0, 0, 0]), int(mapped[0, 0, 1])
        if not (0 <= mx < minimap.shape[1] and 0 <= my < minimap.shape[0]):
            continue

        c = color if color is not None else _player_color(track_id)
        cv2.circle(minimap, (mx, my), 38, (255, 255, 255), 10)  # white ring
        cv2.circle(minimap, (mx, my), 30, c, -1)                # filled dot

    return minimap
