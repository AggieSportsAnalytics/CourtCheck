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
    base_color=(255, 255, 0),  # BGR: yellow
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

    curr_pt = None
    for idx in range(start_idx, frame_idx + 1):
        if (
            idx >= len(ball_track)
            or ball_track[idx] is None
            or ball_track[idx][0] is None
        ):
            continue

        x, y = ball_track[idx]
        pt = (int(x), int(y))

        # Alpha fade: oldest = most transparent, newest = full brightness
        alpha = 1.0 - ((frame_idx - idx) / trace_length)
        color = tuple(int(c * alpha) for c in base_color)

        is_current = (idx == frame_idx)
        radius = 5 if is_current else 3
        cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)

        if is_current:
            curr_pt = pt

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
        # White text on the colored pill — the brand stroke colors are
        # mid-saturated, so black text against them was muddy/unreadable.
        cv2.putText(frame, label, (rx1 + pad, ry2 - pad), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame


# Per-player colors (BGR). Cobalt + clay home/away pair — both outside the
# brand stroke axis (court green / plum / amber), so the bbox color can never
# be confused with whatever stroke pill sits on top of it.
#   P1 = #2A6FB0 cobalt → BGR(176, 111, 42)
#   P2 = #B05B36 clay   → BGR( 54,  91, 176)
_PLAYER_COLORS = [
    (176, 111,  42),   # player 1 — cobalt
    ( 54,  91, 176),   # player 2 — clay
    (200, 210, 220),   # player 3 — cool grey (rare; doubles fallback)
    ( 90,  90, 105),   # player 4 — slate    (rare; doubles fallback)
]


_FAR_PLAYER_COLOR = (54, 91, 176)  # clay (matches P2)


def _player_color(track_id: int):
    # Negative IDs are the canonical far player — always hot pink
    if track_id < 0:
        return _FAR_PLAYER_COLOR
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

        # Slightly thicker stroke so the neutral bboxes still pop against
        # the green court at a glance.
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2, cv2.LINE_AA)

        # Pill label at top-left corner of box. Larger than the original
        # 0.5/pad=4 — coaches were squinting at "P1"/"P2" on 720p replays.
        label = "P2" if track_id < 0 else f"P{track_id}"
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        pad = 7
        lx1, ly1 = x1, y1 - th - pad * 2
        lx2, ly2 = x1 + tw + pad * 2, y1

        # Clamp label above frame edge
        if ly1 < 0:
            ly1, ly2 = y1, y1 + th + pad * 2

        # Semi-transparent pill background
        overlay = frame.copy()
        cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), c, -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # Pick text contrast based on pill luminance: dark text on the
        # light P1 pill, white text on the dark P2 pill.
        text_color = (20, 20, 20) if (c[0] + c[1] + c[2]) > 384 else (245, 245, 245)
        cv2.putText(frame, label, (lx1 + pad, ly2 - pad), font, scale, text_color, thickness, cv2.LINE_AA)

    return frame


# -----------------------------
# Minimap drawing
# -----------------------------

STROKE_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    # Match the brand colors used in the shot-map dashboard tile so the minimap
    # and the courtmap legend agree. The light-mode hexes were too desaturated
    # on a video frame — adjacent strokes looked like "kind of dark green"
    # vs "kind of dark mauve". Use the dark-mode brand variants which are
    # noticeably brighter while staying on the brand axis.
    # Source: docs/brand-drop/tokens.css dark-mode swatches.
    #   --color-court (dark) #6FA88B -> RGB(111,168,139) -> BGR(139,168,111)
    #   --color-plum  (dark) #B584A6 -> RGB(181,132,166) -> BGR(166,132,181)
    #   --color-amber (dark) #DDB166 -> RGB(221,177,102) -> BGR(102,177,221)
    "Forehand":       (139, 168, 111),  # bright court green
    "Backhand":       (166, 132, 181),  # bright plum
    "Serve/Overhead": (102, 177, 221),  # bright amber
    # Slice currently isn't emitted by the live classifier but the swatch is
    # kept in case it's re-enabled — match the brand-mute tone.
    "Slice":          (138, 138, 138),  # ink-mute grey
}
# Neutral bounce color for bounces with no classified stroke (and for all
# near-side bounces, since we don't yet attribute P2's strokes). Used to be
# ink-mute (160) which faded into the lifted minimap bg — now a brand
# tennis-ball yellow so even unclassified bounces read clearly. BGR.
#   #F0D74E -> RGB(240, 215, 78) -> BGR(78, 215, 240)
_STROKE_BOUNCE_DEFAULT = (78, 215, 240)


def draw_minimap_ball_and_bounces(
    minimap,
    homography_inv,
    ball_track,
    frame_idx,
    bounces,
    trace_length=7,
    trace_color=(255, 255, 0),
    trace_min_alpha=0.3,
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
    trace_min_alpha: floor on the per-dot opacity. With a long trace
        (e.g. 45 frames) the natural linear falloff renders the oldest
        dots invisible — clamping at ~0.3 keeps the full comet tail
        readable on the minimap.
    frame_stroke_labels: dict[int, dict[int, str]] mapping frame_idx ->
        {track_id -> stroke label}.  When provided, each bounce dot is
        colored by the stroke type that produced it instead of the
        uniform bounce_color.
    """

    if homography_inv is None:
        return minimap

    # ---- 1. Ball trace ----
    start_idx = max(0, frame_idx - trace_length + 1)
    # Trail is drawn on the full-size court reference, then resized down to
    # ~166x350 for compositing. The pre-resize radii used to be 5/3, which
    # shrunk to ~1.5px and disappeared. Bumped so the trail survives the
    # downscale and reads clearly on the dark minimap.
    HEAD_RADIUS = 14
    TAIL_RADIUS = 9

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

        # Linear falloff, clamped at trace_min_alpha so the back of the
        # comet stays visible even with long trails.
        age = frame_idx - idx
        alpha_linear = 1.0 - (age / max(1, trace_length))
        alpha = max(trace_min_alpha, alpha_linear)
        color = tuple(int(c * alpha) for c in trace_color)
        radius = HEAD_RADIUS if idx == frame_idx else TAIL_RADIUS

        cv2.circle(minimap, (mx, my), radius, color, -1, cv2.LINE_AA)

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

        # Drop the bounds check — cv2.circle clips off-canvas pixels
        # automatically, so a bounce that landed just past the baseline
        # (where the SVG shot map renders it via extendBehind=4) shows
        # as a partial disk at the canvas edge instead of vanishing.
        # Plausibility-implausible coords (way beyond the court) get
        # clipped to nothing — same effective behavior as before.
        # Bottom half of the minimap is the near (P1) side. A bounce
        # there means P2 hit it — and we don't classify P2's strokes
        # yet — so leave it stroke-neutral instead of mislabeling it
        # with whatever swing happened to be active that frame.
        on_near_side = my > minimap.shape[0] // 2

        dot_color = bounce_color
        if on_near_side:
            dot_color = _STROKE_BOUNCE_DEFAULT
        elif frame_stroke_labels is not None:
            labels_at_bounce = frame_stroke_labels.get(bounce_idx, {})
            if labels_at_bounce:
                label = next(iter(labels_at_bounce.values()))
                dot_color = STROKE_COLORS_BGR.get(label, _STROKE_BOUNCE_DEFAULT)

        # Punch up the bounce so it reads at minimap scale. Big colored
        # fill + thick white outline (not a separate white disc) so the
        # color is unambiguously the dot, with a crisp edge against the
        # court surface. After the ~3.4× downscale to 350px tall this
        # renders as a ~14px dot with a ~2.5px white outline.
        cv2.circle(minimap, (mx, my), 90, dot_color, -1)             # colored fill
        cv2.circle(minimap, (mx, my), 90, (255, 255, 255), 18)       # crisp outline

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
