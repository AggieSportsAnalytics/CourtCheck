import cv2
import numpy as np
import os
from backend.vision.court_reference import CourtReference
from backend.vision.drawing import STROKE_COLORS_BGR

_DEFAULT_BOUNCE_COLOR_BGR: tuple[int, int, int] = (0, 220, 200)   # teal (unknown)
_DEFAULT_SHOT_COLOR_BGR:   tuple[int, int, int] = (40, 180, 255)   # amber (no label)
_OUT_COLOR_BGR:            tuple[int, int, int] = (0, 0, 220)      # red  (out of bounds)


def build_heatmap_court_background():
    """Dark navy background with soft white court lines — matches minimap style."""
    raw_court = CourtReference().build_court_reference()
    raw_court = cv2.dilate(raw_court, np.ones((10, 10), dtype=np.uint8))
    bg = np.full((*raw_court.shape, 3), (52, 36, 18), dtype=np.uint8)  # dark navy (BGR)
    bg[raw_court.astype(bool)] = (235, 235, 235)                        # soft white lines
    return bg



def _project_point(x, y, homography):
    """Project a frame-space point to court-space via homography. Returns (cx, cy) or None."""
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    try:
        mapped = cv2.perspectiveTransform(pt, homography)
    except cv2.error:
        return None
    return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])


def _lookup_stroke_label(frame_idx: int, frame_stroke_labels: "dict[int, dict[int, str]] | None", window: int = 5) -> "str | None":
    """
    Look up the stroke label for a given frame from frame_stroke_labels.
    Searches frame_idx and a ±window frame window around it.
    Returns the first label found, or None.
    """
    if not frame_stroke_labels:
        return None
    for offset in range(0, window + 1):
        for delta in ([0] if offset == 0 else [offset, -offset]):
            candidate = frame_idx + delta
            entry = frame_stroke_labels.get(candidate)
            if entry:
                # Take the first player's label (doesn't matter which player)
                return next(iter(entry.values()), None)
    return None


def _draw_bounce_dot(
    image: np.ndarray,
    xx: int,
    yy: int,
    color_bgr: tuple[int, int, int],
    is_out: bool,
) -> np.ndarray:
    """
    Draw a bounce marker on image (in-place copy).
    In-bounds: filled circle (radius 12) with white ring (radius 16, thickness 4).
    Out-of-bounds: red X (two crossing lines).
    Returns the same image (mutations applied).
    """
    if is_out:
        arm = 18
        cv2.line(image, (xx - arm, yy - arm), (xx + arm, yy + arm), _OUT_COLOR_BGR, 5, cv2.LINE_AA)
        cv2.line(image, (xx + arm, yy - arm), (xx - arm, yy + arm), _OUT_COLOR_BGR, 5, cv2.LINE_AA)
    else:
        # Solid filled dot — no white ring (it obscures the color at this scale)
        cv2.circle(image, (xx, yy), 28, color_bgr, -1, cv2.LINE_AA)
    return image


def generate_minimap_heatmaps(
    homography_matrices,
    ball_track,
    bounces,
    player_detections,
    output_bounce_heatmap,
    output_player_heatmap,
    ball_shot_frames=None,
    frame_stroke_labels: "dict[int, dict[int, str]] | None" = None,
    in_bounds_bounces: "set[int] | None" = None,
    **_kwargs,
):
    """
    Generate bounce dot map and player shot-position dot map on a top-down court.

    Bounce heatmap: color-coded dot per detected bounce (stroke type + in/out).
    Player heatmap: dot per shot moment at player foot position.
    """
    court_img = build_heatmap_court_background()
    Hc, Wc = court_img.shape[:2]
    n_frames = len(homography_matrices)

    # ---- Bounce heatmap: direct circle drawing, color-coded by stroke ----
    bounce_overlay = court_img.copy()
    for i in bounces:
        if i >= n_frames or i >= len(ball_track):
            continue
        bpos = ball_track[i]
        bx, by = bpos if bpos is not None else (None, None)
        inv_mat = homography_matrices[i] if i < len(homography_matrices) else None
        if bx is None or inv_mat is None:
            continue

        result = _project_point(bx, by, inv_mat)
        if result is None:
            continue
        xx, yy = int(result[0]), int(result[1])
        if not (0 <= xx < Wc and 0 <= yy < Hc):
            continue

        label = _lookup_stroke_label(i, frame_stroke_labels)
        color_bgr = STROKE_COLORS_BGR.get(label, _DEFAULT_BOUNCE_COLOR_BGR) if label else _DEFAULT_BOUNCE_COLOR_BGR
        is_out = (in_bounds_bounces is not None) and (i not in in_bounds_bounces)
        _draw_bounce_dot(bounce_overlay, xx, yy, color_bgr, is_out)

    # ---- Player shot-position dot map (replaces gaussian blob) ----
    generate_player_shot_dot_map(
        homography_matrices=homography_matrices,
        player_detections=player_detections,
        shot_frames=ball_shot_frames or [],
        frame_stroke_labels=frame_stroke_labels,
        output_path=output_player_heatmap,
    )

    _save_heatmap_png(output_bounce_heatmap, bounce_overlay, "bounce")


def generate_player_shot_dot_map(
    homography_matrices,
    player_detections,
    shot_frames,
    frame_stroke_labels: "dict[int, dict[int, str]] | None",
    output_path: str,
):
    """
    Generate a player position dot map at shot moments on a top-down court.

    For each shot frame, projects each detected player's foot position to court
    space and draws a colored dot (color = stroke type).
    """
    court_img = build_heatmap_court_background()
    Hc, Wc = court_img.shape[:2]
    n_frames = len(homography_matrices)
    overlay = court_img.copy()

    for frame_idx in shot_frames:
        if frame_idx >= n_frames:
            continue

        inv_mat = homography_matrices[frame_idx]
        if inv_mat is None:
            continue

        player_dict = player_detections[frame_idx] if frame_idx < len(player_detections) else None
        if not player_dict:
            continue

        frame_labels: "dict[int, str]" = frame_stroke_labels.get(frame_idx, {}) if frame_stroke_labels else {}

        for track_id, bbox in player_dict.items():
            x1, y1, x2, y2 = bbox
            foot_x = (x1 + x2) / 2
            foot_y = float(y2)

            result = _project_point(foot_x, foot_y, inv_mat)
            if result is None:
                continue
            cx, cy = int(result[0]), int(result[1])
            if not (0 <= cx < Wc and 0 <= cy < Hc):
                continue

            label = frame_labels.get(track_id)
            if label is None:
                label = _lookup_stroke_label(frame_idx, frame_stroke_labels)
            color_bgr = STROKE_COLORS_BGR.get(label, _DEFAULT_SHOT_COLOR_BGR) if label else _DEFAULT_SHOT_COLOR_BGR

            cv2.circle(overlay, (cx, cy), 32, color_bgr, -1, cv2.LINE_AA)       # filled dot

    _save_heatmap_png(output_path, overlay, "player_shot_map")


def _save_heatmap_png(path, image, label):
    """Save a heatmap image to disk as landscape (rotated 90 CW so near baseline is on the right)."""
    heatmap_dir = os.path.dirname(path)
    if heatmap_dir and not os.path.exists(heatmap_dir):
        os.makedirs(heatmap_dir, exist_ok=True)

    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    success = cv2.imwrite(path, image)
    if not success:
        print(f"[Heatmap] ERROR saving {label} heatmap: {path}")
