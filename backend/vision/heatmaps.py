import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from backend.vision.court_reference import CourtReference


def build_heatmap_court_background(style="black"):
    """
    Court background for heatmaps.
    style="black": black bg with white lines (default)
    style="white": white bg with gray lines
    """
    raw_court = CourtReference().build_court_reference()
    raw_court = cv2.dilate(raw_court, np.ones((10, 10), dtype=np.uint8))

    if style == "white":
        background = np.full_like(raw_court, 255, dtype=np.uint8)
        background[raw_court == 1] = 180  # gray lines
    else:
        background = np.zeros_like(raw_court, dtype=np.uint8)
        background[raw_court == 1] = 255  # white lines

    return cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)


def get_rally_frame_mask(num_frames, ball_shot_frames):
    """
    Returns boolean mask where True = frame is during an active rally
    (between consecutive shot frames).
    """
    mask = np.zeros(num_frames, dtype=bool)
    if not ball_shot_frames or len(ball_shot_frames) < 2:
        # If no shot data, treat all frames as valid
        mask[:] = True
        return mask

    for i in range(len(ball_shot_frames) - 1):
        s = int(ball_shot_frames[i])
        e = int(ball_shot_frames[i + 1])
        if 0 <= s < num_frames and 0 <= e < num_frames and e > s:
            mask[s:e + 1] = True
    return mask


def _project_point(x, y, homography):
    """Project a frame-space point to court-space via homography. Returns (cx, cy) or None."""
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    try:
        mapped = cv2.perspectiveTransform(pt, homography)
    except cv2.error:
        return None
    return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])


def generate_minimap_heatmaps(
    homography_matrices,
    ball_track,
    bounces,
    player_detections,
    output_bounce_heatmap,
    output_player_heatmap,
    ball_shot_frames=None,
    alpha=0.72,
    clip_percentile=99.0,
    colormap=cv2.COLORMAP_INFERNO,
    blur_sigma=8.0,
    bins=(350, 170),
    stride=3,
):
    """
    Generate bounce heatmap and player position heatmap on a top-down court.

    Uses homography matrices to project frame-space positions to court-space.
    Player heatmap uses histogram2d binning + gaussian blur for smooth density.

    Args:
        homography_matrices: list of 3x3 matrices (frame -> court), one per frame
        ball_track: list of (x, y) or None per frame
        bounces: set of frame indices where bounces occurred
        player_detections: list of {track_id: [x1,y1,x2,y2]} per frame
        output_bounce_heatmap: path to save bounce heatmap PNG
        output_player_heatmap: path to save player heatmap PNG
        ball_shot_frames: list of frame indices where shots occurred (for rally filtering)
        alpha: blend weight for heatmap overlay
        clip_percentile: percentile for contrast clipping (None to disable)
        colormap: OpenCV colormap constant
        blur_sigma: sigma for gaussian blur on histogram (controls smoothness)
        bins: (y_bins, x_bins) for histogram2d density grid
        stride: process every Nth frame for player positions (speed vs accuracy)
    """
    court_img = build_heatmap_court_background()
    Hc, Wc = court_img.shape[:2]
    n_frames = len(homography_matrices)

    # Rally mask for filtering non-rally frames
    rally_mask = get_rally_frame_mask(n_frames, ball_shot_frames)

    # ---- Bounce heatmap: direct circle drawing ----
    bounce_overlay = court_img.copy()
    for i in bounces:
        if i >= n_frames or i >= len(ball_track):
            continue
        bx, by = ball_track[i] if ball_track[i] is not None else (None, None)
        inv_mat = homography_matrices[i] if i < len(homography_matrices) else None
        if bx is None or inv_mat is None:
            continue

        result = _project_point(bx, by, inv_mat)
        if result is None:
            continue
        xx, yy = int(result[0]), int(result[1])
        if 0 <= xx < Wc and 0 <= yy < Hc:
            cv2.circle(bounce_overlay, (xx, yy), 12, (0, 255, 255), -1)  # yellow fill
            cv2.circle(bounce_overlay, (xx, yy), 12, (0, 0, 255), 2)     # red outline

    # ---- Player position heatmap: histogram2d + gaussian blur ----
    # Collect all projected court-space positions
    xs, ys = [], []
    for i in range(0, n_frames, stride):
        if not rally_mask[i]:
            continue

        inv_mat = homography_matrices[i] if i < len(homography_matrices) else None
        if inv_mat is None:
            continue

        player_dict = player_detections[i] if i < len(player_detections) else None
        if player_dict is None:
            continue

        for track_id, bbox in player_dict.items():
            x1, y1, x2, y2 = bbox
            foot_x = (x1 + x2) / 2
            foot_y = float(y2)

            result = _project_point(foot_x, foot_y, inv_mat)
            if result is None:
                continue
            cx, cy = result
            if 0 <= cx < Wc and 0 <= cy < Hc:
                xs.append(cx)
                ys.append(cy)

    if len(xs) < 10:
        # Not enough data for a meaningful heatmap
        player_overlay = court_img.copy()
    else:
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)

        # histogram2d: (ys, xs) so output shape is [y_bins, x_bins]
        h, _, _ = np.histogram2d(
            ys, xs,
            bins=bins,
            range=[[0.0, float(Hc)], [0.0, float(Wc)]],
        )
        h = h.astype(np.float32)

        # Smooth with gaussian filter
        h = gaussian_filter(h, sigma=blur_sigma)

        # Normalize
        h = h / (h.max() + 1e-9)

        # Contrast clipping
        if clip_percentile is not None:
            p = np.percentile(h, clip_percentile)
            if p > 1e-9:
                h = np.clip(h / p, 0, 1)

        # Convert to colormap and resize to court dimensions
        hm_u8 = (h * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_u8, colormap)
        hm_color = cv2.resize(hm_color, (Wc, Hc), interpolation=cv2.INTER_LINEAR)

        # Blend: only apply heat where there's actual data
        mask = cv2.resize(hm_u8, (Wc, Hc), interpolation=cv2.INTER_LINEAR) > 0
        player_overlay = court_img.copy()
        player_overlay[mask] = cv2.addWeighted(
            hm_color, alpha, court_img, 1.0 - alpha, 0.0
        )[mask]

    # ---- Save PNGs ----
    _save_heatmap_png(output_bounce_heatmap, bounce_overlay, "bounce")
    _save_heatmap_png(output_player_heatmap, player_overlay, "player")


def _save_heatmap_png(path, image, label):
    """Save a heatmap image to disk with logging."""
    heatmap_dir = os.path.dirname(path)
    if heatmap_dir and not os.path.exists(heatmap_dir):
        os.makedirs(heatmap_dir, exist_ok=True)

    success = cv2.imwrite(path, image)
    if success and os.path.exists(path):
        print(f"[Heatmap] {label} heatmap saved: {path}")
    else:
        print(f"[Heatmap] ERROR saving {label} heatmap: {path}")
