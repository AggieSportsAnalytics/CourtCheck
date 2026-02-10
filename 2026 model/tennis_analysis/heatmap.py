# heatmap.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ----------------------------
# Frame selection helpers
# ----------------------------

def get_rally_frame_mask(num_frames: int, ball_shot_frames: List[int]) -> np.ndarray:
    """
    Returns boolean mask with True for frames between consecutive ball-shot frames.
    """
    mask = np.zeros(num_frames, dtype=bool)
    if not ball_shot_frames or len(ball_shot_frames) < 2:
        return mask

    for i in range(len(ball_shot_frames) - 1):
        s = int(ball_shot_frames[i])
        e = int(ball_shot_frames[i + 1])
        if 0 <= s < num_frames and 0 <= e < num_frames and e > s:
            mask[s:e + 1] = True
    return mask


# ----------------------------
# MiniCourt ROIs
# ----------------------------

def get_minicourt_panel_roi(mini_court) -> Tuple[int, int, int, int]:
    """Full white mini-court rectangle (background box) in full-frame coords."""
    x1 = int(mini_court.start_x)
    y1 = int(mini_court.start_y)
    x2 = int(mini_court.end_x)
    y2 = int(mini_court.end_y)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid MiniCourt PANEL ROI. Check MiniCourt init.")
    return x1, y1, x2, y2


def get_minicourt_inner_court_roi(mini_court) -> Tuple[int, int, int, int]:
    """Inner playable court rectangle in full-frame coords."""
    x1 = int(mini_court.court_start_x)
    y1 = int(mini_court.court_start_y)
    x2 = int(mini_court.court_end_x)
    y2 = int(mini_court.court_end_y)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid MiniCourt INNER ROI. Check MiniCourt init.")
    return x1, y1, x2, y2


def get_minicourt_heat_roi(mini_court, margin_px: int = 40) -> Tuple[int, int, int, int]:
    """
    Expanded ROI around the inner court where we allow heat to appear,
    clamped to the panel so it can't spill outside the white rectangle.
    """
    px1, py1, px2, py2 = get_minicourt_panel_roi(mini_court)
    cx1, cy1, cx2, cy2 = get_minicourt_inner_court_roi(mini_court)

    x1 = max(px1, cx1 - margin_px)
    y1 = max(py1, cy1 - margin_px)
    x2 = min(px2, cx2 + margin_px)
    y2 = min(py2, cy2 + margin_px)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid heat ROI after applying margin.")
    return x1, y1, x2, y2


# ----------------------------
# Template rendering
# ----------------------------

def render_minicourt_template(
    mini_court,
    frame_shape: Tuple[int, int, int],
    crop: str = "panel",  # "panel", "inner", or "full"
) -> np.ndarray:
    """
    Draw mini-court on a blank frame then crop.
    - panel: returns the full white mini-court box (recommended for dashboard images)
    - inner: returns only the inner court rectangle
    - full: returns full frame
    """
    h, w = frame_shape[:2]
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    template_full = mini_court.draw_mini_court([blank])[0]

    if crop == "full":
        return template_full

    if crop == "inner":
        x1, y1, x2, y2 = get_minicourt_inner_court_roi(mini_court)
        return template_full[y1:y2, x1:x2].copy()

    # default: panel
    x1, y1, x2, y2 = get_minicourt_panel_roi(mini_court)
    return template_full[y1:y2, x1:x2].copy()


# ----------------------------
# Heatmap building (expanded outside court)
# ----------------------------

def build_heatmap_from_minicourt_positions(
    player_mini_court_detections: List[Dict[int, Tuple[float, float]]],
    player_id: int,
    mini_court,
    rally_mask: Optional[np.ndarray] = None,
    stride: int = 3,
    bins: Tuple[int, int] = (180, 90),  # (y_bins, x_bins)
    blur_sigma: float = 2.0,
    margin_px: int = 40,                # NEW: allow heat outside court
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Returns:
      - heatmap float array in [0,1], shape (y_bins, x_bins)
      - roi_box_full_frame: (x1,y1,x2,y2) in full-frame coords where the heatmap should be pasted.

    Heat ROI is INNER COURT expanded by margin_px, clamped to the PANEL.
    """
    hx1, hy1, hx2, hy2 = get_minicourt_heat_roi(mini_court, margin_px=margin_px)
    roi_w = float(hx2 - hx1)
    roi_h = float(hy2 - hy1)

    xs, ys = [], []
    n = len(player_mini_court_detections)

    for f in range(0, n, stride):
        if rally_mask is not None and not rally_mask[f]:
            continue

        det = player_mini_court_detections[f]
        if not det or player_id not in det:
            continue

        x, y = det[player_id]
        if x is None or y is None:
            continue

        # ROI-local coords
        lx = float(x) - hx1
        ly = float(y) - hy1

        # clamp to expanded ROI (not inner court)
        if not (0.0 <= lx <= roi_w and 0.0 <= ly <= roi_h):
            continue

        xs.append(lx)
        ys.append(ly)

    if len(xs) < 10:
        return None, None

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    # histogram2d with (ys, xs) => output is [y, x]
    h, _, _ = np.histogram2d(
        ys, xs,
        bins=bins,
        range=[[0.0, roi_h], [0.0, roi_w]],
    )
    h = h.astype(np.float32)

    if blur_sigma and blur_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            h = gaussian_filter(h, sigma=blur_sigma)
        except Exception:
            pass

    h = h / (h.max() + 1e-9)
    return h, (hx1, hy1, hx2, hy2)


# ----------------------------
# Rendering / overlay utilities
# ----------------------------

def overlay_heatmap_on_template(
    template_bgr: np.ndarray,
    heatmap01: np.ndarray,
    alpha: float = 0.72,
    colormap: int = cv2.COLORMAP_INFERNO,
    clip_percentile: float = 99.0,
) -> np.ndarray:
    """
    Blends heatmap over the entire template image.
    Use this only when the template size matches the heat ROI size exactly.
    """
    hm = heatmap01.copy()

    if clip_percentile is not None:
        p = np.percentile(hm, clip_percentile)
        if p > 1e-9:
            hm = np.clip(hm / p, 0, 1)

    hm_u8 = (hm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, colormap)
    hm_color = cv2.resize(hm_color, (template_bgr.shape[1], template_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    return cv2.addWeighted(hm_color, alpha, template_bgr, 1.0 - alpha, 0)


# ----------------------------
# Save PNG on panel (with margins + outside-court heat)
# ----------------------------

def save_static_heatmap_png_with_panel(
    out_path: Path | str,
    mini_court,
    frame_shape: Tuple[int, int, int],
    heatmap01: np.ndarray,
    roi_box_full_frame: Tuple[int, int, int, int],  # where the heatmap belongs (expanded ROI)
    alpha: float = 0.72,
    colormap: int = cv2.COLORMAP_INFERNO,
    clip_percentile: float = 99.0,
) -> None:
    """
    Saves a PNG that includes the full mini-court panel.
    Heat is blended into roi_box_full_frame (expanded ROI) so it can appear outside lines.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panel = render_minicourt_template(mini_court, frame_shape, crop="panel")

    px1, py1, px2, py2 = get_minicourt_panel_roi(mini_court)
    x1, y1, x2, y2 = roi_box_full_frame

    # Map ROI (full-frame) -> panel-local coords
    rx1 = int(x1 - px1)
    ry1 = int(y1 - py1)
    rx2 = int(x2 - px1)
    ry2 = int(y2 - py1)

    # Safety clamp to panel bounds
    rx1 = max(0, min(panel.shape[1], rx1))
    rx2 = max(0, min(panel.shape[1], rx2))
    ry1 = max(0, min(panel.shape[0], ry1))
    ry2 = max(0, min(panel.shape[0], ry2))
    if rx2 <= rx1 or ry2 <= ry1:
        raise ValueError("Heat ROI maps outside the panel. Check ROI math.")

    # Contrast clip
    hm = heatmap01.copy()
    if clip_percentile is not None:
        p = np.percentile(hm, clip_percentile)
        if p > 1e-9:
            hm = np.clip(hm / p, 0, 1)

    hm_u8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, colormap)
    hm_color = cv2.resize(hm_color, (rx2 - rx1, ry2 - ry1), interpolation=cv2.INTER_LINEAR)

    out = panel.copy()
    roi = out[ry1:ry2, rx1:rx2]
    out[ry1:ry2, rx1:rx2] = cv2.addWeighted(hm_color, alpha, roi, 1.0 - alpha, 0)

    cv2.imwrite(str(out_path), out)
