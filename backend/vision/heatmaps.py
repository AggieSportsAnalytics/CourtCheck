import cv2
import numpy as np
import os
from backend.vision.court_reference import CourtReference

def build_heatmap_court_background_black():
    """
    Black background with thick white court lines.
    """
    raw_court = CourtReference().build_court_reference()
    raw_court = cv2.dilate(raw_court, np.ones((10, 10), dtype=np.uint8))

    background = np.zeros_like(raw_court, dtype=np.uint8)
    background[raw_court == 1] = 255  # white lines
    return cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)


def build_custom_colormap_black_purple_red_green_yellow():
    """
    Creates a custom color map (256 x 1 x 3) with 5 anchors:
      0   -> black
      64  -> purple
      128 -> red
      192 -> green
      255 -> bright yellow
    This ensures that 'no data' (0) stays black, and high frequency = yellow.
    """
    anchors = [
        (0, (0, 0, 0)),  # black
        (64, (128, 0, 128)),  # purple
        (128, (0, 0, 255)),  # red
        (192, (0, 255, 0)),  # green
        (255, (0, 255, 255)),  # bright yellow
    ]
    ctable = np.zeros((256, 1, 3), dtype=np.uint8)

    def lerp_color(c1, c2, t):
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )

    for i in range(len(anchors) - 1):
        start_idx, start_col = anchors[i]
        end_idx, end_col = anchors[i + 1]
        for x in range(start_idx, end_idx + 1):
            if end_idx == start_idx:
                t = 0
            else:
                t = (x - start_idx) / float(end_idx - start_idx)
            ctable[x, 0] = lerp_color(start_col, end_col, t)

    return ctable


def generate_minimap_heatmaps(
    homography_matrices,
    ball_track,
    bounces,
    persons_top,
    persons_bottom,
    output_bounce_heatmap,
    output_player_heatmap,
    blur_ksize=41,
    alpha=0.5,
):
    """
    1) For ball bounces, draw bigger bright‐yellow circles (radius=8) with red outline.
    2) For player positions, accumulate + blur, then apply a custom colormap:
       black->purple->red->green->yellow, so zero=black, max=yellow.
    3) The background stays black with white lines (where no data is present).
    """

    # (A) Build black court background
    court_img = build_heatmap_court_background_black()
    Hc, Wc = court_img.shape[:2]
    n_frames = len(homography_matrices)

    # (B) Bounces => direct drawing
    bounce_overlay = court_img.copy()
    for i in range(n_frames):
        if i not in bounces:
            continue
        bx, by = ball_track[i]
        inv_mat = homography_matrices[i]
        if bx is None or inv_mat is None:
            continue

        pt = np.array([[[bx, by]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, inv_mat)
        xx, yy = int(mapped[0, 0, 0]), int(mapped[0, 0, 1])
        if 0 <= xx < Wc and 0 <= yy < Hc:
            # Increase radius from 5 to 8 for bigger circles
            cv2.circle(bounce_overlay, (xx, yy), 12, (0, 255, 255), -1)  # fill (yellow)
            cv2.circle(bounce_overlay, (xx, yy), 12, (0, 0, 255), 2)  # outline (red)

    # (C) Players => aggregator -> blur -> custom colormap
    player_acc = np.zeros((Hc, Wc), dtype=np.float32)
    for i in range(n_frames):
        inv_mat = homography_matrices[i]
        if inv_mat is None:
            continue

        # top
        for bbox, center_pt, display_name in persons_top[i]:
            # Only include in heatmap if player is identified as Player 1 or Player 2
            if (
                bbox is not None
                and len(bbox) == 4
                and display_name in ["Player 1", "Player 2"]
            ):
                cx, cy = center_pt
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                mapped = cv2.perspectiveTransform(pt, inv_mat)
                xx, yy = int(mapped[0, 0, 0]), int(mapped[0, 0, 1])
                if 0 <= xx < Wc and 0 <= yy < Hc:
                    cv2.circle(player_acc, (xx, yy), 10, 1.0, -1)

        # bottom
        for bbox, center_pt, display_name in persons_bottom[i]:
            # Only include in heatmap if player is identified as Player 1 or Player 2
            if (
                bbox is not None
                and len(bbox) == 4
                and display_name in ["Player 1", "Player 2"]
            ):
                cx, cy = center_pt
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                mapped = cv2.perspectiveTransform(pt, inv_mat)
                xx, yy = int(mapped[0, 0, 0]), int(mapped[0, 0, 1])
                if 0 <= xx < Wc and 0 <= yy < Hc:
                    cv2.circle(player_acc, (xx, yy), 10, 1.0, -1)

    # blur => smoother distribution
    player_blurred = cv2.GaussianBlur(player_acc, (blur_ksize, blur_ksize), 0)

    # (D) Convert blurred player data -> custom colormap
    mx = player_blurred.max()
    if mx < 1e-8:
        # no data
        player_overlay = court_img.copy()
    else:
        norm = (player_blurred / mx * 255).astype(np.uint8)
        custom_cmap = build_custom_colormap_black_purple_red_green_yellow()
        player_heat = cv2.applyColorMap(norm, custom_cmap)
        player_overlay = cv2.addWeighted(court_img, 1.0, player_heat, alpha, 0.0)

    # (E) Save
    # Ensure output directories exist and add detailed save logging
    bounce_heatmap_dir = os.path.dirname(output_bounce_heatmap)
    if bounce_heatmap_dir and not os.path.exists(bounce_heatmap_dir):
        print(
            f"[Heatmap Save] Creating directory for bounce heatmap: {bounce_heatmap_dir}"
        )
        try:
            os.makedirs(bounce_heatmap_dir, exist_ok=True)
        except Exception as e_mkdir_bounce:
            print(
                f"[Heatmap Save] ERROR creating directory {bounce_heatmap_dir}: {e_mkdir_bounce}"
            )
            # Potentially skip saving if directory creation fails, or handle error as appropriate

    player_heatmap_dir = os.path.dirname(output_player_heatmap)
    if player_heatmap_dir and not os.path.exists(player_heatmap_dir):
        print(
            f"[Heatmap Save] Creating directory for player heatmap: {player_heatmap_dir}"
        )
        try:
            os.makedirs(player_heatmap_dir, exist_ok=True)
        except Exception as e_mkdir_player:
            print(
                f"[Heatmap Save] ERROR creating directory {player_heatmap_dir}: {e_mkdir_player}"
            )
            # Potentially skip saving

    try:
        write_success_bounce = cv2.imwrite(output_bounce_heatmap, bounce_overlay)
        if write_success_bounce:
            print(
                f"[Heatmap Save] cv2.imwrite for bounce heatmap reported SUCCESS. Path: {output_bounce_heatmap}"
            )
            if os.path.exists(output_bounce_heatmap):
                print(
                    f"  [Heatmap Save] Bounce heatmap file VERIFIED on disk: {output_bounce_heatmap}"
                )
            else:
                print(
                    f"  [Heatmap Save] CRITICAL ERROR: Bounce heatmap file NOT FOUND on disk after reported cv2.imwrite success. Path: {output_bounce_heatmap}"
                )
        else:
            print(
                f"[Heatmap Save] ERROR: cv2.imwrite for bounce heatmap reported FAILURE. Path: {output_bounce_heatmap}"
            )
    except Exception as e_bounce_write:
        print(
            f"[Heatmap Save] EXCEPTION during cv2.imwrite for bounce heatmap. Path: {output_bounce_heatmap}. Error: {e_bounce_write}"
        )

    try:
        write_success_player = cv2.imwrite(output_player_heatmap, player_overlay)
        if write_success_player:
            print(
                f"[Heatmap Save] cv2.imwrite for player heatmap reported SUCCESS. Path: {output_player_heatmap}"
            )
            if os.path.exists(output_player_heatmap):
                print(
                    f"  [Heatmap Save] Player heatmap file VERIFIED on disk: {output_player_heatmap}"
                )
            else:
                print(
                    f"  [Heatmap Save] CRITICAL ERROR: Player heatmap file NOT FOUND on disk after reported cv2.imwrite success. Path: {output_player_heatmap}"
                )
        else:
            print(
                f"[Heatmap Save] ERROR: cv2.imwrite for player heatmap reported FAILURE. Path: {output_player_heatmap}"
            )
    except Exception as e_player_write:
        print(
            f"[Heatmap Save] EXCEPTION during cv2.imwrite for player heatmap. Path: {output_player_heatmap}. Error: {e_player_write}"
        )