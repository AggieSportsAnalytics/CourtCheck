# Utility Script


def scene_detect(path_video):
    """
    Split video to disjoint fragments based on color histograms
    using PySceneDetect.
    """
    video_manager = VideoManager([path_video])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)

    if not scene_list:
        scene_list = [
            (video_manager.get_base_timecode(), video_manager.get_current_timecode())
        ]
    scenes = [[x[0].frame_num, x[1].frame_num] for x in scene_list]
    return scenes


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
        for bbox, center_pt in persons_top[i]:
            if bbox is not None and len(bbox) == 4:
                cx, cy = center_pt
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                mapped = cv2.perspectiveTransform(pt, inv_mat)
                xx, yy = int(mapped[0, 0, 0]), int(mapped[0, 0, 1])
                if 0 <= xx < Wc and 0 <= yy < Hc:
                    cv2.circle(player_acc, (xx, yy), 10, 1.0, -1)

        # bottom
        for bbox, center_pt in persons_bottom[i]:
            if bbox is not None and len(bbox) == 4:
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
    cv2.imwrite(output_bounce_heatmap, bounce_overlay)
    cv2.imwrite(output_player_heatmap, player_overlay)
    print(f"Saved bounce heatmap to: {output_bounce_heatmap}")
    print(f"Saved player heatmap to: {output_player_heatmap}")
