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


def ensure_720p(input_path, intermediate_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Original input: {width}x{height}, fps={fps:.2f}")
    if (width != 1280) or (height != 720):
        print(f"Resizing from ({width}x{height}) to (1280x720) -> {intermediate_path}")
        cap_in = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(intermediate_path, fourcc, fps, (1280, 720))

        while True:
            ret, frame = cap_in.read()
            if not ret:
                break
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            out.write(frame)

        cap_in.release()
        out.release()
        print(f"Finished writing intermediate: {intermediate_path}")
        return intermediate_path
    else:
        print("Video is already 1280x720; using input directly.")
        return input_path


def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def get_court_img():
    """Build a 720p-like minimap with white lines on black background."""
    court_ref = CourtReference()
    court = court_ref.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2) * 255).astype(np.uint8)
    return court_img


def draw_court_keypoints_and_lines(frame, kps, frame_width, frame_height):
    """
    Draw tennis court lines (green) and keypoints (red) on 'frame'.
    """
    for start_name, end_name in court_lines:
        try:
            s_idx = keypoint_names.index(start_name)
            e_idx = keypoint_names.index(end_name)
            if kps[s_idx] is None or kps[e_idx] is None:
                continue
            x1 = int(kps[s_idx][0, 0] * frame_width / 1280)
            y1 = int(kps[s_idx][0, 1] * frame_height / 720)
            x2 = int(kps[e_idx][0, 0] * frame_width / 1280)
            y2 = int(kps[e_idx][0, 1] * frame_height / 720)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except ValueError:
            pass

    # Keypoints
    for i, pt in enumerate(kps):
        if pt is None:
            continue
        x = int(pt[0, 0] * frame_width / 1280)
        y = int(pt[0, 1] * frame_height / 720)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        label = keypoint_names[i]
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(
            frame, (x - 5, y - th - 5), (x - 5 + tw, y - 5), (255, 255, 255), -1
        )
        cv2.putText(
            frame, label, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
        )


def main(
    frames,
    scenes,
    bounces,
    ball_track,
    homography_matrices,
    kps_court,
    persons_top,
    persons_bottom,
    draw_trace=True,
    trace=7,
):
    imgs_res = []
    minimap_frames = []  # New list to store minimap frames
    width_minimap = 166
    height_minimap = 350

    light_blue = (255, 255, 0)
    bounce_color = (0, 255, 255)
    box_color = (255, 0, 0)

    for start_idx, end_idx in scenes:
        valid_count = sum(
            1
            for idx in range(start_idx, end_idx)
            if homography_matrices[idx] is not None
        )
        scene_len = end_idx - start_idx
        scene_rate = valid_count / (scene_len + 1e-15)

        if scene_rate > 0.5:
            for i in range(start_idx, end_idx):
                court_img = get_court_img()
                img_res = frames[i]
                inv_mat = homography_matrices[i]

                # ball
                if ball_track[i][0]:
                    bx = int(ball_track[i][0])
                    by = int(ball_track[i][1])
                    if draw_trace:
                        for j in range(trace):
                            idx = i - j
                            if idx < 0:
                                break
                            if ball_track[idx][0]:
                                px, py = ball_track[idx]
                                alpha = 1.0 - (j / trace)
                                color_fade = tuple(int(c * alpha) for c in light_blue)
                                cv2.circle(
                                    img_res, (int(px), int(py)), 3, color_fade, -1
                                )
                    else:
                        cv2.circle(img_res, (bx, by), 5, light_blue, -1)
                        cv2.putText(
                            img_res,
                            "ball",
                            (bx + 8, by + 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            light_blue,
                            2,
                        )

                # ball trace on minimap
                if inv_mat is not None and ball_track[i][0]:
                    for j in range(trace):
                        idx = i - j
                        if idx < 0:
                            break
                        if ball_track[idx][0]:
                            px, py = ball_track[idx]
                            arr = cv2.perspectiveTransform(
                                np.array([[[px, py]]], dtype=np.float32), inv_mat
                            )
                            alpha_ = 1.0 - (j / trace)
                            color_fade = tuple(int(c * alpha_) for c in light_blue)
                            mx = int(arr[0, 0, 0])
                            my = int(arr[0, 0, 1])
                            cv2.circle(court_img, (mx, my), 3, color_fade, 20)

                # court lines
                if kps_court[i] is not None:
                    h, w = img_res.shape[:2]
                    draw_court_keypoints_and_lines(img_res, kps_court[i], w, h)

                # bounces => up to current frame
                if inv_mat is not None:
                    for bf in bounces:
                        if bf <= i and ball_track[bf][0]:
                            bpx, bpy = ball_track[bf]
                            arr = cv2.perspectiveTransform(
                                np.array([[[bpx, bpy]]], dtype=np.float32), inv_mat
                            )
                            mx = int(arr[0, 0, 0])
                            my = int(arr[0, 0, 1])
                            cv2.circle(court_img, (mx, my), 10, bounce_color, 40)

                # players
                minimap = court_img.copy()

                # -- top players => Player 1
                for bbox, center_pt in persons_top[i]:
                    if bbox is not None and len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        # Draw the bounding box on the main frame
                        cv2.rectangle(img_res, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(
                            img_res,
                            "Player 1",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            box_color,
                            2,
                        )
                        # For the minimap, use the bottom-center point from PersonDetector
                        if inv_mat is not None:
                            cx, cy = center_pt  # <--- bottom-center from PersonDetector
                            arr = cv2.perspectiveTransform(
                                np.array([[[cx, cy]]], dtype=np.float32), inv_mat
                            )
                            mx = int(arr[0, 0, 0])
                            my = int(arr[0, 0, 1])
                            if (
                                0 <= mx < minimap.shape[1]
                                and 0 <= my < minimap.shape[0]
                            ):
                                cv2.circle(minimap, (mx, my), 48, box_color, -1)

                # -- bottom players => Player 2
                for bbox, center_pt in persons_bottom[i]:
                    if bbox is not None and len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        # Draw the bounding box on the main frame
                        cv2.rectangle(img_res, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(
                            img_res,
                            "Player 2",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            box_color,
                            2,
                        )
                        # For the minimap, use the bottom-center point from PersonDetector
                        if inv_mat is not None:
                            cx, cy = center_pt
                            arr = cv2.perspectiveTransform(
                                np.array([[[cx, cy]]], dtype=np.float32), inv_mat
                            )
                            mx = int(arr[0, 0, 0])
                            my = int(arr[0, 0, 1])
                            if (
                                0 <= mx < minimap.shape[1]
                                and 0 <= my < minimap.shape[0]
                            ):
                                cv2.circle(minimap, (mx, my), 48, box_color, -1)

                # Store the minimap frame
                minimap_frames.append(
                    cv2.resize(minimap, (width_minimap, height_minimap))
                )

                # place minimap in main frame as before
                H, W = img_res.shape[:2]
                minimap_resized = cv2.resize(minimap, (width_minimap, height_minimap))
                img_res[0:height_minimap, 0:width_minimap] = minimap_resized

                imgs_res.append(img_res)
        else:
            # If the scene homography is mostly invalid, add blank minimaps
            blank_minimap = np.zeros((height_minimap, width_minimap, 3), dtype=np.uint8)
            minimap_frames.extend([blank_minimap] * (end_idx - start_idx))
            imgs_res += frames[start_idx:end_idx]

    return imgs_res, minimap_frames


def write(imgs_res, fps, output_path):
    if not imgs_res:
        print("No frames, skipping write.")
        return
    H, W = imgs_res[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for frame in imgs_res:
        out.write(frame)
    out.release()
    print(f"[write] Finished writing {output_path}")


if __name__ == "__main__":
    path_ball_track_model = "/content/drive/MyDrive/ASA Tennis Bounds Project/models/court_detection_model/detectron2/models/ball_detection_weights/tracknet_weights.pt"
    path_court_model = "/content/drive/MyDrive/ASA Tennis Bounds Project/models/court_detection_model/detectron2/models/court_detection_weights/model_tennis_court_det.pt"
    path_bounce_model = "/content/drive/MyDrive/ASA Tennis Bounds Project/models/court_detection_model/detectron2/models/bounce_detection_weights/bounce_detection_weights.cbm"
    path_input_video = "/content/drive/MyDrive/ASA Tennis Bounds Project/models/court_detection_model/dataset/UCDwten/game1_UCDwten_1280x720.mp4"
    path_intermediate_video = "/content/drive/MyDrive/ASA Tennis Bounds Project/models/court_detection_model/detectron2/post_processing/UCDwten/game1_UCDwten_edited.mp4"
    path_output_video = "/content/drive/MyDrive/ASA Tennis Bounds Project/models/court_detection_model/detectron2/post_processing/UCDwten/game1_UCDwten.mp4"

    path_output_bounce_heatmap = "/content/drive/MyDrive/ASA Tennis Bounds Project/models/court_detection_model/detectron2/post_processing/UCDwten/heatmap/game1_UCDwten_bounce_heatmap.png"
    path_output_player_heatmap = "/content/drive/MyDrive/ASA Tennis Bounds Project/models/court_detection_model/detectron2/post_processing/UCDwten/heatmap/game1_UCDwten_player_heatmap.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()

    # 1) Scale to 720p if needed
    final_input = ensure_720p(path_input_video, path_intermediate_video)

    # 2) Read frames
    frames, fps = read_video(final_input)
    print(f"Loaded {len(frames)} frames at {fps} fps")

    # 3) Scene detection
    video_manager = VideoManager([final_input])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)
    video_manager.release()

    if not scene_list:
        scene_list = [(base_timecode, base_timecode + len(frames) / fps)]
    scenes = []
    for sc in scene_list:
        start_f = sc[0].frame_num
        end_f = sc[1].frame_num
        scenes.append((start_f, end_f))

    # 4) Ball detection
    ball_detector = BallDetector(path_ball_track_model, device)
    ball_track = ball_detector.infer_model(frames)

    # 5) Court detection
    court_detector = CourtDetectorNet(path_court_model, device)
    homography_matrices, kps_court = court_detector.infer_model(frames)

    # 6) Person detection
    person_detector = PersonDetector(device)
    persons_top, persons_bottom = person_detector.track_players(
        frames, homography_matrices, filter_players=False
    )

    # 7) Bounce detection
    bounce_detector = BounceDetector(path_bounce_model)
    x_ball = [bp[0] for bp in ball_track]
    y_ball = [bp[1] for bp in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)

    # 8) Compose frames & write final videos
    print("Composing final frames...")
    imgs_res, minimap_frames = main(
        frames,
        scenes,
        bounces,
        ball_track,
        homography_matrices,
        kps_court,
        persons_top,
        persons_bottom,
        draw_trace=True,
        trace=7,
    )

    # Write main video
    write(imgs_res, fps, path_output_video)

    # Write minimap video
    path_minimap_video = path_output_video.replace(".mp4", "_minimap.mp4")
    write(minimap_frames, fps, path_minimap_video)

    # 9) Generate separate heatmaps for bounces/players
    generate_minimap_heatmaps(
        homography_matrices=homography_matrices,
        ball_track=ball_track,
        bounces=bounces,
        persons_top=persons_top,
        persons_bottom=persons_bottom,
        output_bounce_heatmap=path_output_bounce_heatmap,
        output_player_heatmap=path_output_player_heatmap,
        blur_ksize=41,
        alpha=0.5,
    )

    total_time = time.time() - start_time
    print(f"Done, saved to {path_output_video}. Overall time: {total_time:.2f}s")
