# pipeline/run.py
import cv2
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

from backend.models import BallDetector, CourtLineDetector, PlayerTracker, BounceDetector, ActionRecognition, PoseStrokeClassifier
from backend.vision import HomographyEstimator, CourtReference, draw_ball_trace, draw_court_keypoints_and_lines, draw_minimap_ball_and_bounces, draw_minimap_players, draw_player_bboxes, draw_stroke_labels
from backend.vision import SwingDetector, extract_pose_sequence
from backend.vision.calibration import load_calibration
from backend.vision.heatmaps import generate_minimap_heatmaps, generate_player_shot_dot_map
from backend.vision.postprocess import detect_shot_frames

from backend.pipeline.storage import upload_processed_video, upload_heatmap_png, get_supabase, make_streamable_mp4
from backend.pipeline.config import PipelineConfig


def ensure_720p(input_path, intermediate_path, target_width=1280, target_height=720):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Original input: {width}x{height}, fps={fps:.2f}")
    if (width != target_width) or (height != target_height):
        print(f"Resizing from ({width}x{height}) to ({target_width}x{target_height}) -> {intermediate_path}")
        cap_in = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(intermediate_path, fourcc, fps, (target_width, target_height))

        while True:
            ret, frame = cap_in.read()
            if not ret:
                break
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            out.write(frame)

        cap_in.release()
        out.release()
        print(f"Finished writing intermediate: {intermediate_path}")
        return intermediate_path
    else:
        print(f"Video is already {target_width}x{target_height}; using input directly.")
        return input_path


def calculate_rally_count(shot_frames, fps=30.0, gap_seconds=4):
    """
    Group shot frames into rallies. A gap > gap_seconds between consecutive
    shots is treated as the start of a new rally.
    """
    if not shot_frames:
        return 0
    sorted_frames = sorted(shot_frames)
    gap_frames = fps * gap_seconds
    rallies = 1
    for i in range(1, len(sorted_frames)):
        if sorted_frames[i] - sorted_frames[i - 1] > gap_frames:
            rallies += 1
    return rallies


def count_in_out_bounces(bounces, ball_track, homography_matrices, court_ref):
    """
    Classify each detected bounce as in-bounds or out-of-bounds using the
    court-space coordinates obtained via homography projection.
    """
    left_x   = court_ref.left_court_line[0][0]    # 286
    right_x  = court_ref.right_court_line[0][0]   # 1379
    top_y    = court_ref.baseline_top[0][1]        # 561
    bottom_y = court_ref.baseline_bottom[0][1]     # 2935

    in_bounds = 0
    out_bounds = 0
    in_bounds_set: set[int] = set()
    n_frames = len(homography_matrices)

    for frame_idx in bounces:
        if frame_idx >= len(ball_track) or frame_idx >= n_frames:
            continue
        pos = ball_track[frame_idx]
        H = homography_matrices[frame_idx]
        if pos is None or H is None:
            continue
        bx, by = pos
        if bx is None or by is None:
            continue
        pt = np.array([[[float(bx), float(by)]]], dtype=np.float32)
        try:
            mapped = cv2.perspectiveTransform(pt, H)
        except cv2.error:
            continue
        cx, cy = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

        if left_x <= cx <= right_x and top_y <= cy <= bottom_y:
            in_bounds += 1
            in_bounds_set.add(frame_idx)
        else:
            out_bounds += 1

    return in_bounds, out_bounds, in_bounds_set


def compute_spatial_stats(
    bounces: set,
    ball_track: list,
    homography_matrices: list,
    player_detections: list,
    court_ref,
) -> dict:
    """
    Project bounce and player positions into court reference space and return zone
    statistics used to enrich the AI scouting report.

    Court reference coordinate system (from CourtReference):
      x: 286 (left sideline) → 1379 (right sideline), center=832
      y: 561 (far baseline) → 2935 (near baseline), net=1748
      Singles inner lines: x ∈ [423, 1242]
      Service lines: y=1110 (far), y=2386 (near)
    """
    left_x      = court_ref.left_court_line[0][0]      # 286
    right_x     = court_ref.right_court_line[0][0]     # 1379
    top_y       = court_ref.baseline_top[0][1]          # 561
    bottom_y    = court_ref.baseline_bottom[0][1]       # 2935
    net_y       = court_ref.net[0][1]                   # 1748
    center_x    = court_ref.middle_line[0][0]           # 832
    inner_left  = court_ref.left_inner_line[0][0]      # 423
    inner_right = court_ref.right_inner_line[0][0]     # 1242
    svc_bot_y   = court_ref.bottom_inner_line[0][1]    # 2386  (near service line)


    def project(bx, by, H):
        pt = np.array([[[float(bx), float(by)]]], dtype=np.float32)
        try:
            mapped = cv2.perspectiveTransform(pt, H)
            return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
        except cv2.error:
            return None, None

    def pct(n, total):
        return round(n / total * 100) if total > 0 else 0

    # --- Bounce spatial distribution (in-bounds only) ---
    b_near = b_far = b_left = b_right = b_deep_near = b_alley = valid = 0

    for frame_idx in bounces:
        if frame_idx >= len(ball_track) or frame_idx >= len(homography_matrices):
            continue
        pos = ball_track[frame_idx]
        H = homography_matrices[frame_idx]
        if pos is None or H is None:
            continue
        bx, by = pos
        if bx is None or by is None:
            continue
        cx, cy = project(bx, by, H)
        if cx is None:
            continue
        if not (left_x <= cx <= right_x and top_y <= cy <= bottom_y):
            continue  # out-of-bounds, skip for zone analysis
        valid += 1
        if cy > net_y:
            b_near += 1
            if cy > svc_bot_y:
                b_deep_near += 1
        else:
            b_far += 1
        if cx < center_x:
            b_left += 1
        else:
            b_right += 1
        if cx < inner_left or cx > inner_right:
            b_alley += 1

    bounce_stats = {
        "near_pct":      pct(b_near, valid),
        "far_pct":       pct(b_far, valid),
        "left_pct":      pct(b_left, valid),
        "right_pct":     pct(b_right, valid),
        "deep_near_pct": pct(b_deep_near, b_near) if b_near > 0 else 0,
        "alley_pct":     pct(b_alley, valid),
        "total":         valid,
    }

    # --- Player positioning (near half, sampled every 5 frames) ---
    p_near_left = p_near_right = p_deep = p_forward = p_samples = 0

    for frame_idx in range(0, len(player_detections), 5):
        pd = player_detections[frame_idx]
        H = homography_matrices[frame_idx] if frame_idx < len(homography_matrices) else None
        if not pd or H is None:
            continue
        for bbox in pd.values():
            x1, y1, x2, y2 = bbox
            cx, cy = project((x1 + x2) / 2.0, float(y2), H)
            if cx is None:
                continue
            # Loose bounds check to filter projection artifacts
            if not (left_x - 300 <= cx <= right_x + 300 and top_y - 300 <= cy <= bottom_y + 300):
                continue
            p_samples += 1
            if cy > net_y:
                if cx < center_x:
                    p_near_left += 1
                else:
                    p_near_right += 1
                if cy > svc_bot_y:
                    p_deep += 1
                else:
                    p_forward += 1
    near_samples = p_near_left + p_near_right
    player_stats = {
        "near_pct":         pct(near_samples, p_samples),
        "near_left_pct":    pct(p_near_left, near_samples),
        "near_right_pct":   pct(p_near_right, near_samples),
        "near_deep_pct":    pct(p_deep, near_samples),
        "near_forward_pct": pct(p_forward, near_samples),
        "samples":          p_samples,
    }

    return {"bounces": bounce_stats, "player": player_stats}


def generate_scouting_report(stats: dict, fps: float, num_frames: int, spatial_stats: "dict | None" = None) -> "str | None":
    """Call GPT-4o-mini to generate a markdown scouting report from match stats."""
    try:
        import openai
        duration_sec = round(num_frames / fps) if fps else 0
        duration_str = f"{duration_sec // 60}m {duration_sec % 60}s"

        in_b = stats.get("in_bounds_bounces", 0) or 0
        out_b = stats.get("out_bounds_bounces", 0) or 0
        total_b = in_b + out_b
        acc = f"{round(in_b / total_b * 100)}%" if total_b > 0 else "N/A"

        prompt = (
            "You are a Division I tennis performance analyst.\n"
            "The player being analyzed is the player closer to the camera "
            "(bottom side of player movement map, top side of bounce map).\n\n"
            "Write a clear, data-driven match report under 250 words.\n"
            "Use direct second-person language ('You...').\n"
            "Do NOT mention AI, models, or assumptions.\n"
            "Only use the numbers provided.\n\n"
            "Structure your response EXACTLY with these sections:\n"
            "1) Match Snapshot\n"
            "2) Positioning Tendencies\n"
            "3) Error Patterns\n"
            "4) Strengths\n"
            "5) Areas to Improve\n"
            "6) One-Line Coaching Adjustment\n\n"
            "Be specific and quantitative. Avoid generic advice.\n\n"
            "Match Statistics:\n"
            f"- Duration: {duration_str}\n"
            f"- Total shots: {stats.get('shot_count', 0)}\n"
            f"- Rallies: {stats.get('rally_count', 0)}\n"
            f"- Total bounces: {stats.get('bounce_count', 0)} "
            f"(In-bounds: {in_b}, Out-of-bounds: {out_b})\n"
            f"- In-bounds accuracy: {acc}\n"
            f"- Forehands: {stats.get('forehand_count', 0)}\n"
            f"- Backhands: {stats.get('backhand_count', 0)}\n"
            f"- Serves/Smashes: {stats.get('serve_count', 0)}\n"
        )

        if spatial_stats:
            bs = spatial_stats.get("bounces", {})
            ps = spatial_stats.get("player", {})
            prompt += "\nCourt zone analysis (in-bounds bounces only):\n"
            prompt += f"- Near half (player's side): {bs.get('near_pct', 0)}% of bounces\n"
            prompt += f"- Far half (opponent's side): {bs.get('far_pct', 0)}% of bounces\n"
            prompt += f"- Left side: {bs.get('left_pct', 0)}%, Right side: {bs.get('right_pct', 0)}%\n"
            prompt += f"- Deep near (behind service line, near half): {bs.get('deep_near_pct', 0)}% of near bounces\n"
            prompt += f"- Alley bounces (outside singles): {bs.get('alley_pct', 0)}%\n"
            if ps.get("samples", 0) > 0:
                prompt += "\nPlayer positioning (near half):\n"
                prompt += f"- Time on near half: {ps.get('near_pct', 0)}% of sampled frames\n"
                prompt += f"- Left side of court: {ps.get('near_left_pct', 0)}%, Right side: {ps.get('near_right_pct', 0)}%\n"
                prompt += f"- Deep (behind service line): {ps.get('near_deep_pct', 0)}%, Forward: {ps.get('near_forward_pct', 0)}%\n"

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[SCOUTING] Report generation failed: {e}")
        return None


def run_pipeline(video_path: str, match_id: str, local_mode: bool = False, config: PipelineConfig = None):
    """
    Main pipeline entry point (Modal-compatible).
    """
    print("[PIPELINE v2 — stats-enabled] run_pipeline called")
    try:
        if config is None:
            config = PipelineConfig()

        device = config.device

        supabase = None
        if not local_mode:
            supabase = get_supabase()
            supabase.table("matches").update({
                "status": "processing"
            }).eq("id", match_id).execute()

        def update_progress(value: float):
            if supabase:
                supabase.table("matches").update({
                    "progress": round(float(value), 3)
                }).eq("id", match_id).execute()
            else:
                print(f"Progress: {round(float(value) * 100, 1)}%")

        update_progress(0.01)

        # ---------- Initialize models ----------
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

        print("ball detection")
        ball_detector = BallDetector(
            path_model=os.path.join(WEIGHTS_DIR, "tracknet_weights.pt"),
            device=device,
        )

        print("court detection")
        court_detector = CourtLineDetector(
            model_path=os.path.join(WEIGHTS_DIR, "keypoints_model.pth"),
            device=device,
        )

        print("player tracking")
        player_model_path = os.path.join(WEIGHTS_DIR, config.player_model)
        if not os.path.exists(player_model_path):
            # Not in weights dir — pass name directly so ultralytics auto-downloads
            player_model_path = config.player_model
        player_tracker = PlayerTracker(model_path=player_model_path, device=device, imgsz=config.player_imgsz)

        print("bounce detection")
        bounce_detector = BounceDetector(
            os.path.join(WEIGHTS_DIR, "bounce_detection_weights.cbm")
        )

        # Legacy stroke model (kept for fallback)
        legacy_stroke_path = os.path.join(WEIGHTS_DIR, "stroke_classifier_weights.pth")
        stroke_model = None
        if os.path.exists(legacy_stroke_path):
            try:
                stroke_model = ActionRecognition(model_saved_state=legacy_stroke_path)
                print("stroke recognition (legacy LSTM)")
            except Exception as e:
                print(f"Legacy stroke model load failed: {e}")

        # Pose-based TCN stroke classifier
        tcn_weights_path = None
        if config.stroke_classifier_weights_tcn:
            tcn_weights_path = os.path.join(WEIGHTS_DIR, config.stroke_classifier_weights_tcn)
        pose_stroke_classifier = PoseStrokeClassifier(
            weights_path=tcn_weights_path,
            device=device,
        )
        swing_detector = SwingDetector(
            velocity_threshold=config.swing_velocity_threshold,
            ball_proximity=config.swing_ball_proximity,
        )

        homography_estimator = HomographyEstimator()
        court_ref = CourtReference()

        # ---------- Ensure target resolution ----------
        if config.enforce_720p:
            path_intermediate_video = str(Path(video_path).with_name("intermediate_720p.mp4"))
            video_path = ensure_720p(
                video_path,
                path_intermediate_video,
                target_width=config.target_width,
                target_height=config.target_height
            )
        update_progress(0.05)

        # ---------- Court calibration (skip per-frame detection if available) ----------
        calibrated_H_ref = None
        calibrated_keypoints = None
        if config.calibration_path and config.camera_id:
            cal_H_ref, _cal_H_frame, cal_kps = load_calibration(config.calibration_path, config.camera_id)
            if cal_H_ref is not None:
                calibrated_H_ref = cal_H_ref
                calibrated_keypoints = cal_kps
                print(f"[Pipeline] Using saved calibration for camera '{config.camera_id}' — skipping per-frame court detection")

        # ---------- Pass 1: Ball + Player tracking + pose extraction ----------
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ball_track = []
        player_detections = []
        pose_keypoints_per_frame = []  # list of {track_id: np.ndarray(17,3)}

        for i in tqdm(range(total_frames), desc="Pass 1: Ball + Player tracking"):
            ret, frame = cap.read()
            if not ret:
                break
            ball_track.append(ball_detector.infer_single(frame))
            player_dict, kps_dict = player_tracker.detect_frame(frame)
            player_detections.append(player_dict)
            pose_keypoints_per_frame.append(kps_dict)

            if i % max(1, total_frames // config.progress_update_frequency) == 0:
                update_progress(0.05 + 0.4 * (i / total_frames))
        # Don't release cap — Pass 2 reuses it by seeking to frame 0

        # Filter players using homography: pick 1 player per side of the net
        H_ref_for_filter = calibrated_H_ref
        if H_ref_for_filter is None:
            # No calibration — estimate from first frame
            cap = cv2.VideoCapture(video_path)
            ret, first_frame = cap.read()
            cap.release()
            kps_first = court_detector.infer_single(first_frame)
            if kps_first is not None:
                H_ref_for_filter, _ = homography_estimator.estimate(kps_first)

        if H_ref_for_filter is not None and len(player_detections) > 0:
            player_detections, pose_keypoints_per_frame = player_tracker.choose_and_filter_players(
                H_ref_for_filter,
                player_detections,
                pose_keypoints_per_frame,
            )

        # ---------- Bounce + shot detection ----------
        cap_meta = cv2.VideoCapture(video_path)
        fps = cap_meta.get(cv2.CAP_PROP_FPS)
        cap_meta.release()

        bounces_all = set()
        if bounce_detector.model is not None:
            # Feed screen-space pixel coordinates — the CatBoost model was trained on these.
            # Court-space projection changes the y_diff scale (~10x) and breaks detection.
            x_ball_raw = [float(p[0]) if p and p[0] is not None else None for p in ball_track]
            y_ball_raw = [float(p[1]) if p and p[1] is not None else None for p in ball_track]

            non_none_raw = sum(1 for x in x_ball_raw if x is not None)
            print(f"[Bounce] total frames={len(x_ball_raw)}, non-None ball positions={non_none_raw}, fps={fps:.2f}")

            # Sub-sample to ~30fps if video is higher frame rate
            TARGET_FPS = 30.0
            if fps > TARGET_FPS * 1.1:
                step = fps / TARGET_FPS
                sample_indices = [int(round(i * step)) for i in range(int(len(ball_track) / step))]
                sample_indices = [min(i, len(ball_track) - 1) for i in sample_indices]
                x_sub = [x_ball_raw[i] for i in sample_indices]
                y_sub = [y_ball_raw[i] for i in sample_indices]
                non_none_sub = sum(1 for x in x_sub if x is not None)
                print(f"[Bounce] subsampling: step={step:.2f}, sub frames={len(x_sub)}, non-None sub={non_none_sub}")
                bounces_sub = bounce_detector.predict(x_sub, y_sub, smooth=True)
                bounces_all = {sample_indices[b] for b in bounces_sub if b < len(sample_indices)}
            else:
                bounces_all = bounce_detector.predict(x_ball_raw, y_ball_raw, smooth=True)
            print(f"[Bounce] detected {len(bounces_all)} bounces (screen-space coords)")

        shot_frames = detect_shot_frames(ball_track)
        shot_frames_set = set(shot_frames)
        rally_count = calculate_rally_count(shot_frames, fps=fps)

        # ---------- Pose-based swing detection + TCN stroke classification ----------
        # New 4-class stroke counts (replaces legacy 3-class counts when TCN runs)
        pose_stroke_counts = {"Forehand": 0, "Backhand": 0, "Serve/Overhead": 0, "Slice": 0}
        frame_stroke_labels: dict[int, dict[int, str]] = {}
        swing_events = []
        if config.enable_stroke_recognition:
            print("Detecting swings from pose keypoints...")
            swing_events = swing_detector.detect(
                pose_keypoints_per_frame=pose_keypoints_per_frame,
                ball_track=ball_track,
                player_detections=player_detections,
            )
            print(f"  {len(swing_events)} swing events detected")
            # frame_stroke_labels: {frame_idx: {track_id: label}}
            # Populated for the 30 frames after each swing peak so Pass 2 can draw it.
            frame_stroke_labels: dict[int, dict[int, str]] = {}
            for event in swing_events:
                seq = extract_pose_sequence(
                    pose_keypoints_per_frame,
                    track_id=event["track_id"],
                    window_start=event["window_start"],
                    window_end=event["window_end"],
                )
                _probs, label = pose_stroke_classifier.predict(seq)
                pose_stroke_counts[label] = pose_stroke_counts.get(label, 0) + 1
                # Show label from peak frame for 30 frames
                display_end = min(total_frames - 1, event["peak_frame"] + 30)
                for f in range(event["peak_frame"], display_end + 1):
                    frame_stroke_labels.setdefault(f, {})[event["track_id"]] = label

        update_progress(0.5)

        # ---------- Pass 2: Drawing ----------
        # Reuse same VideoCapture — seek back to start instead of re-opening
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        local_output_path = f"/tmp/{match_id}_processed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(local_output_path, fourcc, fps, (width, height))

        frame_idx = 0
        last_kps = None
        last_H_ref = None
        homography_matrices = [None] * total_frames

        # Legacy stroke recognition state (for when pose-based is not used)
        stroke_counts = {"Forehand": 0, "Backhand": 0, "Service/Smash": 0}

        # Pre-fill homography and keypoints from calibration if available
        if calibrated_H_ref is not None:
            homography_matrices = [calibrated_H_ref] * total_frames
            last_H_ref = calibrated_H_ref
            last_kps = calibrated_keypoints  # use saved keypoints to draw court lines
            print(f"[Pipeline] Homography matrices pre-filled from calibration")

        # Pre-compute static minimap background once — court reference never changes
        _minimap_raw = court_ref.build_court_reference()
        _minimap_raw = cv2.dilate(_minimap_raw, np.ones((10, 10), np.uint8))
        minimap_bg = np.full((*_minimap_raw.shape, 3), (52, 36, 18), dtype=np.uint8)
        minimap_bg[_minimap_raw.astype(bool)] = (235, 235, 235)

        for i in tqdm(range(total_frames), desc="Pass 2: Drawing"):
            ret, frame = cap.read()
            if not ret:
                break

            # Court detection: skip entirely if calibration is loaded
            if calibrated_H_ref is not None:
                kps = last_kps  # may be None — draw functions handle this gracefully
            else:
                process_this_frame = frame_idx % config.court_detection_interval == 0
                if process_this_frame:
                    kps = court_detector.infer_single(frame)
                    if kps is not None:
                        last_kps = kps
                else:
                    kps = last_kps

            output = frame.copy()
            output = draw_ball_trace(output, ball_track, frame_idx,
                                     trace_length=config.trace_length,
                                     base_color=config.ball_trace_color)
            output = draw_court_keypoints_and_lines(output, kps)

            if frame_idx < len(player_detections):
                output = draw_player_bboxes(output, player_detections[frame_idx],
                                            color=config.player_bbox_color)
                output = draw_stroke_labels(output, player_detections[frame_idx],
                                            frame_stroke_labels, frame_idx)

            minimap = minimap_bg.copy()

            if calibrated_H_ref is not None:
                H_ref = calibrated_H_ref
            elif kps is not None:
                H_ref, _ = homography_estimator.estimate(kps)
                if H_ref is not None:
                    last_H_ref = H_ref
                else:
                    H_ref = last_H_ref
            else:
                H_ref = last_H_ref

            homography_matrices[frame_idx] = H_ref

            minimap = draw_minimap_ball_and_bounces(
                minimap=minimap, homography_inv=H_ref, ball_track=ball_track,
                frame_idx=frame_idx, bounces=bounces_all,
                trace_length=config.trace_length,
                frame_stroke_labels=frame_stroke_labels,
            )
            if frame_idx < len(player_detections):
                minimap = draw_minimap_players(
                    minimap=minimap, homography_inv=H_ref,
                    player_dict=player_detections[frame_idx],
                    color=config.player_bbox_color,
                )

            minimap = cv2.resize(minimap, (config.minimap_width, config.minimap_height))
            # Thin border
            cv2.rectangle(minimap, (1, 1), (config.minimap_width - 2, config.minimap_height - 2),
                          (180, 180, 180), 1, cv2.LINE_AA)
            # Alpha-blend into frame (80% minimap, 20% video)
            roi = output[0:config.minimap_height, 0:config.minimap_width]
            cv2.addWeighted(minimap, 0.82, roi, 0.18, 0, roi)
            output[0:config.minimap_height, 0:config.minimap_width] = roi

            out.write(output)

            # Legacy stroke recognition (runs only when pose-based swing detection
            # produced no events AND the legacy model is loaded).
            if config.enable_stroke_recognition and stroke_model is not None and len(swing_events) == 0:
                pd_at_frame = player_detections[frame_idx] if frame_idx < len(player_detections) else {}
                if pd_at_frame:
                    player_box = list(pd_at_frame.values())[0]
                    try:
                        probs, stroke_label = stroke_model.predict_stroke(frame, player_box)
                        if frame_idx in shot_frames_set and probs is not None:
                            stroke_counts[stroke_label] = stroke_counts.get(stroke_label, 0) + 1
                    except Exception as e:
                        print(f"Stroke prediction error at frame {frame_idx}: {e}")

            frame_idx += 1
            if i % max(1, total_frames // config.progress_update_frequency) == 0:
                update_progress(0.50 + 0.45 * (i / total_frames))

        update_progress(0.95)
        cap.release()
        out.release()

        # ---------- In-bounds / out-of-bounds bounce classification ----------
        in_bounds_bounces, out_bounds_bounces, in_bounds_set = count_in_out_bounces(
            bounces_all, ball_track, homography_matrices, court_ref
        )

        # ---------- Spatial zone stats (for scouting report) ----------
        spatial_stats = compute_spatial_stats(
            bounces=bounces_all,
            ball_track=ball_track,
            homography_matrices=homography_matrices,
            player_detections=player_detections,
            court_ref=court_ref,
        )

        # Merge stroke counts: prefer pose-based counts when swing events were detected
        if len(swing_events) > 0:
            final_forehand = pose_stroke_counts.get("Forehand", 0)
            final_backhand = pose_stroke_counts.get("Backhand", 0)
            final_serve    = pose_stroke_counts.get("Serve/Overhead", 0)
        else:
            # Fall back to legacy LSTM counts
            final_forehand = stroke_counts.get("Forehand", 0)
            final_backhand = stroke_counts.get("Backhand", 0)
            final_serve    = stroke_counts.get("Service/Smash", 0)

        # ---------- AI Scouting Report ----------
        scouting_report = generate_scouting_report(
            stats={
                "shot_count": len(shot_frames),
                "rally_count": rally_count,
                "bounce_count": len(bounces_all),
                "in_bounds_bounces": in_bounds_bounces,
                "out_bounds_bounces": out_bounds_bounces,
                "forehand_count": final_forehand,
                "backhand_count": final_backhand,
                "serve_count": final_serve,
            },
            fps=fps,
            num_frames=total_frames,
            spatial_stats=spatial_stats,
        )

        # ---------- Heatmap generation ----------
        bounce_heatmap_path = None
        player_heatmap_path = None
        player_shot_map_path = None

        if config.generate_heatmaps:
            print("Generating heatmaps...")
            heatmap_dir = f"/tmp/{match_id}_heatmaps"
            local_bounce_path = os.path.join(heatmap_dir, "bounce_heatmap.png")
            local_player_path = os.path.join(heatmap_dir, "player_heatmap.png")
            local_shot_map_path = os.path.join(heatmap_dir, "player_shot_map.png")

            generate_minimap_heatmaps(
                homography_matrices=homography_matrices,
                ball_track=ball_track,
                bounces=bounces_all,
                player_detections=player_detections,
                output_bounce_heatmap=local_bounce_path,
                output_player_heatmap=local_player_path,
                ball_shot_frames=shot_frames,
                frame_stroke_labels=frame_stroke_labels,
                in_bounds_bounces=in_bounds_set,
            )

            generate_player_shot_dot_map(
                homography_matrices=homography_matrices,
                player_detections=player_detections,
                shot_frames=shot_frames,
                frame_stroke_labels=frame_stroke_labels,
                output_path=local_shot_map_path,
            )

            if not local_mode:
                if os.path.exists(local_bounce_path):
                    bounce_heatmap_path = upload_heatmap_png(local_bounce_path, match_id, "bounce_heatmap.png")
                if os.path.exists(local_player_path):
                    player_heatmap_path = upload_heatmap_png(local_player_path, match_id, "player_heatmap.png")
                if os.path.exists(local_shot_map_path):
                    player_shot_map_path = upload_heatmap_png(local_shot_map_path, match_id, "player_shot_map.png")
            else:
                print(f"   Bounce heatmap: {local_bounce_path}")
                print(f"   Player heatmap: {local_player_path}")
                print(f"   Player shot map: {local_shot_map_path}")

        # ---------- Make streamable + upload ----------
        streamable_output = make_streamable_mp4(local_output_path)
        results_path = None

        if not local_mode:
            results_path = upload_processed_video(
                local_path=streamable_output,
                match_id=match_id,
            )

            update_data = {
                "status": "done",
                "progress": 1.0,
                "results_path": results_path,
                "fps": fps,
                "num_frames": total_frames,
                # Tennis stats
                "bounce_count": len(bounces_all),
                "shot_count": len(shot_frames),
                "rally_count": rally_count,
                "in_bounds_bounces": in_bounds_bounces,
                "out_bounds_bounces": out_bounds_bounces,
                "forehand_count": final_forehand,
                "backhand_count": final_backhand,
                "serve_count": final_serve,
                "scouting_report": scouting_report,
            }
            if bounce_heatmap_path:
                update_data["bounce_heatmap_path"] = bounce_heatmap_path
            if player_heatmap_path:
                update_data["player_heatmap_path"] = player_heatmap_path
            # player_shot_map_path requires the column to exist in Supabase;
            # skip gracefully if not yet migrated
            if player_shot_map_path:
                try:
                    supabase.table("matches").update({"player_shot_map_path": player_shot_map_path}).eq("id", match_id).execute()
                except Exception as e:
                    print(f"[STATS] Skipping player_shot_map_path (column may not exist yet): {e}")

            print(f"\n[STATS] Saving to DB — match_id={match_id}")
            print(f"[STATS]   bounces={len(bounces_all)}  shots={len(shot_frames)}  rallies={rally_count}")
            print(f"[STATS]   in_bounds={in_bounds_bounces}  out_bounds={out_bounds_bounces}")
            print(f"[STATS]   FH={final_forehand}  BH={final_backhand}  Srv={final_serve}  (pose swings={len(swing_events)})")

            supabase.table("matches").update(update_data).eq("id", match_id).execute()
            print(f"[STATS] DB update successful — status=done")
        else:
            print(f"\n✅ Local processing complete!")
            print(f"   Output: {streamable_output}")
            print(f"   FPS: {fps}, Frames: {total_frames}")
            print(f"   Bounces: {len(bounces_all)} ({in_bounds_bounces} in / {out_bounds_bounces} out)")
            print(f"   Shots: {len(shot_frames)}, Rallies: {rally_count}")
            print(f"   Strokes — FH: {final_forehand}, BH: {final_backhand}, Serve: {final_serve}")
            if len(swing_events) > 0:
                print(f"   (Pose-based: {len(swing_events)} swing events detected)")

        return {
            "status": "done",
            "results_path": results_path,
            "output_file": streamable_output,
            "meta": {
                "fps": fps,
                "num_frames": total_frames,
                "bounce_count": len(bounces_all),
                "shot_count": len(shot_frames),
                "rally_count": rally_count,
            },
        }
    except Exception as e:
        if supabase:
            import logging
            logging.exception("Pipeline failed for match %s", match_id)
            supabase.table("matches").update({
                "status": "failed",
                "error": "Processing failed. Please try again.",
                "progress": 0
            }).eq("id", match_id).execute()
        raise


if __name__ == "__main__":
    import argparse
    import uuid
    import shutil

    parser = argparse.ArgumentParser(description="Run tennis analysis pipeline locally")
    parser.add_argument("--video", type=str, required=True, help="Path to input .mp4 video")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to output video")
    args = parser.parse_args()

    test_match_id = str(uuid.uuid4())
    print(f"🎾 Running tennis analysis pipeline locally")
    print(f"   Input: {args.video}")
    print(f"   Output: {args.output}")
    print(f"   Test match_id: {test_match_id}\n")

    try:
        result = run_pipeline(
            video_path=args.video,
            match_id=test_match_id,
            local_mode=True,
        )
        if result.get("output_file") and os.path.exists(result["output_file"]):
            shutil.copy(result["output_file"], args.output)
            print(f"\n✅ Pipeline complete! Output saved to: {args.output}")
        else:
            print(f"\n⚠️  Warning: Output file not found")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
