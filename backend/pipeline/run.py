# pipeline/run.py
import time
import cv2
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

from backend.models import BallDetector, CourtLineDetector, PlayerTracker, BounceDetector, ActionRecognition
from backend.vision import HomographyEstimator, CourtReference, draw_ball_trace, draw_court_keypoints_and_lines, draw_minimap_ball_and_bounces, draw_minimap_players, draw_player_bboxes
from backend.vision.heatmaps import generate_minimap_heatmaps
from backend.vision.postprocess import detect_shot_frames

from backend.pipeline.storage import upload_processed_video, upload_heatmap_png, get_supabase, make_streamable_mp4
from backend.pipeline.config import PipelineConfig


def load_models() -> None:
  pass


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
        else:
            out_bounds += 1

    return in_bounds, out_bounds


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
    svc_top_y   = court_ref.top_inner_line[0][1]       # 1110  (far service line)

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
    p_near_left = p_near_right = p_deep = p_forward = p_far = p_samples = 0

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
            else:
                p_far += 1

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

        lines = [
            "You are an expert tennis coach. Based on the following match statistics, "
            "write a concise scouting report in markdown with three sections: "
            "**Performance Summary**, **Strengths**, and **Areas to Improve**. "
            "Be specific, actionable, and keep the total under 350 words.\n",
            "Match statistics:",
            f"- Duration: {duration_str}",
            f"- Total shots detected: {stats.get('shot_count', 0)}",
            f"- Rallies: {stats.get('rally_count', 0)}",
            f"- Total bounces: {stats.get('bounce_count', 0)} (in-bounds: {in_b}, out-of-bounds: {out_b})",
            f"- In-bounds accuracy: {acc}",
            f"- Stroke breakdown — Forehand: {stats.get('forehand_count', 0)}, "
            f"Backhand: {stats.get('backhand_count', 0)}, "
            f"Serve/Smash: {stats.get('serve_count', 0)}",
        ]

        if spatial_stats:
            bs = spatial_stats.get("bounces", {})
            ps = spatial_stats.get("player", {})
            lines.append("\nCourt zone analysis (in-bounds bounces only):")
            lines.append(f"- Near half (player's side): {bs.get('near_pct', 0)}% of bounces")
            lines.append(f"- Far half (opponent's side): {bs.get('far_pct', 0)}% of bounces")
            lines.append(f"- Left side: {bs.get('left_pct', 0)}%, Right side: {bs.get('right_pct', 0)}%")
            lines.append(f"- Deep near (behind service line, near half): {bs.get('deep_near_pct', 0)}% of near bounces")
            lines.append(f"- Alley bounces (outside singles): {bs.get('alley_pct', 0)}%")
            if ps.get("samples", 0) > 0:
                lines.append("\nPlayer positioning (near half):")
                lines.append(f"- Time on near half: {ps.get('near_pct', 0)}% of sampled frames")
                lines.append(f"- Left side of court: {ps.get('near_left_pct', 0)}%, Right side: {ps.get('near_right_pct', 0)}%")
                lines.append(f"- Deep (behind service line): {ps.get('near_deep_pct', 0)}%, Forward: {ps.get('near_forward_pct', 0)}%")

        prompt = "\n".join(lines)

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
        player_tracker = PlayerTracker(
            model_path=os.path.join(WEIGHTS_DIR, config.player_model),
            device=device,
        )

        print("bounce detection")
        bounce_detector = BounceDetector(
            os.path.join(WEIGHTS_DIR, "bounce_detection_weights.cbm")
        )

        print("stroke recognition")
        stroke_model = ActionRecognition(
            model_saved_state=os.path.join(WEIGHTS_DIR, "stroke_classifier_weights.pth")
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

        # ---------- Pass 1: Ball + Player tracking ----------
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ball_track = []
        player_detections = []

        for i in tqdm(range(total_frames), desc="Pass 1: Ball + Player tracking"):
            ret, frame = cap.read()
            if not ret:
                break
            ball_track.append(ball_detector.infer_single(frame))
            player_dict = player_tracker.detect_frame(frame)
            player_detections.append(player_dict)

            if i % max(1, total_frames // config.progress_update_frequency) == 0:
                update_progress(0.05 + 0.4 * (i / total_frames))
        cap.release()

        # Filter players to keep only the 2 main players closest to court
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()

        court_keypoints = court_detector.infer_single(first_frame)
        if court_keypoints is not None and len(player_detections) > 0:
            player_detections = player_tracker.choose_and_filter_players(
                court_keypoints,
                player_detections
            )
            print(f"Filtered to 2 main players based on court proximity")

        # ---------- Bounce + shot detection ----------
        bounces_all = set()
        if bounce_detector.model is not None:
            x_ball = [p[0] if p is not None else None for p in ball_track]
            y_ball = [p[1] if p is not None else None for p in ball_track]
            bounces_all = bounce_detector.predict(x_ball, y_ball, smooth=True)

        shot_frames = detect_shot_frames(ball_track)
        shot_frames_set = set(shot_frames)

        # Rally count (derived from shot frames)
        cap_meta = cv2.VideoCapture(video_path)
        fps = cap_meta.get(cv2.CAP_PROP_FPS)
        cap_meta.release()
        rally_count = calculate_rally_count(shot_frames, fps=fps)

        update_progress(0.5)

        # ---------- Pass 2: Drawing + stroke recognition ----------
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        local_output_path = f"/tmp/{match_id}_processed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(local_output_path, fourcc, fps, (width, height))

        frame_idx = 0
        last_H = None
        last_kps = None
        last_H_ref = None
        homography_matrices = [None] * total_frames

        # Stroke recognition state (rolling window via predict_stroke)
        stroke_counts = {"Forehand": 0, "Backhand": 0, "Service/Smash": 0}

        for i in tqdm(range(total_frames), desc="Pass 2: Drawing"):
            ret, frame = cap.read()
            if not ret:
                break

            process_this_frame = frame_idx % config.court_detection_interval == 0

            if process_this_frame:
                kps = court_detector.infer_single(frame)
                if kps is not None:
                    H = homography_estimator.get_trans_matrix(kps)
                    if H is not None:
                        last_H = H
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

            minimap = court_ref.build_court_reference()
            minimap = cv2.dilate(minimap, np.ones((10, 10), np.uint8))
            minimap = (np.stack([minimap] * 3, axis=2) * 255).astype(np.uint8)

            if kps is not None:
                H_ref, _ = homography_estimator.estimate(kps)
                if H_ref is not None:
                    last_H_ref = H_ref
            else:
                H_ref = last_H_ref

            homography_matrices[frame_idx] = H_ref

            minimap = draw_minimap_ball_and_bounces(
                minimap=minimap, homography_inv=H_ref, ball_track=ball_track,
                frame_idx=frame_idx, bounces=bounces_all,
                trace_length=config.trace_length,
            )
            if frame_idx < len(player_detections):
                minimap = draw_minimap_players(
                    minimap=minimap, homography_inv=H_ref,
                    player_dict=player_detections[frame_idx],
                    color=config.player_bbox_color,
                )

            minimap = cv2.resize(minimap, (config.minimap_width, config.minimap_height))
            output[0:config.minimap_height, 0:config.minimap_width] = minimap

            out.write(output)

            # Stroke recognition: feed every frame to maintain rolling temporal window;
            # record the prediction only at shot frames (when a stroke just occurred).
            if config.enable_stroke_recognition:
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
        in_bounds_bounces, out_bounds_bounces = count_in_out_bounces(
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

        # ---------- AI Scouting Report ----------
        scouting_report = generate_scouting_report(
            stats={
                "shot_count": len(shot_frames),
                "rally_count": rally_count,
                "bounce_count": len(bounces_all),
                "in_bounds_bounces": in_bounds_bounces,
                "out_bounds_bounces": out_bounds_bounces,
                "forehand_count": stroke_counts.get("Forehand", 0),
                "backhand_count": stroke_counts.get("Backhand", 0),
                "serve_count": stroke_counts.get("Service/Smash", 0),
            },
            fps=fps,
            num_frames=total_frames,
            spatial_stats=spatial_stats,
        )

        # ---------- Heatmap generation ----------
        bounce_heatmap_path = None
        player_heatmap_path = None

        if config.generate_heatmaps:
            print("Generating heatmaps...")
            heatmap_dir = f"/tmp/{match_id}_heatmaps"
            local_bounce_path = os.path.join(heatmap_dir, "bounce_heatmap.png")
            local_player_path = os.path.join(heatmap_dir, "player_heatmap.png")

            generate_minimap_heatmaps(
                homography_matrices=homography_matrices,
                ball_track=ball_track,
                bounces=bounces_all,
                player_detections=player_detections,
                output_bounce_heatmap=local_bounce_path,
                output_player_heatmap=local_player_path,
                ball_shot_frames=shot_frames,
            )

            if not local_mode:
                if os.path.exists(local_bounce_path):
                    bounce_heatmap_path = upload_heatmap_png(local_bounce_path, match_id, "bounce_heatmap.png")
                if os.path.exists(local_player_path):
                    player_heatmap_path = upload_heatmap_png(local_player_path, match_id, "player_heatmap.png")
            else:
                print(f"   Bounce heatmap: {local_bounce_path}")
                print(f"   Player heatmap: {local_player_path}")

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
                "forehand_count": stroke_counts.get("Forehand", 0),
                "backhand_count": stroke_counts.get("Backhand", 0),
                "serve_count": stroke_counts.get("Service/Smash", 0),
                "scouting_report": scouting_report,
            }
            if bounce_heatmap_path:
                update_data["bounce_heatmap_path"] = bounce_heatmap_path
            if player_heatmap_path:
                update_data["player_heatmap_path"] = player_heatmap_path

            print(f"\n[STATS] Saving to DB — match_id={match_id}")
            print(f"[STATS]   bounces={len(bounces_all)}  shots={len(shot_frames)}  rallies={rally_count}")
            print(f"[STATS]   in_bounds={in_bounds_bounces}  out_bounds={out_bounds_bounces}")
            print(f"[STATS]   FH={stroke_counts.get('Forehand',0)}  BH={stroke_counts.get('Backhand',0)}  Srv={stroke_counts.get('Service/Smash',0)}")

            supabase.table("matches").update(update_data).eq("id", match_id).execute()
            print(f"[STATS] DB update successful — status=done")
        else:
            print(f"\n✅ Local processing complete!")
            print(f"   Output: {streamable_output}")
            print(f"   FPS: {fps}, Frames: {total_frames}")
            print(f"   Bounces: {len(bounces_all)} ({in_bounds_bounces} in / {out_bounds_bounces} out)")
            print(f"   Shots: {len(shot_frames)}, Rallies: {rally_count}")
            print(f"   Strokes — FH: {stroke_counts.get('Forehand',0)}, "
                  f"BH: {stroke_counts.get('Backhand',0)}, "
                  f"Serve: {stroke_counts.get('Service/Smash',0)}")

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
            supabase.table("matches").update({
                "status": "failed",
                "error": str(e),
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
