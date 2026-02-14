# pipeline/run.py
import time
import cv2
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

from backend.models import BallDetector, CourtLineDetector, PlayerTracker, BounceDetector, ActionRecognition
from backend.vision import HomographyEstimator, CourtReference, draw_ball_trace, draw_court_keypoints_and_lines, draw_minimap_ball_and_bounces, draw_minimap_players, draw_player_bboxes

from backend.pipeline.storage import upload_processed_video, get_supabase, make_streamable_mp4
from backend.pipeline.config import PipelineConfig

def load_models() -> None:
  pass

def ensure_720p(input_path, intermediate_path, target_width=1280, target_height=720):
    """
    Ensure video is at target resolution

    Args:
        input_path: Input video path
        intermediate_path: Output path for resized video
        target_width: Target width (default 1280)
        target_height: Target height (default 720)

    Returns:
        Path to video (either original or resized)
    """
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

def run_pipeline(video_path: str, match_id: str, local_mode: bool = False, config: PipelineConfig = None):
    """
    Main pipeline entry point (Modal-compatible).

    Args:
        video_path: local path to downloaded input video
        match_id: Supabase matches.id
        local_mode: If True, skip Supabase operations (for local testing)
        config: Pipeline configuration (uses defaults if None)
    """
    try:
        # Initialize config with defaults if not provided
        if config is None:
            config = PipelineConfig()

        device = config.device

        # Supabase setup (skip if local_mode)
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
                # Silently update - tqdm progress bars show visual progress
            else:
                print(f"Progress: {round(float(value) * 100, 1)}%")

        update_progress(0.01)

        # ---------- Initialize backend.models ----------
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

            # Track ball
            ball_track.append(ball_detector.infer_single(frame))

            # Track players
            player_dict = player_tracker.detect_frame(frame)
            player_detections.append(player_dict)

            if i % max(1, total_frames // config.progress_update_frequency) == 0:
                update_progress(0.05 + 0.4 * (i / total_frames))
        cap.release()

        # Filter players to keep only the 2 main players (closest to court)
        # Use first frame's court keypoints
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

        # ---------- Bounce detection ----------
        bounces_all = set()
        if bounce_detector.model is not None:
            x_ball = [p[0] if p is not None else None for p in ball_track]
            y_ball = [p[1] if p is not None else None for p in ball_track]
            bounces_all = bounce_detector.predict(x_ball, y_ball, smooth=True)
        update_progress(0.5)

        # ---------- Pass 2: Drawing (Ball, Court, Players, Minimap) ----------
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

            output = draw_ball_trace(
                output,
                ball_track,
                frame_idx,
                trace_length=config.trace_length,
                base_color=config.ball_trace_color,
            )

            output = draw_court_keypoints_and_lines(output, kps)

            # Draw player bounding boxes
            if frame_idx < len(player_detections):
                output = draw_player_bboxes(
                    output,
                    player_detections[frame_idx],
                    color=config.player_bbox_color
                )

            minimap = court_ref.build_court_reference()
            minimap = cv2.dilate(minimap, np.ones((10, 10), np.uint8))
            minimap = (np.stack([minimap] * 3, axis=2) * 255).astype(np.uint8)

            if kps is not None:
                H_ref, _ = homography_estimator.estimate(kps)
                if H_ref is not None:
                    last_H_ref = H_ref
            else:
                H_ref = last_H_ref

            minimap = draw_minimap_ball_and_bounces(
                minimap=minimap,
                homography_inv=H_ref,
                ball_track=ball_track,
                frame_idx=frame_idx,
                bounces=bounces_all,
                trace_length=config.trace_length,
            )

            if frame_idx < len(player_detections):
                minimap = draw_minimap_players(
                    minimap=minimap,
                    homography_inv=H_ref,
                    player_dict=player_detections[frame_idx],
                    color=config.player_bbox_color,
                )

            minimap = cv2.resize(minimap, (config.minimap_width, config.minimap_height))
            output[0:config.minimap_height, 0:config.minimap_width] = minimap

            out.write(output)
            frame_idx += 1
            if i % max(1, total_frames // config.progress_update_frequency) == 0:
                update_progress(0.50 + 0.45 * (i / total_frames))

        update_progress(0.95)

        cap.release()
        out.release()

        # make browser-streamable mp4
        streamable_output = make_streamable_mp4(local_output_path)

        results_path = None

        # ---------- Upload to Supabase (skip if local_mode) ----------
        if not local_mode:
            results_path = upload_processed_video(
                local_path=streamable_output,
                match_id=match_id,
            )

            # ---------- Final DB update ----------
            supabase.table("matches").update({
                "status": "done",
                "progress": 1.0,
                "results_path": results_path,
                "fps": fps,
                "num_frames": total_frames,
            }).eq("id", match_id).execute()
        else:
            print(f"\n✅ Local processing complete!")
            print(f"   Output: {streamable_output}")
            print(f"   FPS: {fps}, Frames: {total_frames}")

        return {
            "status": "done",
            "results_path": results_path,
            "output_file": streamable_output,  # For local mode
            "meta": {
                "fps": fps,
                "num_frames": total_frames,
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


# running locally on device for testing (without Supabase)
if __name__ == "__main__":
  import argparse
  import uuid
  import shutil

  parser = argparse.ArgumentParser(description="Run tennis analysis pipeline locally")
  parser.add_argument(
      "--video",
      type=str,
      required=True,
      help="Path to input .mp4 video",
  )
  parser.add_argument(
      "--output",
      type=str,
      default="output.mp4",
      help="Path to output video",
  )

  args = parser.parse_args()

  # Generate a test match_id for local testing
  test_match_id = str(uuid.uuid4())

  print(f"🎾 Running tennis analysis pipeline locally")
  print(f"   Input: {args.video}")
  print(f"   Output: {args.output}")
  print(f"   Test match_id: {test_match_id}\n")

  try:
      result = run_pipeline(
          video_path=args.video,
          match_id=test_match_id,
          local_mode=True,  # Skip Supabase operations
      )

      # Copy the processed video to the desired output location
      if result.get("output_file") and os.path.exists(result["output_file"]):
          shutil.copy(result["output_file"], args.output)
          print(f"\n✅ Pipeline complete! Output saved to: {args.output}")
      else:
          print(f"\n⚠️  Warning: Output file not found")

  except Exception as e:
      print(f"\n❌ Pipeline failed: {e}")
      import traceback
      traceback.print_exc()