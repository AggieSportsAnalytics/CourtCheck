# Pipeline Config class to replace colab global flags
import os
from dataclasses import dataclass
from typing import Optional, Tuple
import torch

# Absolute path to calibration_frames/ regardless of working directory
_CALIBRATION_DIR = os.path.join(os.path.dirname(__file__), '..', 'calibration_frames')
_DEFAULT_CALIBRATION_PATH = os.path.normpath(
    os.path.join(_CALIBRATION_DIR, 'court_calibration.json')
)


@dataclass
class PipelineConfig:
    """Central configuration for the tennis analysis pipeline"""

    # ========== Device ==========
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== Video Processing ==========
    enforce_720p: bool = True
    target_width: int = 1280
    target_height: int = 720

    # ========== Model Settings ==========
    # YOLOv8 pose model for player detection + keypoint extraction.
    player_model: str = 'yolov8m-pose.pt'

    # YOLO inference resolution. Must match or exceed input video resolution for small object
    # detection. At imgsz=640 (default), far players at ~25-35px become ~12-18px at inference
    # — below YOLO's detection floor. imgsz=1280 preserves full 720p detail.
    player_imgsz: int = 960

    # ========== Detection Settings ==========
    # Number of frames at pipeline startup to try court detection on.
    # The frame with the most detected keypoints is used for the entire video.
    # Ignored when a valid calibration is loaded (calibration_path + camera_id).
    court_detection_startup_frames: int = 10

    # Legacy — retained for backwards compatibility but no longer used by the pipeline.
    court_detection_interval: int = 5

    # Run YOLO player detection every Nth frame; interpolate bboxes between.
    # 1 = every frame (original behaviour). 3 = detect every 3rd frame (~3x speedup).
    player_detection_interval: int = 3

    # Run ball detection every Nth frame; use previous result for skipped frames.
    # 1 = every frame. 2 = detect every 2nd frame (~2x throughput on TrackNet).
    ball_detection_interval: int = 2

    # Far player detection thresholds (court-space projection, calibration required).
    # x_margin: how far beyond the court sideline (in court units) a foot projection
    # may land and still count as a valid far player. Zoomed cameras project wider.
    far_player_court_x_margin: float = 500
    # Max bbox pixel height for a far player. Near players are excluded by track_id,
    # but this filters non-player detections with oversized bboxes.
    far_player_max_height: int = 400

    # ========== Court Calibration ==========
    # Path to court_calibration.json produced by backend.tools.calibrate_court.
    # When set together with camera_id, per-frame court detection is skipped.
    calibration_path: Optional[str] = _DEFAULT_CALIBRATION_PATH
    camera_id: Optional[str] = 'uc_davis_court1_zoomed'

    # ========== Stroke Classifier ==========
    # Path to TCN weights trained on pose keypoints (THETIS dataset).
    # When None, stroke classification uses a simple rule-based heuristic.
    stroke_classifier_weights_tcn: Optional[str] = None

    # Swing trigger thresholds for the pose-based swing detector
    swing_velocity_threshold: float = 15.0   # pixels/frame at wrist
    swing_ball_proximity: float = 300.0      # ball must be within N pixels of player

    # ========== Feature Toggles ==========
    generate_heatmaps: bool = True
    enable_stroke_recognition: bool = True

    # ========== Visualization ==========
    # Ball trace settings
    trace_length: int = 7  # Number of frames to show in ball trace
    ball_trace_color: Tuple[int, int, int] = (255, 255, 0)  # BGR: Cyan/Yellow

    # Minimap settings
    minimap_width: int = 166
    minimap_height: int = 350


    # Player bbox color
    player_bbox_color: Tuple[int, int, int] = (0, 0, 255)  # BGR: Red

    # ========== Progress Tracking ==========
    # Number of progress updates to send (more = finer granularity)
    progress_update_frequency: int = 5