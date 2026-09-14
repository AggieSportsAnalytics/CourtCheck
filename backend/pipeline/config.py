# Pipeline Config class to replace colab global flags
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Absolute path to calibration_frames/ regardless of working directory
_CALIBRATION_DIR = os.path.join(os.path.dirname(__file__), '..', 'calibration_frames')
_DEFAULT_CALIBRATION_PATH = os.path.normpath(
    os.path.join(_CALIBRATION_DIR, 'court_calibration.json')
)
_DEFAULT_FINGERPRINTS_PATH = os.path.normpath(
    os.path.join(_CALIBRATION_DIR, 'camera_fingerprints.json')
)


def _default_device() -> str:
    # Importing config and explicitly choosing CPU must work without torch.
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PipelineConfig:
    """Central configuration for the tennis analysis pipeline"""

    # ========== Device ==========
    device: str = field(default_factory=_default_device)

    # ========== Video Processing ==========
    enforce_720p: bool = False
    target_width: int = 1920
    target_height: int = 1080

    # ========== Model Settings ==========
    # YOLOv8 pose model for player detection + keypoint extraction.
    player_model: str = 'yolov8m-pose.pt'

    # TrackNet weights for ball detection.
    # tracknet_v2_official.pt = official NCTU tennis weights (36k broadcast frames, 9 court surfaces)
    # tracknet_weights.pt = original bundled weights (provenance unknown)
    ball_model_weights: str = 'tracknet_v2_official.pt'

    # InpaintNet trajectory rectification — fills missed ball detections using learned trajectory physics.
    # Runs after TrackNet inference, before linear interpolation. Bounce detector
    # uses the raw pre-rectification track, so inpainted points can't fabricate bounces.
    enable_inpaint_net: bool = True
    inpaint_net_weights: str = 'inpaint_net_weights.pt'

    # YOLO inference resolution. Must match or exceed input video resolution for small object
    # detection. At imgsz=640 (default), far players at ~25-35px become ~12-18px at inference
    # — below YOLO's detection floor. imgsz=1280 preserves full 720p detail.
    player_imgsz: int = 1920  # match input resolution so far player isn't downscaled below detection floor

    # ========== Detection Settings ==========
    # Number of frames at pipeline startup to try court detection on.
    # The frame with the most detected keypoints is used for the entire video.
    # Ignored when a valid calibration is loaded (calibration_path + camera_id).
    court_detection_startup_frames: int = 10

    # Legacy — retained for backwards compatibility but no longer used by the pipeline.
    court_detection_interval: int = 5

    # YOLO detection confidence threshold. Lower = more detections (catches small
    # far players) at the cost of more false positives filtered downstream.
    player_conf: float = 0.05

    # Run YOLO player detection every Nth frame; interpolate bboxes between.
    # 1 = every frame (original behaviour). 3 = detect every 3rd frame (~3x speedup).
    player_detection_interval: int = 3

    # Run ball detection every Nth frame; use previous result for skipped frames.
    # 1 = every frame. 2 = detect every 2nd frame (~2x throughput on TrackNet).
    ball_detection_interval: int = 1

    # TrackNet inference resolution. Trained at 640x360; 1280x720 doubles the
    # far-ball's in-model pixel size (no retrain). A/B toggle for far-court recall.
    ball_infer_width: int = 640
    ball_infer_height: int = 360

    # Far player detection thresholds (court-space projection, calibration required).
    # x_margin: how far beyond the court sideline (in court units) a foot projection
    # may land and still count as a valid far player. Zoomed cameras project wider.
    far_player_court_x_margin: float = 100
    # Max bbox pixel height for a far player. Near players are excluded by track_id,
    # but this filters non-player detections with oversized bboxes.
    far_player_max_height: int = 400

    # ---- Temporal stabilizer ----
    # Maximum pixel distance the far player center may jump between consecutive accepted
    # frames before the candidate is treated as noise (when miss streak < 3).
    # 350px handles fast lateral movement at 1080p without letting spectator detections
    # "steal" the far player ID.
    far_player_max_jump_px: float = 350
    # Frames to hold the last known far player bbox when no valid detection is found.
    # Covers brief occlusions, bad lighting, or serve motion blur (typically 3-8 frames).
    far_player_hold_frames: int = 8

    # ========== Court Calibration ==========
    # Path to court_calibration.json produced by backend.tools.calibrate_court.
    # When set together with camera_id, per-frame court detection is skipped.
    calibration_path: Optional[str] = _DEFAULT_CALIBRATION_PATH
    camera_fingerprints_path: str = _DEFAULT_FINGERPRINTS_PATH
    camera_id: Optional[str] = None  # An explicit ID always overrides identification.
    camera_auto_identify: bool = True  # Match startup backgrounds to reference frames.
    camera_match_min_ncc: float = 0.75  # Minimum median background correlation.
    camera_match_min_ncc_margin: float = 0.12  # Required lead over the runner-up.
    camera_match_min_points: int = 10  # Overlapping points for the diagnostic cross-check.
    # Sample evenly across the first court_detection_startup_frames * 6 frames.
    camera_identify_frames: int = 5

    # ========== Bounce Detection ==========
    # CatBoost regressor confidence threshold (0-1). Higher = fewer but more certain bounces.
    # 0.20 = original (too permissive). 0.40 = calibrated for broadcast tennis footage.
    # 0.18 = paired with the far-court ROI ball-tracker pass + plausibility gate
    # downstream. The ROI improves far-side trajectory quality so weaker
    # inflections are still real bounces — raising the recall of P1 shots
    # landing on the opponent's far half. False positives get caught by the
    # compute_bounce_positions() plausibility filter and the side-aware shot
    # pairing, so the cost of a lower threshold is bounded.
    bounce_threshold: float = 0.35

    # Minimum frames between two accepted bounces. Enforces ball physics — a ball cannot
    # bounce twice within < 0.5s (15 frames at 30fps). Filters duplicate detections per event.
    bounce_min_gap_frames: int = 10

    # CatBoost bounce weights filename inside backend/weights/. Defaults to the original
    # broadcast-trained weights; flip to "bounce_detection_weights_ucd.cbm" after the
    # UC-Davis retrain ships.
    bounce_model_weights: str = 'bounce_detection_weights_ucd.cbm'

    # ========== Stroke Classifier ==========
    # Path to TCN weights trained on pose keypoints (THETIS dataset).
    # When None, stroke classification uses a simple rule-based heuristic.
    stroke_classifier_weights_tcn: Optional[str] = "stroke_classifier_tcn.pt"

    # Swing trigger thresholds for the pose-based swing detector
    swing_velocity_threshold: float = 15.0   # pixels/frame at wrist
    swing_ball_proximity: float = 300.0      # ball must be within N pixels of player

    # ========== Feature Toggles ==========
    generate_heatmaps: bool = True
    enable_stroke_recognition: bool = True

    # ========== Visualization ==========
    # Ball trace settings — separate lengths for the two surfaces.
    # MAIN VIDEO: short comet so the live ball doesn't smear across the screen.
    # MINIMAP: long comet (~1.5s @ 30fps) so the small minimap actually shows
    # the recent trajectory at a glance, which is the whole point of the
    # minimap. Was unified at 45 frames; that worked for the minimap but made
    # the main video unreadable so we split the param.
    trace_length_main: int = 10
    trace_length_minimap: int = 45
    ball_trace_color: Tuple[int, int, int] = (5, 250, 210)  # BGR: tennis ball neon yellow-green
    # Floor opacity for the oldest dot in the trail. Below this the alpha
    # falloff at trace_length=45 would render the far end of the minimap tail
    # invisible. Bumped 0.3 -> 0.55 so the whole comet stays legible against
    # the dark minimap surface.
    ball_trace_min_alpha: float = 0.55

    # Minimap settings
    minimap_width: int = 166
    minimap_height: int = 350


    # Player bbox color override — leave None to use per-player theme colors
    # (brand court / clay) from backend.vision.drawing._player_color.
    player_bbox_color: Optional[Tuple[int, int, int]] = None

    # ========== Progress Tracking ==========
    # Number of progress updates to send per per-frame loop pass (more = finer
    # granularity in the UI). Each update is a single supabase UPDATE — cheap.
    progress_update_frequency: int = 40