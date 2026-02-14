# Pipeline Config class to replace colab global flags
from dataclasses import dataclass
from typing import Tuple
import torch


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
    player_model: str = 'yolov8x.pt'  # YOLOv8 model for player detection (downloads to weights/ on first run)

    # ========== Detection Settings ==========
    # Court detection runs every Nth frame to save compute
    court_detection_interval: int = 5

    # Ball detection runs every frame (critical for tracking)
    ball_detection_interval: int = 1

    # ========== Feature Toggles ==========
    detect_bounces: bool = True
    track_players: bool = True
    enable_drawing: bool = True
    generate_heatmaps: bool = False  # Not implemented yet
    enable_pose_detection: bool = False  # Not implemented yet
    enable_stroke_recognition: bool = False  # Not used in pipeline yet

    # ========== Visualization ==========
    draw_ball_trace: bool = True
    draw_court_lines: bool = True
    draw_player_bboxes: bool = True
    draw_minimap: bool = True

    # Ball trace settings
    trace_length: int = 7  # Number of frames to show in ball trace
    ball_trace_color: Tuple[int, int, int] = (255, 255, 0)  # BGR: Cyan/Yellow

    # Minimap settings
    minimap_width: int = 166
    minimap_height: int = 350
    minimap_position: Tuple[int, int] = (0, 0)  # Top-left corner

    # Player bbox color
    player_bbox_color: Tuple[int, int, int] = (0, 0, 255)  # BGR: Red

    # ========== Progress Tracking ==========
    # Number of progress updates to send (more = finer granularity)
    progress_update_frequency: int = 20