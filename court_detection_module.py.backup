"""
Court Detection Module using Detectron2.

This module provides court detection functionality extracted from the
original Colab notebook, refactored for production use.
"""

import cv2
import numpy as np
import torch
from collections import deque
from typing import List, Tuple, Optional, Dict
import os
import logging

try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer, ColorMode
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    logging.warning("Detectron2 not available. Court detection will not work.")

logger = logging.getLogger(__name__)

# Keypoint names for tennis court
KEYPOINT_NAMES = [
    "BTL", "BTLI", "BTRI", "BTR", "BBR", "BBRI", "IBR", "NR", "NM", "ITL",
    "ITM", "ITR", "NL", "BBL", "IBL", "IBM", "BBLI"
]

KEYPOINT_FLIP_MAP = [
    ("BTL", "BTR"), ("BTLI", "BTRI"), ("BBL", "BBR"), ("BBLI", "BBRI"), ("ITL", "ITR"),
    ("ITM", "ITM"), ("NL", "NR"), ("IBL", "IBR"), ("IBM", "IBM"), ("NM", "NM")
]

# Court lines to draw between keypoints
COURT_LINES = [
    ("BTL", "BTLI"), ("BTLI", "BTRI"), ("BTL", "NL"), ("BTLI", "ITL"),
    ("BTRI", "BTR"), ("BTR", "NR"), ("BTRI", "ITR"), ("ITL", "ITM"), ("ITM", "ITR"),
    ("ITL", "IBL"), ("ITM", "NM"), ("ITR", "IBR"), ("NL", "NM"), ("NL", "BBL"),
    ("NM", "IBM"), ("NR", "BBR"), ("NM", "NR"), ("IBL", "IBM"),
    ("IBM", "IBR"), ("IBL", "BBLI"), ("IBR", "BBRI"), ("BBR", "BBRI"),
    ("BBRI", "BBLI"), ("BBL", "BBLI"),
]


class CourtDetectionProcessor:
    """
    Processor for detecting tennis court keypoints using Detectron2.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
        score_threshold: float = 0.5,
    ):
        """
        Initialize the court detection processor.
        
        Args:
            model_path: Path to trained Detectron2 model weights (.pth file)
            config_path: Path to Detectron2 config file (optional, uses default if None)
            device: Device to run inference on ('cuda' or 'cpu')
            score_threshold: Minimum confidence score for detections
        """
        if not DETECTRON2_AVAILABLE:
            raise ImportError("Detectron2 is required for court detection. Install with: pip install 'git+https://github.com/facebookresearch/detectron2.git'")
        
        self.device = device
        self.model_path = model_path
        self.config_path = config_path
        self.score_threshold = score_threshold
        
        # Initialize keypoint history for stabilization
        self.keypoint_history = {name: deque(maxlen=10) for name in KEYPOINT_NAMES}
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the Detectron2 model and predictor."""
        from detectron2 import model_zoo
        
        cfg = get_cfg()
        
        # Load config if provided, otherwise use default
        if self.config_path and os.path.exists(self.config_path):
            cfg.merge_from_file(self.config_path)
        else:
            # Use default COCO-Keypoints config from model zoo
            try:
                cfg.merge_from_file(
                    model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
                )
            except Exception:
                # Fallback: try to load from local path
                default_config = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
                if os.path.exists(default_config):
                    cfg.merge_from_file(default_config)
                else:
                    logger.warning("Could not load config file. Using minimal config.")
                    # Set minimal config manually
                    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
                    cfg.MODEL.KEYPOINT_ON = True
        
        # Set model weights
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model weights not found at: {self.model_path}")
        
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_threshold
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Single class: tennis court
        cfg.MODEL.KEYPOINT_ON = True
        cfg.MODEL.DEVICE = self.device
        
        # Register metadata
        dataset_name = "tennis_court"
        MetadataCatalog.get(dataset_name).keypoint_names = KEYPOINT_NAMES
        MetadataCatalog.get(dataset_name).keypoint_flip_map = KEYPOINT_FLIP_MAP
        MetadataCatalog.get(dataset_name).keypoint_connection_rules = []
        
        # Create predictor
        self.predictor = DefaultPredictor(cfg)
        self.dataset_name = dataset_name
        
        logger.info(f"Court detection model loaded from {self.model_path}")
    
    def detect_keypoints(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect court keypoints in a single image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Array of keypoints with shape (N, 3) where N is number of keypoints,
            and columns are [x, y, confidence]. Returns None if no detection.
        """
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        
        if len(instances) == 0:
            return None
        
        # Get the most confident detection
        max_conf_idx = instances.scores.argmax()
        instance = instances[max_conf_idx]
        
        # Extract keypoints
        keypoints = instance.pred_keypoints.numpy()
        
        return keypoints
    
    def stabilize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Stabilize keypoints by averaging over recent frames.
        
        Args:
            keypoints: Keypoints array with shape (N, 3)
            
        Returns:
            Stabilized keypoints array
        """
        stabilized = []
        for i, keypoint in enumerate(keypoints):
            name = KEYPOINT_NAMES[i]
            self.keypoint_history[name].append(keypoint[:2])
            
            if len(self.keypoint_history[name]) > 1:
                stabilized.append(
                    np.mean(np.array(self.keypoint_history[name]), axis=0)
                )
            else:
                stabilized.append(keypoint[:2])
        
        return np.array(stabilized)
    
    def get_homography_matrix(
        self,
        keypoints: np.ndarray,
        target_width: int = 400,
        target_height: int = 600,
    ) -> Optional[np.ndarray]:
        """
        Compute homography matrix to transform court to 2D top-down view.
        
        Args:
            keypoints: Detected keypoints array
            target_width: Width of target 2D view
            target_height: Height of target 2D view
            
        Returns:
            Homography matrix (3x3) or None if insufficient keypoints
        """
        if keypoints is None or len(keypoints) < 4:
            return None
        
        # Create keypoint dictionary
        kp_dict = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            if i < len(keypoints) and keypoints[i, 2] > 0:  # Check visibility
                kp_dict[name] = keypoints[i, :2]
        
        # Use four corner points for homography
        required_points = ["BTL", "BTR", "BBL", "BBR"]
        if not all(p in kp_dict for p in required_points):
            return None
        
        # Source points (from detected keypoints)
        src_points = np.array([
            kp_dict["BTL"],
            kp_dict["BTR"],
            kp_dict["BBL"],
            kp_dict["BBR"],
        ], dtype=np.float32)
        
        # Destination points (target 2D view)
        margin_x = target_width // 6
        margin_y = target_height // 7
        dst_points = np.array([
            [margin_x, margin_y],  # BTL
            [target_width - margin_x, margin_y],  # BTR
            [margin_x, target_height - margin_y],  # BBL
            [target_width - margin_x, target_height - margin_y],  # BBR
        ], dtype=np.float32)
        
        # Compute homography
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return matrix
    
    def visualize_keypoints(
        self,
        image: np.ndarray,
        keypoints: Optional[np.ndarray] = None,
        draw_lines: bool = True,
    ) -> np.ndarray:
        """
        Visualize detected keypoints on the image.
        
        Args:
            image: Input image
            keypoints: Optional keypoints array (if None, will detect)
            draw_lines: Whether to draw court lines
            
        Returns:
            Image with visualizations
        """
        if keypoints is None:
            keypoints = self.detect_keypoints(image)
        
        if keypoints is None:
            return image.copy()
        
        # Stabilize keypoints
        stabilized = self.stabilize_keypoints(keypoints)
        
        img_copy = image.copy()
        
        # Draw keypoints
        for i, (x, y) in enumerate(stabilized):
            if keypoints[i, 2] > 0:  # Visible keypoint
                cv2.circle(img_copy, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.putText(
                    img_copy,
                    KEYPOINT_NAMES[i],
                    (int(x) + 5, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
        
        # Draw court lines
        if draw_lines:
            for start, end in COURT_LINES:
                try:
                    start_idx = KEYPOINT_NAMES.index(start)
                    end_idx = KEYPOINT_NAMES.index(end)
                    
                    if (start_idx < len(stabilized) and end_idx < len(stabilized) and
                        keypoints[start_idx, 2] > 0 and keypoints[end_idx, 2] > 0):
                        pt1 = tuple(map(int, stabilized[start_idx]))
                        pt2 = tuple(map(int, stabilized[end_idx]))
                        cv2.line(img_copy, pt1, pt2, (0, 255, 0), 2)
                except (ValueError, IndexError):
                    continue
        
        return img_copy
    
    def process_frame(
        self,
        frame: np.ndarray,
    ) -> Dict:
        """
        Process a single frame and return detection results.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary with:
            - keypoints: Detected keypoints
            - homography_matrix: Homography matrix for 2D transformation
            - stabilized_keypoints: Stabilized keypoints
        """
        keypoints = self.detect_keypoints(frame)
        
        result = {
            "keypoints": keypoints,
            "homography_matrix": None,
            "stabilized_keypoints": None,
        }
        
        if keypoints is not None:
            result["stabilized_keypoints"] = self.stabilize_keypoints(keypoints)
            result["homography_matrix"] = self.get_homography_matrix(keypoints)
        
        return result
