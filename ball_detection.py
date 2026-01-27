"""
Ball Detection Module using TrackNet.

This module provides ball tracking functionality for tennis videos.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from typing import List, Tuple, Optional
import logging
import os

# Import TrackNet model
try:
    from CourtCheck.models.TrackNet.tracknet import BallTrackerNet
    TRACKNET_AVAILABLE = True
except ImportError:
    try:
        from models.TrackNet.tracknet import BallTrackerNet
        TRACKNET_AVAILABLE = True
    except ImportError:
        try:
            # For Modal deployment - copy tracknet.py to root or use inline definition
            import sys
            import os
            # Try to add CourtCheck to path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            courtcheck_path = os.path.join(current_dir, "CourtCheck")
            if os.path.exists(courtcheck_path):
                sys.path.insert(0, current_dir)
                from CourtCheck.models.TrackNet.tracknet import BallTrackerNet
                TRACKNET_AVAILABLE = True
            else:
                TRACKNET_AVAILABLE = False
                logging.warning("TrackNet model not available. Ball detection will not work.")
        except ImportError:
            TRACKNET_AVAILABLE = False
            logging.warning("TrackNet model not available. Ball detection will not work.")

logger = logging.getLogger(__name__)


class BallDetector:
    """
    Ball detector using TrackNet model.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        width: int = 640,
        height: int = 360,
    ):
        """
        Initialize the ball detector.
        
        Args:
            model_path: Path to TrackNet model weights (.pt file)
            device: Device to run inference on ('cuda' or 'cpu')
            width: Input image width for model
            height: Input image height for model
        """
        self.device = device
        self.width = width
        self.height = height
        
        # Frame buffer for temporal context (needs 3 frames)
        self.frame_buffer = deque(maxlen=3)
        self.prev_pred = [None, None]
        
        # Load model
        if model_path:
            self._load_model(model_path)
        else:
            self.model = None
            logger.warning("No model path provided. Ball detection will not work.")
    
    def _load_model(self, model_path: str):
        """Load the TrackNet model."""
        if not TRACKNET_AVAILABLE:
            raise ImportError("TrackNet model not available. Check imports.")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}")
        
        self.model = BallTrackerNet(input_channels=9, out_channels=256)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Ball detection model loaded from {model_path}")
    
    def detect_ball(
        self,
        current_frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
        prev_prev_frame: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Detect ball position in a frame.
        
        Args:
            current_frame: Current frame as numpy array
            prev_frame: Previous frame (optional, uses buffer if None)
            prev_prev_frame: Frame before previous (optional, uses buffer if None)
            
        Returns:
            Tuple of (x, y) coordinates or (None, None) if not detected
        """
        if self.model is None:
            return (None, None)
        
        # Use frame buffer if frames not provided
        if prev_frame is None or prev_prev_frame is None:
            self.frame_buffer.append(current_frame)
            if len(self.frame_buffer) < 3:
                return (None, None)
            prev_prev_frame, prev_frame, current_frame = list(self.frame_buffer)
        
        # Preprocess frames
        img = cv2.resize(current_frame, (self.width, self.height))
        img_prev = cv2.resize(prev_frame, (self.width, self.height))
        img_preprev = cv2.resize(prev_prev_frame, (self.width, self.height))
        
        # Concatenate frames
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)
        
        # Model inference
        with torch.no_grad():
            out = self.model(torch.from_numpy(inp).float().to(self.device))
            output = out.argmax(dim=1).detach().cpu().numpy()
        
        # Post-process
        x_pred, y_pred = self.postprocess(output, self.prev_pred)
        
        # Update previous prediction
        self.prev_pred = [x_pred, y_pred]
        
        return (x_pred, y_pred)
    
    def postprocess(
        self,
        feature_map: np.ndarray,
        prev_pred: List[Optional[float]],
        scale: int = 2,
        max_dist: int = 80,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Post-process model output to get ball coordinates.
        
        Args:
            feature_map: Model output feature map
            prev_pred: Previous prediction [x, y]
            scale: Scale factor for coordinates
            max_dist: Maximum distance from previous prediction
            
        Returns:
            Tuple of (x, y) coordinates or (None, None)
        """
        feature_map = feature_map[0]
        feature_map = (feature_map * 255).astype(np.uint8)
        
        # Find ball location
        ret, heatmap = cv2.threshold(feature_map, 170, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(
            heatmap,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=2,
            minRadius=10,
            maxRadius=25,
        )
        
        if circles is not None:
            x_pred = circles[0][0][0] * scale
            y_pred = circles[0][0][1] * scale
            
            # Check distance from previous prediction
            if prev_pred[0] is not None and prev_pred[1] is not None:
                dist = np.sqrt(
                    (x_pred - prev_pred[0]) ** 2 + (y_pred - prev_pred[1]) ** 2
                )
                if dist > max_dist:
                    return (prev_pred[0], prev_pred[1])
            
            return (x_pred, y_pred)
        
        # Use previous prediction if available
        if prev_pred[0] is not None and prev_pred[1] is not None:
            return (prev_pred[0], prev_pred[1])
        
        return (None, None)
    
    def process_video(self, frames: List[np.ndarray]) -> List[Tuple[Optional[float], Optional[float]]]:
        """
        Process a list of frames and return ball tracks.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of (x, y) tuples for each frame
        """
        ball_track = []
        
        # First two frames have no prediction
        ball_track.append((None, None))
        ball_track.append((None, None))
        
        # Process remaining frames
        for i in range(2, len(frames)):
            x, y = self.detect_ball(
                frames[i],
                frames[i - 1],
                frames[i - 2],
            )
            ball_track.append((x, y))
        
        return ball_track
