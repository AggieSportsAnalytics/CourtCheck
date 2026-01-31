"""
Bounce Detection Module
Detects when the tennis ball bounces on the court
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost not available. Install with: pip install catboost")

logger = logging.getLogger(__name__)


class BounceDetector:
    """
    Detector for tennis ball bounces.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.7
    ):
        """
        Initialize bounce detector.
        
        Args:
            model_path: Path to CatBoost model (.cbm file)
            threshold: Confidence threshold for bounce detection
        """
        self.threshold = threshold
        self.model = None
        
        if model_path and CATBOOST_AVAILABLE:
            self._load_model(model_path)
        elif not CATBOOST_AVAILABLE:
            logger.warning("CatBoost not available. Bounce detection will use heuristics.")
    
    def _load_model(self, model_path: str):
        """Load the CatBoost bounce detection model."""
        try:
            self.model = CatBoostClassifier()
            self.model.load_model(model_path)
            logger.info(f"Bounce detection model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading bounce detection model: {e}")
            self.model = None
    
    def extract_features(
        self,
        ball_positions: List[Tuple[Optional[float], Optional[float]]],
        frame_idx: int,
        window_size: int = 5
    ) -> np.ndarray:
        """
        Extract features for bounce detection.
        
        Args:
            ball_positions: List of (x, y) ball positions
            frame_idx: Current frame index
            window_size: Number of frames to consider
            
        Returns:
            Feature vector
        """
        features = []
        
        # Get window of positions
        start_idx = max(0, frame_idx - window_size)
        end_idx = min(len(ball_positions), frame_idx + window_size + 1)
        window = ball_positions[start_idx:end_idx]
        
        # Extract features
        valid_positions = [(x, y) for x, y in window if x is not None and y is not None]
        
        if len(valid_positions) < 2:
            # Not enough data
            return np.zeros(10)
        
        # Velocity features
        velocities_y = []
        for i in range(len(valid_positions) - 1):
            dy = valid_positions[i+1][1] - valid_positions[i][1]
            velocities_y.append(dy)
        
        # Acceleration features
        if len(velocities_y) >= 2:
            accelerations = []
            for i in range(len(velocities_y) - 1):
                acc = velocities_y[i+1] - velocities_y[i]
                accelerations.append(acc)
        else:
            accelerations = [0]
        
        # Feature vector
        features = [
            valid_positions[-1][1] if valid_positions else 0,  # Current Y position
            np.mean(velocities_y) if velocities_y else 0,      # Mean velocity
            np.std(velocities_y) if velocities_y else 0,       # Std velocity
            max(velocities_y) if velocities_y else 0,          # Max velocity
            min(velocities_y) if velocities_y else 0,          # Min velocity
            np.mean(accelerations) if accelerations else 0,    # Mean acceleration
            np.std(accelerations) if accelerations else 0,     # Std acceleration
            len(valid_positions),                              # Window size
            valid_positions[-1][0] if valid_positions else 0,  # Current X position
            valid_positions[0][1] - valid_positions[-1][1] if len(valid_positions) > 0 else 0  # Y displacement
        ]
        
        return np.array(features)
    
    def detect_bounce_heuristic(
        self,
        ball_positions: List[Tuple[Optional[float], Optional[float]]],
        frame_idx: int
    ) -> bool:
        """
        Heuristic-based bounce detection.
        Looks for sudden change in Y velocity (direction reversal).
        
        Args:
            ball_positions: List of (x, y) ball positions
            frame_idx: Current frame index
            
        Returns:
            True if bounce detected
        """
        if frame_idx < 2 or frame_idx >= len(ball_positions) - 1:
            return False
        
        # Get positions
        prev2 = ball_positions[frame_idx - 2]
        prev1 = ball_positions[frame_idx - 1]
        curr = ball_positions[frame_idx]
        next1 = ball_positions[frame_idx + 1]
        
        # Check if all positions are valid
        if any(pos[0] is None or pos[1] is None for pos in [prev2, prev1, curr, next1]):
            return False
        
        # Calculate velocities
        vel_before = curr[1] - prev1[1]
        vel_after = next1[1] - curr[1]
        
        # Detect bounce: velocity direction change and ball moving downward before
        if vel_before > 2 and vel_after < -2:  # Ball was moving down, now moving up
            return True
        
        return False
    
    def detect_bounces(
        self,
        ball_positions: List[Tuple[Optional[float], Optional[float]]],
        use_model: bool = True
    ) -> List[int]:
        """
        Detect all bounces in a sequence of ball positions.
        
        Args:
            ball_positions: List of (x, y) ball positions for each frame
            use_model: Use ML model if available, otherwise use heuristics
            
        Returns:
            List of frame indices where bounces occur
        """
        bounces = []
        
        for frame_idx in range(len(ball_positions)):
            is_bounce = False
            
            # Always use heuristic detection for now
            # The CatBoost model requires specific features that may not match our extraction
            is_bounce = self.detect_bounce_heuristic(ball_positions, frame_idx)
            
            if is_bounce:
                bounces.append(frame_idx)
        
        # Remove consecutive bounces (keep first of each group)
        filtered_bounces = []
        for i, frame_idx in enumerate(bounces):
            if i == 0 or frame_idx - bounces[i-1] > 5:  # At least 5 frames apart
                filtered_bounces.append(frame_idx)
        
        return filtered_bounces
    
    def annotate_bounces(
        self,
        frame: np.ndarray,
        ball_position: Tuple[float, float],
        is_bounce: bool
    ) -> np.ndarray:
        """
        Annotate frame with bounce indicator.
        
        Args:
            frame: Video frame
            ball_position: (x, y) position of ball
            is_bounce: Whether this is a bounce frame
            
        Returns:
            Annotated frame
        """
        if is_bounce and ball_position[0] is not None:
            x, y = int(ball_position[0]), int(ball_position[1])
            
            # Draw bounce indicator
            cv2.circle(frame, (x, y), 30, (0, 255, 255), 3)  # Yellow circle
            cv2.putText(
                frame,
                "BOUNCE",
                (x + 35, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        
        return frame
