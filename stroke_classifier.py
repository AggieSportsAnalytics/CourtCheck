"""
Stroke Classification Module
Classifies tennis strokes (forehand, backhand, serve, etc.)
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class StrokeClassifierLSTM(nn.Module):
    """LSTM model for stroke classification (matches the provided weights)."""
    
    def __init__(self, input_size=2048, hidden_size=90, num_layers=3, num_classes=3):
        super(StrokeClassifierLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.LSTM(x)
        # Take output from last time step
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out


class StrokeClassifier:
    """
    Classifier for tennis strokes.
    """
    
    STROKE_TYPES = [
        "Stroke Type 1",
        "Stroke Type 2", 
        "Stroke Type 3"
    ]
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        input_size: tuple = (224, 224)
    ):
        """
        Initialize stroke classifier.
        
        Args:
            model_path: Path to model weights
            device: Device to run inference on
            input_size: Input image size (width, height)
        """
        self.device = device
        self.input_size = input_size
        self.model = None
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load the stroke classification model."""
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Initialize LSTM model (matches the provided weights)
            self.model = StrokeClassifierLSTM(num_classes=len(self.STROKE_TYPES))
            
            # Load state dict
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Stroke classifier loaded from {model_path}")
            logger.info(f"Training accuracy: {checkpoint.get('train_acc', 'N/A')}")
            logger.info(f"Validation accuracy: {checkpoint.get('valid_acc', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Error loading stroke classifier: {e}")
            self.model = None
    
    def classify_stroke(
        self,
        frame: np.ndarray,
        bbox: Optional[tuple] = None
    ) -> tuple:
        """
        Classify stroke type in a frame.
        
        Note: This is a simplified version. The LSTM model expects temporal sequences,
        but for single-frame inference we'll use a dummy approach or skip.
        
        Args:
            frame: Video frame as numpy array
            bbox: Optional bounding box (x, y, w, h) to crop player
            
        Returns:
            Tuple of (stroke_type, confidence)
        """
        if self.model is None:
            return ("Unknown", 0.0)
        
        try:
            # LSTM expects sequence input, but we only have single frames
            # For now, return Unknown since proper sequence-based classification
            # requires multiple frames in temporal order
            # TODO: Implement proper sequence buffering for LSTM inference
            return ("Unknown", 0.0)
            
        except Exception as e:
            logger.error(f"Error classifying stroke: {e}")
            return ("Unknown", 0.0)
    
    def process_video(
        self,
        frames: List[np.ndarray],
        ball_track: Optional[List] = None
    ) -> List[tuple]:
        """
        Process video frames and classify strokes.
        
        Args:
            frames: List of video frames
            ball_track: Optional ball tracking data
            
        Returns:
            List of (stroke_type, confidence) for each frame
        """
        strokes = []
        
        for i, frame in enumerate(frames):
            # Classify stroke (simplified - could use ball track for better detection)
            stroke, confidence = self.classify_stroke(frame)
            strokes.append((stroke, confidence))
        
        return strokes
