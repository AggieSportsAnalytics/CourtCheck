"""
Pose-based stroke classifier using a Temporal Convolutional Network (TCN).

Input:  sequence of pose keypoint vectors — shape (seq_len, 34)
        where 34 = 17 COCO keypoints x (x, y)
Output: class probabilities over 4 stroke types

Classes (label index):
    0 - Forehand
    1 - Backhand
    2 - Serve / Overhead
    3 - Slice

Architecture: TCN (dilated causal convolutions) — faster than LSTM and
parallelisable. Designed to be pretrained on the THETIS dataset and then
fine-tuned on UC Davis footage.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


STROKE_LABELS = ["Forehand", "Backhand", "Serve/Overhead", "Slice"]
INPUT_DIM = 34   # 17 keypoints x 2 coords (x, y)
SEQ_LEN = 45     # fixed sequence length fed to the model


# --- Model -------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    """One TCN residual block with dilated causal convolutions."""

    def __init__(self, n_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            n_channels, n_channels, kernel_size, padding=padding, dilation=dilation,
        ))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(
            n_channels, n_channels, kernel_size, padding=padding, dilation=dilation,
        ))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self._padding = padding

    def _trim(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self._padding] if self._padding > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self._trim(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self._trim(self.conv2(out)))
        out = self.dropout(out)
        return self.relu(out + x)


class StrokeTCN(nn.Module):
    """
    Temporal Convolutional Network for tennis stroke classification.

    Input shape:  (batch, seq_len, 34)
    Output shape: (batch, 4) -- raw logits
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        n_classes: int = 4,
        n_channels: int = 64,
        kernel_size: int = 3,
        n_levels: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, n_channels)
        blocks = [
            _ResidualBlock(n_channels, kernel_size, 2 ** level, dropout)
            for level in range(n_levels)
        ]
        self.tcn = nn.Sequential(*blocks)
        self.classifier = nn.Linear(n_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        x = self.input_proj(x)    # (B, T, C)
        x = x.permute(0, 2, 1)   # (B, C, T) for Conv1d
        x = self.tcn(x)           # (B, C, T)
        x = x[:, :, -1]           # last timestep (B, C)
        return self.classifier(x) # (B, n_classes)


# --- Wrapper -----------------------------------------------------------------

class PoseStrokeClassifier:
    """
    High-level wrapper around StrokeTCN for use in the pipeline.

    Falls back to a rule-based heuristic when weights are unavailable.
    """

    LABELS = STROKE_LABELS

    def __init__(self, weights_path: str | None = None, device: str = "cpu"):
        self.device = device
        self._model: StrokeTCN | None = None
        self._weights_available = False

        if weights_path is not None:
            self._try_load(weights_path)

        if not self._weights_available:
            print(
                "[Stroke] TCN weights absent — using rule-based fallback"
            )

    def _try_load(self, path: str) -> None:
        if not os.path.exists(path):
            print(f"[StrokeClassifier] Weights not found at {path}")
            return
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
            model = StrokeTCN()
            model.load_state_dict(state)
            model.to(self.device)
            # Switch to inference mode
            for param in model.parameters():
                param.requires_grad_(False)
            model.training = False
            self._model = model
            self._weights_available = True
            print(f"[StrokeClassifier] Loaded TCN weights from {path}")
        except Exception as e:
            print(f"[StrokeClassifier] Failed to load weights: {e}")

    @property
    def is_available(self) -> bool:
        return self._weights_available

    def predict(self, pose_sequence: np.ndarray) -> tuple[np.ndarray, str]:
        """
        Classify a stroke from a pose keypoint sequence.

        Args:
            pose_sequence: Float32 array of shape (seq_len, 34).

        Returns:
            (probs, label) -- softmax probabilities and the predicted class name.
        """
        if self._weights_available and self._model is not None:
            return self._predict_tcn(pose_sequence)
        return self._predict_heuristic(pose_sequence)

    def _predict_tcn(self, pose_sequence: np.ndarray) -> tuple[np.ndarray, str]:
        x = torch.from_numpy(pose_sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(x)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs, self.LABELS[int(np.argmax(probs))]

    def _predict_heuristic(self, pose_sequence: np.ndarray) -> tuple[np.ndarray, str]:
        """
        Simple rule-based fallback when TCN weights are unavailable.

        Logic:
          - Wrist y-position above shoulders at swing peak -> Serve/Overhead
          - Dominant wrist vs torso center -> Forehand vs Backhand
        """
        def xy(kp_idx: int) -> np.ndarray:
            return pose_sequence[:, [2 * kp_idx, 2 * kp_idx + 1]]  # (T, 2)

        right_wrist    = xy(10)
        left_wrist     = xy(9)
        right_shoulder = xy(6)
        left_shoulder  = xy(5)
        right_hip      = xy(12)
        left_hip       = xy(11)

        mid = pose_sequence.shape[0] // 2
        shoulder_y = float(np.mean([right_shoulder[mid, 1], left_shoulder[mid, 1]]))
        torso_cx   = float(np.mean([right_hip[mid, 0], left_hip[mid, 0]]))

        r_vel = float(np.max(np.linalg.norm(np.diff(right_wrist, axis=0), axis=1) + 1e-6))
        l_vel = float(np.max(np.linalg.norm(np.diff(left_wrist, axis=0), axis=1) + 1e-6))
        dominant_wrist = right_wrist if r_vel >= l_vel else left_wrist
        dominant_is_right = r_vel >= l_vel
        dominant_x = float(dominant_wrist[mid, 0])
        dominant_y = float(dominant_wrist[mid, 1])

        # Serve: wrist above shoulders
        if dominant_y < shoulder_y - 20:
            return np.array([0.05, 0.05, 0.85, 0.05], dtype=np.float32), "Serve/Overhead"

        # Forehand vs backhand
        on_right_side = dominant_x > torso_cx
        is_forehand = (dominant_is_right and on_right_side) or (not dominant_is_right and not on_right_side)

        if is_forehand:
            return np.array([0.70, 0.20, 0.05, 0.05], dtype=np.float32), "Forehand"
        return np.array([0.15, 0.70, 0.05, 0.10], dtype=np.float32), "Backhand"
