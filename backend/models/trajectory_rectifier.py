# Trajectory rectification using InpaintNet (TrackNetV3).
#
# Wraps InpaintNet to fill in None entries in a ball_track list produced by
# BallDetector. Runs as a post-processing step after TrackNet inference and
# before linear interpolation in the pipeline.
#
# Only the visualization / shot-detection track is rectified.
# The bounce detector receives ball_track_raw (pre-rectification) so that
# InpaintNet-smoothed gaps cannot create false bounce inflections.

from __future__ import annotations

import torch
import numpy as np

from backend.models.inpaint_net import InpaintNet

# Sequence length InpaintNet was trained with (TrackNetV3 default).
_SEQ_LEN = 16
# Minimum detected frames in a window to attempt inpainting.
_MIN_VISIBLE = 3
# Normalized coordinate threshold — below this is treated as out-of-frame.
_COOR_TH = 0.05


class TrajectoryRectifier:
    """
    Fills in missed ball detections (None entries) using InpaintNet.

    Usage:
        rectifier = TrajectoryRectifier(weights_path, frame_w, frame_h, device)
        ball_track = rectifier.rectify(ball_track_raw)
    """

    def __init__(self, weights_path: str, frame_w: int, frame_h: int, device: str = 'cpu'):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.device = device

        self.model = InpaintNet().to(device)
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        # Handle different checkpoint formats from TrackNetV3:
        # - full training checkpoint: {'epoch': ..., 'model': state_dict, ...}
        # - plain state_dict: {'layer.weight': tensor, ...}
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        elif isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def rectify(
        self,
        ball_track: list[tuple[float, float] | None],
    ) -> list[tuple[float, float] | None]:
        """
        Args:
            ball_track: list of (x, y) or None, length N. Pixel coordinates.
        Returns:
            New list of same length. None entries are replaced where InpaintNet
            predicts a plausible position; frames it cannot recover remain None.
        """
        n = len(ball_track)
        result = list(ball_track)

        for start in range(0, n, _SEQ_LEN):
            end = min(start + _SEQ_LEN, n)
            window = ball_track[start:end]
            window_len = len(window)

            visible = sum(1 for p in window if p is not None)
            if visible < _MIN_VISIBLE:
                continue
            if not any(p is None for p in window):
                continue

            coords = np.zeros((window_len, 2), dtype=np.float32)
            mask = np.zeros((window_len, 1), dtype=np.float32)

            for i, pt in enumerate(window):
                if pt is not None:
                    coords[i, 0] = pt[0] / self.frame_w
                    coords[i, 1] = pt[1] / self.frame_h
                else:
                    mask[i, 0] = 1.0

            x_t = torch.from_numpy(coords).unsqueeze(0).to(self.device)
            m_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)

            # Pad short windows (last window may be < _SEQ_LEN)
            if window_len < _SEQ_LEN:
                pad = _SEQ_LEN - window_len
                x_t = torch.cat([x_t, torch.zeros(1, pad, 2, device=self.device)], dim=1)
                m_t = torch.cat([m_t, torch.zeros(1, pad, 1, device=self.device)], dim=1)

            inpainted = self.model(x_t, m_t)
            inpainted_np = inpainted[0, :window_len].cpu().numpy()

            for i in range(window_len):
                if mask[i, 0] == 0.0:
                    continue  # already detected — keep original

                ix, iy = float(inpainted_np[i, 0]), float(inpainted_np[i, 1])
                if ix < _COOR_TH or iy < _COOR_TH or ix > 1.0 or iy > 1.0:
                    continue  # out-of-frame prediction — discard

                result[start + i] = (ix * self.frame_w, iy * self.frame_h)

        return result
