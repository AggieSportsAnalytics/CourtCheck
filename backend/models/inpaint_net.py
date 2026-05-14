# Ported from github.com/qaz812345/TrackNetV3
# 1D U-Net that fills in missed ball detections using learned trajectory physics.
#
# Input:
#   x — (batch, seq_len, 2)  normalized (x, y) in [0, 1]
#   m — (batch, seq_len, 1)  visibility mask: 1 = needs inpainting, 0 = valid detection
# Output:
#   (batch, seq_len, 2)  inpainted coordinates in [0, 1]
#
# Merge rule (from TrackNetV3 predict.py):
#   final = inpainted * mask + original * (1 - mask)
#   i.e. only replace frames where ball was not detected.

import torch
import torch.nn as nn


class _Conv1DBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding='same', bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class _Double1DConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv_1 = _Conv1DBlock(in_dim, out_dim)
        self.conv_2 = _Conv1DBlock(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_2(self.conv_1(x))


class InpaintNet(nn.Module):
    """1D U-Net trajectory inpainting model from TrackNetV3."""

    def __init__(self):
        super().__init__()
        self.down_1 = _Conv1DBlock(3, 32)
        self.down_2 = _Conv1DBlock(32, 64)
        self.down_3 = _Conv1DBlock(64, 128)
        self.buttleneck = _Double1DConv(128, 256)  # preserved from original (typo in saved weights)
        self.up_1 = _Conv1DBlock(384, 128)
        self.up_2 = _Conv1DBlock(192, 64)
        self.up_3 = _Conv1DBlock(96, 32)
        self.predictor = nn.Conv1d(32, 2, kernel_size=3, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 2) normalized coordinates
            m: (batch, seq_len, 1) inpaint mask (1 = missing, 0 = detected)
        Returns:
            (batch, seq_len, 2) inpainted coordinates in [0, 1]
        """
        # (batch, seq_len, 3) -> (batch, 3, seq_len) for Conv1D
        inp = torch.cat([x, m], dim=2).permute(0, 2, 1)

        x1 = self.down_1(inp)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        xb = self.buttleneck(x3)

        out = self.up_1(torch.cat([xb, x3], dim=1))
        out = self.up_2(torch.cat([out, x2], dim=1))
        out = self.up_3(torch.cat([out, x1], dim=1))
        out = self.sigmoid(self.predictor(out))

        # (batch, 2, seq_len) -> (batch, seq_len, 2)
        return out.permute(0, 2, 1)
