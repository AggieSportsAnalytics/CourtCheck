## Tracknet script & ball detector
import torch
import torch.nn as nn
import cv2
import numpy as np
from collections import deque
from scipy.spatial import distance

class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=pad,
                bias=bias,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module):
    def __init__(self, input_channels=3, out_channels=14):
        super().__init__()
        self.out_channels = out_channels
        self.input_channels = input_channels

        self.conv1 = ConvBlock(in_channels=self.input_channels, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        return x

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


## Ball Detection
class BallDetector:
    def __init__(self, path_model=None, device="cuda"):
        self.model = BallTrackerNet(input_channels=9, out_channels=256)
        self.device = device
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location=device))
            self.model = self.model.to(device)
            if device == "cuda":
                self.model = self.model.half()  # FP16 for ~2x throughput on GPU
            self.model.eval()  # inference mode
        self.width = 640
        self.height = 360
        self.frame_buffer = deque(maxlen=3)
        self.prev_pred = [None, None]

    def infer_single(self, current_frame):
        """Processes a single frame, using internal buffer for context."""
        self.frame_buffer.append(current_frame)

        if len(self.frame_buffer) < 3:
            return (None, None)  # Not enough frames yet

        # Get the last 3 frames from the buffer
        frame_m2, frame_m1, frame_0 = list(self.frame_buffer)

        # Preprocess frames
        img = cv2.resize(frame_0, (self.width, self.height))
        img_prev = cv2.resize(frame_m1, (self.width, self.height))
        img_preprev = cv2.resize(frame_m2, (self.width, self.height))

        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        # Model inference — use FP16 on GPU, FP32 on CPU
        with torch.no_grad():
            tensor = torch.from_numpy(inp)
            tensor = tensor.half() if self.device == "cuda" else tensor.float()
            out = self.model(tensor.to(self.device))
            output = out.argmax(dim=1).detach().cpu().numpy()

        # Post-process using previous prediction for stability
        x_pred, y_pred = self.postprocess(output, self.prev_pred)

        # Update previous prediction state
        self.prev_pred = [x_pred, y_pred]

        return (x_pred, y_pred)

    def postprocess(self, feature_map, prev_pred, scale=2, max_dist=80):
        """
        :params
            feature_map: feature map with shape (1,360,640)
            prev_pred: [x,y] coordinates of ball prediction from previous frame
            scale: scale for conversion to original shape (720,1280)
            max_dist: maximum distance from previous ball detection to remove outliers
        :return
            x,y ball coordinates
        """
        feature_map *= 255
        feature_map = feature_map.reshape((self.height, self.width))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(
            heatmap,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=50,
            param2=2,
            minRadius=1,   # lowered from 2: far-baseline balls are 1-2px in 360x640 map
            maxRadius=7,
        )
        x, y = None, None
        if circles is not None:
            if prev_pred[0]:
                for i in range(len(circles[0])):
                    x_temp = circles[0][i][0] * scale
                    y_temp = circles[0][i][1] * scale
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < max_dist:
                        x, y = x_temp, y_temp
                        break
            else:
                x = circles[0][0][0] * scale
                y = circles[0][0][1] * scale
        return x, y

