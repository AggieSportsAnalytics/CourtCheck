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
    # Far-court ROI pass — mirrors PlayerTracker.detect_frame_with_far_roi.
    # The full-frame inference at 640x360 downsizes a ~15px far-baseline ball
    # to ~5px, which TrackNet's heatmap regression handles poorly and which the
    # downstream CatBoost bounce model then can't detect inflections on. The
    # ROI pass crops the far half of the court at near-native resolution before
    # resizing to TrackNet's 640x360 input, effectively 2-3x'ing the far-side
    # ball's pixel size in the model's view.
    _FAR_COURT_CORNERS = np.array([
        [[286.0, 161.0]], [[1379.0, 161.0]],
        [[286.0, 1778.0]], [[1379.0, 1778.0]],
    ], dtype=np.float32)
    _FAR_ROI_PAD = 40  # image-space padding around projected far-court rect

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

        # Separate state for the ROI pass — its 3-frame buffer + prev_pred must
        # not mix with the main full-frame pass since they're different scales.
        self.roi_frame_buffer = deque(maxlen=3)
        self.roi_prev_pred = [None, None]
        self._roi_bounds = None        # image-space crop (x1,y1,x2,y2); cached since calibration is static
        self._roi_logged = False

    def infer_single(self, current_frame):
        """Processes a single frame, using internal buffer for context."""
        self.frame_buffer.append(current_frame)

        if len(self.frame_buffer) < 3:
            return (None, None)  # Not enough frames yet

        # Get the last 3 frames from the buffer
        frame_m2, frame_m1, frame_0 = list(self.frame_buffer)

        # Capture output dimensions before downscaling for TrackNet
        frame_h, frame_w = frame_0.shape[:2]

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
        x_pred, y_pred = self.postprocess(output, self.prev_pred, frame_w=frame_w, frame_h=frame_h)

        # Update previous prediction state
        self.prev_pred = [x_pred, y_pred]

        return (x_pred, y_pred)

    def postprocess(self, feature_map, prev_pred, frame_w=1280, frame_h=720, max_dist=80):
        """
        :params
            feature_map: feature map with shape (1,360,640)
            prev_pred: [x,y] coordinates of ball prediction from previous frame
            frame_w: original frame width (used to scale TrackNet output back to frame space)
            frame_h: original frame height
            max_dist: maximum distance (in 1280x720-equivalent pixels) to filter outliers
        :return
            x,y ball coordinates in frame pixel space
        """
        # Scale TrackNet 640x360 detections back to original frame dimensions
        scale_x = frame_w / self.width
        scale_y = frame_h / self.height
        # max_dist was tuned at 1280x720; scale it proportionally to frame size
        scaled_max_dist = max_dist * (frame_w / 1280)

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
                    x_temp = circles[0][i][0] * scale_x
                    y_temp = circles[0][i][1] * scale_y
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < scaled_max_dist:
                        x, y = x_temp, y_temp
                        break
            else:
                x = circles[0][0][0] * scale_x
                y = circles[0][0][1] * scale_y
        return x, y

    # ------------------------------------------------------------------
    # ROI pass: tracks the ball on a far-court crop at near-native pixel
    # resolution. Used to backfill frames where the main full-frame pass
    # missed the ball — typically far-side bounce frames where the ball
    # is only ~5px in the 640x360 downscale and the heatmap loses it.
    # ------------------------------------------------------------------
    def _compute_far_court_roi_bounds(self, frame_shape, H_frame):
        """Project the far-court rectangle (court ref space) into image space
        and return (x1, y1, x2, y2) with padding. Cached after first call —
        calibration is static across the run so the bounds never change."""
        if self._roi_bounds is not None:
            return self._roi_bounds
        if H_frame is None:
            return None
        frame_h, frame_w = frame_shape[:2]
        mapped = cv2.perspectiveTransform(self._FAR_COURT_CORNERS, H_frame)
        xs = mapped[:, 0, 0]
        ys = mapped[:, 0, 1]
        x1 = max(0, int(np.min(xs)) - self._FAR_ROI_PAD)
        y1 = max(0, int(np.min(ys)) - self._FAR_ROI_PAD)
        x2 = min(frame_w, int(np.max(xs)) + self._FAR_ROI_PAD)
        y2 = min(frame_h, int(np.max(ys)) + self._FAR_ROI_PAD)
        if x2 - x1 < 50 or y2 - y1 < 50:
            return None  # degenerate crop; calibration probably wrong
        self._roi_bounds = (x1, y1, x2, y2)
        if not self._roi_logged:
            print(f"[BallROI] Far-court crop in image space: ({x1},{y1}) -> ({x2},{y2})  size={x2-x1}x{y2-y1}", flush=True)
            self._roi_logged = True
        return self._roi_bounds

    def _infer_roi_crop(self, current_frame, bounds):
        """TrackNet inference on the far-court crop with its own 3-frame buffer.
        Returns (x, y) in CROP-space pixel coords, or (None, None)."""
        x1, y1, x2, y2 = bounds
        crop = current_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return (None, None)
        self.roi_frame_buffer.append(crop)
        if len(self.roi_frame_buffer) < 3:
            return (None, None)

        frame_m2, frame_m1, frame_0 = list(self.roi_frame_buffer)
        # Each buffered crop must share the same (h, w) for the concat below.
        # Calibration is static, so all crops are bounds-sized — guaranteed.
        crop_h, crop_w = frame_0.shape[:2]

        img        = cv2.resize(frame_0,    (self.width, self.height))
        img_prev   = cv2.resize(frame_m1,   (self.width, self.height))
        img_prev2  = cv2.resize(frame_m2,   (self.width, self.height))

        imgs = np.concatenate((img, img_prev, img_prev2), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        with torch.no_grad():
            tensor = torch.from_numpy(inp)
            tensor = tensor.half() if self.device == "cuda" else tensor.float()
            out = self.model(tensor.to(self.device))
            output = out.argmax(dim=1).detach().cpu().numpy()

        # Postprocess returns coords in the model's input frame (= crop space
        # after resize). Pass crop_w/h so the scaling matches.
        x_pred, y_pred = self.postprocess(
            output, self.roi_prev_pred, frame_w=crop_w, frame_h=crop_h,
        )
        self.roi_prev_pred = [x_pred, y_pred]
        return (x_pred, y_pred)

    def infer_with_far_roi(self, frame, H_frame):
        """Main pass + ROI backfill in one call.

        Strategy:
          - Always run the main full-frame pass (preserves trajectory continuity
            for the near-court ball where TrackNet works well).
          - Also run the ROI pass on the far-court crop every frame so its
            3-frame buffer stays in lockstep with the video.
          - If main missed (returned None), use the ROI result mapped back to
            full-frame coords. If main hit, keep main.

        Falls back to plain infer_single() when H_frame is None (no calibration).
        """
        main_pred = self.infer_single(frame)

        if H_frame is None:
            return main_pred

        bounds = self._compute_far_court_roi_bounds(frame.shape, H_frame)
        if bounds is None:
            return main_pred

        x_main, y_main = main_pred
        # Always feed the ROI buffer so it doesn't go stale.
        x_crop, y_crop = self._infer_roi_crop(frame, bounds)

        if x_main is not None and y_main is not None:
            return main_pred  # main wins when it has a value

        if x_crop is None or y_crop is None:
            return main_pred  # neither pass hit — preserve None

        # Map crop-space coords back to original frame coords.
        x1, y1, _, _ = bounds
        return (x_crop + x1, y_crop + y1)
