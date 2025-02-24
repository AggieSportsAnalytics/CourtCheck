from tracknet import BallTrackerNet
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm


class BallDetector:
    def __init__(self, path_model=None, device="cuda"):
        self.model = BallTrackerNet(input_channels=9, out_channels=256)
        self.device = device
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()
        self.width = 640
        self.height = 360

    def infer_model(self, frames):
        """Run pretrained model on a consecutive list of frames
        :params
            frames: list of consecutive video frames
        :return
            ball_track: list of detected ball points
        """
        ball_track = [(None, None)] * 2
        prev_pred = [None, None]
        for num in tqdm(range(2, len(frames))):
            img = cv2.resize(frames[num], (self.width, self.height))
            img_prev = cv2.resize(frames[num - 1], (self.width, self.height))
            img_preprev = cv2.resize(frames[num - 2], (self.width, self.height))
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            out = self.model(torch.from_numpy(inp).float().to(self.device))
            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = self.postprocess(output, prev_pred)
            prev_pred = [x_pred, y_pred]
            ball_track.append((x_pred, y_pred))
        return ball_track

    def postprocess(self, output, prev_pred, scale=3, max_dist=100):
        """
        :params
            output: feature map with shape (1,360,640)
            prev_pred: [x,y] coordinates of ball prediction from previous frame
        :return
            x,y ball coordinates
        """
        # Reshape and find max probability location
        feature_map = output[0].reshape((self.height, self.width))
        y_pred, x_pred = np.unravel_index(np.argmax(feature_map), feature_map.shape)

        # Scale to original frame size using direct ratios
        x = int(x_pred * (self.width * scale) / self.width)
        y = int(y_pred * (self.height * scale) / self.height)

        # Optional: Keep distance check for consistency
        if prev_pred[0] is not None:
            dist = distance.euclidean((x, y), prev_pred)
            if dist > max_dist:
                return None, None

        return x, y
