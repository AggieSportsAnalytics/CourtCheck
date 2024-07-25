# ball_detection.py
import torch
import cv2
from general import postprocess
from tqdm import tqdm
import numpy as np
from scipy.spatial import distance
from itertools import groupby
from model import BallTrackerNet

tracknet_weights_path = "CourtCheck/models/tracknet/weights/tracknet.pth"


def load_tracknet_model(tracknet_weights_path):
    model = BallTrackerNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(tracknet_weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device


def detect_ball(tracknet_model, device, frame, prev_frame, prev_prev_frame):
    height = 360
    width = 640
    img = cv2.resize(frame, (width, height))
    img_prev = cv2.resize(prev_frame, (width, height))
    img_preprev = cv2.resize(prev_prev_frame, (width, height))
    imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
    imgs = imgs.astype(np.float32) / 255.0
    imgs = np.rollaxis(imgs, 2, 0)
    inp = np.expand_dims(imgs, axis=0)

    out = tracknet_model(torch.from_numpy(inp).float().to(device))
    output = out.argmax(dim=1).detach().cpu().numpy()
    x_pred, y_pred = postprocess(output)
    return x_pred, y_pred


def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps


def remove_outliers(ball_track, dists, max_dist=100):
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if (dists[i + 1] > max_dist) | (dists[i + 1] == -1):
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i - 1] == -1:
            ball_track[i - 1] = (None, None)
    return ball_track


def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    list_det = [0 if x[0] else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []
    for i, (k, l) in enumerate(groups):
        if (k == 1) & (i > 0) & (i < len(groups) - 1):
            dist = distance.euclidean(ball_track[cursor - 1], ball_track[cursor + l])
            if (l >= max_gap) | (dist / l > max_dist_gap):
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                    min_value = cursor + l - 1
        cursor += l
    if len(list_det) - min_value > min_track:
        result.append([min_value, len(list_det)])
    return result


def interpolation(coords):
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
    y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

    nons, yy = nan_helper(x)
    x[nons] = np.interp(yy(nons), yy(~nons), x[~nons])
    nans, xx = nan_helper(y)

    y[nans] = np.interp(xx(nans), xx(~nons), y[~nons])

    track = [*zip(x, y)]
    return track
