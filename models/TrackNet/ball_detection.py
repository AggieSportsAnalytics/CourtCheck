from model import BallTrackerNet
import torch
import cv2
from general import postprocess
from tqdm import tqdm
import numpy as np
from itertools import groupby
from scipy.spatial import distance


def load_tracknet_model(tracknet_weights_path):
    """Load the TrackNet model with the specified weights
    :params
        tracknet_weights_path: path to the model weights file
    :return
        model: loaded model
        device: device used for computation (CPU)
    """
    model = BallTrackerNet()
    device = torch.device("cpu")
    model.load_state_dict(torch.load(tracknet_weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device


def read_video(path_video):
    """Read video file
    :params
        path_video: path to video file
    :return
        frames: list of video frames
        fps: frames per second
    """
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


def postprocess(feature_map, original_shape, model_input_shape=(360, 640)):
    """Postprocess the output of the model to get the ball coordinates
    :params
        feature_map: output feature map from the model
        original_shape: original shape of the video frame
        model_input_shape: shape of the model input
    :return
        x_pred: x-coordinate of the detected ball
        y_pred: y-coordinate of the detected ball
    """
    height, width = model_input_shape
    feature_map = feature_map.reshape((height, width))
    y_pred, x_pred = np.unravel_index(np.argmax(feature_map), feature_map.shape)

    original_height, original_width = original_shape
    x_pred = int(x_pred * original_width / width)
    y_pred = int(y_pred * original_height / height)
    return x_pred, y_pred


def infer_model(frames, model, device):
    """Run pretrained model on a consecutive list of frames
    :params
        frames: list of consecutive video frames
        model: pretrained model
        device: device to run the model on
    :return
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
    """
    height = 360
    width = 640
    original_height, original_width = frames[0].shape[:2]
    dists = [-1] * 2
    ball_track = [(None, None)] * 2
    for num in tqdm(range(2, len(frames))):
        img = cv2.resize(frames[num], (width, height))
        img_prev = cv2.resize(frames[num - 1], (width, height))
        img_preprev = cv2.resize(frames[num - 2], (width, height))
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        out = model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess(output, (original_height, original_width))
        ball_track.append((x_pred, y_pred))

        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)
    return ball_track, dists


def remove_outliers(ball_track, dists, max_dist=100):
    """Remove outliers from model prediction
    :params
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
        max_dist: maximum distance between two neighbouring ball points
    :return
        ball_track: list of ball points
    """
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if (dists[i + 1] > max_dist) | (dists[i + 1] == -1):
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i - 1] == -1:
            ball_track[i - 1] = (None, None)
    return ball_track


def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    """Split ball track into several subtracks in each of which we will perform
    ball interpolation.
    :params
        ball_track: list of detected ball points
        max_gap: maximum number of coherent None values for interpolation
        max_dist_gap: maximum distance at which neighboring points remain in one subtrack
        min_track: minimum number of frames in each subtrack
    :return
        result: list of subtrack indexes
    """
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
    """Run ball interpolation in one subtrack
    :params
        coords: list of ball coordinates of one subtrack
    :return
        track: list of interpolated ball coordinates of one subtrack
    """

    def nan_helper(y):
        """Helper to handle indexes and logical indices of NaNs."""
        return np.isnan(y), lambda z: z.nonzero()[0]

    x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
    y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

    nons, yy = nan_helper(x)
    x[nons] = np.interp(yy(nons), yy(~nons), x[~nons])
    nans, xx = nan_helper(y)
    y[nans] = np.interp(xx(nans), xx(~nans), y[~nans])

    track = [*zip(x, y)]
    return track


def transform_ball_2d(x, y, matrix):
    """Transform ball coordinates to 2D perspective
    :params
        x: x-coordinate of the ball
        y: y-coordinate of the ball
        matrix: transformation matrix
    :return
        transformed_ball_pos: transformed ball position in 2D perspective
    """
    ball_pos = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_ball_pos = cv2.perspectiveTransform(ball_pos, matrix)
    return transformed_ball_pos[0][0]


def write_track(frames, ball_track, path_output_video, fps, trace=7):
    """Write .avi file with detected ball tracks
    :params
        frames: list of original video frames
        ball_track: list of ball coordinates
        path_output_video: path to output video
        fps: frames per second
        trace: number of frames with detected trace
    """
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(
        path_output_video, cv2.VideoWriter_fourcc(*"DIVX"), fps, (width, height)
    )
    for num in range(len(frames)):
        frame = frames[num]
        for i in range(trace):
            if num - i > 0:
                if ball_track[num - i][0]:
                    x = int(ball_track[num - i][0])
                    y = int(ball_track[num - i][1])
                    frame = cv2.circle(
                        frame, (x, y), radius=0, color=(0, 0, 255), thickness=10 - i
                    )
                else:
                    break
        out.write(frame)
    out.release()


def detect_ball(model, device, frame, prev_frame, prev_prev_frame):
    """Detect ball in a given frame using the model
    :params
        model: pretrained model
        device: device to run the model on
        frame: current video frame
        prev_frame: previous video frame
        prev_prev_frame: frame before the previous video frame
    :return
        x_pred: x-coordinate of the detected ball
        y_pred: y-coordinate of the detected ball
    """
    height = 360
    width = 640
    img = cv2.resize(frame, (width, height))
    img_prev = cv2.resize(prev_frame, (width, height))
    img_preprev = cv2.resize(prev_prev_frame, (width, height))
    imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
    imgs = imgs.astype(np.float32) / 255.0
    imgs = np.rollaxis(imgs, 2, 0)
    inp = np.expand_dims(imgs, axis=0)
    out = model(torch.from_numpy(inp).float().to(device))
    output = out.argmax(dim=1).detach().cpu().numpy()
    x_pred, y_pred = postprocess(output, (frame.shape[0], frame.shape[1]))
    return x_pred, y_pred
