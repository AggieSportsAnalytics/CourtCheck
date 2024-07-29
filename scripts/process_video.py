import sys
import os
import warnings
import logging
import cv2
import numpy as np
import torch
from tqdm import tqdm
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from scipy.spatial import distance

# Absolute path to the CourtCheck directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
courtcheck_path = os.path.join(root_dir, "CourtCheck")
sys.path.append(courtcheck_path)

# Add the path to the tracknet directory to the Python path
tracknet_path = os.path.join(root_dir, "CourtCheck/models/tracknet")
sys.path.append(tracknet_path)

# Add the path to the court_detection directory to the Python path
court_detection_path = os.path.join(root_dir, "CourtCheck/models/court_detection")
sys.path.append(court_detection_path)

from court_detection import (
    load_model as load_court_model,
    lines,
    keypoint_names,
    stabilize_keypoints,
    transform_keypoints_to_2d,
    visualize_2d_court_skeleton,
    visualize_predictions_with_lines,
)

from ball_detection import (
    postprocess,
    BallTrackerNet,
    read_video,
    remove_outliers,
    split_track,
    interpolation,
)


def load_tracknet_model(tracknet_weights_path):
    model = BallTrackerNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(tracknet_weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device


warnings.filterwarnings(
    "ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def transform_ball_position_to_2d(x, y, stabilized_keypoints):
    keypoint_dict = {
        keypoint_names[i]: stabilized_keypoints[i, :2]
        for i in range(len(keypoint_names))
    }
    src_points = np.array(
        [
            keypoint_dict["BTL"],
            keypoint_dict["BTR"],
            keypoint_dict["BBL"],
            keypoint_dict["BBR"],
        ],
        dtype=np.float32,
    )
    dst_points = np.array(
        [[17, 37], [217, 37], [17, 407], [217, 407]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    ball_pos = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_ball_pos = cv2.perspectiveTransform(ball_pos, matrix)
    return transformed_ball_pos[0][0]


def combine_results(frames, court_predictor, tracknet_model, device):
    combined_frames = []
    dists = [-1] * 2
    ball_track = [(None, None)] * 2

    for i in tqdm(
        range(2, len(frames)), desc="Combining results", position=0, leave=True
    ):
        frame = frames[i]
        prev_frame = frames[i - 1]
        prev_prev_frame = frames[i - 2]

        outputs = court_predictor(frame)
        instances = outputs["instances"]
        if len(instances) > 0:
            keypoints = instances.pred_keypoints.cpu().numpy()[0]
            processed_frame = visualize_predictions_with_lines(
                frame, court_predictor, keypoint_names, lines
            )
        else:
            keypoints = np.zeros((17, 3))
            processed_frame = frame.copy()

        x_pred, y_pred = detect_ball(
            tracknet_model, device, frame, prev_frame, prev_prev_frame
        )
        ball_track.append((x_pred, y_pred))

        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)

        if x_pred and y_pred:
            for j in range(min(7, len(ball_track))):  # Show trace of last 7 frames
                if ball_track[-j][0] is not None and ball_track[-j][1] is not None:
                    if (
                        0 <= int(ball_track[-j][0]) < processed_frame.shape[1]
                        and 0 <= int(ball_track[-j][1]) < processed_frame.shape[0]
                    ):
                        cv2.circle(
                            processed_frame,
                            (int(ball_track[-j][0]), int(ball_track[-j][1])),
                            max(2, 7 - j),  # Increase the ball size
                            (255, 255, 0),
                            -1,
                        )

        stabilized_keypoints = stabilize_keypoints(keypoints)
        transformed_keypoints = transform_keypoints_to_2d(stabilized_keypoints)
        court_skeleton = visualize_2d_court_skeleton(transformed_keypoints, lines)

        if x_pred and y_pred:
            ball_pos_2d = transform_ball_position_to_2d(
                x_pred, y_pred, stabilized_keypoints
            )
            if (
                0 <= int(ball_pos_2d[0]) < court_skeleton.shape[1]
                and 0 <= int(ball_pos_2d[1]) < court_skeleton.shape[0]
            ):
                cv2.circle(
                    court_skeleton,
                    (int(ball_pos_2d[0]), int(ball_pos_2d[1])),
                    3,
                    (255, 255, 0),
                    -1,
                )

        processed_frame[0 : court_skeleton.shape[0], 0 : court_skeleton.shape[1]] = (
            court_skeleton
        )

        combined_frames.append(processed_frame)

    ball_track = remove_outliers(ball_track, dists)
    subtracks = split_track(ball_track)
    for r in subtracks:
        ball_subtrack = ball_track[r[0] : r[1]]
        ball_subtrack = interpolation(ball_subtrack)
        ball_track[r[0] : r[1]] = ball_subtrack

    return combined_frames


def detect_ball(model, device, frame, prev_frame, prev_prev_frame):
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


def process_video(video_path, output_path, court_predictor, tracknet_model, device):
    frames, fps = read_video(video_path)
    if not frames:
        logger.error(f"Error opening video file {video_path}")
        return

    combined_frames = combine_results(frames, court_predictor, tracknet_model, device)

    height, width = combined_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in combined_frames:
        out.write(frame)

    out.release()
    print(f"Processed video saved to {output_path}")


def main():
    court_config_path = "/Users/macbookairm1/Documents/ASA_s2024/CourtCheck/models/court_detection/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    court_model_weights = "/Users/macbookairm1/Documents/ASA_s2024/CourtCheck/models/model_weights/court_detection_weights.pth"
    tracknet_weights_path = "/Users/macbookairm1/Documents/ASA_s2024/CourtCheck/models/model_weights/tracknet_weights.pt"
    video_path = "/Users/macbookairm1/Documents/ASA_s2024/10s_game2.mp4"
    output_path = "/Users/macbookairm1/Documents/ASA_s2024/CourtCheck/data/processed/game1_combined_output_v31.mp4"

    court_predictor = load_court_model(court_config_path, court_model_weights)
    tracknet_model, device = load_tracknet_model(tracknet_weights_path)

    process_video(video_path, output_path, court_predictor, tracknet_model, device)


if __name__ == "__main__":
    main()
