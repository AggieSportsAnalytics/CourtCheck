import sys
import os
import warnings
import logging
import cv2
import time
import numpy as np
from tqdm import tqdm
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
    load_court_model,
    lines,
    keypoint_names,
    stabilize_points,
    transform_points,
    visualize_2d,
    visualize_predictions,
)

from ball_detection import (
    detect_ball,
    load_tracknet_model,
    transform_ball_2d,
    read_video,
    remove_outliers,
    split_track,
    interpolation,
)

# Suppress specific warnings and set logging levels for certain modules
warnings.filterwarnings(
    "ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release"
)
logging.getLogger("detectron2.checkpoint.detection_checkpoint").setLevel(
    logging.WARNING
)
logging.getLogger("fvcore.common.checkpoint").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def combine_results(frames, court_predictor, tracknet_model, device):
    """Combine court detection and ball tracking results, process each video frame
    :params
        frames: list of video frames
        court_predictor: court detection model
        tracknet_model: ball tracking model
        device: device to run the model on (CPU or GPU)
    :return
        combined_frames: list of processed video frames
    """
    combined_frames = []
    dists = [-1] * 2
    ball_track = [(None, None)] * 2
    keypoints_found = 0
    total_frames = len(frames)
    start_time = time.time()

    for i in tqdm(
        range(2, total_frames), desc="Combining results", position=0, leave=True
    ):
        frame = frames[i]
        prev_frame = frames[i - 1]
        prev_prev_frame = frames[i - 2]

        black_frame_width = (2 * frame.shape[1]) // 13
        black_frame_height = (5 * frame.shape[0]) // 11
        outputs = court_predictor(frame)
        instances = outputs["instances"]
        if len(instances) > 0:
            keypoints = instances.pred_keypoints.cpu().numpy()[0]
            keypoints_found += 1
            processed_frame = visualize_predictions(
                frame,
                court_predictor,
                keypoint_names,
                lines,
                black_frame_width,
                black_frame_height,
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
            for j in range(min(7, len(ball_track))):
                if ball_track[-j][0] is not None and ball_track[-j][1] is not None:
                    if (
                        0 <= int(ball_track[-j][0]) < processed_frame.shape[1]
                        and 0 <= int(ball_track[-j][1]) < processed_frame.shape[0]
                    ):
                        cv2.circle(
                            processed_frame,
                            (int(ball_track[-j][0]), int(ball_track[-j][1])),
                            max(2, 7 - j),
                            (255, 255, 0),
                            -1,
                        )

        stabilized_points = stabilize_points(keypoints)
        transformed_keypoints, matrix = transform_points(
            stabilized_points, black_frame_width, black_frame_height
        )
        court_skeleton = visualize_2d(
            transformed_keypoints, lines, black_frame_width, black_frame_height
        )

        if x_pred and y_pred:
            ball_pos_2d = transform_ball_2d(x_pred, y_pred, matrix)
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

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_frame = total_time / (total_frames - 2)
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    logger.info(f"Keypoints found in {keypoints_found} frames out of {total_frames}")
    logger.info(f"Total frames processed: {total_frames}")
    logger.info(f"Average time per frame: {avg_time_per_frame:.2f} seconds")
    logger.info(
        f"Total processing time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    )

    ball_track = remove_outliers(ball_track, dists)
    subtracks = split_track(ball_track)
    for r in subtracks:
        ball_subtrack = ball_track[r[0] : r[1]]
        ball_subtrack = interpolation(ball_subtrack)
        ball_track[r[0] : r[1]] = ball_subtrack

    return combined_frames


def process_video(video_path, output_path, court_predictor, tracknet_model, device):
    """Process a video by reading frames, combining results, and saving the output video
    :params
        video_path: path to the input video file
        output_path: path to save the processed video
        court_predictor: court detection model
        tracknet_model: ball tracking model
        device: device to run the model on (CPU or GPU)
    """
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
    """Main function to set up models, process a video, and save the result"""
    court_config_path = "/Users/macbookairm1/Documents/ASA_s2024/CourtCheck/models/court_detection/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    court_model_weights = "/Users/macbookairm1/Documents/ASA_s2024/CourtCheck/models/model_weights/court_detection_weights.pth"
    tracknet_weights_path = "/Users/macbookairm1/Documents/ASA_s2024/CourtCheck/models/model_weights/tracknet_weights.pt"
    video_path = "/Users/macbookairm1/Documents/ASA_s2024/_.25s_game1.mp4"
    output_path = "/Users/macbookairm1/Documents/ASA_s2024/CourtCheck/data/processed/game1_combined_output_v38.mp4"

    court_predictor = load_court_model(court_config_path, court_model_weights)
    tracknet_model, device = load_tracknet_model(tracknet_weights_path)

    process_video(video_path, output_path, court_predictor, tracknet_model, device)


if __name__ == "__main__":
    """Execute main function""" ""
    main()
