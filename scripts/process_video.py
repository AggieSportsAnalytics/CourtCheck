import sys
import os
import warnings
import logging
import cv2
import time
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
import torch
from datetime import datetime

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

from models.court_detection.court_detection_net import CourtDetectorNet
from models.court_detection.court_reference import CourtReference
from models.tracknet.ball_detection import BallDetector
from models.bounce_detection.bounce_detection import BounceDetector
from models.person_detection.person_detection import PersonDetector
from models.utils.utils import scene_detect


# Suppress specific warnings and set logging levels for certain modules
warnings.filterwarnings(
    "ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release"
)
logging.getLogger("detectron2.checkpoint.detection_checkpoint").setLevel(
    logging.WARNING
)
logging.getLogger("fvcore.common.checkpoint").setLevel(logging.WARNING)

# Clear any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging to show all messages
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,  # This will override any existing logger
)
logger = logging.getLogger(__name__)

# Add a print statement to verify the code is running
print("Starting video processing...")


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
            ball_point = np.array([[[float(x_pred), float(y_pred)]]], dtype=np.float32)
            ball_2d = cv2.perspectiveTransform(ball_point, matrix)
            ball_2d_pos = (int(ball_2d[0, 0, 0]), int(ball_2d[0, 0, 1]))

            # Add bounce visualization
            if i in bounces:
                cv2.circle(
                    court_skeleton, ball_2d_pos, 5, (0, 255, 255), -1
                )  # Yellow circle for bounces

            # Draw current ball position in minimap
            cv2.circle(
                court_skeleton,
                ball_2d_pos,
                3,
                (0, 255, 0),
                -1,
            )

        processed_frame[0 : court_skeleton.shape[0], 0 : court_skeleton.shape[1]] = (
            court_skeleton
        )

        # Draw ball
        if ball_track[i][0] is not None:
            cv2.circle(
                frame,
                (int(ball_track[i][0]), int(ball_track[i][1])),
                3,
                (0, 255, 0),
                -1,
            )

        # Draw court keypoints
        if kps_court[i] is not None:
            for j in range(len(kps_court[i])):
                kp = kps_court[i][j]
                cv2.circle(
                    frame,
                    (int(kp[0, 0]), int(kp[0, 1])),  # Access numpy array correctly
                    radius=5,
                    color=(0, 0, 255),
                    thickness=2,
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


def process_video(
    video_path,
    output_path,
    court_model_path,
    ball_model_path,
    bounce_model_path,
    device="cuda",
    draw_trace=True,
    trace_length=7,
):
    """Process video with integrated detection systems"""
    # Add immediate print for verification
    print(f"Processing video: {video_path}")

    total_start_time = time.time()
    logger.info("=" * 50)
    logger.info("Starting video processing")
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Output will be saved to: {output_path}")
    logger.info(f"Using device: {device}")
    logger.info("=" * 50)

    # Initialize models
    logger.info("Initializing models...")
    model_start_time = time.time()
    court_detector = CourtDetectorNet(court_model_path, device)
    ball_detector = BallDetector(ball_model_path, device)
    bounce_detector = BounceDetector(bounce_model_path)
    logger.info(f"Models initialized in {time.time() - model_start_time:.2f} seconds")

    # Create court reference for 2D visualization
    court_reference = CourtReference()
    court_img = court_reference.build_court_reference()
    court_img = cv2.dilate(court_img, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court_img, court_img, court_img), axis=2) * 255).astype(
        np.uint8
    )

    # Read video and detect scenes
    logger.info("Reading video...")
    video_start_time = time.time()
    frames, fps = read_video(video_path)
    total_frames = len(frames)
    logger.info(f"Video loaded: {total_frames} frames at {fps} FPS")
    logger.info(f"Video reading time: {time.time() - video_start_time:.2f} seconds")

    # Scene detection
    scene_start_time = time.time()
    scenes = scene_detect(video_path)
    logger.info(
        f"Detected {len(scenes)} scenes in {time.time() - scene_start_time:.2f} seconds"
    )

    # Ball detection
    logger.info("Detecting ball positions...")
    ball_start_time = time.time()
    ball_track = ball_detector.infer_model(frames)
    ball_positions = sum(1 for pos in ball_track if pos[0] is not None)
    logger.info(f"Ball detected in {ball_positions}/{total_frames} frames")
    logger.info(f"Ball detection time: {time.time() - ball_start_time:.2f} seconds")

    # Court detection
    logger.info("Detecting court and calculating transformations...")
    court_start_time = time.time()
    homography_matrices, kps_court = court_detector.infer_model(frames)
    valid_courts = sum(1 for matrix in homography_matrices if matrix is not None)
    logger.info(f"Court detected in {valid_courts}/{total_frames} frames")
    logger.info(f"Court detection time: {time.time() - court_start_time:.2f} seconds")

    # Comment out person detection
    # logger.info("Detecting players...")
    # person_detector = PersonDetector(device)
    # persons_top, persons_bottom = person_detector.track_players(
    #     frames, homography_matrices, filter_players=False
    # )

    logger.info("Detecting bounces...")
    bounce_start_time = time.time()
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)
    logger.info(f"Detected {len(bounces)} bounces")
    logger.info(f"Bounce detection time: {time.time() - bounce_start_time:.2f} seconds")

    # Process frames
    logger.info("Processing frames...")
    process_start_time = time.time()
    processed_frames = []
    width_minimap = 166  # Width of 2D court visualization
    height_minimap = 350  # Height of 2D court visualization
    ball_trail = []

    # At the start of processing
    frame_height, frame_width = frames[0].shape[:2]
    logger.debug(f"Original frame dimensions: {frame_width}x{frame_height}")

    # Define keypoint names to match the actual detection points
    keypoint_names = [
        "BTL",  # 0: Bottom Top Left (top baseline corner)
        "BTR",  # 1: Bottom Top Right (top baseline corner)
        "BBL",  # 2: Bottom Bottom Left (bottom baseline corner)
        "BBR",  # 3: Bottom Bottom Right (bottom baseline corner)
        "BTLI",  # 4: Bottom Top Left Inner (inner top baseline)
        "BBLI",  # 5: Bottom Bottom Left Inner (inner bottom baseline)
        "BTRI",  # 6: Bottom Top Right Inner (inner top baseline)
        "BBRI",  # 7: Bottom Bottom Right Inner (inner bottom baseline)
        "ITL",  # 8: Inner Top Left (top service line corner)
        "ITR",  # 9: Inner Top Right (top service line corner)
        "IBL",  # 10: Inner Bottom Left (bottom service line corner)
        "IBR",  # 11: Inner Bottom Right (bottom service line corner)
        "ITM",  # 12: Inner Top Middle (service line T)
        "IBM",  # 13: Inner Bottom Middle (service line T)
    ]

    # Define court outline connections using the correct indices
    court_lines = [
        ("BTL", "BTLI"),
        ("BTLI", "BTRI"),
        ("BTRI", "BTR"),  # Top line
        ("BTL", "BBL"),
        ("BTR", "BBR"),  # Left and right lines
        ("BBL", "BBLI"),
        ("BBLI", "BBRI"),
        ("BBLI", "IBL"),
        ("BBRI", "IBR"),
        ("BBRI", "BBR"),  # Bottom line
        ("BTLI", "ITL"),
        ("BTRI", "ITR"),  # Top service lines
        ("ITL", "ITM"),
        ("ITM", "IBM"),  # Middle service line
        ("ITL", "IBL"),
        ("ITR", "IBR"),  # Service box sides
        ("IBL", "IBM"),
        ("IBM", "IBR"),
        ("ITM", "ITR"),
    ]

    # Initialize coordinate tracking if not exists
    if not hasattr(process_video, "min_x"):
        process_video.min_x = float("inf")
        process_video.max_x = float("-inf")
        process_video.min_y = float("inf")
        process_video.max_y = float("-inf")

    # Initialize keypoint smoothing buffer if not exists
    if not hasattr(process_video, "keypoint_buffer"):
        process_video.keypoint_buffer = []
        process_video.buffer_size = 5  # Number of frames to consider
        process_video.max_deviation = 20  # Maximum allowed deviation in pixels

    def smooth_keypoints(current_kps, keypoint_buffer, buffer_size, max_deviation):
        if current_kps is None:
            return None

        # Add current keypoints to buffer
        keypoint_buffer.append(current_kps)
        if len(keypoint_buffer) > buffer_size:
            keypoint_buffer.pop(0)

        # If buffer is too small, return current keypoints
        if len(keypoint_buffer) < 2:
            return current_kps

        smoothed_kps = current_kps.copy()

        # For each keypoint
        for j in range(len(current_kps)):
            valid_positions = []

            # Collect valid positions from buffer
            for past_kps in keypoint_buffer[:-1]:  # Exclude current frame
                if past_kps is not None:
                    # Check if the point hasn't moved too much
                    if np.linalg.norm(current_kps[j] - past_kps[j]) < max_deviation:
                        valid_positions.append(past_kps[j])

            # If we have valid past positions, compute weighted average
            if valid_positions:
                weights = np.linspace(
                    0.5, 1.0, len(valid_positions) + 1
                )  # More weight to recent frames
                valid_positions.append(current_kps[j])  # Add current position
                smoothed_kps[j] = np.average(valid_positions, axis=0, weights=weights)

        return smoothed_kps

    for i in range(len(frames)):
        frame = frames[i].copy()
        frame_height, frame_width = frame.shape[:2]

        # Draw ball trajectory in bright neon blue
        if draw_trace:
            for j in range(trace_length):
                if i - j >= 0 and ball_track[i - j][0] is not None:
                    x = int(ball_track[i - j][0])
                    y = int(ball_track[i - j][1])

                    cv2.circle(
                        frame,
                        (x, y),
                        radius=max(2, 7 - j),
                        color=(255, 255, 0),  # Bright neon blue (BGR)
                        thickness=-1,
                    )

        # Draw court keypoints and lines
        if kps_court[i] is not None:
            try:
                # Draw court outline lines in brighter green with increased thickness
                for start, end in court_lines:
                    start_idx = keypoint_names.index(start)
                    end_idx = keypoint_names.index(end)

                    if start_idx < len(kps_court[i]) and end_idx < len(kps_court[i]):
                        x1 = int(kps_court[i][start_idx][0, 0] * frame_width / 1280)
                        y1 = int(kps_court[i][start_idx][0, 1] * frame_height / 720)
                        x2 = int(kps_court[i][end_idx][0, 0] * frame_width / 1280)
                        y2 = int(kps_court[i][end_idx][0, 1] * frame_height / 720)

                        cv2.line(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            color=(0, 255, 0),  # Brighter green (BGR)
                            thickness=3,  # Increased thickness
                        )

                # Draw keypoints with smaller labels
                for j in range(len(kps_court[i])):
                    x = int(kps_court[i][j][0, 0] * frame_width / 1280)
                    y = int(kps_court[i][j][0, 1] * frame_height / 720)

                    # Draw keypoint
                    cv2.circle(
                        frame,
                        (x, y),
                        radius=5,
                        color=(0, 0, 255),  # Red (BGR)
                        thickness=-1,
                    )

                    # Add smaller label with background
                    label = keypoint_names[j]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.4  # Reduced from 0.7
                    thickness = 1  # Reduced from 2

                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )

                    # Draw white background rectangle for text
                    cv2.rectangle(
                        frame,
                        (x + 5, y - text_height - 5),  # Reduced offset
                        (x + 5 + text_width, y - 2),
                        (255, 255, 255),
                        -1,
                    )

                    # Draw text
                    cv2.putText(
                        frame,
                        label,
                        (x + 5, y - 5),  # Reduced offset
                        font,
                        font_scale,
                        (0, 0, 255),  # Red text
                        thickness,
                        cv2.LINE_AA,
                    )

            except Exception as e:
                logger.warning(f"Error drawing court lines: {e}")
                continue

        # Create smaller minimap
        minimap_width = frame_width // 8
        minimap_height = int(minimap_width * court_img.shape[0] / court_img.shape[1])
        minimap = court_img.copy()

        # Keep track of bounce positions in minimap coordinates
        if not hasattr(process_video, "bounce_positions"):
            process_video.bounce_positions = []

        # Draw ball trail and bounces on minimap
        if ball_track[i][0] is not None and homography_matrices[i] is not None:
            print(f"\n=== Frame {i} Debug Info ===")

            # Original dimensions and transformations remain the same
            matrix = homography_matrices[i]
            ball_point = np.array(
                [[[float(ball_track[i][0]), float(ball_track[i][1])]]], dtype=np.float32
            )
            ball_2d = cv2.perspectiveTransform(ball_point, matrix)

            # Calculate initial scaled position
            minimap_scale_x = minimap_width / court_img.shape[1]
            minimap_scale_y = minimap_height / court_img.shape[0]
            scaled_x = ball_2d[0, 0, 0] * minimap_scale_x
            scaled_y = ball_2d[0, 0, 1] * minimap_scale_y

            # Add drawing scale factor (5x)
            draw_scale = 5.0
            draw_x = int(scaled_x * draw_scale)
            draw_y = int(scaled_y * draw_scale)

            print(f"Original scaled position: ({scaled_x:.2f}, {scaled_y:.2f})")
            print(f"Drawing position after {draw_scale}x scale: ({draw_x}, {draw_y})")

            # Draw with scaled coordinates
            ball_2d_pos = (draw_x, draw_y)
            cv2.circle(minimap, ball_2d_pos, 10, (255, 255, 0), -1)  # Yellow ball
            cv2.circle(minimap, ball_2d_pos, 11, (0, 0, 0), 1)  # Black outline

            # Handle bounces with same scaling
            if i in bounces:
                bounce_pos = ball_2d_pos
                process_video.bounce_positions.append(bounce_pos)

            # Draw all previous bounces with scaled coordinates
            print(f"\nDrawing {len(process_video.bounce_positions)} previous bounces")
            for bounce_pos in process_video.bounce_positions:
                cv2.circle(minimap, bounce_pos, 12, (0, 255, 255), -1)  # Yellow bounce
                cv2.circle(minimap, bounce_pos, 13, (0, 0, 0), 1)  # Black outline

        # Resize minimap after all drawing is complete
        minimap = cv2.resize(minimap, (minimap_width, minimap_height))

        # Position minimap in top-left corner (touching edges)
        frame[0:minimap_height, 0:minimap_width] = minimap

        # Apply smoothing to keypoints
        if kps_court[i] is not None:
            kps_court[i] = smooth_keypoints(
                kps_court[i],
                process_video.keypoint_buffer,
                process_video.buffer_size,
                process_video.max_deviation,
            )

        # Write frame
        processed_frames.append(frame)

    # Write output video
    logger.info(f"\nWriting output video:")
    logger.info(f"Output path: {output_path}")

    # Get dimensions from first frame
    height, width = frames[0].shape[:2]
    logger.info(f"Output dimensions: {width}x{height}")
    logger.info(f"Output FPS: {fps}")

    # Make sure output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Add .mp4 extension if not present
    if not output_path.endswith(".mp4"):
        output_path = output_path + ".mp4"
        logger.info(f"Added .mp4 extension. Final path: {output_path}")

    write_start_time = time.time()
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    if processed_frames:
        for frame in processed_frames:
            out.write(frame)
    else:
        logger.warning("No frames were processed, writing original frames")
        for frame in frames:
            out.write(frame)

    out.release()

    # Verify file was created
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
        logger.info(f"Successfully saved video:")
        logger.info(f"- Location: {output_path}")
        logger.info(f"- Size: {file_size:.2f} MB")
        logger.info(f"- Writing time: {time.time() - write_start_time:.2f} seconds")
    else:
        logger.error(f"Failed to save video to {output_path}")

    # Final statistics
    total_time = time.time() - total_start_time
    logger.info("\nFinal Statistics:")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info(f"Input video: {total_frames} frames at {fps} FPS")
    logger.info(f"Ball detection rate: {(ball_positions/total_frames)*100:.1f}%")
    logger.info(f"Court detection rate: {(valid_courts/total_frames)*100:.1f}%")
    logger.info(f"Number of bounces detected: {len(bounces)}")


def main():
    """Main function to set up models and process video"""
    print("Initializing main function...")

    # Model paths
    court_model_path = "/content/drive/MyDrive/ASA Tennis Bounds Project/models/court_detection_model/detectron2/models/court_detection_weights/model_tennis_court_det.pt"
    ball_model_path = "/content/drive/MyDrive/ASA Tennis Bounds Project/models/court_detection_model/detectron2/models/ball_detection_weights/tracknet_weights.pt"
    bounce_model_path = "/content/drive/MyDrive/ASA Tennis Bounds Project/models/court_detection_model/detectron2/models/bounce_detection_weights/bounce_detection_weights.cbm"

    video_path = "/content/drive/MyDrive/ASA Tennis Bounds Project/models/assets/game1_UCDwten.mp4"
    output_path = "/content/drive/MyDrive/ASA Tennis Bounds Project/models/court_detection_model/detectron2/post_processing/UCDwten/game1_UCDwten.mp4"

    print(f"Input video: {video_path}")
    print(f"Output path: {output_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    process_video(
        video_path,
        output_path,
        court_model_path,
        ball_model_path,
        bounce_model_path,
        device,
    )

    print("Processing completed!")


if __name__ == "__main__":
    """Execute main function"""
    main()


def read_video(path_video):
    """Read video file and return frames and fps
    Args:
        path_video (str): Path to video file
    Returns:
        frames (list): List of frames
        fps (int): Frames per second
    """
    logger.info(f"Opening video file: {path_video}")
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Total frames in video: {total_frames}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Read {frame_count}/{total_frames} frames")
        else:
            break
    cap.release()

    logger.info(f"Finished reading video: {len(frames)} frames at {fps} FPS")
    return frames, fps


def scale_coordinates(x, y, original_frame, processed_size=(640, 360)):
    """Scale coordinates from processed size back to original frame size"""
    if x is None or y is None:
        return None, None

    orig_h, orig_w = original_frame.shape[:2]
    proc_w, proc_h = processed_size

    # Scale coordinates
    scaled_x = int((x * orig_w) / proc_w)
    scaled_y = int((y * orig_h) / proc_h)

    return scaled_x, scaled_y
