import sys
import os
import warnings


# Ignore the specific warning
warnings.filterwarnings(
    "ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release"
)

# Absolute path to the CourtCheck directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
courtcheck_path = os.path.join(root_dir, "CourtCheck")
sys.path.append(courtcheck_path)

# Now import dependencies
try:
    from dependencies import *

except ImportError as e:
    print(f"Error importing dependencies: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define keypoint names, flip map, and skeleton
keypoint_names = [
    "BTL",
    "BTLI",
    "BTRI",
    "BTR",
    "BBR",
    "BBRI",
    "IBR",
    "NR",
    "NM",
    "ITL",
    "ITM",
    "ITR",
    "NL",
    "BBL",
    "IBL",
    "IBM",
    "BBLI",
]

keypoint_flip_map = [
    ("BTL", "BTR"),
    ("BTLI", "BTRI"),
    ("BBL", "BBR"),
    ("BBLI", "BBRI"),
    ("ITL", "ITR"),
    ("ITM", "ITM"),
    ("NL", "NR"),
    ("IBL", "IBR"),
    ("IBM", "IBM"),
    ("NM", "NM"),
]

skeleton = []


# Utility function to load the trained model
def load_model(config_path, model_weights):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = (
        11  # Ensure the number of classes matches your dataset
    )
    cfg.MODEL.KEYPOINT_ON = True
    cfg.MODEL.DEVICE = "cpu"  # Use CPU

    MetadataCatalog.get("tennis_game_train").keypoint_names = keypoint_names
    MetadataCatalog.get("tennis_game_train").keypoint_flip_map = keypoint_flip_map
    MetadataCatalog.get("tennis_game_train").keypoint_connection_rules = skeleton

    predictor = DefaultPredictor(cfg)
    return predictor


lines = [
    ("BTL", "BTLI"),
    ("BTLI", "BTRI"),
    ("BTL", "NL"),
    ("BTLI", "ITL"),
    ("BTRI", "BTR"),
    ("BTR", "NR"),
    ("BTRI", "ITR"),
    ("ITL", "ITM"),
    ("ITM", "ITR"),
    ("ITL", "IBL"),
    ("ITM", "NM"),
    ("ITR", "IBR"),
    ("NL", "NM"),
    ("NL", "BBL"),
    ("NM", "IBM"),
    ("NR", "BBR"),
    ("NM", "NR"),
    ("IBL", "IBM"),
    ("IBM", "IBR"),
    ("IBL", "BBLI"),
    ("IBR", "BBRI"),
    ("BBR", "BBRI"),
    ("BBRI", "BBLI"),
    ("BBL", "BBLI"),
]

line_colors = [(0, 255, 0)] * len(lines)

keypoint_history = {name: deque(maxlen=10) for name in keypoint_names}


def stabilize_keypoints(keypoints):
    stabilized_keypoints = []
    for i, keypoint in enumerate(keypoints):
        keypoint_history[keypoint_names[i]].append(keypoint[:2])
        stabilized_keypoints.append(
            np.mean(keypoint_history[keypoint_names[i]], axis=0)
        )
    return np.array(stabilized_keypoints)


def transform_keypoints_to_2d(keypoints):
    keypoint_dict = {
        keypoint_names[i]: keypoints[i, :2] for i in range(len(keypoint_names))
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
    transformed_keypoints = cv2.perspectiveTransform(keypoints[None, :, :2], matrix)[0]
    return transformed_keypoints


def visualize_2d_court_skeleton(transformed_keypoints, lines):
    blank_image = np.zeros((490, 280, 3), np.uint8)
    start_x = 25
    start_y = 25
    end_x = blank_image.shape[1] - 25
    end_y = blank_image.shape[0] - 25

    for start, end in lines:
        start_point = tuple(
            map(int, transformed_keypoints[keypoint_names.index(start)])
        )
        end_point = tuple(map(int, transformed_keypoints[keypoint_names.index(end)]))
        start_point = (start_point[0] + start_x, start_point[1] + start_y)
        end_point = (end_point[0] + start_x, end_point[1] + start_y)
        cv2.line(blank_image, start_point, end_point, (255, 255, 255), 2)

    for point in transformed_keypoints:
        point = tuple(map(int, point))
        point = (point[0] + start_x, point[1] + start_y)
        cv2.circle(blank_image, point, 3, (0, 0, 255), -1)

    return blank_image


def visualize_predictions_with_lines(img, predictor, keypoint_names, lines):
    outputs = predictor(img)
    v = Visualizer(
        img[:, :, ::-1],
        metadata=MetadataCatalog.get("tennis_game_train"),
        scale=0.8,
        instance_mode=ColorMode.IMAGE,
    )
    instances = outputs["instances"].to("cpu")

    if len(instances) > 0:
        max_conf_idx = instances.scores.argmax()
        instances = instances[max_conf_idx : max_conf_idx + 1]

    out = v.draw_instance_predictions(instances)
    keypoints = instances.pred_keypoints.numpy()[0]

    img_copy = img.copy()
    stabilized_keypoints = stabilize_keypoints(keypoints)
    transformed_keypoints = transform_keypoints_to_2d(stabilized_keypoints)
    court_skeleton = visualize_2d_court_skeleton(transformed_keypoints, lines)

    img_copy[0 : court_skeleton.shape[0], 0 : court_skeleton.shape[1]] = court_skeleton

    for idx, keypoint in enumerate(stabilized_keypoints):
        x, y = keypoint
        label = keypoint_names[idx]
        cv2.putText(
            img_copy,
            label,
            (int(x) + 5, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(img_copy, (int(x), int(y)), 3, (0, 0, 255), -1)

    for (start, end), color in zip(lines, line_colors):
        start_idx = keypoint_names.index(start)
        end_idx = keypoint_names.index(end)
        cv2.line(
            img_copy,
            (
                int(stabilized_keypoints[start_idx][0]),
                int(stabilized_keypoints[start_idx][1]),
            ),
            (
                int(stabilized_keypoints[end_idx][0]),
                int(stabilized_keypoints[end_idx][1]),
            ),
            color,
            2,
        )

    return img_copy


def process_video(video_path, output_path, predictor, keypoint_names, lines):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with tqdm(total=frame_count, desc="Processing video frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = visualize_predictions_with_lines(
                frame, predictor, keypoint_names, lines
            )
            out.write(processed_frame)
            pbar.update(1)

    cap.release()
    out.release()
    logger.info(f"Processed video saved to {output_path}")
    print(f"Processed video saved to {output_path}")


def main():
    config_path = "/Users/macbookairm1/Documents/ASA_s2024/CourtCheck/models/court_detection/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    model_weights = "/Users/macbookairm1/Documents/ASA_s2024/CourtCheck/models/model_weights/court_detection_weights.pth"
    video_path = "/Users/macbookairm1/Documents/ASA_s2024/.25s_game1.mp4"  # Change this to the path of the input video
    output_path = "/Users/macbookairm1/Documents/ASA_s2024/CourtCheck/data/processed/game1_short_clip_2Dskeleton_v14.mp4"

    predictor = load_model(config_path, model_weights)
    process_video(video_path, output_path, predictor, keypoint_names, lines)


if __name__ == "__main__":
    main()
