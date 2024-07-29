import warnings
import cv2
import numpy as np
from collections import deque
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# Ignore the specific warning
warnings.filterwarnings(
    "ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release"
)

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


def load_court_model(config_path, model_weights):
    """Utility function to load the trained model
    :params
        config_path: path to the model configuration file
        model_weights: path to the model weights file
    :return
        predictor: loaded model predictor
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = (
        11  # Ensure the number of classes matches your dataset
    )
    cfg.MODEL.KEYPOINT_ON = True
    cfg.MODEL.DEVICE = "cpu"

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


def stabilize_points(keypoints):
    """Stabilize keypoints by averaging over a history of detected points
    :params
        keypoints: list of detected keypoints
    :return
        stabilized_points: list of stabilized keypoints
    """
    stabilized_points = []
    for i, keypoint in enumerate(keypoints):
        keypoint_history[keypoint_names[i]].append(keypoint[:2])
        if len(keypoint_history[keypoint_names[i]]) > 1:
            stabilized_points.append(
                np.mean(np.array(keypoint_history[keypoint_names[i]]), axis=0)
            )
        else:
            stabilized_points.append(keypoint[:2])
    return np.array(stabilized_points)


def transform_points(keypoints, black_frame_width, black_frame_height):
    """Transform keypoints to fit within a black frame
    :params
        keypoints: list of keypoints to transform
        black_frame_width: width of the black frame
        black_frame_height: height of the black frame
    :return
        transformed_keypoints: transformed keypoints
        matrix: perspective transformation matrix
    """
    width_frac = 6
    height_frac = 7

    keypoint_dict = {
        keypoint_names[i]: keypoints[i, :2] for i in range(len(keypoint_names))
    }

    dst_points = np.array(
        [
            [black_frame_width // width_frac, black_frame_height // height_frac],  # BTL
            [
                black_frame_width - black_frame_width // width_frac,
                black_frame_height // height_frac,
            ],  # BTR
            [
                black_frame_width // width_frac,
                black_frame_height - black_frame_height // height_frac,
            ],  # BBL
            [
                black_frame_width - black_frame_width // width_frac,
                black_frame_height - black_frame_height // height_frac,
            ],  # BBR
        ],
        dtype=np.float32,
    )

    src_points = np.array(
        [
            keypoint_dict["BTL"],
            keypoint_dict["BTR"],
            keypoint_dict["BBL"],
            keypoint_dict["BBR"],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_keypoints = cv2.perspectiveTransform(keypoints[None, :, :2], matrix)[0]
    return transformed_keypoints, matrix


def visualize_2d(transformed_keypoints, lines, black_frame_width, black_frame_height):
    """Visualize 2D court skeleton with transformed keypoints
    :params
        transformed_keypoints: list of transformed keypoints
        lines: list of lines connecting keypoints
        black_frame_width: width of the black frame
        black_frame_height: height of the black frame
    :return
        blank_image: image with the visualized 2D court skeleton
    """
    blank_image = np.zeros(
        (black_frame_height, black_frame_width, 3), np.uint8
    )  # Adjust the black frame size

    for start, end in lines:
        start_idx = keypoint_names.index(start)
        end_idx = keypoint_names.index(end)
        start_point = tuple(map(int, transformed_keypoints[start_idx][:2]))
        end_point = tuple(map(int, transformed_keypoints[end_idx][:2]))
        cv2.line(blank_image, start_point, end_point, (255, 255, 255), 2)

    for point in transformed_keypoints:
        point = tuple(map(int, point[:2]))
        cv2.circle(blank_image, point, 4, (0, 0, 255), -1)

    return blank_image


def visualize_predictions(
    img, predictor, keypoint_names, lines, black_frame_width, black_frame_height
):
    """Visualize model predictions on the input image
    :params
        img: input image
        predictor: model predictor
        keypoint_names: list of keypoint names
        lines: list of lines connecting keypoints
        black_frame_width: width of the black frame
        black_frame_height: height of the black frame
    :return
        img_copy: image with visualized predictions and court skeleton
    """
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
    stabilized_points = stabilize_points(keypoints)

    transformed_keypoints, matrix = transform_points(
        stabilized_points, black_frame_width, black_frame_height
    )
    court_skeleton = visualize_2d(
        transformed_keypoints, lines, black_frame_width, black_frame_height
    )

    img_copy[0 : court_skeleton.shape[0], 0 : court_skeleton.shape[1]] = court_skeleton

    for idx, keypoint in enumerate(stabilized_points):
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
        cv2.circle(img_copy, (int(x), int(y)), 5, (0, 0, 255), -1)

    for (start, end), color in zip(lines, line_colors):
        start_idx = keypoint_names.index(start)
        end_idx = keypoint_names.index(end)
        cv2.line(
            img_copy,
            (
                int(stabilized_points[start_idx][0]),
                int(stabilized_points[start_idx][1]),
            ),
            (
                int(stabilized_points[end_idx][0]),
                int(stabilized_points[end_idx][1]),
            ),
            color,
            2,
        )

    return img_copy
