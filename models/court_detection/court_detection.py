import sys

# Append the path to the system path
courtcheck_path = "/Users/macbookairm1/Documents/ASA_s2024/CourtCheck"
sys.path.append(courtcheck_path)

from dependencies import *

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
    scores = instances.scores if instances.has("scores") else [1.0]

    label_offset_x = 5
    label_offset_y = -10

    img_copy = img.copy()
    stabilized_keypoints = stabilize_keypoints(keypoints)

    for idx, (keypoints_per_instance, score) in enumerate(
        zip([stabilized_keypoints], scores)
    ):
        average_kp_score = 0
        for j, keypoint in enumerate(keypoints_per_instance):
            x, y = keypoint
            kp_score = keypoints[j, 2]
            label = keypoint_names[j]
            kp_score = max(0, min(1, kp_score))
            average_kp_score += kp_score
            if kp_score > 0:
                cv2.putText(
                    img_copy,
                    label,
                    (int(x) + label_offset_x, int(y) + label_offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.circle(img_copy, (int(x), int(y)), 3, (0, 0, 255), -1)

        average_kp_score /= len(keypoints_per_instance)
        average_kp_score = max(0, min(1, average_kp_score)) * 100
        cv2.putText(
            img_copy,
            f"Confidence: {average_kp_score:.2f}%",
            (10, 30 + 30 * idx),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

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

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
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
    video_path = "/Users/macbookairm1/Documents/ASA_s2024/game1_short_clip.mp4"  # Change this to the path of the input video
    output_path = "/Users/macbookairm1/Documents/ASA_s2024/CourtCheck/data/processed/game1_short_clip_output.mp4"

    predictor = load_model(config_path, model_weights)
    process_video(video_path, output_path, predictor, keypoint_names, lines)


if __name__ == "__main__":
    main()
