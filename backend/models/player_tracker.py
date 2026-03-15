from ultralytics import YOLO
import numpy as np
import cv2


def _project_foot(bbox, H_ref):
    """Project a player's foot position (bottom-center of bbox) to court reference space."""
    x1, y1, x2, y2 = bbox
    foot = np.array([[[(x1 + x2) / 2, float(y2)]]], dtype=np.float32)
    try:
        mapped = cv2.perspectiveTransform(foot, H_ref)
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
    except cv2.error:
        return None


_FAR_MIN_HEIGHT = 20   # absolute minimum — smaller than this isn't a person
_FAR_MIN_WIDTH  = 15


def _far_player_score(bbox, near_id: int, tid: int, H_ref: np.ndarray, court_bounds: dict, x_margin: float = 500, max_height: int = 400) -> float:
    """
    Return a positive score if this detection is a valid far player, else 0.

    Projects the bbox foot position to court-space using H_ref and checks:
      - Not the near player
      - Bbox size is plausible for a person
      - Foot lands on the far side of the net within court X bounds (+x_margin)

    Score = bbox area (larger = more confident YOLO detection).

    Args:
        court_bounds: dict with keys left_x, right_x, far_y_min, far_y_max
                      (derived from CourtReference at call site)
        x_margin:     allowed overshoot beyond sideline in court units
        max_height:   max bbox pixel height (filters oversized non-player detections)
    """
    if tid == near_id:
        return 0.0

    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w < _FAR_MIN_WIDTH or h < _FAR_MIN_HEIGHT or h > max_height:
        return 0.0

    result = _project_foot(bbox, H_ref)
    if result is None:
        return 0.0

    court_x, court_y = result

    if not (court_bounds['left_x'] - x_margin <= court_x <= court_bounds['right_x'] + x_margin):
        return 0.0

    if not (court_bounds['far_y_min'] <= court_y <= court_bounds['far_y_max']):
        return 0.0

    return w * h


class PlayerTracker:
    def __init__(self, model_path='yolov8m-pose.pt', device='cuda', imgsz: int = 1280):
        """
        Initialize player tracker with a YOLOv8-Pose model.

        Using a pose model allows extracting 17 body keypoints per player
        per frame in the same inference pass — no extra compute cost.

        Args:
            model_path: Path to the YOLOv8-Pose model (e.g. 'yolov8m-pose.pt').
            device: 'cuda' or 'cpu'
            imgsz: YOLO inference resolution. Should match input video resolution
                   to avoid downscaling small far-player detections below threshold.
        """
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        if device == 'cuda':
            self.model.to(device)
        print(f"PlayerTracker initialized with model: {model_path} on {device}, imgsz={imgsz}")

    def detect_frame(self, frame) -> tuple[dict, dict]:
        """
        Detect players in a single frame and extract pose keypoints.

        Args:
            frame: Single video frame (numpy array, BGR).

        Returns:
            player_dict: {track_id: [x1, y1, x2, y2]}
            keypoints_dict: {track_id: np.ndarray shape (17, 3)} where
                            columns are (x, y, confidence)
                            or None if the model did not return keypoints.
        """
        results = self.model.track(frame, persist=True, verbose=False, conf=0.10, iou=0.45, half=True, imgsz=self.imgsz)[0]
        id_name_dict = results.names

        player_dict: dict[int, list] = {}
        keypoints_dict: dict[int, np.ndarray | None] = {}

        if results.boxes is None or results.boxes.id is None:
            return player_dict, keypoints_dict

        # Extract pose keypoints if available (pose model)
        kps_data = None
        if results.keypoints is not None and results.keypoints.data is not None:
            kps_data = results.keypoints.data.cpu().numpy()  # (N, 17, 3)

        for det_idx, box in enumerate(results.boxes):
            track_id = int(box.id.tolist()[0])
            object_cls_name = id_name_dict[box.cls.tolist()[0]]
            if object_cls_name != "person":
                continue

            player_dict[track_id] = box.xyxy.tolist()[0]

            if kps_data is not None and det_idx < len(kps_data):
                keypoints_dict[track_id] = kps_data[det_idx]  # (17, 3)
            else:
                keypoints_dict[track_id] = None

        return player_dict, keypoints_dict

    def choose_and_filter_players(
        self,
        H_ref: np.ndarray,
        player_detections: list[dict],
        pose_keypoints_per_frame: list[dict] | None = None,
        court_ref=None,
        x_margin: float = 500,
        far_player_max_height: int = 400,
    ) -> tuple[list[dict], list[dict]]:
        """
        Filter detections to the 2 actual players.

        Near player: found by track_id vote using homography projection.
            The near player has a large, stable bbox and a reliable court-space
            projection — voting on track_id works well.

        Far player: found PER FRAME using court-space projection.
            YOLO track IDs for the small, distant far player fragment constantly
            (a new ID every few frames is normal). Locking onto any specific ID
            will miss the player in most frames. Instead, every frame we pick the
            single best qualifying detection — the largest bbox that passes the
            court-space projection and isn't the near player. No ID locking.
        """
        if court_ref is not None:
            court_bounds = {
                'net_y':    court_ref.net[0][1],
                'top_y':    court_ref.baseline_top[0][1],
                'bottom_y': court_ref.baseline_bottom[0][1],
                'left_x':   court_ref.left_court_line[0][0],
                'right_x':  court_ref.right_court_line[0][0],
                'far_y_min': court_ref.baseline_top[0][1] - 100,
                'far_y_max': court_ref.net[0][1] + 30,
            }
        else:
            court_bounds = {
                'net_y': 1748, 'top_y': 561, 'bottom_y': 2935,
                'left_x': 286, 'right_x': 1379,
                'far_y_min': 461, 'far_y_max': 1778,
            }

        # --- Step 1: Find near player by track_id vote ---
        near_votes: dict[int, int] = {}

        for frame in player_detections:
            for track_id, bbox in frame.items():
                result = _project_foot(bbox, H_ref)
                if result is None:
                    continue
                cx, court_y = result
                if not (court_bounds['left_x'] - x_margin <= cx <= court_bounds['right_x'] + x_margin):
                    continue
                if court_y > court_bounds['net_y']:
                    near_votes[track_id] = near_votes.get(track_id, 0) + 1

        near_id = max(near_votes, key=near_votes.__getitem__) if near_votes else None

        if near_id:
            print(f"[PlayerTracker] Near player: track_id={near_id} ({near_votes[near_id]} frames)")
        else:
            print("[PlayerTracker] Near player: not found")

        # --- Step 2: Filter frames ---
        # Near player: strict track_id match.
        # Far player: per-frame best candidate by court-space projection + largest bbox area.
        #   No track_id locking — far player ID is too fragmented to be reliable.
        filtered_players: list[dict] = []
        far_count = 0

        for frame in player_detections:
            new_frame: dict[int, list] = {}

            # Include near player by ID
            if near_id is not None and near_id in frame:
                new_frame[near_id] = frame[near_id]

            # Far player: per-frame best candidate by court-space projection + largest bbox area.
            best_tid, best_score = None, 0.0
            for tid, bbox in frame.items():
                score = _far_player_score(bbox, near_id, tid, H_ref, court_bounds, x_margin, far_player_max_height)
                if score > best_score:
                    best_score = score
                    best_tid = tid

            if best_tid is not None:
                new_frame[best_tid] = frame[best_tid]
                far_count += 1

            filtered_players.append(new_frame)

        print(f"[PlayerTracker] Far player: detected in {far_count}/{len(player_detections)} frames (position-based, no ID lock)")

        # --- Step 3: Build filtered pose list ---
        if pose_keypoints_per_frame is None:
            return filtered_players, [{} for _ in player_detections]

        filtered_poses: list[dict] = []
        for i, frame in enumerate(player_detections):
            new_kps: dict[int, np.ndarray | None] = {}
            kps_frame = pose_keypoints_per_frame[i] if i < len(pose_keypoints_per_frame) else {}
            for tid in filtered_players[i]:
                new_kps[tid] = kps_frame.get(tid)
            filtered_poses.append(new_kps)

        return filtered_players, filtered_poses

