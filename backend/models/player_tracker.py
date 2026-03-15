from ultralytics import YOLO
import numpy as np
import cv2

# Court reference coordinate bounds (from CourtReference)
_COURT_NET_Y    = 1748
_COURT_TOP_Y    = 561
_COURT_BOTTOM_Y = 2935
_COURT_LEFT_X   = 286
_COURT_RIGHT_X  = 1379
_COURT_X_MARGIN = 150   # allow slightly outside sideline

# Image-space bounds for the far player in a 1280x720 frame.
# Far players stand at the far baseline — small bbox, upper half of frame.
_FAR_MAX_CENTER_Y = 400   # center_y must be in upper ~55% of frame
_FAR_MIN_HEIGHT   = 20    # far baseline players can be 25-35px tall
_FAR_MAX_HEIGHT   = 250   # far player is always smaller than near player
_FAR_MIN_WIDTH    = 15
_FAR_MIN_CENTER_X = 200   # within court x range — exclude extreme sideline spectators
_FAR_MAX_CENTER_X = 1080


def _project_foot(bbox, H_ref):
    """Project a player's foot position (bottom-center of bbox) to court reference space."""
    x1, y1, x2, y2 = bbox
    foot = np.array([[[(x1 + x2) / 2, float(y2)]]], dtype=np.float32)
    try:
        mapped = cv2.perspectiveTransform(foot, H_ref)
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
    except cv2.error:
        return None


def _far_player_score(bbox, near_id, tid) -> float:
    """
    Return a positive score if this detection qualifies as the far player, else 0.

    Score = bbox area (larger = more confident YOLO detection).
    Returns 0 if the detection doesn't pass the image-space bounds or belongs to
    the near player.
    """
    if tid == near_id:
        return 0.0
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if (
        cy < _FAR_MAX_CENTER_Y
        and _FAR_MIN_HEIGHT <= h <= _FAR_MAX_HEIGHT
        and w >= _FAR_MIN_WIDTH
        and _FAR_MIN_CENTER_X <= cx <= _FAR_MAX_CENTER_X
    ):
        return w * h  # bbox area as confidence proxy
    return 0.0


# Court-space Y range for a valid far player.
# Far side: between far baseline (561) and net (1748).
_FAR_COURT_Y_MIN = _COURT_TOP_Y - 100    # 461
_FAR_COURT_Y_MAX = _COURT_NET_Y + 30     # 1778  (small margin for net players; avoid near-side false positives)


def _far_player_score_with_H(bbox, near_id: int, tid: int, H_ref) -> float:
    """
    Court-space version of _far_player_score.

    Projects the candidate bbox's foot position to court-space using H_ref and
    verifies it lands on the far side of the net within court X bounds.
    Falls back to the original image-space check when H_ref is None.

    Score = bbox area (same as _far_player_score) when the candidate passes.
    Returns 0 if the detection belongs to near_id or fails the court-space check.
    """
    if tid == near_id:
        return 0.0

    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w < _FAR_MIN_WIDTH or h < _FAR_MIN_HEIGHT or h > _FAR_MAX_HEIGHT:
        return 0.0

    if H_ref is None:
        # Fallback: original image-space filter
        return _far_player_score(bbox, near_id, tid)

    result = _project_foot(bbox, H_ref)
    if result is None:
        return 0.0

    court_x, court_y = result

    # Must be within court X bounds (with margin for slight sideline overrun)
    if not (_COURT_LEFT_X - _COURT_X_MARGIN <= court_x <= _COURT_RIGHT_X + _COURT_X_MARGIN):
        return 0.0

    # Must be on far side of net
    if not (_FAR_COURT_Y_MIN <= court_y <= _FAR_COURT_Y_MAX):
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
    ) -> tuple[list[dict], list[dict]]:
        """
        Filter detections to the 2 actual players.

        Near player: found by track_id vote using homography projection.
            The near player has a large, stable bbox and a reliable court-space
            projection — voting on track_id works well.

        Far player: found PER FRAME using court-space projection (falls back to image-space if H_ref is None).
            YOLO track IDs for the small, distant far player fragment constantly
            (a new ID every few frames is normal). Locking onto any specific ID
            will miss the player in most frames. Instead, every frame we pick the
            single best qualifying detection — the largest bbox that passes the
            court-space projection and isn't the near player. No ID locking.
        """
        # --- Step 1: Find near player by track_id vote ---
        near_votes: dict[int, int] = {}

        for frame in player_detections:
            for track_id, bbox in frame.items():
                result = _project_foot(bbox, H_ref)
                if result is None:
                    continue
                cx, court_y = result
                if not (_COURT_LEFT_X - _COURT_X_MARGIN <= cx <= _COURT_RIGHT_X + _COURT_X_MARGIN):
                    continue
                if court_y > _COURT_NET_Y:
                    near_votes[track_id] = near_votes.get(track_id, 0) + 1

        near_id = max(near_votes, key=near_votes.__getitem__) if near_votes else None

        if near_id:
            print(f"[PlayerTracker] Near player: track_id={near_id} ({near_votes[near_id]} frames)")
        else:
            print("[PlayerTracker] Near player: not found")

        # --- Step 2: Filter frames ---
        # Near player: strict track_id match.
        # Far player: per-frame best candidate by image-space bounds + largest bbox area.
        #   No track_id locking — far player ID is too fragmented to be reliable.
        filtered_players: list[dict] = []
        far_count = 0

        for frame in player_detections:
            new_frame: dict[int, list] = {}

            # Include near player by ID
            if near_id is not None and near_id in frame:
                new_frame[near_id] = frame[near_id]

            # Far player: per-frame best candidate by court-space projection + largest bbox area.
            # _far_player_score_with_H falls back to image-space if H_ref is None.
            best_tid, best_score = None, 0.0
            for tid, bbox in frame.items():
                score = _far_player_score_with_H(bbox, near_id, tid, H_ref)
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

