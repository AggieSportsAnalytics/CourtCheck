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


_FAR_MIN_HEIGHT = 40   # min height to reject ball detections (~20px ball at 4x upscale)
_FAR_MIN_WIDTH  = 15
_far_filter_logged: set[int] = set()  # tids already logged — cap noise to 1 line per tid


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

    # Reject non-person shapes: nets, banners, horizontal bars have h << w
    if h < w * 0.3:
        return 0.0

    result = _project_foot(bbox, H_ref)
    if result is None:
        return 0.0

    court_x, court_y = result

    x_ok = court_bounds['left_x'] - x_margin <= court_x <= court_bounds['right_x'] + x_margin
    y_ok = court_bounds['far_y_min'] <= court_y <= court_bounds['far_y_max']

    if not (x_ok and y_ok):
        if tid not in _far_filter_logged:
            _far_filter_logged.add(tid)
            print(
                f"[FAR FILTER] tid={tid} court=({court_x:.0f},{court_y:.0f}) "
                f"x_ok={x_ok} y_ok={y_ok} "
                f"x_range=[{court_bounds['left_x']-x_margin:.0f},{court_bounds['right_x']+x_margin:.0f}] "
                f"y_range=[{court_bounds['far_y_min']:.0f},{court_bounds['far_y_max']:.0f}]"
            )
        return 0.0

    return w * h


class PlayerTracker:
    def __init__(self, model_path='yolov8m-pose.pt', device='cuda', imgsz: int = 1280, conf: float = 0.05):
        """
        Initialize player tracker with a YOLOv8-Pose model.

        Using a pose model allows extracting 17 body keypoints per player
        per frame in the same inference pass — no extra compute cost.

        Args:
            model_path: Path to the YOLOv8-Pose model (e.g. 'yolov8m-pose.pt').
            device: 'cuda' or 'cpu'
            imgsz: YOLO inference resolution. Should match input video resolution
                   to avoid downscaling small far-player detections below threshold.
            conf: Detection confidence threshold. Lower values surface more
                  candidates (needed for small far-court players).
        """
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        if device == 'cuda':
            self.model.to(device)

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
        results = self.model.track(frame, persist=True, verbose=False, conf=self.conf, iou=0.45, half=True, imgsz=self.imgsz)[0]
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

    # Far-court region in court-reference space.
    # y=161 = 400 units above far baseline (catches players standing well behind it)
    # y=1778 = net + 30 (small buffer past the net line)
    _FAR_COURT_CORNERS = np.array([
        [[286.0, 161.0]], [[1379.0, 161.0]],
        [[286.0, 1778.0]], [[1379.0, 1778.0]],
    ], dtype=np.float32)
    _FAR_ROI_UPSCALE = 4.0      # upscale factor applied to the crop
    _FAR_ROI_PAD = 60           # generous padding — far baseline is near top of image
    _FAR_ROI_CONF = 0.005       # very low threshold: far player is tiny and blurry
    _FAR_ROI_ID_OFFSET = -1000  # synthetic track IDs: -1000, -1001, …
    _far_roi_logged = False     # log ROI bounds once per instantiation

    def detect_frame_with_far_roi(
        self,
        frame: np.ndarray,
        H_frame: np.ndarray,
    ) -> tuple[dict, dict]:
        """
        Main detection pass + a second focused pass on the far-court region.

        The far player is typically only 30-60 px tall in a 1080p zoomed frame.
        Cropping the far-court strip (~100 px tall) and upscaling it 4x before
        running YOLO gives the model a ~160 px target — well within its detection
        range — without running the full model at 4× resolution.

        Detections from the ROI pass are added to player_dict with synthetic
        negative track IDs (-1000, -1001, …).  choose_and_filter_players handles
        them correctly because:
          - Negative IDs never win the near-player vote (they project to far side).
          - Far-player selection is per-frame best-score, no ID locking needed.
        """
        player_dict, keypoints_dict = self.detect_frame(frame)

        frame_h, frame_w = frame.shape[:2]

        # Project far-court corners to image space
        mapped = cv2.perspectiveTransform(self._FAR_COURT_CORNERS, H_frame)
        xs = mapped[:, 0, 0]
        ys = mapped[:, 0, 1]

        x1 = max(0, int(np.min(xs)) - self._FAR_ROI_PAD)
        y1 = max(0, int(np.min(ys)) - self._FAR_ROI_PAD)
        x2 = min(frame_w, int(np.max(xs)) + self._FAR_ROI_PAD)
        y2 = min(frame_h, int(np.max(ys)) + self._FAR_ROI_PAD)

        if not self._far_roi_logged:
            print(f"[ROI] Far-court crop in image space: ({x1},{y1}) -> ({x2},{y2})  size={x2-x1}x{y2-y1}")
            self._far_roi_logged = True

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return player_dict, keypoints_dict

        crop_h, crop_w = crop.shape[:2]
        upscaled = cv2.resize(
            crop,
            (int(crop_w * self._FAR_ROI_UPSCALE), int(crop_h * self._FAR_ROI_UPSCALE)),
            interpolation=cv2.INTER_LINEAR,
        )

        results = self.model.predict(upscaled, verbose=False, conf=self._FAR_ROI_CONF, imgsz=1280)[0]

        n_persons = sum(
            1 for box in (results.boxes or [])
            if results.names[int(box.cls.item())] == "person"
        )
        if n_persons > 0:
            print(f"[ROI] Frame detected {n_persons} person(s) in far-court crop")

        if results.boxes is None or len(results.boxes) == 0:
            return player_dict, keypoints_dict

        id_name_dict = results.names
        synthetic_id = self._FAR_ROI_ID_OFFSET
        for box in results.boxes:
            cls_name = id_name_dict[int(box.cls.item())]
            if cls_name != "person":
                continue

            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            # Map back to original frame coordinates
            ox1 = x1 + bx1 / self._FAR_ROI_UPSCALE
            oy1 = y1 + by1 / self._FAR_ROI_UPSCALE
            ox2 = x1 + bx2 / self._FAR_ROI_UPSCALE
            oy2 = y1 + by2 / self._FAR_ROI_UPSCALE

            if synthetic_id not in player_dict:
                player_dict[synthetic_id] = [ox1, oy1, ox2, oy2]
                keypoints_dict[synthetic_id] = None
            synthetic_id -= 1

        return player_dict, keypoints_dict

    def choose_and_filter_players(
        self,
        H_ref: np.ndarray,
        player_detections: list[dict],
        pose_keypoints_per_frame: list[dict] | None = None,
        court_ref=None,
        x_margin: float = 500,
        far_player_max_height: int = 400,
        far_max_jump_px: float = 350,
        far_hold_frames: int = 8,
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
                # 400-unit buffer above far baseline catches players standing behind it
                'far_y_min': court_ref.baseline_top[0][1] - 400,
                'far_y_max': court_ref.net[0][1] + 30,
            }
        else:
            court_bounds = {
                'net_y': 1748, 'top_y': 561, 'bottom_y': 2935,
                'left_x': 286, 'right_x': 1379,
                'far_y_min': 161, 'far_y_max': 1778,
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

        all_tids = {tid for frame in player_detections for tid in frame}
        print(f"[Player] IDs seen: {len(all_tids)} unique {sorted(all_tids)[:10]}")

        near_info = f"near={near_id} ({near_votes[near_id]} frames)" if near_id else "near=none"

        # --- Step 2: Filter frames ---
        # Near player: strict track_id match.
        # Far player: per-frame best candidate by court-space projection + largest bbox area,
        #   with temporal stabilization:
        #     - Proximity boost: candidates close to the last accepted position score higher.
        #     - Jump rejection: implausibly large jumps are suppressed when miss streak < 3.
        #     - Hold: last known bbox is carried forward during brief detection gaps.
        #     - Anchor reset: after a prolonged absence the anchor clears entirely.
        filtered_players: list[dict] = []
        far_count = 0
        far_tid_votes: dict[int, int] = {}

        # Temporal stabilizer state
        far_last_bbox: list | None = None   # last accepted far player bbox (image px)
        far_miss_streak: int = 0            # consecutive frames without an accepted far player
        _PROXIMITY_DECAY_PX = 250.0         # distance at which proximity bonus decays to zero
        _PROXIMITY_BOOST    = 2.5           # max score multiplier for on-target candidate

        for frame in player_detections:
            new_frame: dict[int, list] = {}

            # Include near player by ID
            if near_id is not None and near_id in frame:
                new_frame[near_id] = frame[near_id]

            # Far player: score all valid candidates, apply temporal preference.
            best_tid, best_score = None, 0.0
            for tid, bbox in frame.items():
                score = _far_player_score(bbox, near_id, tid, H_ref, court_bounds, x_margin, far_player_max_height)
                if score <= 0:
                    continue

                # Proximity boost: candidates near the last accepted position score higher.
                if far_last_bbox is not None:
                    lx1, ly1, lx2, ly2 = far_last_bbox
                    bx1, by1, bx2, by2 = bbox
                    dist = (
                        ((bx1 + bx2) / 2 - (lx1 + lx2) / 2) ** 2
                        + ((by1 + by2) / 2 - (ly1 + ly2) / 2) ** 2
                    ) ** 0.5
                    proximity_frac = max(0.0, 1.0 - dist / _PROXIMITY_DECAY_PX)
                    score *= 1.0 + proximity_frac * _PROXIMITY_BOOST

                if score > best_score:
                    best_score = score
                    best_tid = tid

            # Jump rejection: if the winning candidate is implausibly far from the last
            # accepted position AND the miss streak is short (i.e., the anchor is fresh),
            # treat it as noise and skip this frame rather than relocating the far player.
            if best_tid is not None and far_last_bbox is not None and far_miss_streak < 3:
                lx1, ly1, lx2, ly2 = far_last_bbox
                bx1, by1, bx2, by2 = frame[best_tid]
                jump = (
                    ((bx1 + bx2) / 2 - (lx1 + lx2) / 2) ** 2
                    + ((by1 + by2) / 2 - (ly1 + ly2) / 2) ** 2
                ) ** 0.5
                if jump > far_max_jump_px:
                    best_tid = None  # reject — suppress noisy relocation

            if best_tid is not None:
                accepted = frame[best_tid]
                # Always store under the canonical ID regardless of which synthetic
                # detection ID won this frame (-1000, -1001, etc.).
                new_frame[self._FAR_ROI_ID_OFFSET] = accepted
                far_last_bbox = accepted
                far_miss_streak = 0
                far_count += 1
                far_tid_votes[best_tid] = far_tid_votes.get(best_tid, 0) + 1

            elif far_last_bbox is not None and far_miss_streak < far_hold_frames:
                # Hold last known position during brief detection gaps (occlusion, blur).
                new_frame[self._FAR_ROI_ID_OFFSET] = far_last_bbox
                far_miss_streak += 1

            else:
                far_miss_streak += 1
                # Reset anchor after prolonged absence (player left court, dead time).
                if far_miss_streak > far_hold_frames * 4:
                    far_last_bbox = None

            filtered_players.append(new_frame)

        best_tid_overall = max(far_tid_votes, key=far_tid_votes.__getitem__) if far_tid_votes else None
        far_info = f"far={best_tid_overall if far_count > 0 else 'none'} ({far_count}/{len(player_detections)} frames)"
        print(f"[Player] {near_info} | {far_info}")

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

