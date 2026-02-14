from ultralytics import YOLO
import cv2

class PlayerTracker:
    def __init__(self, model_path='yolov8x', device='cuda'):
        """
        Initialize player tracker with YOLO model

        Args:
            model_path: Path to YOLO model or model name
            device: 'cuda' or 'cpu'
        """
        self.model = YOLO(model_path)
        if device == 'cuda':
            self.model.to(device)
        print(f"PlayerTracker initialized with model: {model_path} on {device}")

    def detect_frame(self, frame):
        """
        Detect players in a single frame

        Args:
            frame: Single video frame (numpy array)

        Returns:
            dict: {track_id: [x1, y1, x2, y2]} for each detected player
        """
        results = self.model.track(frame, persist=True, verbose=False)[0]
        id_name_dict = results.names

        player_dict = {}
        if results.boxes is not None and results.boxes.id is not None:
            for box in results.boxes:
                track_id = int(box.id.tolist()[0])
                result = box.xyxy.tolist()[0]
                object_cls_id = box.cls.tolist()[0]
                object_cls_name = id_name_dict[object_cls_id]
                if object_cls_name == "person":
                    player_dict[track_id] = result

        return player_dict

    def choose_players(self, court_keypoints, player_dict):
        """
        Choose the 2 players closest to the court

        Args:
            court_keypoints: List of tuples [(x1, y1), (x2, y2), ...] or None values
            player_dict: {track_id: [x1, y1, x2, y2]}

        Returns:
            list: [track_id1, track_id2] of chosen players
        """
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = self._get_center_of_bbox(bbox)

            min_distance = float('inf')
            for court_keypoint in court_keypoints:
                if court_keypoint is None:
                    continue
                distance = self._measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # Sort the distances in ascending order
        distances.sort(key=lambda x: x[1])
        # Choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]] if len(distances) >= 2 else []
        return chosen_players

    def choose_and_filter_players(self, court_keypoints, player_detections):
        """
        Filter player detections to only keep the 2 main players

        Args:
            court_keypoints: Court keypoints from first frame
            player_detections: List of dicts [{track_id: bbox}, ...] for all frames

        Returns:
            list: Filtered player_detections with only chosen players
        """
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)

        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {
                track_id: bbox
                for track_id, bbox in player_dict.items()
                if track_id in chosen_player
            }
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections

    @staticmethod
    def _get_center_of_bbox(bbox):
        """Get center point of bounding box"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    @staticmethod
    def _measure_distance(p1, p2):
        """Calculate Euclidean distance between two points"""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
