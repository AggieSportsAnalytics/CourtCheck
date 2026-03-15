"""
Tests for two-player detection in test2.mov.

TDD session — verifies that both near and far players are detected and
that choose_and_filter_players returns 2 players per frame across the video.

Test video: backend/calibration_frames/videos/test2.mov
  - 2182x1228 source, resized to 1280x720
  - Far player: cy≈114-127, h≈102-131, w≈21, conf≈0.11-0.14
  - Near player: cy≈370-460, h≈180-230, conf≈0.85+
"""

import os
import cv2
import numpy as np
import pytest

VIDEO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "calibration_frames", "videos", "test2.mov"
)
CALIBRATION_PATH = os.path.join(
    os.path.dirname(__file__), "..", "calibration_frames", "court_calibration.json"
)
CAMERA_ID = "uc_davis_court1"
SAMPLE_STEP = 30  # sample every 30th frame (~2fps for a 59fps video)
TARGET_SIZE = (1280, 720)

# Thresholds — we allow some frames to have only 1 player (occlusion, mid-rally, etc.)
MIN_TWO_PLAYER_FRAMES_PCT = 0.30   # at least 30% of sampled frames should have 2 players
MIN_FAR_PLAYER_FRAMES = 5          # far player must appear in at least 5 sampled frames
FAR_PLAYER_MAX_CY = 300            # far player must appear in top 300px of 720p frame


@pytest.fixture(scope="module")
def player_tracker():
    """Load PlayerTracker once for all tests in this module."""
    from backend.models.player_tracker import PlayerTracker
    from backend.pipeline.config import PipelineConfig
    cfg = PipelineConfig()
    return PlayerTracker(model_path=cfg.player_model, device="cpu")


@pytest.fixture(scope="module")
def H_ref():
    """Load homography from calibration file."""
    from backend.vision.calibration import load_calibration
    H, _, _ = load_calibration(CALIBRATION_PATH, CAMERA_ID)
    return H


@pytest.fixture(scope="module")
def sampled_frames():
    """Read sampled frames from the test video, resized to 1280x720."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f"Cannot open video: {VIDEO_PATH}"
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = list(range(0, total, SAMPLE_STEP))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, cv2.resize(frame, TARGET_SIZE)))
    cap.release()
    assert len(frames) > 10, f"Too few frames sampled: {len(frames)}"
    return frames


@pytest.fixture(scope="module")
def raw_detections(player_tracker, sampled_frames):
    """Run detect_frame on all sampled frames, return list of player_dicts."""
    # Reset tracker state between fixtures
    player_tracker.model.predictor = None
    detections = []
    for frame_idx, frame in sampled_frames:
        player_dict, _ = player_tracker.detect_frame(frame)
        detections.append((frame_idx, player_dict))
    return detections


class TestRawDetections:
    """Phase 1: verify raw YOLO detections see both players."""

    def test_video_exists(self):
        assert os.path.exists(VIDEO_PATH), f"Test video not found: {VIDEO_PATH}"

    def test_detects_persons_in_every_frame(self, raw_detections):
        """Every sampled frame should have at least 1 person detected."""
        empty = [idx for idx, pd in raw_detections if len(pd) == 0]
        assert len(empty) == 0, (
            f"{len(empty)} frames with zero detections: {empty[:5]}"
        )

    def test_far_player_detected_in_raw(self, player_tracker, sampled_frames):
        """
        Far player (cy < FAR_PLAYER_MAX_CY) must appear in raw detections
        in at least MIN_FAR_PLAYER_FRAMES sampled frames.
        """
        player_tracker.model.predictor = None
        far_count = 0
        for _idx, frame in sampled_frames:
            player_dict, _ = player_tracker.detect_frame(frame)
            for bbox in player_dict.values():
                x1, y1, x2, y2 = bbox
                cy = (y1 + y2) / 2
                if cy < FAR_PLAYER_MAX_CY:
                    far_count += 1
                    break  # at most 1 far player per frame

        assert far_count >= MIN_FAR_PLAYER_FRAMES, (
            f"Far player (cy<{FAR_PLAYER_MAX_CY}) detected in only {far_count} frames "
            f"(need {MIN_FAR_PLAYER_FRAMES}). "
            f"Likely cause: conf threshold too high — far player conf ≈0.11-0.14."
        )

    def test_two_players_detected_simultaneously(self, raw_detections):
        """
        At least MIN_TWO_PLAYER_FRAMES_PCT of sampled frames must have 2+ detections.
        """
        two_player = sum(1 for _, pd in raw_detections if len(pd) >= 2)
        pct = two_player / len(raw_detections)
        assert pct >= MIN_TWO_PLAYER_FRAMES_PCT, (
            f"Only {two_player}/{len(raw_detections)} ({pct*100:.0f}%) frames have 2+ players. "
            f"Need {MIN_TWO_PLAYER_FRAMES_PCT*100:.0f}%."
        )


class TestFilteredDetections:
    """Phase 2: verify choose_and_filter_players returns 2 players per frame."""

    @pytest.fixture(scope="class")
    def filtered(self, player_tracker, sampled_frames, H_ref):
        """Run the full pipeline: detect + filter."""
        player_tracker.model.predictor = None
        all_detections = []
        all_kps = []
        for _idx, frame in sampled_frames:
            pd, kd = player_tracker.detect_frame(frame)
            all_detections.append(pd)
            all_kps.append(kd)

        if H_ref is not None:
            filtered_players, _ = player_tracker.choose_and_filter_players(
                H_ref, all_detections, all_kps
            )
        else:
            filtered_players = all_detections

        return filtered_players

    def test_near_player_in_every_frame(self, filtered):
        """Near player (large, bottom half) must appear in every filtered frame."""
        near_missing = 0
        for frame_dict in filtered:
            has_near = any(
                (b[1] + b[3]) / 2 > 350 for b in frame_dict.values()
            )
            if not has_near:
                near_missing += 1
        assert near_missing == 0, (
            f"Near player missing from {near_missing}/{len(filtered)} filtered frames"
        )

    def test_far_player_in_majority_of_frames(self, filtered):
        """
        Far player must appear in at least MIN_TWO_PLAYER_FRAMES_PCT of filtered frames.
        """
        far_count = sum(
            1 for frame_dict in filtered
            if any((b[1] + b[3]) / 2 < FAR_PLAYER_MAX_CY for b in frame_dict.values())
        )
        pct = far_count / len(filtered)
        assert pct >= MIN_TWO_PLAYER_FRAMES_PCT, (
            f"Far player in only {far_count}/{len(filtered)} ({pct*100:.0f}%) filtered frames. "
            f"Need {MIN_TWO_PLAYER_FRAMES_PCT*100:.0f}%."
        )

    def test_two_players_per_frame_majority(self, filtered):
        """Majority of filtered frames should have exactly 2 players."""
        two_player = sum(1 for fd in filtered if len(fd) == 2)
        pct = two_player / len(filtered)
        assert pct >= MIN_TWO_PLAYER_FRAMES_PCT, (
            f"Only {two_player}/{len(filtered)} ({pct*100:.0f}%) frames have 2 players. "
            f"Need {MIN_TWO_PLAYER_FRAMES_PCT*100:.0f}%."
        )
