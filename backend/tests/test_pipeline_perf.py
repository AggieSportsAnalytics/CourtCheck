"""Performance optimization tests for the CourtCheck pipeline."""


def test_default_model_is_yolov8m():
    from backend.pipeline.config import PipelineConfig
    cfg = PipelineConfig()
    assert "yolov8m" in cfg.player_model, (
        f"Expected yolov8m model, got: {cfg.player_model}"
    )


def test_player_detection_interval_default():
    from backend.pipeline.config import PipelineConfig
    cfg = PipelineConfig()
    assert cfg.player_detection_interval == 3


def test_interpolate_player_detections_fills_gaps():
    from backend.pipeline.run import _interpolate_player_detections
    # Frames 0 and 3 have player data; frames 1, 2 are empty ({})
    detections = [
        {1: [0.0, 10.0, 20.0, 30.0]},   # frame 0
        {},                               # frame 1: empty (skipped YOLO)
        {},                               # frame 2: empty (skipped YOLO)
        {1: [3.0, 13.0, 23.0, 33.0]},   # frame 3
    ]
    result = _interpolate_player_detections(detections)
    assert len(result) == 4
    # Frame 1: 1/3 of way from frame 0 to frame 3
    assert 1 in result[1], "track_id=1 should be interpolated into frame 1"
    x1, y1, x2, y2 = result[1][1]
    assert abs(x1 - 1.0) < 0.1, f"x1 interpolation wrong: {x1}"
    assert abs(y1 - 11.0) < 0.1, f"y1 interpolation wrong: {y1}"
    # Frame 2: 2/3 of the way
    x1_2, y1_2, x2_2, y2_2 = result[2][1]
    assert abs(x1_2 - 2.0) < 0.1
    # Frame 3 unchanged
    assert result[3][1] == [3.0, 13.0, 23.0, 33.0]


def test_interpolate_backfills_before_first_anchor():
    from backend.pipeline.run import _interpolate_player_detections
    detections = [
        {},                               # frame 0: before first detection
        {1: [10.0, 20.0, 30.0, 40.0]},  # frame 1: first anchor
        {},                               # frame 2
    ]
    result = _interpolate_player_detections(detections)
    # Frame 0 should be back-filled from frame 1
    assert 1 in result[0]
    # Frame 2 should be forward-filled from frame 1
    assert 1 in result[2]


def test_interpolate_all_anchors_unchanged():
    from backend.pipeline.run import _interpolate_player_detections
    # All frames have data — nothing should change
    detections = [
        {1: [float(i), float(i), float(i+10), float(i+10)]} for i in range(5)
    ]
    result = _interpolate_player_detections(detections)
    for i in range(5):
        assert result[i][1] == detections[i][1]


def test_mjpeg_codec_available():
    """MJPEG codec must be available in this OpenCV build for the intermediate file."""
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    assert fourcc != 0, "MJPEG codec (MJPG) not available in this OpenCV build"


def test_court_detection_startup_frames_config():
    from backend.pipeline.config import PipelineConfig
    cfg = PipelineConfig()
    assert hasattr(cfg, "court_detection_startup_frames")
    assert cfg.court_detection_startup_frames == 10


def test_detect_court_once_returns_best_result():
    from backend.pipeline.run import _detect_court_once
    import numpy as np

    class FakeDetector:
        def infer_single(self, frame):
            return [(100.0, 50.0)] * 14  # 14 fake keypoints, all valid

    class FakeEstimator:
        def estimate(self, kps):
            return np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)

    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(5)]
    H_ref, kps = _detect_court_once(frames, FakeDetector(), FakeEstimator())
    assert H_ref is not None, "Should return homography from mock frames"
    assert kps is not None

def test_detect_court_once_handles_empty_frames():
    from backend.pipeline.run import _detect_court_once

    class FailingDetector:
        def infer_single(self, frame):
            return None  # always fails

    class FakeEstimator:
        def estimate(self, kps):
            import numpy as np
            return np.eye(3), np.eye(3)

    import numpy as np
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]
    H_ref, kps = _detect_court_once(frames, FailingDetector(), FakeEstimator())
    assert H_ref is None
    assert kps is None


def test_make_streamable_prefers_nvenc(tmp_path, monkeypatch):
    """make_streamable_mp4 should attempt h264_nvenc before libx264."""
    import subprocess
    from backend.pipeline.storage import make_streamable_mp4

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "h264_nvenc" in cmd:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    input_file = tmp_path / "test.avi"
    input_file.write_bytes(b"fake")

    make_streamable_mp4(str(input_file))
    assert len(calls) >= 1
    assert "h264_nvenc" in calls[0], f"First attempt should use h264_nvenc, got: {calls[0]}"
    if len(calls) > 1:
        assert "libx264" in calls[1], f"Fallback should use libx264, got: {calls[1]}"


def test_pass2_videowriter_uses_mjpeg():
    """Pass 2 VideoWriter should use MJPG codec."""
    import inspect
    from backend.pipeline import run
    source = inspect.getsource(run.run_pipeline)
    assert '"MJPG"' in source or "'MJPG'" in source, (
        "Pass 2 VideoWriter should use MJPG codec"
    )
    assert "_processed.avi" in source, (
        "Pass 2 output file should use .avi extension"
    )
