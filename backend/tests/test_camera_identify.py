"""Camera matching tests use synthetic signatures and fake detectors, never models."""
from pathlib import Path

import numpy as np
import pytest

from backend.vision.camera_identify import (
    CameraMatch,
    identify_by_background,
    identify_by_keypoints,
    identify_from_frames,
    load_fingerprints,
    ncc,
    score_against,
)
from backend.vision import camera_identify as matcher


CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration_frames"
CAMERAS = ("uc_davis_court2", "uc_davis_court4", "uc_davis_court6")
GATES = {"max_distance": 60.0, "min_margin_ratio": 0.8, "min_points": 10}
NCC_GATES = {"min_ncc": 0.75, "min_margin": 0.12}
FRAME_SIZE = (1920, 1080)


@pytest.fixture
def calibrations():
    # Stand-ins for outputs of the same detector on each reference frame.
    base = np.column_stack((np.linspace(200, 1400, 14), np.linspace(150, 800, 14)))
    return {camera: (base + [offset, 0]).tolist() for camera, offset in zip(CAMERAS, (0, 60, -60))}


def _signature(pattern):
    centered = pattern - pattern.mean()
    return centered / centered.std()


@pytest.fixture
def fingerprints(monkeypatch):
    rng = np.random.default_rng(7)
    monkeypatch.setattr(matcher, "thumbnail_signature", lambda frame: frame)
    return {camera: {"signature": _signature(rng.normal(size=(54, 96))), "frame_path": f"{camera}.png"}
            for camera in CAMERAS}


def test_fingerprint_manifest_lists_only_courts_with_reference_frames():
    import json

    entries = json.loads((CALIBRATION_DIR / "camera_fingerprints.json").read_text())
    assert entries == {camera: {"frame": f"court{camera[-1]}.png"} for camera in CAMERAS}
    assert all((CALIBRATION_DIR / entry["frame"]).is_file() for entry in entries.values())


@pytest.mark.parametrize("camera", CAMERAS)
def test_identical_signature_matches_itself(fingerprints, camera):
    signature = fingerprints[camera]["signature"]
    assert ncc(signature, signature) == pytest.approx(1.0)
    match = identify_by_background([signature], fingerprints, **NCC_GATES)
    assert match.camera_id == camera
    assert match.reason == "matched"
    assert match.best == pytest.approx(1.0)
    assert list(match.scores.values()) == sorted(match.scores.values(), reverse=True)


def test_noisy_signature_still_matches(fingerprints):
    query = _signature(fingerprints[CAMERAS[0]]["signature"] + np.random.default_rng(0).normal(0, 0.5, (54, 96)))
    match = identify_by_background([query], fingerprints, **NCC_GATES)
    assert match.camera_id == CAMERAS[0]
    assert match.best - match.runner_up >= NCC_GATES["min_margin"]


def test_similar_backgrounds_are_ambiguous(fingerprints):
    base = fingerprints[CAMERAS[0]]["signature"]
    other = fingerprints[CAMERAS[1]]["signature"]
    orthogonal = _signature(other - ncc(base, other) * base)
    similar = 0.9 * base + np.sqrt(1 - 0.9 ** 2) * orthogonal
    references = {"base": {"signature": base}, "similar": {"signature": similar}}
    match = identify_by_background([base], references, **NCC_GATES)
    assert match.runner_up == pytest.approx(0.9)
    assert match.camera_id is None
    assert match.reason == "ambiguous"


def test_uncorrelated_background_is_too_low(fingerprints):
    query = _signature(np.random.default_rng(8).normal(size=(54, 96)))
    match = identify_by_background([query], fingerprints, **NCC_GATES)
    assert match.camera_id is None
    assert match.reason == "too_low"


def test_background_without_references(fingerprints):
    match = identify_by_background([fingerprints[CAMERAS[0]]["signature"]], {}, **NCC_GATES)
    assert match.camera_id is None
    assert match.reason == "no_fingerprints"


@pytest.mark.parametrize("constant", [False, True])
def test_thumbnail_preprocessing_with_numpy_opencv_stub(monkeypatch, constant):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    thumbnail = np.ones((54, 96)) if constant else np.arange(54 * 96).reshape(54, 96)
    canonical = np.zeros((1080, 1920, 3), dtype=np.uint8)
    gray = canonical[:, :, 0]
    cv2 = SimpleNamespace(COLOR_BGR2GRAY=6, INTER_AREA=3,
                          resize=Mock(side_effect=[canonical, thumbnail]), cvtColor=Mock(return_value=gray))
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    result = matcher.thumbnail_signature(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert cv2.resize.call_args_list[0].args[1] == FRAME_SIZE
    assert cv2.resize.call_args_list[1].args[1] == (96, 54)
    assert cv2.resize.call_args_list[1].kwargs == {"interpolation": cv2.INTER_AREA}
    assert cv2.cvtColor.call_args.args[0] is canonical
    assert cv2.cvtColor.call_args.args[1] == cv2.COLOR_BGR2GRAY
    assert result.mean() == pytest.approx(0, abs=1e-12)
    assert np.isfinite(result).all()
    assert result.std() == pytest.approx(0 if constant else 1)


def test_detector_fingerprints_runs_once_per_reference(monkeypatch, fingerprints, calibrations):
    calls = []

    def reference_frame(path):
        calls.append(path)
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

    monkeypatch.setattr(matcher, "_reference_frame", reference_frame)
    detector = FakeDetector([calibrations[camera] for camera in CAMERAS])
    result = matcher.detector_fingerprints(detector, fingerprints)
    assert result == calibrations
    assert calls == [fingerprints[camera]["frame_path"] for camera in CAMERAS]


def test_keypoint_cross_check_uses_median(monkeypatch, fingerprints, calibrations):
    monkeypatch.setattr(matcher, "detector_fingerprints", lambda detector, references: calibrations)
    detector = FakeDetector([calibrations[CAMERAS[0]], [(9999, 9999)] * 14, calibrations[CAMERAS[0]]])
    frames = [fingerprints[CAMERAS[0]]["signature"]] * 3
    match = identify_from_frames(frames, detector, fingerprints, frame_size=FRAME_SIZE, min_points=10, **NCC_GATES)
    assert match.keypoint_agrees is True
    assert match.keypoint_scores[CAMERAS[0]] == 0


@pytest.mark.parametrize("cross_check,agrees", [("uc_davis_court4", True), ("uc_davis_court2", False)])
def test_background_median_decides_and_keypoints_only_cross_check(monkeypatch, fingerprints, calibrations, cross_check, agrees):
    frames = [fingerprints["uc_davis_court4"]["signature"], fingerprints["uc_davis_court2"]["signature"],
              fingerprints["uc_davis_court4"]["signature"]]
    monkeypatch.setattr(matcher, "detector_fingerprints", lambda detector, references: calibrations)
    match = identify_from_frames(frames, FakeDetector([calibrations[cross_check]] * 3), fingerprints,
                                 frame_size=FRAME_SIZE, min_points=10, **NCC_GATES)
    assert match.camera_id == "uc_davis_court4"
    assert match.best == pytest.approx(1.0)
    assert match.keypoint_camera_id == cross_check
    assert match.keypoint_agrees is agrees
    assert match.keypoint_scores[cross_check] == 0
    assert list(match.keypoint_scores.values()) == sorted(match.keypoint_scores.values())


def test_background_rejection_never_uses_keypoints(fingerprints):
    query = _signature(np.random.default_rng(8).normal(size=(54, 96)))
    # An empty detector iterator would raise if inference ran.
    match = identify_from_frames([query], FakeDetector([]), fingerprints, min_points=10, **NCC_GATES)
    assert match.reason == "too_low"
    assert match.keypoint_agrees is None
    assert match.keypoint_scores == {}


def test_cross_check_failure_keeps_background_match(monkeypatch, fingerprints, capsys):
    def fail(*args):
        raise RuntimeError("Cross-check unavailable")

    monkeypatch.setattr(matcher, "detector_fingerprints", fail)
    match = identify_from_frames([fingerprints[CAMERAS[0]]["signature"]], FakeDetector([]), fingerprints,
                                 min_points=10, **NCC_GATES)
    assert match.camera_id == CAMERAS[0]
    assert match.reason == "matched"
    assert match.keypoint_agrees is None
    assert "Cross-check unavailable" in capsys.readouterr().out


@pytest.mark.parametrize("camera", CAMERAS)
def test_detector_reference_identifies_itself(calibrations, camera):
    match = identify_by_keypoints(calibrations[camera], calibrations, FRAME_SIZE, **GATES)
    assert match.camera_id == camera
    assert match.best == 0
    assert match.valid_points == 14
    assert match.reason == "matched"
    assert list(match.scores.values()) == sorted(match.scores.values())


@pytest.mark.parametrize("camera", CAMERAS)
def test_noise_preserves_identity(calibrations, camera):
    detected = np.asarray(calibrations[camera]) + np.random.default_rng(0).normal(0, 4, (14, 2))
    match = identify_by_keypoints(detected.tolist(), calibrations, FRAME_SIZE, **GATES)
    assert match.camera_id == camera
    assert match.reason == "matched"


def test_shifted_court2_distance_gate(calibrations):
    detected = (np.asarray(calibrations["uc_davis_court2"]) + [40, 0]).tolist()
    # Isolate the optional absolute distance gate for the keypoint path.
    only_court2 = {"uc_davis_court2": calibrations["uc_davis_court2"]}
    strict = identify_by_keypoints(detected, only_court2, FRAME_SIZE, **{**GATES, "max_distance": 30})
    assert strict.camera_id is None
    assert strict.reason == "too_far"
    relaxed = identify_by_keypoints(detected, only_court2, FRAME_SIZE, **GATES)
    assert relaxed.camera_id == "uc_davis_court2"


def test_halfway_between_courts_is_ambiguous(calibrations):
    halfway = (np.asarray(calibrations["uc_davis_court2"]) + calibrations["uc_davis_court6"]) / 2
    match = identify_by_keypoints(halfway.tolist(), calibrations, FRAME_SIZE, **GATES)
    assert match.camera_id is None
    assert match.reason == "ambiguous"


def test_too_few_detected_points(calibrations):
    detected = calibrations["uc_davis_court2"][:8] + [None] * 6
    match = identify_by_keypoints(detected, calibrations, FRAME_SIZE, **GATES)
    assert match.camera_id is None
    assert match.reason == "no_detection"
    assert match.valid_points == 8


@pytest.mark.parametrize("camera", CAMERAS)
def test_frame_size_scaling(calibrations, camera):
    detected = (np.asarray(calibrations[camera]) * (2 / 3)).tolist()
    match = identify_by_keypoints(detected, calibrations, (1280, 720), **GATES)
    assert match.camera_id == camera
    assert match.best < 1


class FakeDetector:
    def __init__(self, detections):
        self.detections = iter(detections)

    def infer_single(self, frame):
        return next(self.detections)


@pytest.mark.parametrize("detected", [None, [], [None] * 14])
def test_no_detection(calibrations, detected):
    match = identify_by_keypoints(detected, calibrations, FRAME_SIZE, **GATES)
    assert match.camera_id is None
    assert match.reason == "no_detection"


def test_no_calibrations(calibrations):
    match = identify_by_keypoints(calibrations["uc_davis_court4"], {}, FRAME_SIZE, **GATES)
    assert match.camera_id is None
    assert match.reason == "no_calibrations"


def test_empty_frames(fingerprints):
    match = identify_from_frames([], FakeDetector([]), fingerprints, min_points=10, **NCC_GATES)
    assert match.reason == "no_detection"


def test_detector_returns_none(monkeypatch, fingerprints, calibrations):
    monkeypatch.setattr(matcher, "detector_fingerprints", lambda detector, references: calibrations)
    match = identify_from_frames([fingerprints[CAMERAS[0]]["signature"]], FakeDetector([None]), fingerprints,
                                 frame_size=FRAME_SIZE, min_points=10, **NCC_GATES)
    assert match.reason == "matched"
    assert match.keypoint_camera_id is None
    assert match.keypoint_agrees is None


def test_score_uses_only_shared_finite_points():
    detected = [(3, 4), None, (np.nan, 1), (1, 1)]
    stored = [(0, 0), (0, 0), (0, 0), None]
    assert score_against(detected, stored, FRAME_SIZE) == (5.0, 1)
    assert score_against([None], [(0, 0)], FRAME_SIZE) == (float("inf"), 0)


def test_sparse_calibration_cannot_win(calibrations):
    detected = calibrations["uc_davis_court4"]
    entries = {"sparse": detected[:8] + [None] * 6, **calibrations}
    match = identify_by_keypoints(detected, entries, FRAME_SIZE, **GATES)
    assert match.camera_id == "uc_davis_court4"


def test_single_calibration_can_match(calibrations):
    points = calibrations["uc_davis_court4"]
    match = identify_by_keypoints(points, {"only": points}, FRAME_SIZE, **GATES)
    assert match.camera_id == "only"
    assert match.runner_up is None


def test_duplicate_calibrations_are_ambiguous(calibrations):
    points = calibrations["uc_davis_court4"]
    match = identify_by_keypoints(points, {"first": points, "second": points}, FRAME_SIZE, **GATES)
    assert match.camera_id is None
    assert match.reason == "ambiguous"


def test_loader_skips_missing_reference_frames(tmp_path, capsys):
    path = tmp_path / "fingerprints.json"
    path.write_text('{"missing": {"frame": "missing.png"}}')
    assert load_fingerprints(tmp_path, path) == {}
    assert "missing.png" in capsys.readouterr().out


def test_loader_skips_unreadable_reference_frames(tmp_path, monkeypatch, capsys):
    path = tmp_path / "fingerprints.json"
    path.write_text('{"broken": {"frame": "broken.png"}}')
    (tmp_path / "broken.png").touch()

    def unreadable(path):
        raise ValueError(f"Cannot read camera reference frame: {path}")

    monkeypatch.setattr(matcher, "_reference_frame", unreadable)
    assert load_fingerprints(tmp_path, path) == {}
    assert "broken.png" in capsys.readouterr().out


@pytest.mark.parametrize("contents", ['[]', '{"bad": []}', '{"bad": {"frame": null}}'])
def test_loader_rejects_malformed_entries(tmp_path, contents):
    path = tmp_path / "calibration.json"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_fingerprints(tmp_path, path)


def test_numpy_only_imports_and_config_without_torch():
    import subprocess
    import sys

    code = '''
import importlib.abc
import sys
class NoOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {"torch", "cv2", "scipy", "pandas"}:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, NoOptionalDependencies())
from backend.vision.camera_identify import identify_by_background
from backend.pipeline.config import PipelineConfig
assert "torch" not in sys.modules
config = PipelineConfig()
assert config.device == "cpu"
assert config.camera_id is None
assert config.camera_auto_identify is True
assert config.camera_match_min_ncc == 0.75
assert config.camera_match_min_ncc_margin == 0.12
assert config.camera_fingerprints_path.endswith("calibration_frames/camera_fingerprints.json")
assert config.camera_match_min_points == 10
assert config.camera_identify_frames == 5
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_identify_camera_for_synthetic_video(tmp_path, monkeypatch, calibrations):
    cv2 = pytest.importorskip("cv2")
    import backend.pipeline.run as pipeline
    from backend.pipeline.config import PipelineConfig

    class FakeCourtDetector:
        def __init__(self, **kwargs):
            self.calls = 0

        def infer_single(self, frame):
            self.calls += 1
            assert (frame.shape[1], frame.shape[0]) == FRAME_SIZE
            if self.calls <= len(CAMERAS):
                return calibrations[CAMERAS[self.calls - 1]]
            return calibrations["uc_davis_court4"]

    monkeypatch.setattr(pipeline, "CourtLineDetector", FakeCourtDetector)
    path = tmp_path / "Court2.avi"  # A misleading hint cannot override the detector.
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, FRAME_SIZE)
    assert writer.isOpened()
    reference_frame = cv2.imread(str(CALIBRATION_DIR / "court4.png"))
    try:
        for _ in range(12):
            writer.write(reference_frame)
    finally:
        writer.release()
    detector = pipeline.CourtLineDetector()
    config = PipelineConfig(device="cpu")
    match = pipeline._identify_camera_for_video(str(path), config, detector)
    assert match.camera_id == "uc_davis_court4"
    assert match.best > 0.99
    assert match.keypoint_agrees is True
    assert detector.calls == config.camera_identify_frames + len(CAMERAS)
    assert config.camera_id is None


@pytest.fixture
def pipeline_startup(monkeypatch):
    """Run real calibration wiring, stopping before tracking or external writes."""
    pytest.importorskip("cv2")
    import sys
    from types import ModuleType, SimpleNamespace
    from unittest.mock import Mock
    import backend.models as models
    import backend.pipeline.run as pipeline

    class StopBeforeTracking(BaseException):
        pass

    state = SimpleNamespace(detectors=[], fallback_calls=[], identify_calls=[], payloads=[])
    for name in models.__all__:
        if name != "CourtLineDetector":
            monkeypatch.setitem(vars(models), name, Mock())
    storage = ModuleType("backend.pipeline.storage")
    for name in ("upload_processed_video", "upload_heatmap_png", "get_supabase", "make_streamable_mp4", "upload_results_parallel"):
        setattr(storage, name, Mock())
    storage.ProcessedVideoUploadError = RuntimeError
    storage.PROCESSED_VIDEO_UPLOAD_ERROR = "Upload failed"
    monkeypatch.setitem(sys.modules, "backend.pipeline.storage", storage)
    monkeypatch.setattr(pipeline, "CourtLineDetector", lambda **kwargs: state.detectors.append(object()) or state.detectors[-1])
    monkeypatch.setattr(pipeline, "HomographyEstimator", Mock())
    monkeypatch.setattr(pipeline, "CourtReference", Mock())
    monkeypatch.setattr(pipeline, "_detect_court_once", lambda frames, detector, estimator: state.fallback_calls.append(detector) or (np.eye(3), []))
    monkeypatch.setattr(pipeline, "tqdm", Mock(side_effect=StopBeforeTracking))
    capture = Mock()
    capture.read.return_value = (True, np.zeros((2, 2, 3), dtype=np.uint8))
    capture.get.return_value = 1
    monkeypatch.setattr(pipeline.cv2, "VideoCapture", Mock(return_value=capture))
    state.pipeline, state.stop, state.storage = pipeline, StopBeforeTracking, storage
    return state


@pytest.mark.parametrize("explicit,auto,identified,hint,raises,expected,fallback", [
    ("uc_davis_court1_zoomed", True, "uc_davis_court4", "Court2", False, "uc_davis_court1_zoomed", False),
    (None, True, "uc_davis_court4", "Court2", False, "uc_davis_court4", False),
    (None, True, None, "Court2", False, "uc_davis_court2", False),
    (None, True, None, "unknown", False, None, True),
    (None, True, None, "Court2", True, "uc_davis_court2", False),
    (None, True, None, "unknown", True, None, True),
    (None, False, None, "Court2", False, None, True),
])
def test_pipeline_camera_fallback_order(pipeline_startup, monkeypatch, capsys, explicit, auto, identified, hint, raises, expected, fallback):
    from backend.pipeline.config import PipelineConfig

    state = pipeline_startup
    config = PipelineConfig(device="cpu", camera_id=explicit, camera_auto_identify=auto)

    def identify(path, cfg, detector):
        state.identify_calls.append(detector)
        if raises:
            raise RuntimeError("Detector failed")
        return CameraMatch(identified, {}, 1.0 if identified else None, None, 14, "matched" if identified else "too_low")

    monkeypatch.setattr(state.pipeline, "_identify_camera_for_video", identify)
    with pytest.raises(state.stop):
        state.pipeline.run_pipeline(f"{hint}.mp4", "local-test", local_mode=True, config=config)
    assert config.camera_id == expected
    assert bool(state.fallback_calls) == fallback
    assert len(state.detectors) == (0 if explicit else 1)
    assert len(state.identify_calls) == (1 if auto and not explicit else 0)
    if fallback:
        assert state.fallback_calls[0] is state.detectors[0]
    output = capsys.readouterr().out
    if identified and not explicit:
        assert "filename hint disagrees" in output
    if expected == "uc_davis_court2":
        assert "[Camera] falling back to filename hint" in output
    state.storage.get_supabase.assert_not_called()


def test_pipeline_unavailable_hint_uses_detected_homography(pipeline_startup, monkeypatch, tmp_path):
    from backend.pipeline.config import PipelineConfig

    state = pipeline_startup
    config = PipelineConfig(device="cpu", calibration_path=str(tmp_path / "missing.json"))
    monkeypatch.setattr(state.pipeline, "_identify_camera_for_video", lambda *args: CameraMatch(None, {}, None, None, 0, "no_fingerprints"))
    with pytest.raises(state.stop):
        state.pipeline.run_pipeline("Court2.mp4", "local-test", local_mode=True, config=config)
    assert config.camera_id is None
    assert state.fallback_calls == state.detectors


def test_camera_metadata_is_json_safe_and_missing_columns_only_log(monkeypatch, capsys):
    pytest.importorskip("cv2")
    import json
    from unittest.mock import Mock
    import backend.pipeline.run as pipeline

    database = Mock()
    match = CameraMatch(None, {"sparse": float("nan")}, None, float("nan"), 0, "no_detection",
                        keypoint_scores={"sparse": float("inf")})
    pipeline._record_camera_match(database, "match-id", "uc_davis_court2", match, "uc_davis_court2")
    payload = database.table.return_value.update.call_args.args[0]
    json.dumps(payload, allow_nan=False)
    assert payload == {"camera_id": "uc_davis_court2", "camera_match": {
        "reason": "no_detection", "best": None, "runner_up": None,
        "hint": "uc_davis_court2", "scores": {"sparse": None},
        "keypoint_camera_id": None, "keypoint_scores": {"sparse": None}, "keypoint_agrees": None,
    }}
    database.table.return_value.update.return_value.eq.assert_called_with("id", "match-id")
    database.table.return_value.update.return_value.eq.return_value.execute.side_effect = RuntimeError("column camera_match does not exist")
    pipeline._record_camera_match(database, "match-id", None, match, None)
    assert "Decision write failed" in capsys.readouterr().out


def test_pipeline_stores_camera_decision_outside_local_mode(pipeline_startup, monkeypatch, capsys):
    from backend.pipeline.config import PipelineConfig

    state = pipeline_startup
    match = CameraMatch("uc_davis_court4", {"uc_davis_court4": 0.93}, 0.93, 0.65, 14, "matched",
                        keypoint_camera_id="uc_davis_court2", keypoint_scores={"uc_davis_court2": 18.3}, keypoint_agrees=False)
    monkeypatch.setattr(state.pipeline, "_identify_camera_for_video", lambda *args: match)
    database = state.storage.get_supabase.return_value
    database.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = {}
    with pytest.raises(state.stop):
        state.pipeline.run_pipeline("Court2.mp4", "saved-match", config=PipelineConfig(device="cpu"))
    updates = [call.args[0] for call in database.table.return_value.update.call_args_list]
    decision = next(update for update in updates if "camera_match" in update)
    assert decision["camera_id"] == "uc_davis_court4"
    assert decision["camera_match"]["hint"] == "uc_davis_court2"
    assert decision["camera_match"]["reason"] == "matched"
    assert decision["camera_match"]["best"] == 0.93
    assert decision["camera_match"]["runner_up"] == 0.65
    assert decision["camera_match"]["keypoint_camera_id"] == "uc_davis_court2"
    assert decision["camera_match"]["keypoint_scores"] == {"uc_davis_court2": 18.3}
    assert decision["camera_match"]["keypoint_agrees"] is False
    output = capsys.readouterr().out
    assert "best=0.93 runner_up=0.65" in output
    assert "keypoint_agrees=False" in output
    assert "keypoint_scores={'uc_davis_court2': 18.3}" in output


@pytest.mark.parametrize("total,expected", [(120, [0, 14, 29, 44, 59]), (3, [0, 1, 2])])
def test_pipeline_samples_evenly_and_releases_capture(monkeypatch, calibrations, total, expected):
    cv2 = pytest.importorskip("cv2")
    from unittest.mock import Mock
    import backend.pipeline.run as pipeline
    from backend.pipeline.config import PipelineConfig

    capture = Mock()
    capture.get.return_value = total
    capture.read.return_value = (True, cv2.imread(str(CALIBRATION_DIR / "court4.png")))
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=capture))
    detector = FakeDetector([calibrations[camera] for camera in CAMERAS] + [calibrations["uc_davis_court4"]] * len(expected))
    match = pipeline._identify_camera_for_video("input.mp4", PipelineConfig(device="cpu"), detector)
    assert match.camera_id == "uc_davis_court4"
    assert [call.args[1] for call in capture.set.call_args_list] == expected
    capture.release.assert_called_once()


def test_cli_samples_only_first_ten_seconds(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    from unittest.mock import Mock
    from backend.tools import identify_camera as cli

    path = tmp_path / "video.mp4"
    path.touch()
    capture = Mock()
    capture.get.side_effect = lambda prop: 30 if prop == cv2.CAP_PROP_FPS else 600
    capture.read.return_value = (True, np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=capture))
    assert len(cli._load_frames(path, 5)) == 5
    assert [call.args[1] for call in capture.set.call_args_list] == [0, 74, 149, 224, 299]
    capture.release.assert_called_once()


def test_cli_help_needs_no_model_runtime():
    import subprocess
    import sys

    code = '''
import importlib.abc
import sys
class NoModelRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {"torch", "torchvision", "cv2", "numpy"}:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, NoModelRuntime())
from backend.tools.identify_camera import main
sys.exit(main(["--help"]))
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--frames" in result.stdout


def test_cli_json_is_sorted_and_continues_after_bad_input(monkeypatch, calibrations, fingerprints, capsys):
    import json
    from backend.tools import identify_camera as cli

    def load(path, count):
        if path.name == "bad.mp4":
            raise ValueError("Unreadable input")
        return [np.zeros((1080, 1920, 3), dtype=np.uint8)] * count

    monkeypatch.setattr(cli, "_load_frames", load)
    monkeypatch.setattr(matcher, "load_fingerprints", lambda *args: fingerprints)
    monkeypatch.setattr(matcher, "thumbnail_signature", lambda frame: fingerprints["uc_davis_court4"]["signature"])
    monkeypatch.setattr(matcher, "detector_fingerprints", lambda detector, references: calibrations)
    monkeypatch.setattr(cli, "_create_detector", lambda: FakeDetector([calibrations["uc_davis_court4"]] * 3))
    assert cli.main(["bad.mp4", "Court2.mp4", "--frames", "3", "--json"]) == 0
    output = capsys.readouterr()
    results = json.loads(output.out)
    assert results[0]["reason"] == "error"
    assert results[1]["camera_id"] == "uc_davis_court4"
    assert results[1]["hint"] == "uc_davis_court2"
    scores = list(results[1]["scores"].values())
    assert scores == sorted(scores, reverse=True)
    assert results[1]["best"] == pytest.approx(1.0)
    assert results[1]["keypoint_camera_id"] == "uc_davis_court4"
    assert results[1]["keypoint_agrees"] is True
    distances = list(results[1]["keypoint_scores"].values())
    assert distances == sorted(distances)
    assert "Unreadable input" in output.err


def test_cli_prints_both_signal_tables(capsys):
    from backend.tools import identify_camera as cli

    cli._print_table([{"input": "Court2.mp4", "camera_id": "uc_davis_court2", "reason": "matched", "hint": "uc_davis_court2",
                       "scores": {"uc_davis_court2": 0.93}, "keypoint_camera_id": "uc_davis_court2",
                       "keypoint_scores": {"uc_davis_court2": 18.3}, "keypoint_agrees": True}])
    output = capsys.readouterr().out
    assert "Background NCC" in output
    assert "Keypoint cross-check" in output
    assert "0.93" in output and "18.30" in output
    assert "matched" in output and "True" in output
