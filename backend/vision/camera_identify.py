"""Identify cameras by background correlation, with detector keypoints as a cross-check."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np


CALIBRATION_FRAME_SIZE = (1920, 1080)
THUMBNAIL_SIZE = (96, 54)
SIGNATURE_EPSILON = 1e-6
_OPPONENT_COURT_MAP = {"2": "uc_davis_court2", "4": "uc_davis_court4", "6": "uc_davis_court6"}


def filename_camera_hint(video_path: str | Path) -> str | None:
    """Share the advisory filename convention between the pipeline and CLI."""
    match = re.search(r'[Cc]ourt(\d+)', Path(video_path).stem)
    return _OPPONENT_COURT_MAP.get(match.group(1)) if match else None


@dataclass(frozen=True)
class CameraMatch:
    camera_id: str | None
    scores: dict[str, float]  # Median background NCC, descending (higher is better).
    best: float | None
    runner_up: float | None
    valid_points: int
    reason: str
    keypoint_camera_id: str | None = None
    keypoint_scores: dict[str, float] = field(default_factory=dict)
    keypoint_agrees: bool | None = None


@dataclass(frozen=True)
class _KeypointMatch:
    """Keep diagnostic pixel distances separate from CameraMatch's NCC scores."""
    camera_id: str | None
    scores: dict[str, float]
    best: float | None
    runner_up: float | None
    valid_points: int
    reason: str


def _reference_frame(frame_path: str | Path) -> np.ndarray:
    import cv2

    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise ValueError(f"Cannot read camera reference frame: {frame_path}")
    if (frame.shape[1], frame.shape[0]) != CALIBRATION_FRAME_SIZE:
        frame = cv2.resize(frame, CALIBRATION_FRAME_SIZE)
    return frame


def thumbnail_signature(frame_bgr, size=THUMBNAIL_SIZE) -> np.ndarray:
    """Normalize a grayscale background thumbnail; import OpenCV only when needed."""
    import cv2

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or not frame_bgr.size:
        raise ValueError("Camera signature requires a non-empty BGR frame")
    if len(size) != 2 or any(not isinstance(value, int) or value < 1 for value in size):
        raise ValueError("Thumbnail size must be a positive (width, height) pair")
    frame = frame_bgr
    if (frame.shape[1], frame.shape[0]) != CALIBRATION_FRAME_SIZE:
        frame = cv2.resize(frame, CALIBRATION_FRAME_SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thumbnail = cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(float)
    return (thumbnail - thumbnail.mean()) / (thumbnail.std() + SIGNATURE_EPSILON)


def ncc(a, b) -> float:
    """Normalized correlation of two already standardized, equally sized signatures."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Signatures must have the same non-empty shape and finite values")
    return float(np.mean(a * b))


def load_fingerprints(calibration_dir, fingerprints_path) -> dict:
    """Load reference thumbnails, logging and skipping missing/unreadable frames."""
    with Path(fingerprints_path).open() as source:
        entries = json.load(source)
    if not isinstance(entries, dict):
        raise ValueError("Camera fingerprints must be an object keyed by camera ID")
    result = {}
    for camera, entry in entries.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("frame"), str) or not entry["frame"]:
            raise ValueError(f"Fingerprint {camera!r} must name a reference frame")
        frame_path = (Path(calibration_dir) / entry["frame"]).resolve()
        if not frame_path.is_file():
            print(f"[Camera] Skipping missing reference frame: {frame_path}", flush=True)
            continue
        try:
            frame = _reference_frame(frame_path)
        except ValueError as exc:
            print(f"[Camera] Skipping reference frame: {exc}", flush=True)
            continue
        result[camera] = {"signature": thumbnail_signature(frame), "frame_path": str(frame_path)}
    return result


def identify_by_background(frames, fingerprints, *, min_ncc, min_margin) -> CameraMatch:
    """Choose the highest median NCC when its absolute score and lead pass the gates."""
    if not np.isfinite(min_ncc) or not -1 <= min_ncc <= 1:
        raise ValueError("min_ncc must be between -1 and 1")
    if not np.isfinite(min_margin) or not 0 <= min_margin <= 2:
        raise ValueError("min_margin must be between 0 and 2")
    if not frames:
        return CameraMatch(None, {}, None, None, 0, "no_detection")
    if not fingerprints:
        return CameraMatch(None, {}, None, None, 0, "no_fingerprints")
    signatures = [thumbnail_signature(frame) for frame in frames]
    scores = {camera: float(np.median([ncc(signature, entry["signature"]) for signature in signatures]))
              for camera, entry in fingerprints.items()}
    ordered = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    candidates = list(ordered)
    best = ordered[candidates[0]]
    runner_up = ordered[candidates[1]] if len(candidates) > 1 else None
    camera_id = None
    if best < min_ncc:
        reason = "too_low"
    elif runner_up is not None and (best == runner_up or best - runner_up < min_margin):
        reason = "ambiguous"
    else:
        camera_id, reason = candidates[0], "matched"
    return CameraMatch(camera_id, ordered, best, runner_up, 0, reason)


def detector_fingerprints(court_detector, fingerprints) -> dict:
    """Run the same detector once per reference frame, always at 1920x1080."""
    return {camera: court_detector.infer_single(_reference_frame(entry["frame_path"]))
            for camera, entry in fingerprints.items()}


def _points_array(points) -> np.ndarray:
    """Keep missing/non-finite points unusable without changing their indices."""
    if points is None or len(points) == 0:
        return np.empty((0, 2), dtype=float)
    array = np.asarray([(np.nan, np.nan) if p is None else p for p in points], dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Court keypoints must be pairs of coordinates or None")
    return array


def score_against(detected: list[tuple[float, float] | None], stored: list[tuple[float, float] | None],
                  frame_size: tuple[int, int]) -> tuple[float, int]:
    """Mean pixel distance at shared valid indices, scaling stored points first.

    Return (inf, 0) for no overlap. The caller enforces its minimum point count.
    """
    size = np.asarray(frame_size, dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError("Frame size must be a positive (width, height) pair")
    predicted, reference = _points_array(detected), _points_array(stored)
    overlap = min(len(predicted), len(reference))
    predicted = predicted[:overlap]
    reference = reference[:overlap] * (size / CALIBRATION_FRAME_SIZE)
    valid = np.isfinite(predicted).all(axis=1) & np.isfinite(reference).all(axis=1)
    count = int(valid.sum())
    distance = float(np.linalg.norm(predicted[valid] - reference[valid], axis=1).mean()) if count else float("inf")
    return distance, count


def _pick_keypoint_match(scores, valid_points, *, min_points, max_distance=None, min_margin_ratio=None) -> _KeypointMatch:
    if max_distance is not None and (not np.isfinite(max_distance) or max_distance < 0):
        raise ValueError("max_distance must be finite and non-negative")
    if min_margin_ratio is not None and (not np.isfinite(min_margin_ratio) or not 0 < min_margin_ratio <= 1):
        raise ValueError("min_margin_ratio must be between zero (exclusive) and one")
    if not isinstance(min_points, int) or min_points < 1:
        raise ValueError("min_points must be a positive integer")
    ordered = dict(sorted(scores.items(), key=lambda item: item[1]))
    candidates = [(camera, distance) for camera, distance in ordered.items() if np.isfinite(distance)]
    best = candidates[0][1] if candidates else None
    runner_up = candidates[1][1] if len(candidates) > 1 else None
    camera_id = None
    if not scores:
        reason = "no_calibrations"
    elif valid_points < min_points or best is None:
        reason = "no_detection"
    elif max_distance is not None and best > max_distance:
        reason = "too_far"
    elif runner_up is not None and (best == runner_up or (min_margin_ratio is not None and best > min_margin_ratio * runner_up)):
        reason = "ambiguous"
    else:
        camera_id, reason = candidates[0][0], "matched"
    return _KeypointMatch(camera_id, ordered, best, runner_up, valid_points, reason)


def identify_by_keypoints(detected: list[tuple[float, float] | None] | None, calibrations: dict,
                          frame_size: tuple[int, int], *, min_points: int,
                          max_distance=None, min_margin_ratio=None) -> _KeypointMatch:
    """Rank detector-vs-detector distances; optional distance gates are diagnostic only."""
    valid_points = int(np.isfinite(_points_array(detected)).all(axis=1).sum())
    scores = {}
    for camera_id, stored in calibrations.items():
        distance, count = score_against(detected, stored, frame_size)
        scores[camera_id] = distance if count >= min_points else float("inf")
    match = _pick_keypoint_match(scores, valid_points, max_distance=max_distance,
                        min_margin_ratio=min_margin_ratio, min_points=min_points)
    if not valid_points:
        return _KeypointMatch(None, match.scores, None, None, 0, "no_detection")
    return match


def _keypoint_cross_check(frames, court_detector, fingerprints, *, frame_size=None, min_points) -> _KeypointMatch:
    """Median detector-vs-detector distances across usable query frames.

    valid_points is the median detected point count across usable frames.
    """
    calibrations = detector_fingerprints(court_detector, fingerprints)
    distances = {camera: [] for camera in calibrations}
    counts = []
    for frame in frames:
        detected = court_detector.infer_single(frame)
        size = frame_size if frame_size is not None else (frame.shape[1], frame.shape[0])
        match = identify_by_keypoints(detected, calibrations, size, min_points=min_points)
        if match.valid_points < min_points:
            continue
        counts.append(match.valid_points)
        for camera, distance in match.scores.items():
            if np.isfinite(distance):
                distances[camera].append(distance)
    scores = {camera: float(np.median(values)) if values else float("inf") for camera, values in distances.items()}
    match = _pick_keypoint_match(scores, int(np.median(counts)) if counts else 0, min_points=min_points)
    if not counts:
        return _KeypointMatch(None, match.scores, None, None, 0, "no_detection")
    return match


def identify_from_frames(frames: list[np.ndarray], court_detector, fingerprints: dict, *,
                         min_ncc: float, min_margin: float, min_points: int, frame_size=None) -> CameraMatch:
    """Background NCC decides; keypoint agreement is recorded without changing it."""
    match = identify_by_background(frames, fingerprints, min_ncc=min_ncc, min_margin=min_margin)
    if match.camera_id is None or court_detector is None:
        return match
    try:
        keypoints = _keypoint_cross_check(frames, court_detector, fingerprints,
                                         frame_size=frame_size, min_points=min_points)
    except Exception as exc:
        print(f"[Camera] Keypoint cross-check failed; keeping background match: {exc}", flush=True)
        return match
    return replace(match, valid_points=keypoints.valid_points, keypoint_camera_id=keypoints.camera_id,
                   keypoint_scores=keypoints.scores,
                   keypoint_agrees=keypoints.camera_id == match.camera_id if keypoints.camera_id else None)
