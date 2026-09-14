"""Inspect camera matches locally: python3 -m backend.tools.identify_camera --help."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = BACKEND_DIR / "weights/keypoints_model.pth"
VIDEO_SAMPLE_SECONDS = 10
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _positive_frames(value: str) -> int:
    frames = int(value)
    if frames < 1:
        raise argparse.ArgumentTypeError("--frames must be positive")
    return frames


def _load_frames(path: Path, count: int) -> list:
    import cv2
    import numpy as np

    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")
    if path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        return [image]
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cannot determine video frame rate: {path}")
        limit = max(1, math.ceil(fps * VIDEO_SAMPLE_SECONDS))
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if math.isfinite(total) and total > 0:
            limit = min(limit, int(total))
        frames = []
        for index in np.linspace(0, limit - 1, min(count, limit), dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
            else:
                print(f"[Camera] Cannot read frame {index} from {path}", file=sys.stderr)
        return frames
    finally:
        cap.release()


def _create_detector():
    # This import loads torch only when the model is constructed, after argparse.
    from backend.models.court_line_detector import CourtLineDetector

    return CourtLineDetector(model_path=str(WEIGHTS_PATH), device="cpu")


def _inspect_inputs(inputs: list[str], frame_count: int) -> list[dict]:
    from backend.pipeline.config import PipelineConfig
    from backend.vision.camera_identify import filename_camera_hint, identify_from_frames, load_fingerprints

    config = PipelineConfig(device="cpu")
    results = []
    detector = None
    for source in inputs:
        result = {"input": source, "camera_id": None, "reason": "error", "scores": {}, "hint": filename_camera_hint(source),
                  "best": None, "runner_up": None, "keypoint_camera_id": None, "keypoint_scores": {}, "keypoint_agrees": None}
        try:
            frames = _load_frames(Path(source), frame_count)
            fingerprints = load_fingerprints(Path(config.camera_fingerprints_path).parent, config.camera_fingerprints_path)
            if detector is None:
                detector = _create_detector()
            match = identify_from_frames(
                frames, detector, fingerprints,
                min_ncc=config.camera_match_min_ncc,
                min_margin=config.camera_match_min_ncc_margin,
                min_points=config.camera_match_min_points,
            )
            result.update(camera_id=match.camera_id, reason=match.reason, best=match.best, runner_up=match.runner_up,
                          scores=match.scores, keypoint_camera_id=match.keypoint_camera_id, keypoint_agrees=match.keypoint_agrees,
                          keypoint_scores={camera: distance if math.isfinite(distance) else None for camera, distance in match.keypoint_scores.items()})
        except Exception as exc:
            result["error"] = str(exc)
            print(f"[Camera] {source}: {exc}", file=sys.stderr)
        results.append(result)
    return results


def _print_table(results: list[dict]) -> None:
    print("Background NCC (higher is better)")
    print("input\tidentified camera\treason\tper-camera median NCC\tfilename hint")
    for result in results:
        scores = ", ".join(f"{camera}={distance:.2f}" if distance is not None else f"{camera}=n/a"
                           for camera, distance in result["scores"].items())
        print(f"{result['input']}\t{result['camera_id'] or 'none'}\t{result['reason']}\t{scores or '-'}\t{result['hint'] or 'none'}")
    print("\nKeypoint cross-check (lower is better)")
    print("input\tkeypoint camera\tagrees\tper-camera median distance (px)")
    for result in results:
        scores = ", ".join(f"{camera}={distance:.2f}" if distance is not None else f"{camera}=n/a"
                           for camera, distance in result["keypoint_scores"].items())
        print(f"{result['input']}\t{result['keypoint_camera_id'] or 'none'}\t{result['keypoint_agrees']}\t{scores or '-'}")


def main(argv: list[str] | None = None) -> int:
    from backend.pipeline.config import PipelineConfig

    parser = argparse.ArgumentParser(description="Identify cameras by background NCC with a CPU keypoint cross-check.")
    parser.add_argument("inputs", nargs="+", metavar="video-or-image")
    parser.add_argument("--frames", type=_positive_frames, default=PipelineConfig.camera_identify_frames,
                        help="frames sampled evenly within the first 10 seconds (default: %(default)s)")
    parser.add_argument("--json", action="store_true", help="print results as JSON instead of a table")
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 0  # Diagnostic-only CLI, including help and argument errors.
    try:
        with contextlib.redirect_stdout(sys.stderr):
            results = _inspect_inputs(args.inputs, args.frames)
    except Exception as exc:
        # Missing optional runtime dependencies must still produce a diagnostic.
        results = [{"input": source, "camera_id": None, "reason": "error", "scores": {}, "hint": None, "error": str(exc),
                    "best": None, "runner_up": None, "keypoint_camera_id": None, "keypoint_scores": {}, "keypoint_agrees": None}
                   for source in args.inputs]
        print(f"[Camera] {exc}", file=sys.stderr)
    if args.json:
        print(json.dumps(results, indent=2, allow_nan=False))
    else:
        _print_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
