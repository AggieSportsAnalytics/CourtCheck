"""Import each model only when requested, including the standalone court CLI."""
from importlib import import_module


_EXPORTS = {
    "BallDetector": "ball_tracker",
    "CourtLineDetector": "court_line_detector",
    "PlayerTracker": "player_tracker",
    "BounceDetector": "bounce_detector",
    "ActionRecognition": "stroke_detector",
    "PoseStrokeClassifier": "stroke_classifier_tcn",
    "TrajectoryRectifier": "trajectory_rectifier",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
