"""Load vision exports on demand so keypoint matching only requires numpy."""
from importlib import import_module


_EXPORTS = {
    "homography": ("HomographyEstimator",),
    "court_reference": ("CourtReference",),
    "drawing": ("draw_ball_trace", "draw_court_keypoints_and_lines", "draw_minimap_ball_and_bounces",
                "draw_minimap_players", "draw_player_bboxes", "draw_stroke_labels"),
    "postprocess": ("detect_shot_frames",),
    "swing_detector": ("SwingDetector", "extract_pose_sequence"),
    "calibration": ("load_calibration", "save_calibration", "visualize_keypoints"),
}
__all__ = [name for names in _EXPORTS.values() for name in names]


def __getattr__(name):
    for module, names in _EXPORTS.items():
        if name in names:
            value = getattr(import_module(f"{__name__}.{module}"), name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
