"""Rally state machine.

Reconstructs structured rallies from the unordered streams of bounces +
swing events + in-bounds set produced earlier in the pipeline. Output
shape and segmentation rules are documented in
`docs/RALLY_TRACKING_SPEC.md`.
"""
from __future__ import annotations

import numpy as np
import cv2

# 4-second gap between consecutive events starts a new rally.
# Shared with `calculate_rally_count` in run.py — keep in one place.
RALLY_GAP_SECONDS = 4

# Plausibility gate copied from build_error_summary. Bounces outside this
# box are detector noise (projection blew up past the stands).
_PLAUSIBLE_SVG_X = (-3.0, 30.0)
_PLAUSIBLE_SVG_Y = (-6.0, 84.0)

# Net zone — bounces at y in [37, 41] are side-ambiguous and labeled "net".
_NET_Y_MIN = 37.0
_NET_Y_MAX = 41.0

# In-bounds court box (matches count_in_out_bounces semantics).
_COURT_X_MIN = 1.0
_COURT_X_MAX = 26.0
_COURT_Y_MIN = 0.0
_COURT_Y_MAX = 78.0

# Stroke label normalization (raw classifier label -> frontend key).
_STROKE_MAP = {
    "Forehand": "forehand",
    "Backhand": "backhand",
    "Serve/Overhead": "serve",
}

# Window (seconds) after a bounce in which the receiver could swing
# without it counting as "no follow-up".
_FOLLOWUP_WINDOW_S = 1.5
