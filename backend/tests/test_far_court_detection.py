"""
Tests for the far-court detection fixes:
  Task 1 — court-space far-player filter
  Task 2 — geometric bounce detector
  Task 3 — Hough minRadius=1
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Task 1 helpers
# ---------------------------------------------------------------------------

def _make_H_ref():
    """Return the real uc_davis_court1 H_ref from calibration."""
    from backend.vision.calibration import load_calibration
    import os
    cal_path = os.path.join(
        os.path.dirname(__file__), "..", "calibration_frames", "court_calibration.json"
    )
    H_ref, _, _ = load_calibration(cal_path, "uc_davis_court1")
    return H_ref  # may be None if file absent


class TestFarPlayerCourtSpaceFilter:
    def test_far_player_accepted_when_in_far_court(self):
        """A bbox whose foot projects to far-court should pass the filter."""
        from backend.models.player_tracker import _far_player_score_with_H

        H_ref = _make_H_ref()
        if H_ref is None:
            pytest.skip("Calibration file not available")

        # Far player bbox: small, in upper frame (~cy=120, h=115)
        far_bbox = [503.0, 62.0, 524.0, 177.0]  # typical far player on uc_davis_court1
        near_id = 99  # some other id

        score = _far_player_score_with_H(far_bbox, near_id=near_id, tid=1, H_ref=H_ref)
        assert score > 0, (
            f"Far player at {far_bbox} should score > 0 with court-space filter, got {score}"
        )

    def test_spectator_rejected_when_outside_court(self):
        """A bbox whose foot projects outside the court should be rejected."""
        from backend.models.player_tracker import _far_player_score_with_H

        H_ref = _make_H_ref()
        if H_ref is None:
            pytest.skip("Calibration file not available")

        # Spectator bbox: near sideline, upper frame — outside court x bounds
        spectator_bbox = [50.0, 80.0, 100.0, 200.0]
        near_id = 99

        score = _far_player_score_with_H(spectator_bbox, near_id=near_id, tid=2, H_ref=H_ref)
        assert score == 0.0, (
            f"Spectator bbox {spectator_bbox} should score 0, got {score}"
        )

    def test_near_player_id_always_excluded(self):
        """A bbox belonging to near_id should always score 0."""
        from backend.models.player_tracker import _far_player_score_with_H

        H_ref = _make_H_ref()
        if H_ref is None:
            pytest.skip("Calibration file not available")

        far_bbox = [503.0, 62.0, 524.0, 177.0]
        score = _far_player_score_with_H(far_bbox, near_id=1, tid=1, H_ref=H_ref)
        assert score == 0.0
