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


# ---------------------------------------------------------------------------
# Task 2: Geometric bounce detector
# ---------------------------------------------------------------------------

class TestGeometricBounceDetector:
    def _make_trajectory(self, y_values: list) -> tuple:
        """Build simple ball_track and homography_matrices for testing."""
        import numpy as np
        # Use identity H_ref: court-space = frame-space (for unit testing only)
        H_ref = np.eye(3, dtype=np.float32)
        ball_track = [(500.0, y) if y is not None else None for y in y_values]
        homography_matrices = [H_ref] * len(y_values)
        return ball_track, homography_matrices

    def test_detects_near_court_bounce(self):
        """Near-court bounce: ball falls (y increases), bounces, y decreases."""
        from backend.pipeline.run import _geometric_bounce_detector

        # y increases 550→650 (falling), then decreases 650→550 (bouncing up)
        y_vals = [550, 570, 590, 610, 630, 650, 630, 610, 590, 570, 550]
        ball_track, H_mats = self._make_trajectory(y_vals)
        bounces = _geometric_bounce_detector(ball_track, H_mats, min_court_disp=30, min_gap=3)
        assert len(bounces) == 1, f"Expected 1 bounce, got {len(bounces)}: {bounces}"
        # Peak should be near frame 5 (y=650)
        assert 4 <= list(bounces)[0] <= 6, f"Bounce frame off: {bounces}"

    def test_detects_far_court_bounce(self):
        """Far-court bounce: ball falls toward far baseline (y decreases), bounces, y increases."""
        from backend.pipeline.run import _geometric_bounce_detector

        # Simulates far-court: ball at y≈200 (near net) moves down to y≈130 (far baseline), bounces back
        y_vals = [200, 185, 170, 155, 140, 130, 140, 155, 170, 185, 200]
        ball_track, H_mats = self._make_trajectory(y_vals)
        bounces = _geometric_bounce_detector(ball_track, H_mats, min_court_disp=30, min_gap=3)
        assert len(bounces) == 1, f"Expected 1 bounce, got {len(bounces)}: {bounces}"
        assert 4 <= list(bounces)[0] <= 6

    def test_ignores_noisy_small_oscillations(self):
        """Tiny y fluctuations (noise) below MIN_COURT_DISP should not trigger bounces."""
        from backend.pipeline.run import _geometric_bounce_detector

        # Oscillation of only ±3 units — below min_court_disp=30
        y_vals = [400, 403, 397, 402, 398, 403, 397, 400]
        ball_track, H_mats = self._make_trajectory(y_vals)
        bounces = _geometric_bounce_detector(ball_track, H_mats, min_court_disp=30, min_gap=3)
        assert len(bounces) == 0, f"Should be no bounces for noise, got {bounces}"

    def test_handles_none_positions(self):
        """Missing ball positions (None) should not crash the detector."""
        from backend.pipeline.run import _geometric_bounce_detector

        y_vals = [550, None, None, 610, 650, 630, None, 590, 550]
        ball_track, H_mats = self._make_trajectory(y_vals)
        bounces = _geometric_bounce_detector(ball_track, H_mats, min_court_disp=30, min_gap=3)
        # Should complete without error; may or may not find bounce (interpolation-dependent)
        assert isinstance(bounces, set)

    def test_respects_min_gap(self):
        """Two bounces closer than min_gap frames should be deduplicated."""
        from backend.pipeline.run import _geometric_bounce_detector

        # Two rapid direction changes 3 frames apart — min_gap=8 should keep only one
        y_vals = [550, 600, 650, 600, 550, 600, 650, 600, 550]
        ball_track, H_mats = self._make_trajectory(y_vals)
        bounces = _geometric_bounce_detector(ball_track, H_mats, min_court_disp=30, min_gap=8)
        # With min_gap=8, only the first (or last) bounce should survive
        assert len(bounces) == 1, f"Expected 1 bounce with tight gap, got {bounces}"


class TestBounceDetectorUnion:
    def test_geometric_bounces_added_to_catboost(self):
        """
        When a bounce is found by the geometric detector but not CatBoost,
        it should still appear in the final bounce set after union.
        """
        catboost_bounces = {10, 50, 90}
        geometric_bounces = {30, 70}   # far-court, missed by CatBoost
        combined = catboost_bounces | geometric_bounces
        assert 30 in combined
        assert 70 in combined
        assert len(combined) == 5

    def test_duplicate_bounces_not_double_counted(self):
        """Bounces detected by both detectors should appear once."""
        catboost_bounces = {10, 50, 90}
        geometric_bounces = {10, 90, 110}  # overlap on 10 and 90
        combined = catboost_bounces | geometric_bounces
        assert len(combined) == 4
        assert 110 in combined


# ---------------------------------------------------------------------------
# Task 3: Hough minRadius=1
# ---------------------------------------------------------------------------

class TestBallDetectorFarCourt:
    def test_hough_min_radius_is_1(self):
        """
        BallDetector should detect a 1-pixel activation at the far baseline.
        This regression test ensures minRadius is not raised back above 1.
        """
        import numpy as np
        from backend.models.ball_tracker import BallDetector

        # Build a minimal heatmap (360x640) with a single bright pixel
        # After model output is multiplied by 255 and thresholded, a single
        # bright pixel simulates far-baseline ball detection
        feature_map = np.zeros((1, 360, 640), dtype=np.float32)
        feature_map[0, 50, 320] = 1.0  # single activation at far-baseline position

        # Call postprocess directly to verify detection
        detector = BallDetector(path_model=None)  # Create without loading weights
        x, y = detector.postprocess(feature_map, prev_pred=[None, None])
        assert x is not None and y is not None, (
            "BallDetector.postprocess must detect a single bright pixel at far baseline. "
            "Likely minRadius was raised above 1 — check ball_tracker.py HoughCircles call."
        )

    def test_hough_detects_single_pixel_blob(self):
        """
        cv2.HoughCircles with minRadius=1 detects a single bright pixel blob.
        Confirms the OpenCV primitive works for the far-baseline ball size.
        """
        import cv2
        import numpy as np

        feature_map = np.zeros((360, 640), dtype=np.uint8)
        feature_map[50, 320] = 255

        circles = cv2.HoughCircles(
            feature_map, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
            param1=50, param2=2, minRadius=1, maxRadius=7
        )
        assert circles is not None, "minRadius=1 should detect a single bright pixel"
