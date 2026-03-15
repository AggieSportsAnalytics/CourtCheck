# Far-Court Detection Fix Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix player detection and bounce detection failures on the far side of the court (opponent's half, top of the video frame).

**Architecture:** Three independent fixes targeting the three root causes. (1) Replace brittle image-space bounds in far-player filtering with court-space projection using the existing homography. (2) Add a geometric bounce detector that works in court-space and is perspective-invariant, running alongside CatBoost and unioning results. (3) Lower the ball detector's minimum circle radius so sub-3-pixel blobs at the far baseline are detected.

**Tech Stack:** Python 3.10, OpenCV, NumPy, CatBoost (existing), homography matrices from existing calibration

---

## Root Causes

### Root Cause 1 — Far player filtered by hardcoded image-space bounds
`player_tracker.py:34-56` (`_far_player_score`) filters far-player candidates by fixed pixel thresholds:
```python
_FAR_MAX_CENTER_Y = 400   # only top 55% of 720px frame
_FAR_MIN_HEIGHT   = 20
_FAR_MAX_CENTER_X = 1080
_FAR_MIN_CENTER_X = 200
```
On courts 2, 4, 6 (different camera angles), the far player can appear outside these ranges. The fix: replace every image-space check with a court-space projection check using the homography (`H_ref`) that is already computed at pipeline startup. The same mechanism used to identify the **near** player (project foot → check `court_y > net_y`) applies symmetrically to the far player (`court_y < net_y`).

### Root Cause 2 — Bounce detector blind to far-court trajectory changes
`bounce_detector.py:22-76` feeds **pixel-space** coordinates to CatBoost. At the far baseline (Y_frame ≈ 120), a ball bouncing at match speed produces `y_diff` values ~3-4× smaller than the same bounce near the baseline (Y_frame ≈ 450). The model's fixed threshold of `0.20` was tuned on near-court bounces and misses the proportionally smaller far-court signals.

The fix: add a **geometric bounce detector** that projects the ball trajectory to court-space (perspective-invariant), finds sign reversals in `dy_court/dt`, and unions those detections with the CatBoost output. No retraining required.

### Root Cause 3 — Hough circle detection misses 1-pixel balls at far baseline
`ball_tracker.py:175`: `minRadius=2` at 360×640 scale. At the far baseline the ball is ~1-2 px wide in the 360×640 feature map. A `minRadius=2` circle can't fit around a 1-px blob, so the detection is skipped. The fix: lower `minRadius` to `1`.

---

## File Map

| File | Change |
|---|---|
| `backend/models/player_tracker.py` | Replace image-space far-player filter with court-space projection |
| `backend/models/ball_tracker.py` | Lower Hough `minRadius` from 2 to 1 |
| `backend/pipeline/run.py` | Add `_geometric_bounce_detector()`, union with CatBoost results |
| `backend/tests/test_far_court_detection.py` | New test file covering all three fixes |

---

## Task 1: Replace image-space far-player filter with court-space projection

**Files:**
- Modify: `backend/models/player_tracker.py:13-56`
- Test: `backend/tests/test_far_court_detection.py`

The current `_far_player_score` rejects a bbox if `cy >= _FAR_MAX_CENTER_Y` (400 px). The fix keeps the same scoring (area) but gates it on a court-space check: project the foot to court-space and verify the player is on the far side of the net.

`choose_and_filter_players` already receives `H_ref`. We pass it through to the per-frame far-player selection.

### Step 1a: Write the failing test

- [ ] **Step 1: Create test file**

```python
# backend/tests/test_far_court_detection.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/Brian/PycharmProjects/CourtCheck
python -m pytest backend/tests/test_far_court_detection.py::TestFarPlayerCourtSpaceFilter -v
```

Expected: FAIL — `ImportError: cannot import name '_far_player_score_with_H'`

### Step 1b: Add the court-space scoring function

- [ ] **Step 3: Add `_far_player_score_with_H` to player_tracker.py**

In `backend/models/player_tracker.py`, add this function directly after `_far_player_score()`:

```python
# Court-space Y range for a valid far player.
# Far side: between far baseline (561) and net (1748), with 100-unit margin each side.
_FAR_COURT_Y_MIN = _COURT_TOP_Y - 100    # 461
_FAR_COURT_Y_MAX = _COURT_NET_Y + 100    # 1848  (slight overlap over net handles mid-court)


def _far_player_score_with_H(bbox, near_id: int, tid: int, H_ref) -> float:
    """
    Court-space version of _far_player_score.

    Projects the candidate bbox's foot position to court-space using H_ref and
    verifies it lands on the far side of the net within court X bounds.
    Falls back to the original image-space check when H_ref is None.

    Score = bbox area (same as _far_player_score) when the candidate passes.
    Returns 0 if the detection belongs to near_id or fails the court-space check.
    """
    if tid == near_id:
        return 0.0

    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w < _FAR_MIN_WIDTH or h < _FAR_MIN_HEIGHT or h > _FAR_MAX_HEIGHT:
        return 0.0

    if H_ref is None:
        # Fallback: original image-space filter
        return _far_player_score(bbox, near_id, tid)

    result = _project_foot(bbox, H_ref)
    if result is None:
        return 0.0

    court_x, court_y = result

    # Must be within court X bounds (with margin for slight sideline overrun)
    if not (_COURT_LEFT_X - _COURT_X_MARGIN <= court_x <= _COURT_RIGHT_X + _COURT_X_MARGIN):
        return 0.0

    # Must be on far side of net
    if not (_FAR_COURT_Y_MIN <= court_y <= _FAR_COURT_Y_MAX):
        return 0.0

    return w * h
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest backend/tests/test_far_court_detection.py::TestFarPlayerCourtSpaceFilter -v
```

Expected: PASS (or skip if calibration file absent — acceptable in CI without weights)

### Step 1c: Wire the new function into choose_and_filter_players

- [ ] **Step 5: Update choose_and_filter_players to use H_ref for far player scoring**

In `backend/models/player_tracker.py`, in `choose_and_filter_players()`, replace lines 176-186 (the `_far_player_score` loop):

```python
            # Pick best far-court candidate in this frame (largest qualifying bbox)
            best_tid, best_score = None, 0.0
            for tid, bbox in frame.items():
                score = _far_player_score(bbox, near_id, tid)
                if score > best_score:
                    best_score = score
                    best_tid = tid
```

with:

```python
            # Pick best far-court candidate using court-space projection.
            # _far_player_score_with_H falls back to image-space if H_ref is None.
            best_tid, best_score = None, 0.0
            for tid, bbox in frame.items():
                score = _far_player_score_with_H(bbox, near_id, tid, H_ref)
                if score > best_score:
                    best_score = score
                    best_tid = tid
```

- [ ] **Step 6: Run all player tracking tests**

```bash
python -m pytest backend/tests/test_player_tracking.py backend/tests/test_far_court_detection.py -v --timeout=60
```

Expected: all pass (player tracking tests use calibration so H_ref will be available)

- [ ] **Step 7: Commit**

```bash
git add backend/models/player_tracker.py backend/tests/test_far_court_detection.py
git commit -m "fix: replace image-space far-player filter with court-space projection"
```

---

## Task 2: Geometric bounce detector for far-court

**Files:**
- Modify: `backend/pipeline/run.py` — add `_geometric_bounce_detector()`, union with CatBoost
- Test: `backend/tests/test_far_court_detection.py`

The CatBoost model sees pixel coordinates and its `y_diff` features are proportionally smaller for far-court bounces. The geometric detector works in court-space (perspective-invariant) and finds bounces from the physical sign reversal of `dy_court/dt`.

**How it works:**
1. Project each non-None ball position to court-space using `homography_matrices[i]`
2. Compute `dy_court = y_court[i+1] - y_court[i]` for each frame
3. A bounce is a frame where `dy_court` changes sign (+ → − near side, − → + far side)
4. Filter: minimum vertical displacement `|y_court[i] - y_court[i-N]| >= MIN_COURT_DISPLACEMENT`
5. Filter: minimum frames between bounces (debounce)
6. Filter: ball must be within court boundaries

**Constants (in court-space units from `CourtReference`):**
- `MIN_COURT_DISP = 80` — minimum y-displacement in court units between sign-change neighbors (~5% of court length; very small bounce = skip)
- `MIN_BOUNCE_GAP = 8` — minimum frames between bounces (avoid double-counting)

### Step 2a: Write failing test

- [ ] **Step 1: Add geometric bounce detector tests**

```python
# backend/tests/test_far_court_detection.py  (append to existing file)

class TestGeometricBounceDetector:
    def _make_trajectory(self, y_values: list[float | None]) -> tuple:
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
        assert len(bounces) <= 1, f"Expected ≤1 bounce with tight gap, got {bounces}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest backend/tests/test_far_court_detection.py::TestGeometricBounceDetector -v
```

Expected: FAIL — `ImportError: cannot import name '_geometric_bounce_detector'`

### Step 2b: Implement the geometric bounce detector

- [ ] **Step 3: Add `_geometric_bounce_detector` to run.py**

Add this function in `backend/pipeline/run.py` after `_interpolate_player_detections()` (or after `_detect_court_once()` if Task 3 of the performance plan is implemented):

```python
def _geometric_bounce_detector(
    ball_track: list,
    homography_matrices: list,
    min_court_disp: float = 80.0,
    min_gap: int = 8,
) -> set:
    """
    Perspective-invariant bounce detector using court-space geometry.

    Projects ball positions to court-space via homography and detects
    frames where the ball's y velocity changes sign (direction reversal = bounce).
    Works on both near- and far-court equally because court-space coordinates
    remove the perspective scaling that makes far-court y_diff smaller.

    Args:
        ball_track:         List of (x, y) tuples or None, one per frame.
        homography_matrices: Per-frame H_ref matrices (frame → court-space).
                             None entries use the last valid matrix.
        min_court_disp:     Minimum total y-displacement (in court units) across
                             the sign-change window to count as a real bounce.
                             Prevents noise-triggered false positives.
        min_gap:            Minimum frames between two accepted bounces.

    Returns:
        Set of frame indices where a bounce was detected.
    """
    n = len(ball_track)
    if n == 0:
        return set()

    # Project all ball positions to court-space
    court_y: list[float | None] = [None] * n
    last_H = None
    for i, pos in enumerate(ball_track):
        H = homography_matrices[i] if i < len(homography_matrices) else None
        if H is not None:
            last_H = H
        if pos is None or last_H is None:
            continue
        bx, by = pos
        if bx is None or by is None:
            continue
        pt = np.array([[[float(bx), float(by)]]], dtype=np.float32)
        try:
            mapped = cv2.perspectiveTransform(pt, last_H)
            court_y[i] = float(mapped[0, 0, 1])
        except cv2.error:
            pass

    # Compute smoothed dy: dy[i] = court_y[i+1] - court_y[i] (skip Nones)
    # Build a list of (frame_idx, court_y_val) for non-None positions only
    valid = [(i, cy) for i, cy in enumerate(court_y) if cy is not None]
    if len(valid) < 4:
        return set()

    # Find sign reversals in the dy sequence over valid frames
    bounces: set[int] = set()
    last_accepted = -min_gap

    for j in range(1, len(valid) - 1):
        i_prev, cy_prev = valid[j - 1]
        i_curr, cy_curr = valid[j]
        i_next, cy_next = valid[j + 1]

        dy_before = cy_curr - cy_prev
        dy_after  = cy_next - cy_curr

        if dy_before == 0 or dy_after == 0:
            continue

        # Sign reversal: direction of y changed
        if (dy_before > 0) == (dy_after > 0):
            continue

        # Require minimum displacement across the local window
        # Use a ±3 frame window for robustness
        lo = max(0, j - 3)
        hi = min(len(valid) - 1, j + 3)
        y_vals_window = [valid[k][1] for k in range(lo, hi + 1)]
        disp = max(y_vals_window) - min(y_vals_window)
        if disp < min_court_disp:
            continue

        # Enforce minimum gap
        if i_curr - last_accepted < min_gap:
            continue

        bounces.add(i_curr)
        last_accepted = i_curr

    return bounces
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest backend/tests/test_far_court_detection.py::TestGeometricBounceDetector -v
```

Expected: PASS

### Step 2c: Union geometric bounces with CatBoost bounces

- [ ] **Step 5: Write integration test**

```python
# backend/tests/test_far_court_detection.py  (append)

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
```

- [ ] **Step 6: Run test to verify it passes** (pure set logic, no imports needed)

```bash
python -m pytest backend/tests/test_far_court_detection.py::TestBounceDetectorUnion -v
```

Expected: PASS

- [ ] **Step 7: Wire geometric bounces into run_pipeline()**

In `backend/pipeline/run.py`, find the bounce detection section (around line 440). After `bounces_all` is computed from CatBoost, add the geometric union:

```python
        print(f"[Bounce] detected {len(bounces_all)} bounces (screen-space coords)")

        # ---------- Geometric bounce detector (far-court supplement) ----------
        # CatBoost operates in pixel-space and underdetects far-court bounces because
        # the y_diff features are ~3x smaller there due to perspective foreshortening.
        # The geometric detector projects to court-space (perspective-invariant) and
        # catches those missed sign reversals.
        if calibrated_H_ref is not None:
            # Use the pre-filled homography matrices (all identical for fixed camera)
            geo_bounces = _geometric_bounce_detector(
                ball_track,
                homography_matrices,   # pre-filled from calibration in the perf plan
                min_court_disp=80.0,
                min_gap=8,
            )
            new_geo = geo_bounces - bounces_all
            if new_geo:
                print(f"[Bounce] geometric detector added {len(new_geo)} far-court bounces: {sorted(new_geo)[:10]}")
            bounces_all = bounces_all | geo_bounces
            print(f"[Bounce] total after geometric union: {len(bounces_all)}")
```

**Important:** `homography_matrices` must be populated before this point. In the current pipeline, `homography_matrices` is filled during Pass 2. If running the geometric detector after Pass 1 (before Pass 2), use a separate pre-fill from calibration:

```python
        # Pre-fill homography for geometric bounce detector (uses calibrated H_ref)
        _H_for_geo = [calibrated_H_ref] * total_frames if calibrated_H_ref is not None else [None] * total_frames
        if calibrated_H_ref is not None:
            geo_bounces = _geometric_bounce_detector(
                ball_track, _H_for_geo, min_court_disp=80.0, min_gap=8
            )
            new_geo = geo_bounces - bounces_all
            if new_geo:
                print(f"[Bounce] geometric detector added {len(new_geo)} far-court bounces: {sorted(new_geo)[:10]}")
            bounces_all = bounces_all | geo_bounces
            print(f"[Bounce] total after geometric union: {len(bounces_all)}")
```

- [ ] **Step 8: Run all detection tests**

```bash
python -m pytest backend/tests/test_far_court_detection.py -v --timeout=60
```

Expected: all pass

- [ ] **Step 9: Commit**

```bash
git add backend/pipeline/run.py backend/tests/test_far_court_detection.py
git commit -m "fix: add geometric bounce detector for far-court coverage, union with CatBoost"
```

---

## Task 3: Lower Hough minRadius to detect 1-pixel balls at far baseline

**Files:**
- Modify: `backend/models/ball_tracker.py:175`
- Test: `backend/tests/test_far_court_detection.py`

At the far baseline, BallTrackerNet outputs a single activated pixel in its 360×640 feature map. The current `minRadius=2` Hough circle requires at least a 5-pixel-wide blob (diameter ≥ 5), which a 1-pixel activation can't satisfy. Lowering to `minRadius=1` allows detection of single-pixel activations.

The `max_dist=80` proximity filter (at 1280×720 scale = 40 at 640×360) already prevents false positives from unrelated activations.

### Step 3a: Write failing test

- [ ] **Step 1: Add Hough radius test**

```python
# backend/tests/test_far_court_detection.py  (append)

class TestBallDetectorFarCourt:
    def test_hough_min_radius_is_1(self):
        """
        BallDetector's Hough circle detection should use minRadius=1 so that
        single-pixel activations at the far baseline (where the ball is 1-2px wide
        in the 360x640 feature map) are detected.
        """
        import inspect
        from backend.models.ball_tracker import BallDetector
        source = inspect.getsource(BallDetector.postprocess)
        assert "minRadius=1" in source, (
            "BallDetector.postprocess should use minRadius=1 for far-court detection. "
            "Found: " + source[source.find("minRadius"):source.find("minRadius")+20]
        )

    def test_hough_detects_single_pixel_blob(self):
        """
        A feature map with a single white pixel should produce a Hough circle
        detection when minRadius=1 (and fail with minRadius=2).
        """
        import cv2
        import numpy as np

        feature_map = np.zeros((360, 640), dtype=np.uint8)
        # Place single bright pixel at far-court position (upper frame)
        feature_map[50, 320] = 255

        circles_tight = cv2.HoughCircles(
            feature_map, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
            param1=50, param2=2, minRadius=1, maxRadius=7
        )
        circles_strict = cv2.HoughCircles(
            feature_map, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
            param1=50, param2=2, minRadius=2, maxRadius=7
        )
        assert circles_tight is not None, (
            "minRadius=1 should detect a single bright pixel"
        )
        # minRadius=2 may or may not detect — just verify minRadius=1 works
```

- [ ] **Step 2: Run tests to verify the radius test fails**

```bash
python -m pytest backend/tests/test_far_court_detection.py::TestBallDetectorFarCourt::test_hough_min_radius_is_1 -v
```

Expected: FAIL — `AssertionError: BallDetector.postprocess should use minRadius=1`

- [ ] **Step 3: Change minRadius in ball_tracker.py**

In `backend/models/ball_tracker.py` lines 168-177, change:

```python
        circles = cv2.HoughCircles(
            heatmap,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=50,
            param2=2,
            minRadius=2,
            maxRadius=7,
        )
```

to:

```python
        circles = cv2.HoughCircles(
            heatmap,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=50,
            param2=2,
            minRadius=1,   # lowered from 2: far-baseline balls are 1-2px in 360x640 map
            maxRadius=7,
        )
```

- [ ] **Step 4: Run all ball detector tests**

```bash
python -m pytest backend/tests/test_far_court_detection.py::TestBallDetectorFarCourt -v
```

Expected: both pass

- [ ] **Step 5: Commit**

```bash
git add backend/models/ball_tracker.py backend/tests/test_far_court_detection.py
git commit -m "fix: lower Hough minRadius from 2 to 1 to detect 1px ball at far baseline"
```

---

## Final Verification

- [ ] **Run full test suite**

```bash
python -m pytest backend/tests/ -v --timeout=120
```

Expected: all tests pass

- [ ] **Smoke test on failing clip** (if a test video is available locally)

```bash
python -m backend.pipeline.run --video backend/calibration_frames/videos/test2.mov --output /tmp/test_output.mp4
```

Check Modal logs for:
- `[PlayerTracker] Far player: detected in X/Y frames` — X should be > 50% of Y
- `[Bounce] geometric detector added N far-court bounces` — N > 0 on a match with rallies
- Far court overlay: bounce dots appearing near far baseline in minimap
- Far player bbox: visible in top portion of frame during far-baseline rallies

- [ ] **Deploy to Modal**

```bash
modal deploy backend/app.py
```

Confirm in Modal function logs on next video:
- Far player bbox consistently drawn in upper portion of frame
- Bounce heatmap has coverage in the far-court half (not just near-court)
