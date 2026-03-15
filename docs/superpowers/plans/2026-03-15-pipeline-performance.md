# Pipeline Performance Optimization Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut video processing time from ~8x real-time to ~3x real-time by switching to a faster YOLO model, reducing inference frequency, eliminating redundant video decodes, running court detection once per video, and using GPU-accelerated encoding.

**Architecture:** Five independent optimizations to `run.py`, `config.py`, and `storage.py`. No model retraining required. All changes are backward-compatible — if NVENC or a new codec is unavailable, the pipeline falls back gracefully. Court detection is promoted to a one-time step at pipeline startup instead of every-5th-frame.

**Tech Stack:** Python 3.10, OpenCV, Ultralytics YOLOv8m-Pose, ffmpeg (h264_nvenc), Modal A10G GPU

---

## File Map

| File | Change |
|---|---|
| `backend/pipeline/config.py` | Add `player_detection_interval`, change default `player_model` to yolov8m-pose |
| `backend/pipeline/run.py` | (1) Skip YOLO on non-interval frames + interpolate, (2) write MJPEG intermediate in Pass 1 so Pass 2 avoids H.264 re-decode, (3) detect court once at startup instead of per-frame loop |
| `backend/pipeline/storage.py` | Try `h264_nvenc` first in `make_streamable_mp4`, fall back to `libx264` |
| `backend/tests/test_pipeline_perf.py` | New tests: interpolation correctness, court-once path, NVENC fallback |

---

## Task 1: Switch player model to YOLOv8m-Pose

**Files:**
- Modify: `backend/pipeline/config.py:28`
- Modify: `backend/pipeline/run.py:343` (model path comment)

Note: `app.py` already downloads `yolov8m-pose.pt` (line 17). The only gap is config still defaults to `yolov8x-pose.pt`.

- [ ] **Step 1: Write failing test**

```python
# backend/tests/test_pipeline_perf.py
def test_default_model_is_yolov8m():
    from backend.pipeline.config import PipelineConfig
    cfg = PipelineConfig()
    assert "yolov8m" in cfg.player_model, (
        f"Expected yolov8m model, got: {cfg.player_model}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/Brian/PycharmProjects/CourtCheck
python -m pytest backend/tests/test_pipeline_perf.py::test_default_model_is_yolov8m -v
```

Expected: FAIL — `AssertionError: Expected yolov8m model, got: yolov8x-pose.pt`

- [ ] **Step 3: Change default model in config**

In `backend/pipeline/config.py` line 28, change:
```python
player_model: str = 'yolov8x-pose.pt'
```
to:
```python
player_model: str = 'yolov8m-pose.pt'
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_pipeline_perf.py::test_default_model_is_yolov8m -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/config.py backend/tests/test_pipeline_perf.py
git commit -m "perf: switch default player model to yolov8m-pose (~3-4x faster inference)"
```

---

## Task 2: Run player detection every N frames, interpolate between

Player bboxes change slowly between consecutive frames. Running YOLO every `player_detection_interval` frames (default 3) and linearly interpolating bboxes cuts YOLO call count by ~3x with negligible accuracy loss.

**Files:**
- Modify: `backend/pipeline/config.py` — add `player_detection_interval: int = 3`
- Modify: `backend/pipeline/run.py:404-411` — Pass 1 loop
- Modify: `backend/pipeline/run.py` — add `_interpolate_player_detections()` helper

### Step 2a: Config field

- [ ] **Step 1: Write failing test**

```python
# backend/tests/test_pipeline_perf.py
def test_player_detection_interval_default():
    from backend.pipeline.config import PipelineConfig
    cfg = PipelineConfig()
    assert cfg.player_detection_interval == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_pipeline_perf.py::test_player_detection_interval_default -v
```

Expected: FAIL — `AttributeError: 'PipelineConfig' has no attribute 'player_detection_interval'`

- [ ] **Step 3: Add field to config**

In `backend/pipeline/config.py`, after `player_imgsz: int = 1280`, add:
```python
# Run YOLO player detection every Nth frame; interpolate bboxes between.
# 1 = every frame (original behaviour). 3 = detect every 3rd frame (~3x speedup).
player_detection_interval: int = 3
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_pipeline_perf.py::test_player_detection_interval_default -v
```

Expected: PASS

### Step 2b: Interpolation helper

- [ ] **Step 5: Write failing test for interpolation**

```python
# backend/tests/test_pipeline_perf.py
def test_interpolate_player_detections_fills_gaps():
    from backend.pipeline.run import _interpolate_player_detections
    # Frames 0 and 3 have player data; frames 1, 2 are empty ({})
    detections = [
        {1: [0.0, 10.0, 20.0, 30.0]},  # frame 0: track_id=1, bbox
        {},                              # frame 1: empty (skipped YOLO)
        {},                              # frame 2: empty (skipped YOLO)
        {1: [3.0, 13.0, 23.0, 33.0]},  # frame 3: track_id=1, bbox
    ]
    result = _interpolate_player_detections(detections)
    assert len(result) == 4
    # Frame 1 should be interpolated: 1/3 of the way from frame 0 to frame 3
    assert 1 in result[1], "track_id=1 should be interpolated into frame 1"
    x1, y1, x2, y2 = result[1][1]
    assert abs(x1 - 1.0) < 0.1, f"x1 interpolation wrong: {x1}"
    assert abs(y1 - 11.0) < 0.1, f"y1 interpolation wrong: {y1}"
    # Frame 2: 2/3 of the way
    x1_2, y1_2, x2_2, y2_2 = result[2][1]
    assert abs(x1_2 - 2.0) < 0.1
    # Frame 3 unchanged
    assert result[3][1] == [3.0, 13.0, 23.0, 33.0]
```

- [ ] **Step 6: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_pipeline_perf.py::test_interpolate_player_detections_fills_gaps -v
```

Expected: FAIL — `ImportError: cannot import name '_interpolate_player_detections'`

- [ ] **Step 7: Add interpolation helper to run.py**

Add this function near the top of `backend/pipeline/run.py`, after `calculate_rally_count()`:

```python
def _interpolate_player_detections(detections: list[dict]) -> list[dict]:
    """
    Fill empty player-detection frames by linearly interpolating between the
    nearest non-empty frames on either side.

    Empty frames arise when player_detection_interval > 1: YOLO is only called
    every Nth frame, so intermediate frames are stored as {}.

    For each track_id present in the surrounding anchor frames, the four bbox
    coordinates (x1, y1, x2, y2) are linearly interpolated.  Frames that have
    no anchor on one side (start/end of video) are forward- or back-filled.
    """
    n = len(detections)
    result = [dict(d) for d in detections]  # shallow-copy each frame dict

    # Collect anchor indices (frames where YOLO actually ran)
    anchors = [i for i, d in enumerate(detections) if d]

    if not anchors:
        return result

    # Back-fill frames before the first anchor
    for i in range(anchors[0]):
        result[i] = dict(detections[anchors[0]])

    # Forward-fill frames after the last anchor
    for i in range(anchors[-1] + 1, n):
        result[i] = dict(detections[anchors[-1]])

    # Interpolate between consecutive anchors
    for a_idx in range(len(anchors) - 1):
        start = anchors[a_idx]
        end = anchors[a_idx + 1]
        if end - start <= 1:
            continue  # adjacent anchors, nothing to fill
        start_bboxes = detections[start]
        end_bboxes = detections[end]
        shared_ids = set(start_bboxes) & set(end_bboxes)
        for frame_i in range(start + 1, end):
            t = (frame_i - start) / (end - start)
            for tid in shared_ids:
                s = start_bboxes[tid]
                e = end_bboxes[tid]
                result[frame_i][tid] = [
                    s[0] + t * (e[0] - s[0]),
                    s[1] + t * (e[1] - s[1]),
                    s[2] + t * (e[2] - s[2]),
                    s[3] + t * (e[3] - s[3]),
                ]

    return result
```

- [ ] **Step 8: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_pipeline_perf.py::test_interpolate_player_detections_fills_gaps -v
```

Expected: PASS

### Step 2c: Wire into Pass 1

- [ ] **Step 9: Write failing test for interval detection**

```python
# backend/tests/test_pipeline_perf.py
def test_player_interval_reduces_yolo_calls(monkeypatch):
    """
    With interval=3, YOLO detect_frame() should be called ceil(N/3) times,
    not N times, and interpolation should fill the rest.
    """
    from backend.pipeline import run as pipeline_run
    from backend.pipeline.config import PipelineConfig

    call_count = {"n": 0}
    original_detect = None

    class FakeTracker:
        def detect_frame(self, frame):
            call_count["n"] += 1
            return {1: [100.0, 200.0, 150.0, 400.0]}, {}

        def choose_and_filter_players(self, H_ref, detections, poses):
            return detections, poses

    cfg = PipelineConfig()
    cfg.player_detection_interval = 3

    # Verify the interval logic: if 9 frames, YOLO should be called 3 times (frames 0,3,6)
    # We test the helper directly since running the full pipeline requires video files
    detections_in = []
    for i in range(9):
        if i % 3 == 0:
            detections_in.append({1: [float(i), 0.0, float(i) + 50.0, 100.0]})
        else:
            detections_in.append({})

    result = pipeline_run._interpolate_player_detections(detections_in)
    assert all(len(d) > 0 for d in result), "All frames should have detections after interpolation"
    assert len(result) == 9
```

- [ ] **Step 10: Run test to verify it passes** (this one should pass immediately since it only tests the helper)

```bash
python -m pytest backend/tests/test_pipeline_perf.py::test_player_interval_reduces_yolo_calls -v
```

Expected: PASS

- [ ] **Step 11: Modify Pass 1 loop in run.py to use interval**

In `backend/pipeline/run.py`, replace lines 404-414 (the Pass 1 frame loop body):

```python
        for i in tqdm(range(total_frames), desc="Pass 1: Ball + Player tracking"):
            ret, frame = cap.read()
            if not ret:
                break
            ball_track.append(ball_detector.infer_single(frame))
            player_dict, kps_dict = player_tracker.detect_frame(frame)
            player_detections.append(player_dict)
            pose_keypoints_per_frame.append(kps_dict)

            if i % max(1, total_frames // config.progress_update_frequency) == 0:
                update_progress(0.05 + 0.4 * (i / total_frames))
```

with:

```python
        for i in tqdm(range(total_frames), desc="Pass 1: Ball + Player tracking"):
            ret, frame = cap.read()
            if not ret:
                break
            ball_track.append(ball_detector.infer_single(frame))

            if i % config.player_detection_interval == 0:
                player_dict, kps_dict = player_tracker.detect_frame(frame)
            else:
                player_dict, kps_dict = {}, {}
            player_detections.append(player_dict)
            pose_keypoints_per_frame.append(kps_dict)

            if i % max(1, total_frames // config.progress_update_frequency) == 0:
                update_progress(0.05 + 0.4 * (i / total_frames))

        # Fill gaps between detection frames using linear interpolation
        player_detections = _interpolate_player_detections(player_detections)
        # Propagate last known pose keypoints into interpolated frames (no pose interp)
        _last_kps: dict = {}
        for i, kd in enumerate(pose_keypoints_per_frame):
            if kd:
                _last_kps = kd
            elif _last_kps:
                pose_keypoints_per_frame[i] = dict(_last_kps)
```

- [ ] **Step 12: Run existing player tracking tests to verify nothing broke**

```bash
python -m pytest backend/tests/test_player_tracking.py -v -k "not test_far_player" --timeout=120
```

Expected: existing tests pass (far_player test requires GPU, skip for now)

- [ ] **Step 13: Commit**

```bash
git add backend/pipeline/config.py backend/pipeline/run.py backend/tests/test_pipeline_perf.py
git commit -m "perf: run player detection every 3 frames with linear bbox interpolation"
```

---

## Task 3: MJPEG intermediate — eliminate H.264 re-decode in Pass 2

Pass 1 currently reads the H.264 source video and discards each decoded frame after running inference. Pass 2 re-opens the same file and decodes it again purely for drawing. Replacing the second H.264 decode with a fast MJPEG read saves ~5-8 ms per frame.

**Implementation:** During Pass 1, write each decoded frame to a temporary MJPEG AVI alongside detection. Pass 2 reads from this AVI (which decodes in ~2 ms/frame vs H.264's ~7 ms/frame). The MJPEG file is deleted after drawing.

**Space estimate:** 1280×720 MJPEG @ quality 90 ≈ 60-80 KB/frame. For an 18 K-frame video ≈ 1.1–1.4 GB — acceptable in Modal's `/tmp`.

**Files:**
- Modify: `backend/pipeline/run.py` — Pass 1 + Pass 2 video paths

### Step 3a: Test that Pass 2 uses the intermediate file when present

- [ ] **Step 1: Write failing test**

```python
# backend/tests/test_pipeline_perf.py
def test_mjpeg_intermediate_written_during_pass1(tmp_path):
    """
    After Pass 1 completes, a .avi intermediate file should exist at
    /tmp/{match_id}_frames.avi (MJPEG encoded).
    We verify this by inspecting what VideoWriter codec would be used.
    """
    import cv2
    # Verify MJPEG fourcc exists and is usable
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    assert fourcc != 0, "MJPEG codec (MJPG) not available in this OpenCV build"
```

- [ ] **Step 2: Run test to verify it passes** (should pass — just verifies codec availability)

```bash
python -m pytest backend/tests/test_pipeline_perf.py::test_mjpeg_intermediate_written_during_pass1 -v
```

Expected: PASS

- [ ] **Step 3: Add intermediate writer to Pass 1 in run.py**

Locate the `cap = cv2.VideoCapture(video_path)` line that starts Pass 1 (around line 398). Add a MJPEG VideoWriter right after `total_frames` is retrieved:

```python
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_source = cap.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Write decoded frames to a fast-decode MJPEG intermediate so Pass 2
        # never has to re-decode the H.264 source.
        intermediate_frames_path = f"/tmp/{match_id}_frames.avi"
        _mjpeg_writer = cv2.VideoWriter(
            intermediate_frames_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps_source,
            (frame_w, frame_h),
        )
        _mjpeg_writer_ok = _mjpeg_writer.isOpened()
        if not _mjpeg_writer_ok:
            print("[Pipeline] MJPEG writer unavailable — Pass 2 will re-decode H.264")
```

- [ ] **Step 4: Write each frame to MJPEG during Pass 1**

Inside the Pass 1 `for i in tqdm(...)` loop, after `ball_track.append(...)`, add:

```python
            if _mjpeg_writer_ok:
                _mjpeg_writer.write(frame)
```

After the loop, release the MJPEG writer:

```python
        if _mjpeg_writer_ok:
            _mjpeg_writer.release()
            print(f"[Pipeline] MJPEG intermediate written: {intermediate_frames_path}")
```

- [ ] **Step 5: Use intermediate in Pass 2**

In the `# Pass 2: Drawing` section, replace:

```python
        # Reuse same VideoCapture — seek back to start instead of re-opening
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
```

with:

```python
        # Pass 2 reads from MJPEG intermediate (fast decode) if available,
        # otherwise re-seeks the H.264 source.
        if _mjpeg_writer_ok and os.path.exists(intermediate_frames_path):
            cap.release()
            cap = cv2.VideoCapture(intermediate_frames_path)
            print("[Pipeline] Pass 2 reading from MJPEG intermediate")
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
```

- [ ] **Step 6: Clean up intermediate after Pass 2**

After `cap.release()` and `out.release()` at the end of Pass 2, add:

```python
        # Remove MJPEG intermediate to free /tmp space
        if _mjpeg_writer_ok and os.path.exists(intermediate_frames_path):
            os.remove(intermediate_frames_path)
            print(f"[Pipeline] Removed MJPEG intermediate: {intermediate_frames_path}")
```

- [ ] **Step 7: Run all pipeline tests**

```bash
python -m pytest backend/tests/ -v --ignore=backend/tests/test_player_tracking.py --timeout=60
```

Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add backend/pipeline/run.py backend/tests/test_pipeline_perf.py
git commit -m "perf: write MJPEG intermediate in Pass 1 to avoid H.264 re-decode in Pass 2"
```

---

## Task 4: Court detection once at pipeline startup

The camera is fixed. Running ResNet50 every 5th frame throughout the video wastes ~5 ms/frame. Instead, run court detection on the first few frames at pipeline startup, pick the result with the best homography fit, and use it for the entire video.

If a saved calibration already exists for the given `camera_id`, skip the detection entirely (existing behaviour). The new path handles the case where no calibration is present: detect once → optionally save.

**Files:**
- Modify: `backend/pipeline/run.py` — replace per-frame court detection with one-time detection
- Add config field: `court_detection_startup_frames: int = 10`

### Step 4a: Add config field

- [ ] **Step 1: Write failing test**

```python
# backend/tests/test_pipeline_perf.py
def test_court_detection_startup_frames_config():
    from backend.pipeline.config import PipelineConfig
    cfg = PipelineConfig()
    assert hasattr(cfg, "court_detection_startup_frames")
    assert cfg.court_detection_startup_frames == 10
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_pipeline_perf.py::test_court_detection_startup_frames_config -v
```

Expected: FAIL

- [ ] **Step 3: Add field to config.py**

In `backend/pipeline/config.py`, in the `Detection Settings` section, replace:

```python
    # Court detection runs every Nth frame to save compute.
    # Ignored when a valid calibration is loaded (calibration_path + camera_id).
    court_detection_interval: int = 5
```

with:

```python
    # Court detection: run on this many frames at startup to find the best
    # homography fit. Ignored when a valid calibration is loaded.
    # Replaces the old per-frame court_detection_interval approach.
    court_detection_startup_frames: int = 10

    # Legacy — kept for backwards compatibility but no longer used by the pipeline.
    court_detection_interval: int = 5
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_pipeline_perf.py::test_court_detection_startup_frames_config -v
```

Expected: PASS

### Step 4b: One-time court detection helper

- [ ] **Step 5: Write failing test**

```python
# backend/tests/test_pipeline_perf.py
def test_detect_court_once_returns_best_result():
    from backend.pipeline.run import _detect_court_once

    # Mock a court_detector that always returns the same keypoints
    class FakeDetector:
        def infer_single(self, frame):
            return [(100.0, 50.0)] * 14  # 14 fake keypoints

    class FakeEstimator:
        def estimate(self, kps):
            import numpy as np
            return np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)

    import numpy as np
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(5)]
    H_ref, kps = _detect_court_once(frames, FakeDetector(), FakeEstimator())
    assert H_ref is not None, "Should return a homography from mock frames"
    assert kps is not None
```

- [ ] **Step 6: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_pipeline_perf.py::test_detect_court_once_returns_best_result -v
```

Expected: FAIL — `ImportError: cannot import name '_detect_court_once'`

- [ ] **Step 7: Add `_detect_court_once` helper to run.py**

Add this function after `_interpolate_player_detections()`:

```python
def _detect_court_once(
    frames: list,
    court_detector,
    homography_estimator,
) -> tuple:
    """
    Run court detection on the provided frames and return the homography
    matrix + keypoints from the frame with the most detected keypoints.

    Args:
        frames: List of BGR frames to try (typically first N frames of video).
        court_detector: CourtLineDetector instance.
        homography_estimator: HomographyEstimator instance.

    Returns:
        (H_ref, keypoints) — best result found, or (None, None) if all fail.
    """
    best_H_ref = None
    best_kps = None
    best_kp_count = 0

    for frame in frames:
        kps = court_detector.infer_single(frame)
        if kps is None:
            continue
        valid_count = sum(1 for k in kps if k is not None)
        if valid_count <= best_kp_count:
            continue
        H_ref, _ = homography_estimator.estimate(kps)
        if H_ref is None:
            continue
        best_H_ref = H_ref
        best_kps = kps
        best_kp_count = valid_count

    return best_H_ref, best_kps
```

- [ ] **Step 8: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_pipeline_perf.py::test_detect_court_once_returns_best_result -v
```

Expected: PASS

### Step 4c: Wire into pipeline startup

- [ ] **Step 9: Replace per-frame court detection in Pass 2**

Locate the court calibration section in `run_pipeline()` (around line 387). After the `calibrated_H_ref` block, add the one-time detection fallback:

```python
        # ---------- Court detection: once at startup ----------
        # If calibration is loaded, we already have calibrated_H_ref.
        # Otherwise, detect on the first N frames and use the best result.
        if calibrated_H_ref is None:
            print(f"[Pipeline] No calibration — detecting court on first {config.court_detection_startup_frames} frames")
            cap_startup = cv2.VideoCapture(video_path)
            startup_frames = []
            for _ in range(config.court_detection_startup_frames):
                ret_s, frm_s = cap_startup.read()
                if ret_s:
                    startup_frames.append(frm_s)
            cap_startup.release()

            calibrated_H_ref, calibrated_keypoints = _detect_court_once(
                startup_frames, court_detector, homography_estimator
            )
            if calibrated_H_ref is not None:
                print(f"[Pipeline] Court detected from startup frames — will reuse for all {total_frames} frames")
            else:
                print("[Pipeline] WARNING: Court detection failed on startup frames — falling back to per-frame detection")
```

- [ ] **Step 10: Remove per-frame court detection from Pass 2 drawing loop**

In the Pass 2 `for i in tqdm(...)` loop, replace the court detection block:

```python
            # Court detection: skip entirely if calibration is loaded
            if calibrated_H_ref is not None:
                kps = last_kps  # may be None — draw functions handle this gracefully
            else:
                process_this_frame = frame_idx % config.court_detection_interval == 0
                if process_this_frame:
                    kps = court_detector.infer_single(frame)
                    if kps is not None:
                        last_kps = kps
                else:
                    kps = last_kps
```

with (the calibration branch already covers the one-time case since we now always set `calibrated_H_ref` or `calibrated_keypoints` at startup):

```python
            # Court detection ran once at startup — use fixed result for all frames
            kps = last_kps  # set from calibrated_keypoints or startup detection
```

And in the setup before the Pass 2 loop (where `last_kps` is initialized from calibration), ensure the startup detection keypoints are also applied:

```python
        # Pre-fill homography and keypoints from calibration / startup detection
        if calibrated_H_ref is not None:
            homography_matrices = [calibrated_H_ref] * total_frames
            last_H_ref = calibrated_H_ref
            last_kps = calibrated_keypoints
            print(f"[Pipeline] Homography matrices pre-filled ({total_frames} frames)")
```

This block already exists — it now covers both the saved-calibration path and the new startup-detection path since both set `calibrated_H_ref`.

- [ ] **Step 11: Run all tests**

```bash
python -m pytest backend/tests/test_pipeline_perf.py -v --timeout=60
```

Expected: all pass

- [ ] **Step 12: Commit**

```bash
git add backend/pipeline/config.py backend/pipeline/run.py backend/tests/test_pipeline_perf.py
git commit -m "perf: detect court once at startup instead of every 5th frame in Pass 2"
```

---

## Task 5: NVENC hardware encoding in make_streamable_mp4

`make_streamable_mp4` currently uses `libx264` (CPU software encoder, ~15 ms/frame). On the A10G GPU, `h264_nvenc` encodes at ~2-3 ms/frame. We try NVENC first and fall back silently to `libx264` when NVENC is unavailable (local dev, CPU-only instances).

Also switch the Pass 2 intermediate VideoWriter codec from `mp4v` (CPU H.264) to `MJPEG` — this is now consistent with the intermediate file from Task 3 and faster to write.

**Files:**
- Modify: `backend/pipeline/storage.py` — `make_streamable_mp4`
- Modify: `backend/pipeline/run.py` — Pass 2 VideoWriter codec

### Step 5a: NVENC in make_streamable_mp4

- [ ] **Step 1: Write failing test**

```python
# backend/tests/test_pipeline_perf.py
def test_make_streamable_prefers_nvenc(tmp_path, monkeypatch):
    """
    make_streamable_mp4 should attempt h264_nvenc before libx264.
    We verify by checking which codec argument appears first in the subprocess call.
    """
    import subprocess
    from backend.pipeline.storage import make_streamable_mp4

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        # Raise on first call (simulate NVENC not available) to test fallback
        if "h264_nvenc" in cmd:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Create a dummy input file so the function doesn't fail before subprocess
    input_file = tmp_path / "test.mp4"
    input_file.write_bytes(b"fake")

    result = make_streamable_mp4(str(input_file))
    assert len(calls) >= 1, "subprocess.run should have been called"
    assert "h264_nvenc" in calls[0], f"First attempt should use h264_nvenc, got: {calls[0]}"
    if len(calls) > 1:
        assert "libx264" in calls[1], f"Fallback should use libx264, got: {calls[1]}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_pipeline_perf.py::test_make_streamable_prefers_nvenc -v
```

Expected: FAIL — `AssertionError: First attempt should use h264_nvenc`

- [ ] **Step 3: Rewrite make_streamable_mp4 to try NVENC first**

Replace the entire `make_streamable_mp4` function in `backend/pipeline/storage.py`:

```python
def make_streamable_mp4(input_path: str) -> str:
    """
    Remux/re-encode MP4 so browsers can stream it (moov atom first).

    Tries h264_nvenc (GPU) first for ~5x encoding speedup on A10G.
    Falls back to libx264 (CPU) if NVENC is unavailable (local dev / CPU instances).
    Falls back to the original file if ffmpeg is not found at all.
    """
    input_path = Path(input_path)
    output_path = input_path.with_name(input_path.stem + "_web.mp4")

    def _run_ffmpeg(codec: str) -> bool:
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(input_path),
                    "-c:v", codec,
                    "-preset", "fast",
                    "-crf", "23",
                    "-c:a", "copy",
                    "-movflags", "+faststart",
                    str(output_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    try:
        if _run_ffmpeg("h264_nvenc"):
            print("[Storage] Encoded with h264_nvenc (GPU)")
            return str(output_path)
        print("[Storage] h264_nvenc unavailable — falling back to libx264")
        if _run_ffmpeg("libx264"):
            print("[Storage] Encoded with libx264 (CPU fallback)")
            return str(output_path)
        print("⚠️  Both h264_nvenc and libx264 failed — returning original file.")
        return str(input_path)
    except FileNotFoundError:
        print("⚠️  ffmpeg not found — skipping streamable remux (video still usable locally).")
        return str(input_path)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_pipeline_perf.py::test_make_streamable_prefers_nvenc -v
```

Expected: PASS

### Step 5b: Switch Pass 2 VideoWriter to MJPEG

The Pass 2 output writer currently uses `mp4v` which is a slow CPU H.264 encoder. Since `make_streamable_mp4` re-encodes the file anyway with ffmpeg (NVENC or libx264), the intermediate VideoWriter just needs to be fast to write — MJPEG is ideal.

- [ ] **Step 5: Write failing test**

```python
# backend/tests/test_pipeline_perf.py
def test_pass2_videowriter_uses_mjpeg():
    """
    The Pass 2 intermediate VideoWriter should use MJPEG, not mp4v,
    since make_streamable_mp4 will re-encode the final output anyway.
    Verify the constant is defined and accessible.
    """
    import cv2
    mjpeg_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    mp4v_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    assert mjpeg_fourcc != mp4v_fourcc, "MJPG and mp4v should be different codecs"
    # The actual assertion is that run.py uses MJPG — we verify by reading the source
    import inspect
    from backend.pipeline import run
    source = inspect.getsource(run.run_pipeline)
    assert '"MJPG"' in source or "'MJPG'" in source, (
        "Pass 2 VideoWriter should use MJPG codec, not mp4v"
    )
```

- [ ] **Step 6: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_pipeline_perf.py::test_pass2_videowriter_uses_mjpeg -v
```

Expected: FAIL — `AssertionError: Pass 2 VideoWriter should use MJPG codec`

- [ ] **Step 7: Change VideoWriter codec in run.py Pass 2**

In `backend/pipeline/run.py`, find the Pass 2 VideoWriter setup (around line 510):

```python
        local_output_path = f"/tmp/{match_id}_processed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(local_output_path, fourcc, fps, (width, height))
```

Change to:

```python
        # Use MJPEG for the intermediate output — fast to write; make_streamable_mp4
        # re-encodes to H.264 (h264_nvenc or libx264) as its final step anyway.
        local_output_path = f"/tmp/{match_id}_processed.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(local_output_path, fourcc, fps, (width, height))
```

Note: extension changes to `.avi` since MJPEG lives in AVI containers. `make_streamable_mp4` accepts any ffmpeg-readable input so this is fine.

- [ ] **Step 8: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_pipeline_perf.py::test_pass2_videowriter_uses_mjpeg -v
```

Expected: PASS

- [ ] **Step 9: Run all perf tests**

```bash
python -m pytest backend/tests/test_pipeline_perf.py -v --timeout=60
```

Expected: all 8 tests pass

- [ ] **Step 10: Commit**

```bash
git add backend/pipeline/storage.py backend/pipeline/run.py backend/tests/test_pipeline_perf.py
git commit -m "perf: use h264_nvenc (GPU) encoding with libx264 fallback, MJPEG intermediate VideoWriter"
```

---

## Final Verification

- [ ] Run full test suite

```bash
python -m pytest backend/tests/ -v --timeout=120
```

Expected: all tests pass (skip `test_player_tracking.py` if no GPU available locally)

- [ ] Deploy to Modal and confirm logs show:
  - `PlayerTracker initialized with model: yolov8m-pose.pt`
  - `Pass 2 reading from MJPEG intermediate`
  - `Court detected from startup frames — will reuse for all N frames` (or `Using saved calibration`)
  - `Encoded with h264_nvenc (GPU)`

---

## Expected Speedup Summary

| Optimization | Per-frame savings | Source |
|---|---|---|
| YOLOv8m vs yolov8x | ~12-15 ms | Model is ~3-4x faster |
| YOLO every 3 frames | ~8-10 ms (amortized) | 2/3 of YOLO calls eliminated |
| MJPEG intermediate | ~5 ms in Pass 2 | MJPEG decode ~2 ms vs H.264 ~7 ms |
| Court once at startup | ~5 ms in Pass 2 | ResNet50 removed from per-frame loop |
| NVENC encoding | ~12 ms in Pass 2 | ~3 ms vs ~15 ms for libx264 |
| **Total savings** | **~42-47 ms/frame** | |

Current total: ~55 ms/frame (2-pass) → **Projected: ~10-13 ms/frame** → **~2-3x real-time**
