# CourtCheck Model & Analytics Upgrade Plan

## Context
CourtCheck now has quality UC Davis match footage. The current stroke classifier (Inception-V3 + LSTM on raw image patches) is slow and inaccurate. Heatmaps are basic. The pipeline has no awareness of sets, scoring, or side switches. This plan upgrades all three systems.

---

## Priority 1: Stroke Classification Rebuild (Top Priority)

### Problem
- Current: Inception-V3 extracts 2048-dim features from a 300×300 image patch **every frame** — extremely slow
- LSTM receives noisy, appearance-based features → low accuracy
- 3 classes only: Forehand / Backhand / Serve+Smash

### Approach: YOLOv8-Pose + Swing Trigger + TCN Classifier

**No labeling of UC Davis footage required.** Use a pretrained TCN trained on the public THETIS tennis dataset, with YOLOv8-Pose keypoints as input.

#### Step 1: Swap feature extraction — Inception-V3 → YOLOv8-Pose
- YOLOv8 is already in the pipeline for player tracking (`backend/models/player_tracker.py`)
- Upgrade to `yolov8x-pose.pt` to extract 17 body keypoints per player per frame
- Input to classifier: 17 keypoints × 2 coords = 34 features per frame (vs 2048 from Inception)
- Run pose extraction **in the same YOLO pass** as player detection (no extra inference cost)

**File:** `backend/models/player_tracker.py` — switch `model.track()` to a pose model

#### Step 2: Swing trigger detection (only call classifier when player swings)
- Monitor **wrist keypoint velocity** (keypoints 9 & 10 in COCO format = left/right wrist)
- Compute frame-over-frame velocity: `velocity = ||wrist_t - wrist_{t-1}||`
- Trigger: velocity > threshold (e.g. 15px/frame) AND ball is within N pixels of player
- Only extract pose sequences and run classifier during triggered windows
- This alone will 10x reduce classifier calls vs current every-frame approach

**New file:** `backend/vision/swing_detector.py`

#### Step 3: Stroke classifier — TCN on pose keypoints
- Input: sequence of 34-feature pose vectors (30–60 frames around swing peak)
- Architecture: Temporal Convolutional Network (TCN) — faster than LSTM, no sequential dependency
- Pretrained on THETIS dataset (public tennis stroke dataset: forehand, backhand, slice)
- Classes: Forehand / Backhand / Serve / Slice (add Slice as 4th class)
- Fine-tune on UC Davis footage later once enough clips are collected

**New file:** `backend/models/stroke_classifier_tcn.py`
**Replace:** `backend/models/stroke_detector.py` (ActionRecognition / LSTM_model / Inception approach)

#### Step 4: Update pipeline orchestration
- In `backend/pipeline/run.py`: replace stroke recognition block (lines 437–508)
- In Pass 1: collect pose keypoints alongside player bboxes
- In shot detection: feed pose keypoints + ball positions to swing detector
- Record stroke type per shot frame

**Modified file:** `backend/pipeline/run.py`
**Modified file:** `backend/pipeline/config.py` — add `stroke_classifier_weights_tcn` path

---

## Priority 2: Improved Heatmaps

### Problem
- Bounce heatmap: undifferentiated circles, no shot type or in/out info
- Player heatmap: gaussian density blob, doesn't show positioning at moment of shot

### 2A: Bounce dot map by shot type + in/out
- Replace gaussian circles with colored dots per bounce:
  - Forehand → green dot
  - Backhand → blue dot
  - Serve → yellow dot
  - Out of bounds → red outline/X
- Draw directly on court reference image (no gaussian blur needed)
- Store bounce data as structured array: `[{frame, x, y, stroke_type, in_bounds}]`

**Modified file:** `backend/vision/heatmaps.py` — new `generate_bounce_dot_map()` function
**Modified file:** `backend/pipeline/storage.py` — store bounce detail array to Supabase (new `bounce_data` JSONB column)

### 2B: Player position dots at shot moments
- Instead of histogram2d density over all frames, render a dot for each shot frame at player's projected court position
- Dot color: stroke type (matches bounce dot colors for consistency)
- Makes spacing errors and positioning patterns immediately visible
- Keep existing gaussian heatmap as optional toggle ("Density" vs "Shot Positions" tab)

**Modified file:** `backend/vision/heatmaps.py` — new `generate_player_shot_dot_map()` function

### 2C: Frontend visualization update
- Add stroke type legend to heatmap component
- Add "Dot Map" tab alongside existing "Ball Bounces" and "Player Positions" tabs
- Update `HeatMaps.tsx` to render new image types

**Modified file:** `frontend/web/components/features/dashboard/HeatMaps.tsx`
**Modified file:** `frontend/web/app/api/heatmaps/latest/route.ts`

---

## Priority 3: Court Detection Calibration

### Problem
- Current court detector (`CourtLineDetector`) struggles with the new UC Davis camera angle
- Runs every 5 frames which is slow and produces inconsistent homography matrices

### Approach: One-time calibration per camera angle
Since the UC Davis cameras are fixed and stable, we only need to solve this once per court/camera setup.

- Run court detector on a single reference frame from the new footage
- Add a **calibration tool**: visualize detected keypoints overlaid on the frame, allow manual correction of misplaced keypoints via a simple UI or script
- Save the verified homography as a `court_calibration.json` file (camera ID → H matrix)
- In the pipeline: if a calibration file exists for the video's source camera, **skip per-frame court detection** and use the saved homography — major speed gain
- Fall back to per-frame detection for uncalibrated cameras

**New file:** `backend/vision/calibration.py` — `save_calibration()`, `load_calibration()`, `visualize_keypoints()`
**New file:** `backend/tools/calibrate_court.py` — standalone script to run on a single frame and verify/correct keypoints
**Modified file:** `backend/pipeline/run.py` — check for calibration file before running court detection
**Modified file:** `backend/pipeline/config.py` — add `calibration_path` field

---

## Priority 4: Cuts / Sets / Side-switching (Hybrid Approach)

### Problem
- Pipeline has no awareness of sets, points, or players switching sides
- `is_scene_cut()` already exists in `run.py` but outputs are discarded
- Stats aggregate across entire match regardless of set/rally context

### 3A: Auto-detect side switches
- After each scene cut, check if player positions have swapped halves (near/far)
- Player side: determined by y-position relative to net (y=1748 in court reference)
- If Player A was in near half and is now in far half → side switch detected
- Segment all subsequent stats (bounces, shots, player positions) by current side

**Modified file:** `backend/pipeline/run.py` — use existing `is_scene_cut()`, add `detect_side_switch()`
**New function:** `backend/vision/postprocess.py` — `detect_side_switch(prev_player_positions, curr_player_positions)`

### 3B: Manual annotation UI in video player
- Add a floating annotation toolbar in the recordings video player
- Controls:
  - "Point" button → marks current timestamp as a point scored (prompts: who won?)
  - "Set" button → marks current timestamp as set change
  - "Side Switch" button → marks side switch (auto-suggested after scene cut detection)
  - "Note" button → opens a text input to attach a free-text coaching note at the current timestamp (e.g. "watch footwork here", "double fault pattern")
- Annotations and notes stored in Supabase: new `match_annotations` table
- Notes rendered as markers on the video timeline, expandable on hover/click

**New table:** `match_annotations` — `{id, match_id, user_id, timestamp_ms, type: 'point'|'set'|'side_switch'|'note', note_text: text|null, metadata: JSONB}`
**Modified file:** `frontend/web/app/recordings/[id]/page.tsx` — add annotation toolbar + timeline markers
**New file:** `frontend/web/app/api/annotations/route.ts` — CRUD for annotations
**New file:** `frontend/web/components/features/recordings/AnnotationToolbar.tsx`

### 3C: Segment stats by set
- Once annotations exist, pipeline results page can break down stats per set
- e.g. "Set 1: 12 FH, 8 BH" vs "Set 2: 7 FH, 14 BH"
- Requires re-slicing stored bounce_data and shot timestamps against annotation timestamps

**Modified file:** `frontend/web/app/recordings/[id]/page.tsx` — set-by-set stats breakdown section

---

## Implementation Order

1. **Court calibration** (quick win, unblocks accurate homography for everything else)
   - Calibration script + tool
   - Pipeline integration
2. **Stroke classifier** (highest impact, unblocks heatmap coloring)
   - Pose extraction in player tracker
   - Swing detector
   - TCN model + pretrained weights from THETIS
   - Pipeline integration
3. **Heatmap upgrades** (depends on stroke classifier for shot-type coloring)
   - Bounce dot map
   - Player shot dot map
   - Frontend tabs
4. **Cuts/Sets/Annotations**
   - Auto side switch detection
   - Annotation + notes UI
   - Per-set stats display

---

## Critical Files

| File | Change |
|---|---|
| `backend/models/player_tracker.py` | Switch to YOLOv8-Pose, expose keypoints |
| `backend/models/stroke_detector.py` | Replace with TCN-based classifier |
| `backend/models/stroke_classifier_tcn.py` | New TCN model class |
| `backend/vision/swing_detector.py` | New swing trigger from wrist velocity |
| `backend/vision/heatmaps.py` | Add dot map functions |
| `backend/vision/postprocess.py` | Add `detect_side_switch()` |
| `backend/pipeline/run.py` | Wire up new stroke + side-switch logic |
| `backend/pipeline/storage.py` | Store `bounce_data` JSONB |
| `frontend/web/components/features/dashboard/HeatMaps.tsx` | New tabs + legend |
| `frontend/web/app/recordings/[id]/page.tsx` | Annotation UI + set stats |
| `frontend/web/app/api/annotations/route.ts` | New annotation API |
| `backend/vision/calibration.py` | Court calibration save/load |
| `backend/tools/calibrate_court.py` | One-time calibration script |

---

## Verification

1. Run pipeline on a UC Davis match video, confirm stroke types are logged and plausible (forehand/backhand ratio should roughly match what's visible)
2. Compare speed: old vs new pipeline runtime per minute of footage
3. Inspect heatmap PNGs — bounces should be colored dots, player map should show shot positions
4. Test annotation UI: mark a point, verify it saves and appears on timeline
5. Verify side switch auto-detection triggers after a scene cut when players swap halves
