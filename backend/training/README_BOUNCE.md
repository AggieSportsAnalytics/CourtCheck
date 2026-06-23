# Bounce CatBoost retrain — workflow

End-to-end loop for retraining the bounce regressor on UC Davis labeled clips.
Owns the path from "I have a 1-hour annotation window" to "new weights live on Modal."

Pre-existing weights `bounce_detection_weights.cbm` were trained on broadcast tennis
(36k frames, 9 surfaces). On UC Davis footage they catch ~3 of 6 far-side bounces
because TrackNet loses the ball at the bounce moment and CatBoost only sees
trajectory inflections — see `docs/HANDOFF.md` and JarvisEA handoff 2026-05-21.

## One-time setup

```bash
cd /Users/Brian/PycharmProjects/CourtCheck
mkdir -p data/bounce_train/clips
```

## Workflow

### 1. Cut 2-minute training clips from the raw videos

Three raw matches available, one per calibrated camera:

```bash
python -m backend.training.cut_bounce_clips --source data/raw_videos/StMarys_Court2.mp4 --segments 3
python -m backend.training.cut_bounce_clips --source data/raw_videos/StMarys_Court4.mp4 --segments 3
python -m backend.training.cut_bounce_clips --source data/raw_videos/StMarys_Court6.mp4 --segments 3
```

Result: 9 clips in `data/bounce_train/clips/court{2,4,6}_seg{01,02,03}.mp4`,
each H.264-encoded so OpenCV (annotate.py) and TrackNet read them cleanly.

`--lead` defaults to 300 s (skip the warm-up), `--duration` to 120 s. Tune if
the raw match has long idle stretches.

### 2. Pre-compute ball trajectories

Slow step (TrackNet runs at ~30 fps on CPU). Do this once per clip; annotate.py
doesn't need it but train_bounce.py does.

```bash
python -m backend.training.extract_ball_trajectory \
  --glob 'data/bounce_train/clips/*.mp4' \
  --skip-existing
```

Writes `<clip>.ball.npz` next to each clip.

Camera id auto-detects from filename (`court2` → `uc_davis_court2`,
`court1_zoom` → `uc_davis_court1_zoomed`). Override with `--camera-id` if needed.
The ROI pass requires a valid camera id; without one the script falls back to
the main TrackNet pass and warns.

### 3a. Pre-label with the current detector (NEW — review mode)

Seeds `<clip>.gt.json` with the existing CatBoost's bounce predictions so the
annotation pass becomes review-and-correct instead of cold-annotation. Catches
roughly half the bounces correctly on UC Davis footage, eliminating the
frame-hunting half of the work.

```bash
python -m backend.training.pre_label_bounces \
  --glob 'data/bounce_train/clips/*.mp4'
```

Each pre-labeled bounce gets `"auto_labeled": true` so annotate.py's HUD can
tag them `[AUTO]` and you can see at a glance what the machine wrote vs what
you confirmed/added. train_bounce.py ignores the flag.

### 3b. Annotate (review the seeds)

```bash
python -m backend.eval.annotate data/bounce_train/clips/court2_seg01.mp4
```

Review workflow per clip (~5–7 min):
1. Press `n` — jump to next pre-labeled bounce.
2. HUD shows `[AUTO] bounce IN @ frame X`. Eyeball the contact frame.
   - ✓ Correct → press `n` to advance.
   - ✗ Wrong → press `d` to delete (deletes the bounce nearest current frame within ±15).
   - ✗ Should be OUT → press `d`, then `o` to re-stamp out-of-bounds.
3. Between seeds, skim with `.` for any bounces the detector missed; stamp with `i` or `o`.
4. `w` to save without quitting; `q` to save and quit.

Cold annotation (no pre-label) still works — same `i` / `o` hotkeys. Player
tagging (`1`/`2`/`f`/`b`/`s`/`v`) is irrelevant for bounce training; ignore.

Output: `<clip>.gt.json` with `{frame, in_bounds}` entries (+ optional
`auto_labeled` flag on pre-labeled rows).

Target: 100–300 bounces total across 6–9 clips. Below ~100 the model is
guessing; above ~300 you're past the data-quality ceiling.

### 4. Train

```bash
python -m backend.training.train_bounce \
  --clips-glob 'data/bounce_train/clips/*.mp4' \
  --output backend/weights/bounce_detection_weights_ucd.cbm
```

Runs in ~1-3 min on CPU. Writes:
- `backend/weights/bounce_detection_weights_ucd.cbm` — new weights
- `backend/training/runs/bounce_retrain_<ts>.md` — frame- and bounce-level
  metrics across thresholds 0.10 / 0.18 / 0.25 / 0.35 / 0.50

Clip-level group split (a clip is fully in train or val) so the val metrics
are not contaminated by trajectory autocorrelation within a clip.

What to look at in the report:
- **Frame-level F1 @ 0.18** — direct comparable to the legacy threshold.
- **Bounce-level recall @ 0.18** — the headline. Target ≥ 0.80 on the
  StMarys-class footage (vs 0.50 we're seeing today).
- If train recall is high but val recall is low, you don't have enough clips.
  Add 2–4 more from the under-represented camera and re-run.

### 5. Validate against the live eval harness

```bash
# Swap the weights pointer (one-line config flip)
# Edit backend/pipeline/config.py:
#   bounce_model_weights: str = 'bounce_detection_weights_ucd.cbm'

# Annotate the StMarys regression clip if you haven't already
python -m backend.eval.annotate data/StMarys_Court2_Aileena_vs_StMarys.mp4

# Run pipeline + diff vs ground truth
python -m backend.eval.run_eval data/StMarys_Court2_Aileena_vs_StMarys.mp4
```

Look for: 6/6 bounces detected on the opp-side ground truth clip.

### 6. Deploy

`feedback_test_locally_before_main` applies — only push after step 5 passes
locally.

```bash
# from a clean working tree:
git add backend/pipeline/config.py backend/weights/bounce_detection_weights_ucd.cbm
git commit -m "feat(bounce): retrain CatBoost on UC Davis labels"
git push origin main
modal deploy backend/app.py
```

Re-process the StMarys test match via the UI Reprocess button and confirm the
shotmap fills in.

## Why this loop

`bounce_detector.py` accepts a CatBoostRegressor with a 12-feature trajectory
input. The features are pure ball-coord deltas / divisions at lag ±1, ±2, so
the trainer only needs:
- ball trajectories (we extract those once per clip)
- frame-level bounce labels (annotate.py)

Skipping the full pipeline during training data prep keeps the iteration time
at ~5 min per clip retrain cycle.

## Gotchas

- **prepare_features drops NaN-context rows.** The feature row index is NOT
  the frame index. Labels are aligned via the `num_frames` list the function
  returns, not by position. Don't refactor this assumption away.
- **±2 frame tolerance** means each GT bounce produces ≤ 5 positive rows. With
  300 GT bounces that's ~1.5k positives — fine for CatBoost. Bump to ±3 if you
  see chronic 1-frame annotator slop in the val report.
- **min_gap_frames=10** at inference collapses near-duplicate detections.
  Training labels don't need this — the smoothing happens in postprocess().
- **Don't use the swing-annotation clips in data/annotation/clips/** — they
  are 45-frame swing windows, not continuous rallies. Far too sparse for
  bounce training.
- **Camera id matters.** The ROI backfill only fires with a loaded
  calibration. Without it, far-side ball detection collapses to the
  broadcast-quality main pass — and that's the very failure mode this retrain
  exists to fix.
