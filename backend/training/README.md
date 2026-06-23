# Stroke Classifier Training

CourtCheck-only TCN stroke classifier. No external pretraining — UC Davis
labels are the only source of truth.

## Setup

```bash
pip install ultralytics catboost scikit-learn torch tensorboard tqdm
```

## Pipeline

### 1. Export labels from Supabase

```bash
python -c "import sys; sys.path.insert(0,'.'); \
  from backend.tools._swing_io import db_get_all, write_manifest; from pathlib import Path; \
  write_manifest(Path('data/annotation/labeled_export.csv'), db_get_all())"
```

### 2. Build the TCN training manifest

```bash
python -m backend.tools.build_courtcheck_manifest \
    --labels-csv data/annotation/labeled_export.csv \
    --clips-dir  data/annotation/clips \
    --output     backend/training/data/courtcheck_manifest.json
```

Filters to forehand/backhand/serve labels; drops volley (no TCN class).
Detects clips that already have `.npy` keypoint files and marks them so
extraction skips them on the next step.

### 3. Extract keypoints

```bash
python -m backend.training.data.extract_keypoints \
    --manifest backend/training/data/courtcheck_manifest.json \
    --model yolov8m-pose.pt \
    --device mps
```

Resume-safe: skips clips with existing `.npy` files. ~30 s/clip on MPS.

### 4. Train TCN

```bash
python -m backend.training.train_tcn \
    --manifest backend/training/data/courtcheck_manifest.json
```

Defaults: 85/15 stratified train/val split, AdamW with cosine LR, label
smoothing 0.1, weight decay 5e-2, class-weighted CE, augmentation enabled.
Best weights save to `backend/weights/stroke_classifier_tcn.pt`.

For more robust evaluation use 5-fold stratified CV (best fold becomes the
canonical weights file):

```bash
python -m backend.training.train_tcn --cv
```

Monitor:

```bash
tensorboard --logdir backend/training/runs/
```

### 5. Evaluate

```bash
python -m backend.training.eval_tcn \
    --weights backend/weights/stroke_classifier_tcn.pt
```

Writes a markdown report to `backend/training/runs/eval_<timestamp>.md`
covering top-1 / top-2 accuracy, per-class P/R/F1, confusion matrix, and
top mis-predicted clip paths for spot-checking.

## Augmentation recipe (defaults)

Applied to raw `(T, 17, 3)` keypoints before hip-center normalization:

- Temporal jitter (±3 frames)
- Horizontal flip (50% prob, **suppressed for Serve** — service motion is handedness-specific)
- Rotation ±15° around pose centroid
- Anisotropic scaling 0.9-1.1×
- Shear ±6°
- Gaussian joint jitter (σ ≈ 0.02 × frame height)
- Joint masking (15% per joint, hips and shoulders protected)
- JMDA — joint-subset mixing with another same-class sample (30% prob, 50% joints swapped)

Knobs live in `backend/training/config.py` under `TrainingConfig.aug_*`.

## Activating in pipeline

Set in `PipelineConfig`:

```python
stroke_classifier_weights_tcn = "stroke_classifier_tcn.pt"
```

The pipeline resolves this against `backend/weights/`.

## Notes

- THETIS pretraining was removed (2026-05-12) — produced 0% transfer to UC Davis broadcast footage; pretraining was actively harmful.
- `train_catboost.py` is a fast feature-quality baseline; same manifest format works.
- `num_classes=3` (FH / BH / Serve). Volley dropped. Slice was never represented.
