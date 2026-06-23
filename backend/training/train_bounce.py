"""
Retrain the CatBoost bounce regressor on UC Davis labeled clips.

Closes the loop between:
    annotate.py          -> <clip>.gt.json  (frame-level bounce labels)
    extract_ball_trajectory.py
                         -> <clip>.ball.npz (per-frame ball xy)
    bounce_detector.py    (12-feature CatBoost input)

For each clip we:
    1. Load (x_ball, y_ball) from .ball.npz
    2. Smooth + extrapolate using BounceDetector.smooth_predictions (same as inference path)
    3. Run BounceDetector.prepare_features to get the 12-feature matrix
       and the surviving frame indices.
    4. Mark each surviving frame as positive (1.0) if its frame index is within
       ±tolerance of any bounce in the .gt.json, else 0.0.
    5. Concat across clips.
    6. Split train/val. Train CatBoostRegressor (matches the existing
       BounceDetector signature: regressor + threshold).
    7. Save bounce_detection_weights_ucd.cbm and a markdown eval report.

Usage:
    python -m backend.training.train_bounce \
        --clips-glob 'data/bounce_train/clips/*.mp4' \
        --output backend/weights/bounce_detection_weights_ucd.cbm

The trained weights drop into the live pipeline by flipping
`PipelineConfig.bounce_model_weights` to the new filename.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from catboost import CatBoostRegressor
except ImportError as exc:
    raise ImportError("catboost is required: pip install catboost") from exc

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit

from backend.models.bounce_detector import BounceDetector


@dataclass
class ClipSample:
    clip_id: str
    features: np.ndarray  # (n, 12)
    labels: np.ndarray    # (n,)  float32 0/1
    frames: np.ndarray    # (n,) int — frame index for each feature row
    n_bounces_gt: int
    n_pos_rows: int       # rows labeled positive (= n_bounces_gt * (2*tol+1) minus collisions)


def _load_gt_bounces(gt_path: Path) -> list[dict]:
    if not gt_path.exists():
        return []
    payload = json.loads(gt_path.read_text())
    return list(payload.get("bounces", []))


def _label_rows(frame_indices: list[int], gt_frames: list[int], tolerance: int) -> np.ndarray:
    """Return 0/1 labels for each feature row, positive if within ±tolerance of any GT bounce."""
    if not gt_frames:
        return np.zeros(len(frame_indices), dtype=np.float32)
    gt_arr = np.array(sorted(gt_frames), dtype=np.int64)
    frames = np.array(frame_indices, dtype=np.int64)
    # For each frame, find the nearest GT bounce frame
    idx = np.searchsorted(gt_arr, frames)
    left = np.clip(idx - 1, 0, len(gt_arr) - 1)
    right = np.clip(idx, 0, len(gt_arr) - 1)
    dl = np.abs(frames - gt_arr[left])
    dr = np.abs(frames - gt_arr[right])
    nearest = np.minimum(dl, dr)
    return (nearest <= tolerance).astype(np.float32)


def build_clip_sample(
    clip_path: Path,
    tolerance: int,
    detector: BounceDetector,
) -> ClipSample | None:
    ball_path = clip_path.with_suffix(clip_path.suffix + ".ball.npz")
    gt_path = clip_path.with_suffix(clip_path.suffix + ".gt.json")

    if not ball_path.exists():
        print(f"[train_bounce] SKIP {clip_path.name}: no .ball.npz (run extract_ball_trajectory first)")
        return None
    if not gt_path.exists():
        print(f"[train_bounce] SKIP {clip_path.name}: no .gt.json (run annotate.py first)")
        return None

    data = np.load(ball_path)
    x_raw = data["x_ball"].astype(np.float32)
    y_raw = data["y_ball"].astype(np.float32)

    # smooth_predictions wants Python lists with None for missing values (matches inference path).
    x_list = [None if np.isnan(v) else float(v) for v in x_raw]
    y_list = [None if np.isnan(v) else float(v) for v in y_raw]
    x_smooth, y_smooth = detector.smooth_predictions(x_list, y_list)

    features_df, frame_idx_list = detector.prepare_features(x_smooth, y_smooth)
    if features_df.empty:
        print(f"[train_bounce] SKIP {clip_path.name}: 0 rows survived feature build (ball mostly missing)")
        return None

    feats = features_df.to_numpy(dtype=np.float32)

    gt_bounces = _load_gt_bounces(gt_path)
    gt_frames = [int(b["frame"]) for b in gt_bounces]
    labels = _label_rows(frame_idx_list, gt_frames, tolerance)

    sample = ClipSample(
        clip_id=clip_path.stem,
        features=feats,
        labels=labels,
        frames=np.array(frame_idx_list, dtype=np.int64),
        n_bounces_gt=len(gt_frames),
        n_pos_rows=int(labels.sum()),
    )
    print(
        f"[train_bounce] {clip_path.name}: {feats.shape[0]} rows, "
        f"{sample.n_bounces_gt} GT bounces, {sample.n_pos_rows} positive rows "
        f"({sample.n_pos_rows / max(1, feats.shape[0]):.2%})"
    )
    return sample


def _frame_level_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict:
    preds = (y_pred > threshold).astype(np.int32)
    targets = (y_true > 0.5).astype(np.int32)
    p, r, f1, _ = precision_recall_fscore_support(targets, preds, average="binary", zero_division=0)
    cm = confusion_matrix(targets, preds, labels=[0, 1]).tolist()
    try:
        ap = average_precision_score(targets, y_pred)
    except Exception:
        ap = float("nan")
    return {
        "threshold": threshold,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "avg_precision": float(ap),
        "confusion": cm,
        "n_true_pos": int(targets.sum()),
        "n_pred_pos": int(preds.sum()),
    }


def _bounce_level_metrics(
    samples: list[ClipSample],
    y_pred: np.ndarray,
    threshold: float,
    min_gap_frames: int,
    tolerance: int,
) -> dict:
    """Group rows by clip, apply postprocess() like inference, then match against GT bounces."""
    cursor = 0
    tp = fp = fn = 0
    detector_stub = BounceDetector(threshold=threshold, min_gap_frames=min_gap_frames)
    for s in samples:
        n = s.features.shape[0]
        sub_preds = y_pred[cursor : cursor + n]
        cursor += n
        ind = np.where(sub_preds > threshold)[0]
        if len(ind) > 0:
            ind = detector_stub.postprocess(ind, sub_preds)
        pred_frames = sorted(int(s.frames[i]) for i in ind)
        gt_frames = []  # rebuild from labels: any positive row whose neighbours are not also same bounce
        # Recover GT frames from labels by re-loading would be cleaner — but we kept them implicit in
        # the .gt.json. Reload from the npz/json round-trip is overkill; we approximate by collapsing
        # contiguous positive runs.
        in_run = False
        run_start = None
        positives = np.where(s.labels > 0.5)[0]
        for i in positives:
            f = int(s.frames[i])
            if not in_run or f - (run_start if run_start is not None else f) > tolerance * 2 + 1:
                gt_frames.append(f)
                in_run = True
                run_start = f
            else:
                run_start = f
        # Match pred -> gt within tolerance + min_gap window
        matched_gt = set()
        for pf in pred_frames:
            best, best_d = None, tolerance + min_gap_frames + 1
            for gi, gf in enumerate(gt_frames):
                if gi in matched_gt:
                    continue
                d = abs(pf - gf)
                if d < best_d:
                    best, best_d = gi, d
            if best is not None and best_d <= tolerance + min_gap_frames // 2:
                matched_gt.add(best)
                tp += 1
            else:
                fp += 1
        fn += len(gt_frames) - len(matched_gt)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "threshold": threshold,
        "min_gap_frames": min_gap_frames,
    }


def _markdown_report(
    clips: list[ClipSample],
    train_clips: list[str],
    val_clips: list[str],
    frame_metrics: list[dict],
    bounce_metrics: list[dict],
    output: Path,
    weights_path: Path,
    args: argparse.Namespace,
) -> str:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Bounce CatBoost retrain — {ts}",
        "",
        f"- weights: `{weights_path}`",
        f"- clips: {len(clips)}",
        f"- train clips: {len(train_clips)}, val clips: {len(val_clips)}",
        f"- tolerance: ±{args.tolerance} frames",
        f"- iterations: {args.iterations}, depth: {args.depth}, lr: {args.learning_rate}",
        "",
        "## Per-clip GT counts",
        "",
        "| clip | rows | GT bounces | positive rows |",
        "|------|------|------------|---------------|",
    ]
    for s in clips:
        lines.append(f"| {s.clip_id} | {s.features.shape[0]} | {s.n_bounces_gt} | {s.n_pos_rows} |")
    lines.append("")
    lines.append("## Frame-level validation metrics")
    lines.append("")
    lines.append("| threshold | precision | recall | f1 | avg_precision | TP rows | pred pos |")
    lines.append("|-----------|-----------|--------|----|---------------|---------|----------|")
    for m in frame_metrics:
        lines.append(
            f"| {m['threshold']:.2f} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} | {m['avg_precision']:.3f} | {m['n_true_pos']} | {m['n_pred_pos']} |"
        )
    lines.append("")
    lines.append("## Bounce-level validation metrics (postprocess + min_gap)")
    lines.append("")
    lines.append("| threshold | TP | FP | FN | precision | recall | f1 |")
    lines.append("|-----------|----|----|----|-----------|--------|----|")
    for m in bounce_metrics:
        lines.append(
            f"| {m['threshold']:.2f} | {m['tp']} | {m['fp']} | {m['fn']} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
        )
    lines.append("")
    lines.append("## Train / val split")
    lines.append("")
    lines.append("- train clips: " + ", ".join(train_clips))
    lines.append("- val clips: " + ", ".join(val_clips))
    lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    return "\n".join(lines)


def train(args: argparse.Namespace) -> int:
    clip_paths = [Path(p).resolve() for p in sorted(args.clips)]
    if args.clips_glob:
        import glob as glob_mod
        clip_paths.extend(sorted(Path(p).resolve() for p in glob_mod.glob(args.clips_glob)))
    clip_paths = list({p: None for p in clip_paths})

    if not clip_paths:
        print("no clips provided", file=sys.stderr)
        return 1

    detector = BounceDetector(threshold=args.threshold, min_gap_frames=args.min_gap_frames)

    samples: list[ClipSample] = []
    for cp in clip_paths:
        s = build_clip_sample(cp, tolerance=args.tolerance, detector=detector)
        if s is not None:
            samples.append(s)

    if not samples:
        print("no usable samples (need both .gt.json + .ball.npz per clip)", file=sys.stderr)
        return 1

    total_pos = sum(s.n_pos_rows for s in samples)
    total_rows = sum(s.features.shape[0] for s in samples)
    if total_pos == 0:
        print(f"no positive rows across {len(samples)} clips — check annotation timing / tolerance", file=sys.stderr)
        return 1

    print(
        f"\n[train_bounce] total: {len(samples)} clips, {total_rows} rows, "
        f"{total_pos} positive ({total_pos / total_rows:.2%})"
    )

    if len(samples) < 2:
        print("only one clip available — cannot split, validating on train", file=sys.stderr)
        train_idx = [0]
        val_idx = [0]
    else:
        # Group split by clip so a clip is fully in train or val
        groups = np.arange(len(samples))
        gss = GroupShuffleSplit(n_splits=1, test_size=max(1 / len(samples), args.val_size), random_state=42)
        X_dummy = np.zeros((len(samples), 1))
        y_dummy = np.zeros(len(samples))
        (train_idx_arr, val_idx_arr) = next(gss.split(X_dummy, y_dummy, groups=groups))
        train_idx = train_idx_arr.tolist()
        val_idx = val_idx_arr.tolist()

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    X_train = np.concatenate([s.features for s in train_samples], axis=0)
    y_train = np.concatenate([s.labels for s in train_samples], axis=0)
    X_val = np.concatenate([s.features for s in val_samples], axis=0)
    y_val = np.concatenate([s.labels for s in val_samples], axis=0)

    n_pos_train = int(y_train.sum())
    n_neg_train = int((y_train < 0.5).sum())
    print(f"[train_bounce] train: {X_train.shape[0]} rows ({n_pos_train} pos / {n_neg_train} neg)")
    print(f"[train_bounce] val:   {X_val.shape[0]} rows ({int(y_val.sum())} pos)")

    model = CatBoostRegressor(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=max(1, args.iterations // 10),
        random_seed=42,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output))
    print(f"[train_bounce] saved weights -> {output}")

    # Validation metrics across a sweep of thresholds
    y_val_pred = model.predict(X_val)
    frame_metrics = [
        _frame_level_metrics(y_val, y_val_pred, t)
        for t in (0.10, 0.18, 0.25, 0.35, 0.50)
    ]
    bounce_metrics = [
        _bounce_level_metrics(val_samples, y_val_pred, t, min_gap_frames=args.min_gap_frames, tolerance=args.tolerance)
        for t in (0.10, 0.18, 0.25, 0.35, 0.50)
    ]

    runs_dir = Path(__file__).resolve().parent / "runs"
    report_path = runs_dir / f"bounce_retrain_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_text = _markdown_report(
        samples,
        [s.clip_id for s in train_samples],
        [s.clip_id for s in val_samples],
        frame_metrics,
        bounce_metrics,
        report_path,
        output,
        args,
    )
    print(f"\n[train_bounce] report -> {report_path}\n")
    print(report_text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrain bounce CatBoost on UC Davis labels")
    ap.add_argument("--clips", nargs="*", default=[], help="explicit clip paths")
    ap.add_argument("--clips-glob", default=None, help="glob for clips")
    ap.add_argument(
        "--output",
        default="backend/weights/bounce_detection_weights_ucd.cbm",
        help="output .cbm path",
    )
    ap.add_argument("--tolerance", type=int, default=2, help="GT label tolerance in frames (±)")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=0.18, help="threshold used for postprocess metrics")
    ap.add_argument("--min-gap-frames", type=int, default=10)
    ap.add_argument("--val-size", type=float, default=0.2, help="fraction of clips held out for validation")
    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
