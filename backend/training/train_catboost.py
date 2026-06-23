"""
CatBoost baseline trainer for stroke classification.

Validates feature quality quickly before investing in full TCN training.
Trains on extract_clip_features() vectors from keypoint .npy files.

Usage:
    python -m backend.training.train_catboost \\
        --manifest backend/training/data/courtcheck_manifest_2026-05-12.json \\
        --output backend/weights/stroke_catboost.cbm
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from catboost import CatBoostClassifier
except ImportError as e:
    raise ImportError("catboost is required: pip install catboost") from e

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from backend.models.stroke_classifier_tcn import STROKE_LABELS
from backend.training.features import extract_clip_features


LABEL_TO_IDX = {label: i for i, label in enumerate(STROKE_LABELS)}


def load_manifest(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_feature_matrix(
    entries: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Load keypoints and compute feature vectors for all valid entries.

    Returns:
        X: (N, N_FEATURES) feature matrix
        y: (N,) integer class labels
        skipped_indices: manifest indices skipped due to missing/invalid data
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    skipped: list[int] = []

    for i, entry in enumerate(entries):
        kp_path = entry.get("keypoints_path")
        label = entry.get("mapped_label")

        if not kp_path or not os.path.exists(kp_path):
            skipped.append(i)
            continue

        if not entry.get("valid", True):
            skipped.append(i)
            continue

        if label not in LABEL_TO_IDX:
            skipped.append(i)
            continue

        keypoints = np.load(kp_path).astype(np.float32)
        if keypoints.ndim != 3 or keypoints.shape[1] != 17:
            skipped.append(i)
            continue

        try:
            features = extract_clip_features(keypoints)
        except Exception as e:
            print(f"[train_catboost] Feature extraction failed for {kp_path}: {e}")
            skipped.append(i)
            continue

        X_list.append(features)
        y_list.append(LABEL_TO_IDX[label])

    if not X_list:
        raise RuntimeError(
            "No valid samples found. Run extract_keypoints.py first."
        )

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int32), skipped


def print_class_distribution(y: np.ndarray) -> None:
    counts: dict[int, int] = defaultdict(int)
    for label in y:
        counts[int(label)] += 1
    print("\nClass distribution:")
    for idx, name in enumerate(STROKE_LABELS):
        print(f"  {name:<20} {counts[idx]:>4}")


def train_and_evaluate(
    manifest_path: str,
    output_path: str,
) -> None:
    """Full training and evaluation pipeline."""
    print(f"[train_catboost] Loading manifest: {manifest_path}")
    entries = load_manifest(manifest_path)

    print(f"[train_catboost] Building feature matrix from {len(entries)} entries...")
    X, y, skipped = build_feature_matrix(entries)
    print(f"[train_catboost] Feature matrix: {X.shape}, skipped: {len(skipped)}")
    print_class_distribution(y)

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    print(f"\n[train_catboost] Train: {len(X_train)}, Test: {len(X_test)}")

    print("\n[train_catboost] Training CatBoostClassifier...")
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        verbose=50,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=42,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # Evaluation
    y_pred = model.predict(X_test).flatten().astype(int)
    accuracy = float(np.mean(y_pred == y_test))
    print(f"\n[train_catboost] Test accuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=STROKE_LABELS,
        zero_division=0,
    ))

    print("Confusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    header = "".join(f"{name[:8]:>10}" for name in STROKE_LABELS)
    print(f"{'':>12}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>10}" for v in row)
        print(f"{STROKE_LABELS[i][:12]:>12}{row_str}")

    # Save model
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_path)
    print(f"\n[train_catboost] Model saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CatBoost baseline stroke classifier"
    )
    parser.add_argument(
        "--manifest",
        default="backend/training/data/courtcheck_manifest_2026-05-12.json",
        help="Path to manifest.json with keypoints_path entries",
    )
    parser.add_argument(
        "--output",
        default="backend/weights/stroke_catboost.cbm",
        help="Output path for the trained CatBoost model",
    )
    args = parser.parse_args()

    train_and_evaluate(args.manifest, args.output)


if __name__ == "__main__":
    main()
