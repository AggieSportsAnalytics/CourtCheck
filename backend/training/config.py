"""
Training configuration for the CourtCheck stroke classifier.

All hyperparameters in one place. Can be serialized to/from JSON for
reproducible training runs.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    courtcheck_manifest: str = "backend/training/data/courtcheck_manifest_2026-05-12.json"
    weights_output_dir: str = "backend/weights/"
    val_split_ratio: float = 0.15
    n_folds: int = 5  # used only when --cv is passed
    random_seed: int = 42

    # -------------------------------------------------------------------------
    # Model — must match StrokeTCN constants in stroke_classifier_tcn.py
    # -------------------------------------------------------------------------
    seq_len: int = 45
    input_dim: int = 68   # 17 keypoints * 2 coords * 2 (position + velocity)
    num_classes: int = 3  # Forehand, Backhand, Serve/Overhead
    dropout: float = 0.3

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 120
    patience: int = 15        # early stopping on val loss
    weight_decay: float = 5e-2
    label_smoothing: float = 0.1
    augment: bool = True
    device: str = "mps"

    # -------------------------------------------------------------------------
    # Augmentation strengths
    # -------------------------------------------------------------------------
    aug_flip_prob: float = 0.5            # disabled for serve class at dataset level
    aug_rotation_deg: float = 15.0
    aug_scale_min: float = 0.9
    aug_scale_max: float = 1.1
    aug_shear_deg: float = 6.0
    aug_joint_jitter_sigma: float = 0.02  # in normalized units
    aug_joint_mask_prob: float = 0.15
    aug_temporal_jitter: int = 3
    aug_jmda_prob: float = 0.3            # joint-mixing aug
    aug_jmda_ratio: float = 0.5

    def to_json(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def weights_path(self, filename: str) -> str:
        return str(Path(self.weights_output_dir) / filename)
