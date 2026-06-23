"""
TCN stroke-classifier training on CourtCheck annotations.

Trained from scratch on UC Davis labels. No THETIS pretraining.

Recipe:
- Stratified train/val split within the CourtCheck manifest
- Augmentation: temporal jitter, rotation, scale, shear, joint jitter,
  joint masking, horizontal flip (suppressed for Serve), JMDA joint mixing
- Class-weighted CE with label smoothing, WeightedRandomSampler over classes
- AdamW + cosine LR + early stopping on val loss

Usage:
    # Single train/val run (default 85/15 split, stratified)
    python -m backend.training.train_tcn \\
        --manifest backend/training/data/courtcheck_manifest_2026-05-12.json

    # 5-fold stratified cross-validation
    python -m backend.training.train_tcn --cv \\
        --manifest backend/training/data/courtcheck_manifest_2026-05-12.json

    # Custom config
    python -m backend.training.train_tcn --config path/to/config.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from backend.models.stroke_classifier_tcn import DERIVATIVE_ORDERS, STROKE_LABELS, StrokeTCN
from backend.training.config import TrainingConfig
from backend.training.features import normalize_keypoints, temporal_derivatives


LABEL_TO_IDX = {label: i for i, label in enumerate(STROKE_LABELS)}
SERVE_IDX = STROKE_LABELS.index("Serve/Overhead")
BACKHAND_IDX = STROKE_LABELS.index("Backhand")

# Backhand gets extra sampler weight to counter the persistent BH->FH bias
# observed in evaluation. Pure inverse-frequency sampling alone wasn't enough.
BACKHAND_OVERSAMPLE_FACTOR = 1.25

# COCO-17 left/right pairs to swap on horizontal flip
FLIP_PAIRS = [
    (5, 6),   # shoulders
    (7, 8),   # elbows
    (9, 10),  # wrists
    (11, 12), # hips
    (13, 14), # knees
    (15, 16), # ankles
]


# ---------------------------------------------------------------------------
# Augmentation primitives — operate on (T, 17, 3) raw keypoints (x, y, conf)
# ---------------------------------------------------------------------------

def _horizontal_flip(kp: np.ndarray) -> np.ndarray:
    out = kp.copy()
    for li, ri in FLIP_PAIRS:
        out[:, li, :] = kp[:, ri, :]
        out[:, ri, :] = kp[:, li, :]
    out[:, :, 0] = -out[:, :, 0]
    return out


def _apply_affine(kp: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply 2x2 affine matrix M to (x, y) channels. Confidence untouched."""
    out = kp.copy()
    xy = out[:, :, :2]                    # (T, 17, 2)
    cx = xy[:, :, 0].mean()
    cy = xy[:, :, 1].mean()
    centered = xy - np.array([cx, cy])
    transformed = centered @ M.T
    out[:, :, :2] = transformed + np.array([cx, cy])
    return out


def _rotate(kp: np.ndarray, deg: float) -> np.ndarray:
    theta = math.radians(deg)
    c, s = math.cos(theta), math.sin(theta)
    M = np.array([[c, -s], [s, c]], dtype=np.float32)
    return _apply_affine(kp, M)


def _scale(kp: np.ndarray, sx: float, sy: float) -> np.ndarray:
    M = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float32)
    return _apply_affine(kp, M)


def _shear(kp: np.ndarray, deg: float) -> np.ndarray:
    t = math.tan(math.radians(deg))
    M = np.array([[1.0, t], [0.0, 1.0]], dtype=np.float32)
    return _apply_affine(kp, M)


def _temporal_jitter(kp: np.ndarray, max_shift: int) -> np.ndarray:
    t = kp.shape[0]
    if max_shift <= 0 or t <= max_shift * 2 + 1:
        return kp
    start = np.random.randint(0, max_shift + 1)
    end = t - np.random.randint(0, max_shift + 1)
    return kp[start:end]


def _joint_jitter(kp: np.ndarray, sigma_norm: float, scale: float) -> np.ndarray:
    """Add Gaussian noise to (x, y). sigma is in normalized units, scaled to pixels."""
    out = kp.copy()
    noise = np.random.normal(0, sigma_norm * scale, kp[:, :, :2].shape).astype(np.float32)
    out[:, :, :2] += noise
    return out


_PROTECTED_JOINTS = {5, 6, 11, 12}  # shoulders + hips — required by hip-center normalization


def _joint_mask(kp: np.ndarray, prob: float) -> np.ndarray:
    """Zero a random subset of joints across the whole sequence.
    Hips and shoulders are protected because hip-center normalization depends on them.
    """
    out = kp.copy()
    for j in range(kp.shape[1]):
        if j in _PROTECTED_JOINTS:
            continue
        if np.random.random() < prob:
            out[:, j, :2] = 0.0
    return out


def _jmda(kp_a: np.ndarray, kp_b: np.ndarray, ratio: float) -> np.ndarray:
    """Joint-Mixing Data Aug: replace a random subset of joints in A with B's."""
    if kp_a.shape != kp_b.shape:
        return kp_a
    n_joints = kp_a.shape[1]
    n_swap = max(1, int(n_joints * ratio))
    swap_idx = np.random.choice(n_joints, size=n_swap, replace=False)
    out = kp_a.copy()
    out[:, swap_idx, :] = kp_b[:, swap_idx, :]
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StrokeDataset(Dataset):
    """Loads keypoint sequences from .npy files and applies the augmentation
    pipeline. Returns (seq_len, 34) tensors after hip-center normalization."""

    def __init__(
        self,
        entries: list[dict],
        cfg: TrainingConfig,
        augment: bool,
        pose_pixel_scale: float = 1080.0,
    ):
        self.cfg = cfg
        self.augment = augment
        self.pose_pixel_scale = pose_pixel_scale
        self.samples: list[tuple[str, int]] = []

        for e in entries:
            kp_path = e.get("keypoints_path")
            label = e.get("mapped_label")
            if not kp_path or not os.path.exists(kp_path):
                continue
            if not e.get("valid", True):
                continue
            if label not in LABEL_TO_IDX:
                continue
            self.samples.append((kp_path, LABEL_TO_IDX[label]))

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def labels(self) -> list[int]:
        return [lbl for _, lbl in self.samples]

    def class_counts(self) -> np.ndarray:
        counts = np.zeros(len(STROKE_LABELS), dtype=np.int64)
        for _, lbl in self.samples:
            counts[lbl] += 1
        return counts

    def _augment_kp(self, kp: np.ndarray, label: int) -> np.ndarray:
        cfg = self.cfg

        kp = _temporal_jitter(kp, cfg.aug_temporal_jitter)

        # Horizontal flip is disabled — flipping a forehand produces a
        # backhand-like pose, creating direct label noise between the two
        # most-confused classes. Earlier results: 48/53 errors were FH<->BH.

        # Light rotation only — large rotations distort body orientation,
        # which is the primary discriminator between Forehand and Backhand.
        if np.random.random() < 0.5:
            small_rot = cfg.aug_rotation_deg * 0.5  # was 15deg, now 7.5
            kp = _rotate(kp, np.random.uniform(-small_rot, small_rot))

        if np.random.random() < 0.5:
            sx = np.random.uniform(cfg.aug_scale_min, cfg.aug_scale_max)
            sy = np.random.uniform(cfg.aug_scale_min, cfg.aug_scale_max)
            kp = _scale(kp, sx, sy)

        # Shear removed — it distorts the body orientation that FH/BH classification depends on.

        if np.random.random() < 0.8:
            kp = _joint_jitter(kp, cfg.aug_joint_jitter_sigma, self.pose_pixel_scale)

        if np.random.random() < 0.5:
            kp = _joint_mask(kp, cfg.aug_joint_mask_prob)

        return kp

    def _load_kp(self, idx: int) -> tuple[np.ndarray, int]:
        kp_path, lbl = self.samples[idx]
        kp = np.load(kp_path).astype(np.float32)
        return kp, lbl

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        kp, label = self._load_kp(idx)

        if self.augment:
            kp = self._augment_kp(kp, label)

            # JMDA: mix with another same-class sample
            if np.random.random() < self.cfg.aug_jmda_prob:
                same_class = [i for i, (_, l) in enumerate(self.samples) if l == label and i != idx]
                if same_class:
                    j = random.choice(same_class)
                    kp_b, _ = self._load_kp(j)
                    if kp_b.shape[0] >= kp.shape[0]:
                        kp_b = kp_b[: kp.shape[0]]
                    else:
                        # pad by repeating last frame
                        pad = np.repeat(kp_b[-1:], kp.shape[0] - kp_b.shape[0], axis=0)
                        kp_b = np.concatenate([kp_b, pad], axis=0)
                    kp = _jmda(kp, kp_b, self.cfg.aug_jmda_ratio)

        seq = normalize_keypoints(kp, self.cfg.seq_len)  # (seq_len, 34)
        # Clamp to handle outliers from heavy aug + degenerate torso detections.
        np.clip(seq, -5.0, 5.0, out=seq)
        # Append temporal derivatives (velocity carries direction-of-motion
        # signal critical for FH vs BH discrimination).
        seq = temporal_derivatives(seq, orders=DERIVATIVE_ORDERS)
        return torch.from_numpy(seq), label


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def load_manifest_entries(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def filter_usable(entries: list[dict]) -> list[dict]:
    """Drop entries with no keypoints, invalid flag, or non-supported label."""
    out = []
    for e in entries:
        kp_path = e.get("keypoints_path")
        if not kp_path or not os.path.exists(kp_path):
            continue
        if e.get("valid") is False:
            continue
        if e.get("mapped_label") not in LABEL_TO_IDX:
            continue
        out.append(e)
    return out


def stratified_split(entries: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Per-class stratified split."""
    rng = np.random.RandomState(seed)
    by_class: dict[int, list[dict]] = {}
    for e in entries:
        by_class.setdefault(LABEL_TO_IDX[e["mapped_label"]], []).append(e)
    train, val = [], []
    for cls, items in by_class.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio)))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def stratified_kfold(entries: list[dict], n_folds: int, seed: int) -> list[tuple[list[dict], list[dict]]]:
    rng = np.random.RandomState(seed)
    by_class: dict[int, list[dict]] = {}
    for e in entries:
        by_class.setdefault(LABEL_TO_IDX[e["mapped_label"]], []).append(e)
    for items in by_class.values():
        rng.shuffle(items)

    folds: list[tuple[list[dict], list[dict]]] = []
    for k in range(n_folds):
        train, val = [], []
        for cls, items in by_class.items():
            fold_size = len(items) / n_folds
            start = int(round(k * fold_size))
            end = int(round((k + 1) * fold_size))
            val.extend(items[start:end])
            train.extend(items[:start] + items[end:])
        rng.shuffle(train)
        rng.shuffle(val)
        folds.append((train, val))
    return folds


def build_loaders(
    train_entries: list[dict],
    val_entries: list[dict],
    cfg: TrainingConfig,
) -> tuple[DataLoader, DataLoader, StrokeDataset, StrokeDataset]:
    train_ds = StrokeDataset(train_entries, cfg, augment=cfg.augment)
    val_ds = StrokeDataset(val_entries, cfg, augment=False)

    counts = train_ds.class_counts().astype(np.float64)
    counts = np.clip(counts, 1, None)
    class_weights = 1.0 / counts
    # Extra weight for backhand to counter persistent BH->FH bias.
    class_weights[BACKHAND_IDX] *= BACKHAND_OVERSAMPLE_FACTOR
    sample_weights = np.array([class_weights[lbl] for _, lbl in train_ds.samples], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=(cfg.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(cfg.device == "cuda"),
    )
    return train_loader, val_loader, train_ds, val_ds


# ---------------------------------------------------------------------------
# Train / eval epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.set_grad_enabled(is_train):
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            logits = model(sequences)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum())
            total += len(labels)
    return total_loss / max(total, 1), correct / max(total, 1)


# ---------------------------------------------------------------------------
# Single-run training
# ---------------------------------------------------------------------------

def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return requested


def train_one_split(
    cfg: TrainingConfig,
    train_entries: list[dict],
    val_entries: list[dict],
    weights_filename: str,
    run_name: str,
) -> dict:
    device = resolve_device(cfg.device)
    print(f"[train] device={device}")

    train_loader, val_loader, train_ds, val_ds = build_loaders(train_entries, val_entries, cfg)

    counts = train_ds.class_counts()
    print(f"[train] train samples: {len(train_ds)}, val samples: {len(val_ds)}")
    print(f"[train] train class counts: " + ", ".join(
        f"{STROKE_LABELS[i]}={counts[i]}" for i in range(len(STROKE_LABELS))
    ))

    model = StrokeTCN(
        input_dim=cfg.input_dim,
        n_classes=cfg.num_classes,
        dropout=cfg.dropout,
    ).to(device)

    # Focal loss focuses gradient on hard examples — specifically the FH<->BH
    # boundary that is the dominant remaining failure mode. Built as a wrapper
    # around CrossEntropyLoss with label smoothing.
    base_ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing, reduction="none")

    def criterion(logits, targets):
        ce = base_ce(logits, targets)
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_(1e-6, 1.0)
        focal_weight = (1.0 - pt) ** 2.0
        return (focal_weight * ce).mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    weights_dir = Path(cfg.weights_output_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_path = weights_dir / weights_filename

    log_dir = f"backend/training/runs/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[train] tensorboard: {log_dir}")

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, False)
        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})

        improved = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
        marker = ""
        if improved:
            best_val_acc = max(best_val_acc, val_acc)
            best_val_loss = min(best_val_loss, val_loss)
            torch.save(model.state_dict(), best_path)
            patience = 0
            marker = "  [+ saved]"
        else:
            patience += 1

        print(
            f"Epoch {epoch:3d}/{cfg.epochs} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f}{marker}"
        )

        if patience >= cfg.patience:
            print(f"[train] Early stopping at epoch {epoch} (no val-acc improvement for {patience} epochs).")
            break

    writer.close()
    print(f"[train] best val acc: {best_val_acc:.4f}  weights: {best_path}")
    return {
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "weights_path": str(best_path),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Cross-validation orchestration
# ---------------------------------------------------------------------------

def run_cv(cfg: TrainingConfig, entries: list[dict]) -> None:
    folds = stratified_kfold(entries, cfg.n_folds, cfg.random_seed)
    accs: list[float] = []
    fold_paths: list[str] = []
    for k, (train_entries, val_entries) in enumerate(folds, 1):
        print(f"\n=== Fold {k}/{cfg.n_folds} ===")
        result = train_one_split(
            cfg,
            train_entries,
            val_entries,
            weights_filename=f"stroke_classifier_tcn_fold{k}.pt",
            run_name=f"tcn_cv_fold{k}",
        )
        accs.append(result["best_val_acc"])
        fold_paths.append(result["weights_path"])

    mean = float(np.mean(accs))
    std = float(np.std(accs))
    print(f"\n[cv] {cfg.n_folds}-fold val acc: mean={mean:.4f} std={std:.4f}")
    print(f"[cv] per-fold: " + ", ".join(f"{a:.4f}" for a in accs))

    # Best fold becomes the canonical weights file
    best_k = int(np.argmax(accs))
    best_path = fold_paths[best_k]
    canonical = Path(cfg.weights_output_dir) / "stroke_classifier_tcn.pt"
    import shutil
    shutil.copyfile(best_path, canonical)
    print(f"[cv] best fold: {best_k + 1} (acc {accs[best_k]:.4f}) -> copied to {canonical}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train CourtCheck stroke TCN")
    parser.add_argument("--config", default=None, help="Path to TrainingConfig JSON")
    parser.add_argument("--manifest", default=None, help="Override courtcheck manifest path")
    parser.add_argument("--cv", action="store_true", help="Run 5-fold stratified cross-validation")
    parser.add_argument("--device", default=None, help="Override device (cuda/mps/cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    args = parser.parse_args()

    cfg = TrainingConfig.from_json(args.config) if args.config else TrainingConfig()
    if args.manifest:
        cfg.courtcheck_manifest = args.manifest
    if args.device:
        cfg.device = args.device
    if args.epochs:
        cfg.epochs = args.epochs

    # Reproducibility
    np.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    entries = load_manifest_entries(cfg.courtcheck_manifest)
    entries = filter_usable(entries)
    print(f"[train] usable manifest entries: {len(entries)}")

    if not entries:
        print("[train] No usable entries. Run keypoint extraction first.")
        return

    by_class: dict[str, int] = {}
    for e in entries:
        by_class[e["mapped_label"]] = by_class.get(e["mapped_label"], 0) + 1
    print("[train] usable entries by class:")
    for label, count in sorted(by_class.items()):
        print(f"  {label}: {count}")

    if args.cv:
        run_cv(cfg, entries)
    else:
        train_entries, val_entries = stratified_split(entries, cfg.val_split_ratio, cfg.random_seed)
        train_one_split(
            cfg,
            train_entries,
            val_entries,
            weights_filename="stroke_classifier_tcn.pt",
            run_name="tcn_single",
        )


if __name__ == "__main__":
    main()
