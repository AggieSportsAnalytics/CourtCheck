"""
Evaluate a trained CourtCheck stroke TCN on its val split.

Loads saved weights, runs predictions on the same stratified val split used
during training (same seed + ratio), then writes a markdown report:
- top-1 / top-2 accuracy
- per-class precision / recall / F1
- confusion matrix
- top mis-predicted clip paths (for spot-checking)

Usage:
    python -m backend.training.eval_tcn \
        --weights backend/weights/stroke_classifier_tcn.pt \
        --manifest backend/training/data/courtcheck_manifest_2026-05-12.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from backend.models.stroke_classifier_tcn import DERIVATIVE_ORDERS, STROKE_LABELS, StrokeTCN
from backend.training.config import TrainingConfig
from backend.training.features import normalize_keypoints, temporal_derivatives
from backend.training.train_tcn import (
    LABEL_TO_IDX,
    filter_usable,
    load_manifest_entries,
    resolve_device,
    stratified_split,
)


def predict_all(entries, model, cfg, device):
    """Run the model on every entry. Returns (y_true, y_pred, probs, used_entries)."""
    y_true = []
    y_pred = []
    all_probs = []
    used = []

    model.train(False)
    with torch.no_grad():
        for e in entries:
            kp_path = e["keypoints_path"]
            label = LABEL_TO_IDX[e["mapped_label"]]
            kp = np.load(kp_path).astype(np.float32)
            seq = normalize_keypoints(kp, cfg.seq_len)
            seq = temporal_derivatives(seq, orders=DERIVATIVE_ORDERS)
            x = torch.from_numpy(seq).unsqueeze(0).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pred = int(np.argmax(probs))
            y_true.append(label)
            y_pred.append(pred)
            all_probs.append(probs)
            used.append(e)

    return (
        np.array(y_true, dtype=np.int64),
        np.array(y_pred, dtype=np.int64),
        np.stack(all_probs, axis=0),
        used,
    )


def per_class_prf(y_true, y_pred):
    out = []
    for cls_idx, cls_name in enumerate(STROKE_LABELS):
        tp = int(((y_pred == cls_idx) & (y_true == cls_idx)).sum())
        fp = int(((y_pred == cls_idx) & (y_true != cls_idx)).sum())
        fn = int(((y_pred != cls_idx) & (y_true == cls_idx)).sum())
        support = int((y_true == cls_idx).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append({
            "label": cls_name, "support": support,
            "precision": precision, "recall": recall, "f1": f1,
        })
    return out


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def top_k_accuracy(y_true, probs, k):
    top_k = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(int(y in row) for y, row in zip(y_true, top_k))
    return correct / len(y_true)


def render_report(cfg, weights_path, manifest_path, y_true, y_pred, probs, entries):
    lines = []
    lines.append("# CourtCheck Stroke TCN Eval - " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    lines.append("")
    lines.append("- Weights: `" + str(weights_path) + "`")
    lines.append("- Manifest: `" + str(manifest_path) + "`")
    lines.append("- Val samples: " + str(len(y_true)))
    lines.append("")

    top1 = float((y_true == y_pred).mean())
    top2 = top_k_accuracy(y_true, probs, k=2)
    lines.append("## Accuracy")
    lines.append("")
    lines.append(f"- **Top-1:** {top1:.4f}")
    lines.append(f"- **Top-2:** {top2:.4f}")
    lines.append("")

    lines.append("## Per-class P / R / F1")
    lines.append("")
    lines.append("| Class | Support | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in per_class_prf(y_true, y_pred):
        lines.append(
            f"| {row['label']} | {row['support']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )
    lines.append("")

    lines.append("## Confusion matrix (rows = true, cols = predicted)")
    lines.append("")
    header = "| true \\ pred | " + " | ".join(STROKE_LABELS) + " |"
    sep = "|---|" + "---:|" * len(STROKE_LABELS)
    lines.append(header)
    lines.append(sep)
    cm = confusion_matrix(y_true, y_pred, len(STROKE_LABELS))
    for i, row_name in enumerate(STROKE_LABELS):
        cells = " | ".join(str(int(cm[i, j])) for j in range(len(STROKE_LABELS)))
        lines.append(f"| {row_name} | {cells} |")
    lines.append("")

    mis_idx = np.where(y_true != y_pred)[0]
    confidences = probs[mis_idx, y_pred[mis_idx]]
    order = np.argsort(-confidences)
    lines.append(f"## Top mis-predicted clips ({len(mis_idx)} total)")
    lines.append("")
    lines.append("Sorted by model confidence in the wrong class. Spot-check these.")
    lines.append("")
    lines.append("| true | pred | conf | clip |")
    lines.append("|---|---|---:|---|")
    for rank in order[:15]:
        i = mis_idx[rank]
        true_lbl = STROKE_LABELS[int(y_true[i])]
        pred_lbl = STROKE_LABELS[int(y_pred[i])]
        conf = float(probs[i, int(y_pred[i])])
        clip_path = entries[i].get("clip_path", "?")
        annotator = entries[i].get("annotator", "")
        suffix = f" ({annotator})" if annotator else ""
        lines.append(f"| {true_lbl} | {pred_lbl} | {conf:.3f} | `{clip_path}`{suffix} |")
    lines.append("")

    correct_conf = probs[np.arange(len(y_true)), y_pred][y_true == y_pred]
    wrong_conf = probs[np.arange(len(y_true)), y_pred][y_true != y_pred]
    lines.append("## Confidence stats")
    lines.append("")
    lines.append(f"- Correct preds: mean confidence {float(np.mean(correct_conf)):.3f} (n={len(correct_conf)})")
    if len(wrong_conf):
        lines.append(f"- Wrong preds:  mean confidence {float(np.mean(wrong_conf)):.3f} (n={len(wrong_conf)})")
    else:
        lines.append("- Wrong preds:  none (100% accuracy)")
    lines.append("")

    lines.append("## Failure mode summary")
    lines.append("")
    confusions = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t != p:
            confusions[(int(t), int(p))] += 1
    ranked = sorted(confusions.items(), key=lambda kv: -kv[1])
    for (t, p), n in ranked[:6]:
        lines.append(f"- {STROKE_LABELS[t]} -> {STROKE_LABELS[p]}: {n} cases")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate stroke TCN on the val split")
    parser.add_argument("--weights", default="backend/weights/stroke_classifier_tcn.pt")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = TrainingConfig.from_json(args.config) if args.config else TrainingConfig()
    if args.manifest:
        cfg.courtcheck_manifest = args.manifest
    if args.device:
        cfg.device = args.device

    device = resolve_device(cfg.device)
    entries = filter_usable(load_manifest_entries(cfg.courtcheck_manifest))
    if not entries:
        raise SystemExit("No usable entries in manifest. Run keypoint extraction first.")

    _, val_entries = stratified_split(entries, cfg.val_split_ratio, cfg.random_seed)
    print(f"[eval_tcn] val entries: {len(val_entries)} (split seed {cfg.random_seed})")

    model = StrokeTCN(
        input_dim=cfg.input_dim,
        n_classes=cfg.num_classes,
        dropout=cfg.dropout,
    ).to(device)
    state = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.train(False)

    y_true, y_pred, probs, used = predict_all(val_entries, model, cfg, device)
    print(f"[eval_tcn] top-1: {(y_true == y_pred).mean():.4f}")

    report = render_report(
        cfg, args.weights, cfg.courtcheck_manifest,
        y_true, y_pred, probs, used,
    )

    if args.report_out:
        report_path = Path(args.report_out)
    else:
        runs_dir = Path("backend/training/runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = runs_dir / ("eval_" + ts + ".md")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"[eval_tcn] report -> {report_path}")


if __name__ == "__main__":
    main()
