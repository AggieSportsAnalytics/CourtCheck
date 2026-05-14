"""
Eval harness: run pipeline on a test clip, diff against hand-labeled ground truth.

Usage:
    python -m backend.eval.run_eval data/StMarys_Court2_4950_clip.mp4
    python -m backend.eval.run_eval data/StMarys_Court2_4950_clip.mp4 --skip-pipeline

Requires <video>.gt.json next to the clip (produced by annotate.py).

Outputs:
    <video>.pred.json           cached pipeline events (skip --skip-pipeline to refresh)
    backend/eval/runs/<ts>.md   markdown diff report
"""

import argparse
import json
import sys
import time
from pathlib import Path


STROKE_TOLERANCE_FRAMES = 5
BOUNCE_TOLERANCE_FRAMES = 10

PRED_LABEL_TO_GT = {
    "Forehand": "FH",
    "Backhand": "BH",
    "Serve/Overhead": "Serve",
    "Slice": "Volley",
}


def normalize_pred_label(label: str) -> str:
    return PRED_LABEL_TO_GT.get(label, label)


def match_strokes(gt: list, pred: list):
    matched_pred = set()
    correct, misclassified = [], []

    for g in gt:
        best_pj, best_dist = None, STROKE_TOLERANCE_FRAMES + 1
        for pj, p in enumerate(pred):
            if pj in matched_pred:
                continue
            d = abs(g["frame"] - p["frame"])
            if d < best_dist:
                best_dist, best_pj = d, pj
        if best_pj is None:
            continue
        matched_pred.add(best_pj)
        p = pred[best_pj]
        offset = p["frame"] - g["frame"]
        record = {"gt": g, "pred": p, "offset": offset}
        if normalize_pred_label(p["label"]) == g["type"]:
            correct.append(record)
        else:
            misclassified.append(record)

    matched_gt_frames = {c["gt"]["frame"] for c in correct} | {m["gt"]["frame"] for m in misclassified}
    misses = [g for g in gt if g["frame"] not in matched_gt_frames]
    false_positives = [p for pj, p in enumerate(pred) if pj not in matched_pred]
    return correct, misclassified, misses, false_positives


def match_bounces(gt: list, pred: list):
    matched_pred = set()
    matches = []

    for g in gt:
        best_pj, best_dist = None, BOUNCE_TOLERANCE_FRAMES + 1
        for pj, p in enumerate(pred):
            if pj in matched_pred:
                continue
            d = abs(g["frame"] - p["frame"])
            if d < best_dist:
                best_dist, best_pj = d, pj
        if best_pj is None:
            continue
        matched_pred.add(best_pj)
        matches.append({"gt": g, "pred": pred[best_pj], "offset": pred[best_pj]["frame"] - g["frame"]})

    matched_gt_frames = {m["gt"]["frame"] for m in matches}
    misses = [g for g in gt if g["frame"] not in matched_gt_frames]
    false_positives = [p for pj, p in enumerate(pred) if pj not in matched_pred]
    return matches, misses, false_positives


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def build_report(video_path: Path, gt: dict, pred: dict) -> str:
    correct, misclass, miss_strokes, fp_strokes = match_strokes(gt["strokes"], pred["strokes"])
    bounce_matches, miss_bounces, fp_bounces = match_bounces(gt["bounces"], pred["bounces"])

    n_gt = len(gt["strokes"])
    n_pred = len(pred["strokes"])
    n_matched = len(correct) + len(misclass)
    det_recall = n_matched / n_gt if n_gt else 0
    det_precision = n_matched / n_pred if n_pred else 0
    det_f1 = f1(det_precision, det_recall)
    cls_acc = len(correct) / n_matched if n_matched else 0

    classes = sorted({s["type"] for s in gt["strokes"]} | {normalize_pred_label(s["label"]) for s in pred["strokes"]})
    confusion = {c: {c2: 0 for c2 in classes} for c in classes}
    for record in correct + misclass:
        gt_c = record["gt"]["type"]
        pred_c = normalize_pred_label(record["pred"]["label"])
        confusion[gt_c][pred_c] = confusion[gt_c].get(pred_c, 0) + 1

    per_class = {}
    for c in classes:
        tp = confusion[c].get(c, 0)
        fn = sum(v for k, v in confusion[c].items() if k != c)
        fp = sum(confusion[c2].get(c, 0) for c2 in classes if c2 != c)
        rec = tp / (tp + fn) if (tp + fn) else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        per_class[c] = {"tp": tp, "fn": fn, "fp": fp, "precision": prec, "recall": rec}

    bn_gt = len(gt["bounces"])
    bn_pred = len(pred["bounces"])
    bn_matched = len(bounce_matches)
    b_recall = bn_matched / bn_gt if bn_gt else 0
    b_precision = bn_matched / bn_pred if bn_pred else 0
    b_f1 = f1(b_precision, b_recall)

    in_correct = sum(1 for m in bounce_matches if m["gt"]["in_bounds"] == m["pred"]["in_bounds"])
    in_acc = in_correct / bn_matched if bn_matched else 0

    fps = gt.get("fps", pred.get("fps", 30.0))

    def t(frame: int) -> str:
        return f"{frame / fps:.2f}s"

    L = []
    L.append(f"# Eval Report")
    L.append("")
    L.append(f"**Video:** `{video_path}`")
    L.append(f"**FPS:** {fps}   **Frames:** {gt.get('total_frames')}")
    L.append(f"**Tolerances:** stroke = ±{STROKE_TOLERANCE_FRAMES} frames, bounce = ±{BOUNCE_TOLERANCE_FRAMES} frames")
    L.append("")
    L.append("## Headline")
    L.append("")
    L.append(f"- Strokes: P={det_precision:.2f}  R={det_recall:.2f}  F1={det_f1:.2f}  Class-acc={cls_acc:.2f}  ({len(correct)}/{n_matched} matched events classified correctly)")
    L.append(f"- Bounces: P={b_precision:.2f}  R={b_recall:.2f}  F1={b_f1:.2f}  In/Out-acc={in_acc:.2f}")
    L.append("")
    L.append("## Strokes")
    L.append("")
    L.append(f"- GT events: {n_gt}   Pred events: {n_pred}   Matched: {n_matched}")
    L.append("")
    L.append("### Per-class")
    L.append("")
    L.append("| Class | TP | FN | FP | Precision | Recall |")
    L.append("|-------|----|----|----|-----------|--------|")
    for c in classes:
        m = per_class[c]
        L.append(f"| {c} | {m['tp']} | {m['fn']} | {m['fp']} | {m['precision']:.2f} | {m['recall']:.2f} |")
    L.append("")

    if classes:
        L.append("### Confusion (rows=GT, cols=Pred)")
        L.append("")
        header = "| GT \\ Pred | " + " | ".join(classes) + " |"
        L.append(header)
        L.append("|" + "---|" * (len(classes) + 1))
        for r in classes:
            row = [r] + [str(confusion[r].get(c, 0)) for c in classes]
            L.append("| " + " | ".join(row) + " |")
        L.append("")

    L.append(f"### Misses ({len(miss_strokes)}) — GT events with no pred within tolerance")
    L.append("")
    for m in miss_strokes:
        L.append(f"- frame {m['frame']} ({t(m['frame'])})  P{m['player']}  {m['type']}")
    L.append("")

    L.append(f"### False positives ({len(fp_strokes)}) — pred events not in GT")
    L.append("")
    for f_ev in fp_strokes:
        L.append(f"- frame {f_ev['frame']} ({t(f_ev['frame'])})  P{f_ev.get('player','?')}  {normalize_pred_label(f_ev['label'])}")
    L.append("")

    L.append(f"### Misclassifications ({len(misclass)})")
    L.append("")
    for r in misclass:
        L.append(f"- frame {r['gt']['frame']} ({t(r['gt']['frame'])}): GT={r['gt']['type']}  Pred={normalize_pred_label(r['pred']['label'])}  (offset {r['offset']:+d}f)")
    L.append("")

    L.append("## Bounces")
    L.append("")
    L.append(f"- GT: {bn_gt}   Pred: {bn_pred}   Matched: {bn_matched}")
    L.append("")
    L.append(f"### Bounce misses ({len(miss_bounces)})")
    L.append("")
    for m in miss_bounces:
        L.append(f"- frame {m['frame']} ({t(m['frame'])})  {'IN' if m['in_bounds'] else 'OUT'}")
    L.append("")
    L.append(f"### Bounce false positives ({len(fp_bounces)})")
    L.append("")
    for f_ev in fp_bounces:
        L.append(f"- frame {f_ev['frame']} ({t(f_ev['frame'])})  {'IN' if f_ev.get('in_bounds') else 'OUT'}")
    L.append("")

    L.append(f"### Bounce in/out mismatches")
    L.append("")
    mismatches = [m for m in bounce_matches if m["gt"]["in_bounds"] != m["pred"]["in_bounds"]]
    if not mismatches:
        L.append("- none")
    else:
        for m in mismatches:
            L.append(f"- frame {m['gt']['frame']} ({t(m['gt']['frame'])}): GT={'IN' if m['gt']['in_bounds'] else 'OUT'}  Pred={'IN' if m['pred']['in_bounds'] else 'OUT'}")
    L.append("")

    return "\n".join(L)


def print_console_summary(gt: dict, pred: dict) -> None:
    correct, misclass, miss, fp = match_strokes(gt["strokes"], pred["strokes"])
    bm, _bmiss, _bfp = match_bounces(gt["bounces"], pred["bounces"])
    n_gt = len(gt["strokes"]) or 1
    n_pred = len(pred["strokes"]) or 1
    n_match = len(correct) + len(misclass)
    sp = n_match / n_pred
    sr = n_match / n_gt
    sa = (len(correct) / n_match) if n_match else 0
    bn_gt = len(gt["bounces"]) or 1
    bn_pred = len(pred["bounces"]) or 1
    bp = len(bm) / bn_pred
    br = len(bm) / bn_gt
    print()
    print("=" * 64)
    print(f"STROKES  P={sp:.2f}  R={sr:.2f}  F1={f1(sp, sr):.2f}  Class-acc={sa:.2f}")
    print(f"BOUNCES  P={bp:.2f}  R={br:.2f}  F1={f1(bp, br):.2f}")
    print("=" * 64)


def run_pipeline_on_clip(video_path: Path) -> dict:
    from backend.pipeline.run import run_pipeline

    result = run_pipeline(str(video_path), match_id="eval-test", local_mode=True)
    events = result.get("events")
    if not events:
        raise RuntimeError("Pipeline returned no events. Make sure run.py exports `events` in local_mode.")
    return events


def main():
    ap = argparse.ArgumentParser(description="Eval pipeline output against hand-labeled ground truth.")
    ap.add_argument("video", help="Path to test clip")
    ap.add_argument("--skip-pipeline", action="store_true", help="Reuse cached <video>.pred.json instead of re-running pipeline.")
    args = ap.parse_args()

    video_path = Path(args.video).resolve()
    gt_path = video_path.with_suffix(video_path.suffix + ".gt.json")
    pred_path = video_path.with_suffix(video_path.suffix + ".pred.json")

    if not gt_path.exists():
        print(f"ERROR: ground truth not found at {gt_path}", file=sys.stderr)
        print(f"  Run: python -m backend.eval.annotate {video_path}", file=sys.stderr)
        sys.exit(1)
    gt = json.loads(gt_path.read_text())

    if args.skip_pipeline:
        if not pred_path.exists():
            print(f"ERROR: --skip-pipeline set but no cache at {pred_path}", file=sys.stderr)
            sys.exit(1)
        pred = json.loads(pred_path.read_text())
        print(f"Loaded cached pipeline output from {pred_path}")
    else:
        print(f"Running pipeline on {video_path} ...")
        pred = run_pipeline_on_clip(video_path)
        pred_path.write_text(json.dumps(pred, indent=2))
        print(f"Cached pipeline output -> {pred_path}")

    runs_dir = Path(__file__).resolve().parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    out_path = runs_dir / f"{timestamp}_{video_path.stem}.md"
    out_path.write_text(build_report(video_path, gt, pred))

    print_console_summary(gt, pred)
    print(f"Full report -> {out_path}")


if __name__ == "__main__":
    main()
