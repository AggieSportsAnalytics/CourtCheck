"""
Seed `<clip>.gt.json` with the current CatBoost detector's bounce predictions.

The annotation review workflow then becomes:
    1. Open clip in annotate.py
    2. Press `n` to jump to next pre-labeled bounce
    3. ✓ correct → press `n` to advance
    4. ✗ wrong  → press `d` to delete it
    5. Missed bounce → press `i` / `o` at the contact frame
    6. `q` to save and quit

This turns annotation (~12 min/clip) into review (~5 min/clip) because the
detector already finds ~half the bounces correctly — you just verify + patch.

Pre-labeled bounces are written with `"auto_labeled": true` so train_bounce.py
can ignore the marker (it only reads frame + in_bounds) while annotate.py can
surface the distinction in the HUD.

Default in_bounds=True since most bounces are in. Brian flips to OUT by
deleting + re-stamping with `o` during review.

Usage:
    python -m backend.training.pre_label_bounces \
        --glob 'data/bounce_train/clips/*.mp4'

    python -m backend.training.pre_label_bounces --clip path/to/clip.mp4
"""
from __future__ import annotations

import argparse
import glob as glob_mod
import json
import os
import sys
from pathlib import Path

import numpy as np

from backend.models.bounce_detector import BounceDetector


_DEFAULT_WEIGHTS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "weights", "bounce_detection_weights.cbm")
)


def _seed_clip(
    clip_path: Path,
    detector: BounceDetector,
    overwrite_auto: bool,
) -> int:
    """Pre-label bounces into <clip>.gt.json. Returns count of seeds written."""
    ball_path = clip_path.with_suffix(clip_path.suffix + ".ball.npz")
    gt_path = clip_path.with_suffix(clip_path.suffix + ".gt.json")

    if not ball_path.exists():
        print(f"[pre_label] SKIP {clip_path.name}: no .ball.npz "
              "(run extract_ball_trajectory first)")
        return 0

    data = np.load(ball_path)
    x_ball = [float(v) if not np.isnan(v) else None for v in data["x_ball"]]
    y_ball = [float(v) if not np.isnan(v) else None for v in data["y_ball"]]
    fps = float(data["fps"])
    total = int(data["total"])

    bounce_frames = sorted(detector.predict(x_ball, y_ball, smooth=True))
    seeds = [
        {"frame": int(f), "in_bounds": True, "auto_labeled": True}
        for f in bounce_frames
    ]

    if gt_path.exists():
        existing = json.loads(gt_path.read_text())
        manual = [b for b in existing.get("bounces", []) if not b.get("auto_labeled")]
        prior_auto = [b for b in existing.get("bounces", []) if b.get("auto_labeled")]
        if prior_auto and not overwrite_auto:
            print(f"[pre_label] SKIP {clip_path.name}: gt.json already has "
                  f"{len(prior_auto)} auto-labeled bounces (--overwrite-auto to replace)")
            return 0
        merged = sorted(manual + seeds, key=lambda b: b["frame"])
        payload = {
            "video": existing.get("video", str(clip_path)),
            "fps": existing.get("fps", fps),
            "total_frames": existing.get("total_frames", total),
            "strokes": existing.get("strokes", []),
            "bounces": merged,
        }
        manual_kept = len(manual)
    else:
        payload = {
            "video": str(clip_path),
            "fps": fps,
            "total_frames": total,
            "strokes": [],
            "bounces": seeds,
        }
        manual_kept = 0

    gt_path.write_text(json.dumps(payload, indent=2))
    print(f"[pre_label] {clip_path.name}: seeded {len(seeds)} bounces "
          f"(kept {manual_kept} manual) → {gt_path.name}")
    return len(seeds)


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed gt.json with CatBoost bounce predictions")
    ap.add_argument("--clip", help="single clip .mp4")
    ap.add_argument("--glob", help="glob for multiple clips")
    ap.add_argument("--weights", default=_DEFAULT_WEIGHTS, help="CatBoost weights path")
    ap.add_argument("--threshold", type=float, default=0.18,
                    help="detection threshold (lower = more seeds, less precision)")
    ap.add_argument("--min-gap-frames", type=int, default=10,
                    help="minimum frames between adjacent bounce seeds")
    ap.add_argument("--overwrite-auto", action="store_true",
                    help="replace existing auto-labeled bounces (preserves manual ones)")
    args = ap.parse_args()

    if not args.clip and not args.glob:
        ap.error("--clip or --glob required")

    clips: list[Path] = []
    if args.clip:
        clips.append(Path(args.clip).resolve())
    if args.glob:
        clips.extend(sorted(Path(p).resolve() for p in glob_mod.glob(args.glob)))

    if not clips:
        print("no clips matched", file=sys.stderr)
        return 1

    detector = BounceDetector(
        path_model=args.weights,
        threshold=args.threshold,
        min_gap_frames=args.min_gap_frames,
    )

    total_seeds = 0
    for clip in clips:
        total_seeds += _seed_clip(clip, detector, args.overwrite_auto)
    print(f"\n[pre_label] done — {total_seeds} bounces seeded across {len(clips)} clips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
