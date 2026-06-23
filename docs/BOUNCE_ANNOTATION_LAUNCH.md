# Bounce annotation — launch instructions

## Status

**All 9 clips ready for review.** 545 bounces seeded total. ~$0.30 of Modal credits spent on extraction.

| Clip | Source | Seeds |
|------|--------|------:|
| `court2_pick01.mp4` | Court 2, 45:20 | 63 |
| `court2_pick02.mp4` | Court 2, 54:20 | 73 |
| `court2_pick03.mp4` | Court 2, 1:16:40 | 61 |
| `court4_pick01.mp4` | Court 4, 51:50 | 53 |
| `court4_pick02.mp4` | Court 4, 1:27:00 | 53 |
| `court4_pick03.mp4` | Court 4, 2:01:10–2:03:44 | 74 |
| `court6_pick01.mp4` | Court 6, 51:50 | 60 |
| `court6_pick02.mp4` | Court 6, 1:11:10 | 48 |
| `court6_pick03.mp4` | Court 6, 2:16:10 | 60 |

Pre-labeled at CatBoost threshold 0.30 — the current detector over-fires (real bounces per 2-min clip are ~30-50), so expect ~30-40% of seeds to be false positives you'll delete with `d`. Each delete is <1 sec; net review per clip is ~5-7 min.

## Try one clip first (~5-7 min)

```bash
cd /Users/Brian/PycharmProjects/CourtCheck
python -m backend.eval.annotate data/bounce_train/clips/court2_pick01.mp4
```

### Hotkeys (only the ones you need for review)

| Key | Action |
|-----|--------|
| `n` | Jump to next pre-labeled bounce |
| `p` | Jump to previous pre-labeled bounce |
| `d` | Delete the bounce nearest current frame |
| `i` | Add bounce IN-bounds at current frame |
| `o` | Add bounce OUT-of-bounds at current frame |
| `j` / `l` | Single-frame back / forward |
| `,` / `.` | ±10 frames |
| `u` | Undo |
| `w` | Save (stay open) |
| `q` | Save and quit |

HUD shows when you're on/near a seed: green text `[AUTO] bounce IN @ frame 1284 (Δ+0)`. `Δ+0` means current frame == bounce frame; `Δ-3` means seed is 3 frames ahead.

### Review loop per clip

1. Press `n` → jump to first seed.
2. Look at the ball at the highlighted frame.
   - ✓ **Yes, ball bounced here, in** → press `n` to advance.
   - ✗ **No bounce** → press `d` to delete, then `n`.
   - ✗ **Bounce was OUT-of-bounds** → press `d`, then `o` (re-stamps as OUT at current frame).
3. After the last seed (HUD says "no bounce after this frame"), scrub back with `,` for one final pass — stamp any missed bounces with `i` / `o`.
4. `q` to save and quit.

If the seed count feels way off — too many false positives or too many missed — ping me and I'll re-seed at a different threshold.

## After all clips: train + deploy (I drive)

When you've reviewed all 9 clips (3 done + 6 to come):

1. Train CatBoost on UCD labels:
   ```bash
   python -m backend.training.train_bounce \
     --clips-glob 'data/bounce_train/clips/*.mp4' \
     --output backend/weights/bounce_detection_weights_ucd.cbm
   ```
2. Read the auto-generated report (`backend/training/runs/bounce_retrain_<ts>.md`). Bar: **bounce-level recall ≥ 0.80 @ threshold 0.18 on val clips**.
3. If recall passes: flip `PipelineConfig.bounce_model_weights` to `'bounce_detection_weights_ucd.cbm'`, commit, push, deploy Modal.
4. Reprocess StMarys regression match via Reprocess button. Confirm the missing far-side bounces appear on the shotmap.

## If you want to re-seed at a different threshold

```bash
# Tighter (fewer seeds, fewer false positives, more missed bounces to add):
python -m backend.training.pre_label_bounces \
  --glob 'data/bounce_train/clips/court2_pick*.mp4' \
  --overwrite-auto --threshold 0.40

# Looser (more seeds, more false positives, fewer missed):
python -m backend.training.pre_label_bounces \
  --glob 'data/bounce_train/clips/court2_pick*.mp4' \
  --overwrite-auto --threshold 0.18
```

`--overwrite-auto` preserves any manual bounces you've added — only replaces the `[AUTO]` ones.
