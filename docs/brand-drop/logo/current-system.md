# The logo system (locked)

> Final canonical logo assets. Use these files. Do not iterate.

---

## Assets

```
logo/
├── CourtCheckLogoLight.png       ← Light master (cream canvas)
├── CourtCheckLogoDark.png        ← Dark master (near-black canvas)
└── CourtCheckAnimation.mp4    ← Splash animation (5s, ends on locked state)
```

All assets are the source-of-truth versions. Do not regenerate, redraw, or substitute.

---

## The mark, the wordmark, the lockup

The full lockup has three elements that always appear together:

1. **The mark** — hand-drawn checkmark stroke + lime tennis ball at the apex
2. **The wordmark** — "CourtCheck" with "Court" upright dark, "Check" italic lime
3. **The lockup** — mark and wordmark side by side with consistent optical spacing

The mark is hand-drawn editorial (matches the brand's illustration register). The wordmark is set in Newsreader serif. Together they read as one unit.

---

## Variants we ship

| Variant | File | Use |
|---|---|---|
| Light master | `CourtCheckLogoLight.png` | Default on light backgrounds (cream / paper / surface) |
| Dark master | `CourtCheckLogoDark.png` | Default on dark backgrounds (near-black / surface-dark) |
| Animated splash | `CourtCheckAnimation.mp4` | Page-load intro, splash screen, first-session moments |

---

## What we don't ship (and won't, until requested)

- Monogram-only variant (no mark without wordmark)
- Badge/circular-stamp variant
- Watermark variant
- Animated loop (the animation is a one-shot splash, not a continuous loop)

If a use case genuinely needs one of these, surface it as a request rather than improvising.

---

## Display rules

- Minimum size: lockup renders cleanly down to about 80px wide. Below that, crop to mark-only (no wordmark) using the upper-left quadrant of the master image.
- Padding: leave at least the height of the mark as clear space on all sides
- Background: only on `--cream`, `--paper`, or `--surface` (light mode) — only on `--cream-dark`, `--paper-dark`, or `--surface-dark` (dark mode). Never on `--clay`, `--court`, or any saturated color
- Rotation: do not rotate, skew, or warp
- Recoloring: do not. Light and dark variants are exhaustive

---

## How the animation works

`CourtCheckAnimation.mp4` plays once and settles on the static lockup. Sequence:

1. Ball enters from off-screen right with an aerial arc
2. Ball descends the short left checkmark stroke (ink draws behind)
3. Ball bounces at the pivot (clay puff)
4. Ball ascends the long right stroke (ink draws)
5. Ball settles at the apex
6. "Court" inks in left-to-right
7. "Check" inks in left-to-right (italic lime)
8. Final state holds — matches the static light master pixel-for-pixel

5 seconds total. No audio. Loops would be inappropriate — this is a one-time reveal moment, not a continuous animation.

---

*This file replaces the previous logo system inventory. The 26 SVG experiments referenced in the prior version are obsolete. This is the canonical record.*
