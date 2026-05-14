# Logo: locked

> The CourtCheck logo system is finalized.
> Use the assets in this folder as-is. Do not iterate, refine, or regenerate.

---

## Locked assets in this folder

| File | What it is |
|---|---|
| `CourtCheckLogoLight.png` | Master lockup, light mode (cream paper canvas) |
| `CourtCheckLogoDark.png` | Master lockup, dark mode (near-black canvas) |
| `CourtCheckAnimation.mp4` | Splash animation, ball enters from off-screen right, traces the checkmark path, wordmark reveals with ink-fill effect |

---

## The mark, in plain language

- A single hand-drawn warm-grey ink stroke forms a checkmark with a short left descending arm and a long right ascending arm
- A lime tennis ball sits at the far tip of the right ascending arm, suspended above and to the right of the V
- The wordmark "CourtCheck" sits to the right of the mark
- "Court" in upright Newsreader serif (weight 500, opsz 60), color `--ink`
- "Check" in italic Newsreader serif (weight 500, opsz 60), color `--lime` — matching the ball

The mark is hand-drawn, slightly imperfect, brush-pen feel. Editorial register. No glow, no chrome, no soft-fade trails.

---

## Color values

### Light mode
- Canvas: `#F4F0E6` (cream)
- Mark stroke: `#43403A` (warm grey ink)
- Ball: `#BFC846` (muted lime)
- "Court": `#1A1815` (near-black ink)
- "Check": `#BFC846` (lime, italic)

### Dark mode
- Canvas: `#0A0E14` (near-black)
- Mark stroke: `#F4F0E6` (cream, inverted)
- Ball: `#DDE970` (brighter lime for dark contrast)
- "Court": `#F4F0E6` (cream, inverted)
- "Check": `#DDE970` (brighter lime, italic)

---

## How to use

- **Web nav bar:** `CourtCheckLogoLight.png` (or dark) scaled to 36px tall, full lockup
- **Favicon / app icon:** crop the master to just the mark + ball (no wordmark), render at the target size
- **Splash / intro:** `CourtCheckAnimation.mp4` plays once on first load
- **Loading state:** loop a clipped portion of the animated mark, or use it as the canonical loading visual

---

## What NOT to do

- Do not redraw, re-render, or "improve" the mark
- Do not add a clay splotch beneath the checkmark — it was tested, dropped, locked out
- Do not extend the mark with additional ink strokes, trails, or glow effects
- Do not substitute Helvetica or any other typeface for Newsreader in the wordmark
- Do not break the "Court" upright + "Check" italic-lime treatment — it is the locked wordmark
- If a use case feels like it needs a logo variant that doesn't exist (e.g., monogram, badge, alternate color), raise it as a request — do not silently invent one
