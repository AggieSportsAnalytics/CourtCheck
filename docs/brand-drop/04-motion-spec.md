# Motion specification

> The brand's motion DNA. Locked.
> Animation is feedback, not decoration. Ambient motion is forbidden.

---

## Locked animation assets

Two animations are locked. They are the only motion the brand ships with at the system level. Both are reference videos in this brand drop:

| Asset | File | Use |
|---|---|---|
| **Hero bounce loop** | `motion/Bounce_Animated.mp4` | Public landing page hero — full-bleed or contained background |
| **Logo splash** | `logo/CourtCheckAnimation.mp4` | One-shot intro on app load, splash screens, first-session moments |

The hero bounce loops seamlessly (6 seconds). The logo splash plays once and ends on the static lockup. Both are hand-drawn editorial register.

Additional reference: `motion/Bounce_Still.png` is the source frame of the hero animation — also the poster image and reduced-motion fallback for the hero.

---

## Easing tokens

| Token | Curve | When |
|---|---|---|
| `--ease-out` | `cubic-bezier(0.2, 0.8, 0.2, 1)` | Default. Reveals, fades, hovers. |
| `--ease-in-out` | `cubic-bezier(0.4, 0, 0.2, 1)` | State changes. Toggles, drawer open/close, theme switch. |
| `--ease-spring` | `cubic-bezier(0.34, 1.56, 0.64, 1)` | Hero moments only. Big number reveals, primary CTA hover lift. Sparingly. |
| `linear` | `linear` | Progress, count-ups, drawing along a path. |

## Duration tokens

| Token | Value | When |
|---|---|---|
| `--dur-quick` | 160ms | Hovers, button presses |
| `--dur-base` | 240ms | Standard transitions — card lift, color change |
| `--dur-reveal` | 480ms | Element reveals on load or scroll into view |
| `--dur-hero` | 800ms | Big page entrance moments (one per page max) |
| `--dur-loop` | 2400ms | Looping ambient assets (skeleton shimmer) |

---

## What we animate

### 1. Hero bounce (LOCKED)

`Bounce_Animated.mp4` is the canonical hero animation. It plays as a muted, autoplay, loop background or contained-element video on the public landing page. The source illustration is hand-drawn editorial (warm ink on cream paper, side-perspective tennis court, lime tennis ball bouncing in a continuous arc). 6 seconds, seamless loop, no audio.

The video is the brand's "we observe, we track" promise made visible.

### 2. Logo splash (LOCKED)

`CourtCheckAnimation.mp4` plays once on first load, splash transitions, or any "brand moment" where the full lockup needs to assemble itself in front of the user. 5 seconds, plays-once, ends on the static lockup pixel-matched to `CourtCheckLogoLight.png`.

Sequence: ball enters from off-screen right → traces the checkmark path → wordmark inks in left-to-right.

### 3. Initial reveal (page load)

Hero stack reveals top-to-bottom with `--dur-reveal` + `--ease-out`. 80ms stagger between elements. One-shot — does not loop.

### 4. Hover lift (cards)

Cards translate `-2px` and shadow deepens. `--dur-base` + `--ease-out`.

### 5. Number count-up

Hero stats count from 0 → value over 720ms with linear easing. Newsreader serif. Optional: 80ms scale-pulse on leading digit when it crosses a power of ten (the Robinhood "kinetic number" detail).

### 6. Court overlay data

Shot dots fade in by zone with 30ms stagger between dots. Each dot: opacity 0 → 1 + scale 0.4 → 1, `--dur-reveal`, `--ease-out`.

### 7. Bar chart growth

Bars grow from 0 → target width over `--dur-reveal`. Stagger 60ms between bars (top to bottom). Value labels fade in at the end of each bar's growth.

### 8. Heatmap fill

All zones fade in simultaneously over `--dur-reveal`. No stagger.

---

## What we never animate

- Body text after initial reveal
- Tooltips (must appear instantly, `--dur-quick` max)
- **Ambient backgrounds** — no pulses, drifting glows, idle wind effects
- Scroll-jacking — no parallax beyond the dedicated pinned scroll-through section on the landing page
- Idle elements "breathing" for attention

If something moves while the user is not interacting with it, the only acceptable reason is: new information is arriving, OR it's one of the two locked brand animations above.

---

## Reduced motion

Honor `prefers-reduced-motion: reduce`:
- All animations downgrade to `--dur-quick` or instant
- The hero bounce video pauses on its still frame (`motion/Bounce_Still.png`) — used as poster + fallback
- The logo splash skips to the final static state immediately
- Hover lifts retained (small, not disorienting)
- Number count-ups become instant

---

## Skeleton shimmer (one allowed ambient loop)

For loading placeholders. Calm 2.4s pacing — never the anxious 1s shimmer most products use.

```css
@keyframes shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
.skeleton {
  background: linear-gradient(90deg, var(--shade) 0%, var(--paper) 50%, var(--shade) 100%);
  background-size: 200% 100%;
  animation: shimmer var(--dur-loop) linear infinite;
  border-radius: var(--radius-lg);
}
```

---

*Locked. Distilled from the brand's original motion DNA, now codified around the locked hero + splash animations.*
