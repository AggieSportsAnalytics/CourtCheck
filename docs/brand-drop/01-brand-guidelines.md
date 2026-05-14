---
brand: CourtCheck
tagline: "See every shot. Know every move."
positioning: "Tennis analytics that reads the film for you. Built for college tennis coaches who'd rather be on the court than rewinding it."
archetype: Sage (with Caregiver overlay)
mission: "A coach should walk away with a real insight in under 5 minutes of opening the app for the first time."
audience:
  primary: College tennis head coaches (D1 / D2 / D3)
  secondary: Assistant coaches, players, performance analysts
voice:
  - Calm, never urgent
  - Observed, never inferred
  - Specific, never generic
  - Coaching, never selling
  - Plain English, never jargon
voice_dont:
  - Exclamation marks
  - Marketing-speak ("revolutionize", "supercharge", "unleash")
  - Pro-grade jargon ("BP saved %", "kick spin variance")
  - Hype copy
  - Emoji
hard_constraints:
  - Never use pure red (#FF0000). Clay terracotta only — even for errors and destructive actions.
  - Never use traffic-light coloring (red/yellow/green) for performance.
  - Headlines in display type always have exactly one italic word, rendered in clay terracotta.
  - All numerics render in Newsreader 500 with tabular numerals (font-feature-settings tnum).
  - Lime appears at most once per screen. It's "the ball" — the single moment of impact. ONE LOCKED EXCEPTION: the wordmark renders "Check" in italic lime to match the ball — this is the single brand-locked italic-lime element and the only place italic + lime co-occur.
  - Maximum 6 categorical colors per chart (court/plum/amber/clay/slate/lime).
  - Categorical colors lock product-wide — forehand is always Court green, every chart, every screen.
  - Animation is feedback, not decoration. No idle pulses, drifting glows, or scroll-jacking.
  - Tables exist for export only — every screen leads with a chart, court map, or single big number.
colors:
  primary: "#2E5341"      # Court green — the sport, observed, in your favor
  accent:  "#B05B36"      # Clay terracotta — clay courts, attention, opponent strength
  highlight: "#BFC846"    # Lime — the ball, the apex, the moment of impact
  ink:     "#1A1815"      # Type and primary line
  cream:   "#F4F0E6"      # Background (light mode)
  paper:   "#FBF8F1"      # Cards / elevated surfaces (light mode)
  shade:   "#ECE6D5"      # Inset surfaces (input backgrounds, mini-bars tracks)
  line:    "#DDD7C5"      # Borders
  ink_soft:  "#43403A"
  ink_mute:  "#837F77"
typography:
  display: "Newsreader (variable serif, opsz 6–144 axis used deliberately)"
  body: "Inter Tight"
  system: "JetBrains Mono"
motion_principle: "Reveals draw. Hovers respond. Numbers tick. Loading is the bounce arc. Nothing else moves."
---

# CourtCheck Brand Guidelines

> Single source of truth for Claude Design. Read this first; everything else is supporting context.
> The brand is already opinionated. This doc tells you what to keep, what to refine, and what is locked.

---

## 1 · The brand in one paragraph

CourtCheck is the film room, automated. A coach uploads a match, we surface every shot, pattern, and percentage worth knowing — without them watching it back. The brand DNA is **Robinhood meets SwingVision**: take the analytical depth of pro-grade tennis tools and package it in a calm, plain-spoken, visually obvious interface. Coaches finishing a match at 9pm shouldn't navigate a Bloomberg terminal to find that their #2's first-serve percentage dropped 8% on the deuce side. We *show* it.

We are not a feed. We are not a hype tool. We are quiet and right.

---

## 2 · Audience (who reads this)

**Primary: College tennis head coaches.**
- Mid-30s to early 60s, often former players
- Watch tape on Sundays, drill on Wednesdays, recruit on Saturdays
- Time-poor, opinion-rich
- Use of analytics is "I'll trust it if it tells me something I can act on Wednesday"
- Allergic to: SaaS jargon, dashboards with 14 KPIs, "configurable" anything
- Trust signal: serif typography for numbers (reads as "observed, signed off on")

**Secondary: Assistant coaches, players, analysts.**
- Younger, more comfortable with tech
- Read the same dashboards but click deeper

The brand voice is **a thoughtful assistant coach who's already watched the film** — not a product manager, not a hype-house, not a tech bro.

---

## 3 · The six principles

These are non-negotiable. Every screen passes all six, or we redesign before ship.

1. **One thing per screen.** Every screen has exactly one primary action. The next step is never ambiguous.
2. **Visual-first, tables last.** Replace numbers with shapes. Replace tables with court overlays.
3. **The gentle palette (no panic colors).** Clay for "needs attention." Court green for "in favor." Lime is the highlight only. Never red.
4. **Animation is feedback, not decoration.** Every tap, swipe, hover gets a visible response. Ambient motion is forbidden.
5. **The court is the universal frame.** Whenever data can be shown on the court, it is. The overhead court is the mental model coaches already use.
6. **Onboarding earns trust in five minutes.** Sign in, upload match, get one real insight back. If we can't deliver in the first session, the user never returns.

---

## 4 · Color system

### Light mode (default, "the cream paper")

| Token | Hex | Role |
|---|---|---|
| `--cream` | `#F4F0E6` | Page background |
| `--paper` | `#FBF8F1` | Card / elevated surface |
| `--surface` | `#FFFFFF` | Highest elevation (popovers, focus states) |
| `--shade` | `#ECE6D5` | Inset surfaces (input bg, bar track) |
| `--line` | `#DDD7C5` | Borders |
| `--line-soft` | `#EAE4D2` | Subtle dividers |
| `--ink` | `#1A1815` | Primary type, lines |
| `--ink-soft` | `#43403A` | Body type |
| `--ink-mute` | `#837F77` | Captions, system labels |
| `--court` | `#2E5341` | Primary brand — "in favor" |
| `--court-deep` | `#1F3D2E` | Hover/pressed states |
| `--court-light` | `#6E8C7C` | Tinted variants |
| `--clay` | `#B05B36` | Accent — "needs attention" |
| `--clay-soft` | `#D08866` | Tinted variants |
| `--lime` | `#BFC846` | Highlight — "the ball" (use sparingly) |
| `--plum` | `#7B4E6E` | Categorical slot 5 |
| `--amber` | `#C8923A` | Categorical slot 6 |
| `--slate` | `#5C6B7A` | Categorical slot 7 ("other") |

### Dark mode ("Stadium at Night")

| Token | Hex | Role |
|---|---|---|
| `--cream` | `#07080F` | Page background (deep void) |
| `--paper` | `#1F2940` | Elevated section / card |
| `--surface` | `#2B374F` | Top-tier card / spotlight panel |
| `--shade` | `#11161F` | Subtle wash |
| `--line` | `#2F3B52` | Visible divider |
| `--line-soft` | `#1E2735` | Subtle divider |
| `--ink` | `#F4F0E6` | Primary type |
| `--ink-soft` | `#A8B0C0` | Secondary type |
| `--ink-mute` | `#6B7388` | Tertiary type, captions |
| `--court` | `#6FA88B` | Primary brand — the court, lit at night |
| `--court-deep` | `#4A8268` | Hover/pressed states |
| `--court-light` | `#9FC6B0` | Tinted variants |
| `--court-surface` | `#6B3A24` | Court playing surface (warm clay) |
| `--clay` | `#E07A52` | Accent (warmer for dark) |
| `--clay-soft` | `#F0A07F` | Tinted variants |
| `--lime` | `#DDE970` | Highlight — "the ball" |
| `--plum` | `#B584A6` | Categorical |
| `--amber` | `#DDB166` | Categorical |
| `--slate` | `#889DB3` | Categorical "other" |

**Critical:** dark mode keeps the court green and pulls warm clay into the surface accent — the same palette, lit at night. The hue family does not change with the lights, only its tone. Light mode = clay courts in afternoon sun. Dark mode = the same court after dusk, quieter and warmer.

### Outcome semantics (the gentle palette rule)

| Meaning | Color |
|---|---|
| In your favor / working / won / on track | `--court` (Court green, both modes) |
| Needs attention / opponent strength / decision point | `--clay` |
| The moment / the ball / the apex | `--lime` |
| System error / destructive confirmation | `--clay` (NEVER red) |

### Categorical palette (locked product-wide, 6 slots max)

Canonical source: `tokens.css` lines 30-34 (light) and 109-113 (dark).

| Slot | Token | Light hex | Dark hex | Default role |
|---|---|---|---|---|
| 1 | `--court` | `#2E5341` | `#6FA88B` | Forehand |
| 2 | `--plum` | `#7B4E6E` | `#B584A6` | Backhand |
| 3 | `--amber` | `#C8923A` | `#DDB166` | Serve |
| 4 | `--clay` | `#B05B36` | `#E07A52` | Volley |
| 5 | `--slate` | `#5C6B7A` | `#889DB3` | Other / neutral |
| 6 | `--lime` | `#BFC846` | `#DDE970` | Ball / accent (sparing) |

**Rule:** these lock. Forehand is always Court green, both modes. Coaches build muscle memory around it. Never combine outcome semantics and categorical semantics in the same chart. Lime stays scarce (one ball per screen) even when used as a categorical slot.

---

## 5 · Typography

Three typefaces. Locked. **Do not introduce a fourth.**

### Newsreader (variable serif, display + numerics)

The signature move. Newsreader has a true optical-size axis — we use it as a deliberate recipe, not a default. This is why the brand reads as editorial, not generic SaaS.

| Use | opsz axis | Weight | Size |
|---|---|---|---|
| Hero display (h1) | **144** | 500 | clamp(56px, 9vw, 142px) |
| Section display (h2) | **96** | 500 | clamp(40px, 5.6vw, 76px) |
| Card title | **72** | 500 | 1.7rem |
| Big stat number | **72** | 500, tnum | clamp(72px, 9vw, 144px) |
| Eyebrow stat number | **36** | 500, tnum | 1.05rem |
| Body italic accent | **18** | 400 italic | inherit |
| Footnote italic | **6** | 400 italic | 0.85rem |

**Why a serif for numbers:** a serif number reads as *observed, considered, signed off on*. A sans number reads as *auto-generated, machine-output*. Our brand promise is "we watched the film for you." The typography reinforces the promise.

**The italic-clay rule (locked):**
> Every page title has exactly one italic word, always rendered in `--clay` (light) / `--clay-soft` (dark). The only italic word ever rendered in lime is the `Check` half of the CourtCheck wordmark — that is a single brand-locked exception.

The italic word is the *verb* of the sentence ("automated", "lives", "opens") or the *action* being claimed ("Know every move", "Get back in"). Coaches will start to recognize the pattern unconsciously. It is the brand's voice register made visible.

### Inter Tight (sans, body)

- Body copy, button labels, leads, lede paragraphs
- Weights: 400, 500, 600, 700
- Letter-spacing: -0.005em (tight) for body, -0.014em for labels
- Line-height: 1.5 for body, 1.3 for short

### JetBrains Mono (monospace, system voice)

The "machine voice." Used everywhere the *product itself* is narrating its own state.

- Font-size: 0.66–0.72rem
- Letter-spacing: 0.14em–0.18em
- Always uppercase
- Color: `--court` for eyebrows, `--ink-mute` for labels

**Rule:** anything the product says about itself (timestamps, status, version, axis labels, "tracking", "live", "v0.1", zone names, eyebrows above headlines) speaks in mono caps. Anything *coaches* would speak (headlines, callouts, the human-spoken word) speaks in Newsreader. This is the brand's two-voice system.

### Eyebrow (the signature label)

```
[●] FOR COLLEGE TENNIS COACHES
```

- JetBrains Mono, 0.72rem, letter-spacing 0.18em, uppercase
- Color: `--court`
- Always preceded by a 6×6 clay dot via `::before`
- Used for: section labels above headings, card category labels, eyebrows in the brand mark area

---

## 6 · Spacing, radius, shadow

| Token | Value |
|---|---|
| `--radius-sm` | 8px |
| `--radius-md` | 12px |
| `--radius-lg` | 14px |
| `--radius-xl` | 18px (cards) |
| `--shadow-card` | `0 1px 0 rgba(26,24,21,0.04), 0 18px 48px -24px rgba(26,24,21,0.16)` |
| `--shadow-pop` | `0 1px 0 rgba(26,24,21,0.06), 0 24px 60px -28px rgba(26,24,21,0.22)` |

Buttons are pills (`border-radius: 100px`). Cards are 18px. Inputs are 12px. There's deliberate hierarchy — pill = action, card = content, input = entry. Never blur the distinction.

---

## 7 · Motion DNA

### Easing tokens

| Token | Curve | When |
|---|---|---|
| `--ease-out` | `cubic-bezier(0.2, 0.8, 0.2, 1)` | **Default.** Reveals, fades, hovers. |
| `--ease-spring` | `cubic-bezier(0.34, 1.56, 0.64, 1)` | Hero moments only. Big number reveals, primary CTA hover. |
| `linear` | `linear` | Progress, count-ups, drawing along a path. |

### Duration tokens

| Token | Value | When |
|---|---|---|
| `--dur-quick` | 160ms | Hovers, button presses |
| `--dur-base` | 240ms | Standard transitions |
| `--dur-reveal` | 480ms | Element reveals on load/scroll |
| `--dur-hero` | 800ms | Big page entrance (one per page max) |
| `--dur-loop` | 2400ms | The bounce arc loop, skeleton shimmer |

### What we animate

1. **Initial reveal** — hero stack staggers in top-to-bottom, 80ms between elements
2. **Hover lift** — cards translate `-2px`, shadow deepens
3. **Number count-up** — Newsreader serif counts 0 → value, Robinhood-style with a tiny tick on leading-digit changes
4. **Court overlay data** — shot dots fade in by zone, 30ms stagger
5. **Bar growth** — bars 0 → target width with 60ms top-to-bottom stagger
6. **Loading** — the bounce arc loop. **This is the brand's signature animation.** It is the only loading metaphor we use product-wide.

### What we never animate

- Body text after initial reveal
- Tooltips (must appear instantly)
- **Ambient backgrounds** — no pulses, drifting glows, idle wind effects
- Anything tied to scroll position (no scroll-jacking)

`prefers-reduced-motion: reduce` honored everywhere — animations downgrade to instant.

---

## 8 · The court overlay (universal frame)

Tennis singles court is **78ft × 27ft**. We use those numbers as SVG units. Display orientation is **vertical** — length top-to-bottom, the standard tennis broadcast/coaching view. Player at bottom (y=78), opponent at top (y=0), net horizontal across middle (y=39).

| Element | Position |
|---|---|
| Boundary | (1,1) → (26,77) |
| Net | y=39, full width |
| Service lines | y=18, y=60 |
| Center service line | (13.5, 18) → (13.5, 60) |

Surface color: `--court-surface` `#A85E3A` (light) / `#6B3A24` (dark), the warm clay court color, same hue family in both modes. Lines always white.

**The court is on every dashboard tile.** Pattern cards, match summaries, player profiles. It's our equivalent of Robinhood's portfolio chart — the one persistent visual that anchors the brand.

---

## 9 · The brand mark (LOCKED)

> The logo system is finalized. See `logo/concept-notes.md` and `logo/current-system.md` for the canonical reference.

The lockup has three elements that always appear together:

1. **The mark** — hand-drawn warm-grey ink checkmark with a short left descending arm + a long right ascending arm. A lime tennis ball sits at the far tip of the right ascending arm, suspended above and to the right of the V.
2. **The wordmark** — "CourtCheck" in Newsreader serif. "Court" upright in dark ink, "Check" italic in lime (matching the ball color exactly).
3. **The composition** — mark and wordmark side by side with consistent optical spacing, both in the editorial hand-drawn register.

**Locked assets in `logo/`:**

| File | What |
|---|---|
| `CourtCheckLogoLight.png` | Light master (cream canvas) |
| `CourtCheckLogoDark.png` | Dark master (near-black canvas) |
| `CourtCheckAnimation.mp4` | Splash animation (5s, plays once, ends on static lockup) |

**Color values:**

| Element | Light mode | Dark mode |
|---|---|---|
| Mark stroke | `#43403A` warm grey | `#F4F0E6` cream |
| Lime ball | `#BFC846` | `#DDE970` |
| "Court" | `#1A1815` ink | `#F4F0E6` cream |
| "Check" (italic) | `#BFC846` lime | `#DDE970` lime |
| Background | `#F4F0E6` cream | `#07080F` deep void |

**The wordmark "Check" exception:** lime appears in the wordmark as a one-time brand-locked exception to the "lime is the ball only" rule. The reasoning: the wordmark's "Check" half *is* the brand's lime moment in the typographic register — semantically, "Check" is the verb-of-impact that the ball performs. Everywhere ELSE in the product (headlines, body, accents), the "lime is the ball only" rule holds.

**Do not** redraw, regenerate, or "improve" the mark. Use the master PNGs as-is.

---

## 10 · The categorical chart system

Charts that show categorical data (shot map, heatmap, distribution) **must** be filterable via the legend itself — no separate filter dropdown.

- **Default**: all categories shown, legend chips active-but-not-highlighted
- **Click a chip**: that category becomes the only active one. Others dim to 12% opacity (preserves spatial context — coach can still see "where the rest went")
- **Click again**: reset
- **Hover**: chip lifts 1px
- **Counts in chip**: each chip shows the count for that category in JetBrains Mono — read like a stats line

**The three-card court row** (canonical dashboard layout):

| Card | Question | Data |
|---|---|---|
| **Shot map** (opponent's half) | *Where did my shots land?* | Bounce dots, colored by stroke |
| **Spacing** (player's half) | *How clean was my contact?* | Player↔ball lines colored by ideal/squeezed/long |
| **Coverage** (player's half) | *Where did I live on the court?* | Single-color heatmap, intensity = time spent |

Progression: **landed → contacted → positioned**. Offensive intel, technical intel, movement intel.

---

## 11 · Voice & tone

Speak like a thoughtful assistant coach who's already watched the film.

**Specific, not generic:**
- ✅ "Loses the cross 64% on second-serve return."
- ❌ "AI-powered insights."

**Calm, never urgent:**
- ✅ "Pattern detected. Lin's slice serve wins 71% of openers."
- ❌ "🎉 Amazing pattern unlocked!"

**Coaching, not selling:**
- ✅ "Here's what to work on Wednesday."
- ❌ "Crush your competition with CourtCheck."

**Plain, not jargon:**
- ✅ "First serve in: 68% — up 4 points from your season."
- ❌ "Serve efficiency delta: +4.2 ppt vs trailing 30 baseline."

**Numbers wrapped:** in body copy, numbers always wrap in `<span class="num">68%</span>` so they render in Newsreader serif with tabular numerals.

---

## 12 · Notification voice (Sonner toasts)

Same coach voice, applied to in-product feedback. Four levels, never red:

| Level | Color | Use |
|---|---|---|
| Success | `--court` accent | "Match analyzed" |
| Info | `--ink-mute` accent | "Reset link sent" |
| Warn | `--clay` | "Video is 480p — quality may suffer" |
| Error | `--clay` (NEVER red), persistent | "Couldn't reach the server" |

Lead with what happened, then what to do (if anything).

✅ "Match analyzed. View Stanford on your dashboard."
❌ "🎉 Your match has been successfully analyzed!"

✅ "Couldn't reach the server. Check your connection and try again."
❌ "An error occurred. Error code: NET_FAIL_1138."

---

## 13 · The Robinhood Test (use before shipping)

For every screen, every feature, every flow:

1. **5-second test** — Can a coach identify the next action in under 5 seconds?
2. **Pulse test** — When they tap, does something visibly respond?
3. **One-step test** — Is there exactly one action that's clearly the next step?
4. **Picture test** — Did we replace a table with a visual?
5. **Coach voice test** — Does the copy sound like an assistant coach, not a SaaS product manager?

If any answer is no — redesign before ship.

---

## 14 · Anti-patterns (will never ship)

| Don't | Do |
|---|---|
| 14 KPIs on a dashboard | 1 hero stat + 3 supporting |
| Red/green traffic-light cells | Clay/court — calmer, more useful |
| Important actions buried in menus | Surface the next step on the page itself |
| Animations that move when idle | Animations that respond to user action |
| "Configure your report" | Generate the report; let them edit |
| Pro tooltips, "i" icons, hover help | Inline copy that explains in plain English |
| Dense tables | Court overlays, big numbers, comparison cards |
| Multiple ways to do the same thing | One blessed path, executed beautifully |
| Pure red for errors | `--clay` always |
| Helvetica wordmark | Newsreader 500, opsz 60 |
| Neon green ball gradients | Flat lime, single subtle highlight |
| Ambient pulse animations | Animation is feedback only |

---

## 15 · What we want from Claude Design

**Top priority — logo system:**
1. A new wordmark in Newsreader 500 (opsz 60), letter-spacing -0.018em, with the "C" weighted to function as a standalone monomark when needed.
2. A standalone mark that refines the **seam-as-checkmark** concept (single flowing stroke, ball-seam doubling as a check). Single ink color or single accent.
3. A new bounce-arc loading animation as Lottie JSON — calm 2.4s loop, ball traces a court arc, refined ball treatment (no neon, flat lime fill with one subtle highlight).
4. Full lockup at 32px, 48px, 80px, 200px sizes.
5. Light + dark variants for everything.

**Secondary — refined dashboard mockups:**
6. Refresh the dashboard, dashboard-empty, upload-flow, match-detail, and player-profile mocks with the new logo system applied.
7. Treat the `mocks/` folder as our current state. Don't redesign the layouts — refine the visual execution against the brand laws above.

**What NOT to change:**
- The cream/clay/court/lime palette (locked)
- Newsreader/Inter Tight/JetBrains Mono (locked)
- The court-as-frame principle (locked)
- The italic-clay-headline rule (locked) — with the wordmark `Check` italic-lime as the single locked exception
- The locked categorical palette (forehand=court, backhand=plum, etc.)
- The two-voice system (Newsreader = coach, Mono caps = system)
- The "no red, ever" rule
- The "one lime per screen" rule
- The motion DNA (no ambient motion, bounce arc is the only loop)

If you want to propose a *concept change*, propose it in writing first. Don't ship a redesigned palette or fontstack as a fait accompli.

---

*This is a living document. The brand has been built; this is refinement.*
