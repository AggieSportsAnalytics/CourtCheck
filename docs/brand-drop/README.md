# CourtCheck brand drop (v2 — locked)

> Folder packaged for Claude Design fresh-project onboarding (claude.ai/design, Anthropic Labs).
> Last updated 2026-05-11.

## What this is

The **locked** brand system for **CourtCheck** — a tennis analytics product for college tennis coaches — packaged as a fresh ingestion bundle for Claude Design.

This is **v2**. Earlier iterations refined the brand to its current locked state. The brand drop reflects that final state. Use these assets and rules as-is.

## What's locked

- **Logo system** — light + dark master lockups + a one-shot splash animation
- **Hero animation** — 6-second seamlessly looping hand-drawn court bounce video
- **Color palette** — cream/clay/court/lime in light, near-black/cream/pacific/lime in dark
- **Typography stack** — Newsreader serif (display + numerics), Inter Tight (body), JetBrains Mono UPPERCASE (system meta)
- **Italic-clay rule** on every section headline (one italic word, always clay)
- **Italic-lime wordmark exception** on "Check" (locked, sole exception to the "lime is the ball only" rule)
- **No-red rule** — never use red, clay for all errors and accents
- **Motion DNA** — animation is feedback, not decoration; ambient motion forbidden except for the two locked brand animations

## What's in this folder

```
brand-drop/
├── README.md                          ← this file
├── 00-first-prompt.md                 ← fresh onboarding message to paste into Claude Design
├── 01-brand-guidelines.md             ← THE primary spec (YAML front matter + full system)
├── 02-philosophy.md                   ← Robinhood-meets-SwingVision brand DNA
├── 03-voice.md                        ← Coach voice + microcopy + toast voice
├── 04-motion-spec.md                  ← Motion DNA + locked hero + splash animations
├── 05-peer-references.md              ← Editorial register reference brands
├── tokens.css                         ← Implemented globals.css (current state in code)
│
├── fonts/                             ← Variable font TTFs (6 files)
│   ├── Newsreader-Variable.ttf
│   ├── Newsreader-Italic-Variable.ttf
│   ├── InterTight-Variable.ttf
│   ├── InterTight-Italic-Variable.ttf
│   ├── JetBrainsMono-Variable.ttf
│   └── JetBrainsMono-Italic-Variable.ttf
│
├── logo/                              ← LOCKED logo assets
│   ├── CourtCheckLogoLight.png        ← Light master lockup (cream canvas)
│   ├── CourtCheckLogoDark.png         ← Dark master lockup (near-black canvas)
│   ├── CourtCheckAnimation.mp4     ← Splash animation (5s, plays once)
│   ├── concept-notes.md               ← "Locked, do not iterate" doc
│   └── current-system.md              ← Asset inventory + display rules
│
├── motion/                            ← LOCKED motion assets
│   ├── Bounce_Animated.mp4            ← Hero animation (6s, seamless loop)
│   └── Bounce_Still.png               ← Hero source frame (poster + reduced-motion fallback)
│
└── mocks/                             ← Brand applied across 10 product surfaces
    ├── landing.html                   ← Public landing page mock
    ├── auth.html                      ← Login / signup / forgot-password
    ├── dashboard.html                 ← Populated dashboard
    ├── dashboard-empty.html           ← First-session empty dashboard
    ├── visuals.html                   ← Per-recording visualization reference
    ├── upload-flow.html               ← 4 upload states
    ├── match-detail.html              ← Per-recording analysis (clip detail)
    ├── player-profile.html            ← Single player profile
    ├── players-list.html              ← Roster index
    └── matches-list.html              ← Recordings index
```

## How to use this drop with Claude Design

1. **Open a new Claude Design project** (fresh, not a continuation of any prior project)
2. **Drag the entire `brand-drop/` folder** into the file uploader. Claude Design ingests the folder structure
3. **Open `00-first-prompt.md`** and paste the `\`\`\`` block verbatim as your first message
4. **Wait for Claude Design's acknowledgment** before approving any asset generation
5. **Build surfaces section-by-section** using the follow-up prompts in `00-first-prompt.md`

## What Claude Design should produce

Product surfaces, not brand assets:

- Refreshed landing page using the locked logo + locked hero video
- Refreshed dashboards using the brand system
- Refreshed auth, match detail, player profile, etc.

## What Claude Design should NOT do

- **Do not regenerate, redraw, or "improve" the locked logo** (it's done)
- **Do not regenerate the hero animation** (use the locked MP4 as-is)
- **Do not change the color palette, fontstack, or any brand law**
- **Do not invent variants that don't exist** (monogram, badge, alternate color, etc.) — flag as request if needed
- **Do not "iterate" on rejected prior directions** (3D court, before/after photo, fake testimonials, fake school logo strip, vintage newspaper)

If Claude Design tries to silently re-litigate any locked decision, re-paste the relevant section of `01-brand-guidelines.md` or `logo/concept-notes.md`.

## Preview

A local preview of the locked assets lives at `preview.html` in this folder. Open it in a browser to see the logo lockups, splash animation, and hero animation rendered against the brand's cream paper background.

---

*The brand is locked. Build forward.*
