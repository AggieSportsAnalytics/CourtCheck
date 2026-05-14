# CourtCheck — Session Handoff

*Last updated: 2026-05-12 (final pre-port audit complete — mocks port-ready)*

> Hand-off for the next Claude Code session working on CourtCheck. Brand is locked. The mocks have been audited and cleaned for the React port. Tokens.css and all brand-drop markdown docs are in sync with the canonical mock palette.

## 🎯 Port-ready as of 2026-05-12

A thorough internal audit ran today. All CRITICAL items fixed, all 12 HIGH items fixed, MEDIUM items addressed. The React port can begin with confidence in the brand-drop bundle as ground truth.

**Today's audit + cleanup pass:**

1. **Dark palette restored to GREEN court** across all 10 mocks + tokens.css. The original drop incorrectly flipped `--court` from green to Pacific Blue (#3B8FE8) in dark mode, which drained the brand identity. Now `--court: #6FA88B` (green-bright) in dark, with `--court-surface: #6B3A24` (warm clay, same hue family as light). `--plum`, `--amber`, `--slate` added to dark mode where missing.
2. **Surface elevation amplified** — dark `--cream #07080F` → `--paper #1F2940` → `--surface #2B374F` (~30 luminance units between tiers, vs the prior ~10). Adjacent sections in dark mode now read as distinct planes instead of a flat navy wall.
3. **Landing problem-wrap fix** — `body.dark .problem-wrap` now uses `var(--surface)` instead of `var(--paper)` so the "Tennis has always been" section renders as the brightest spotlight panel, mirroring light mode's dramatic ink-black contrast (just inverted for dark).
4. **Brand markdown docs rewritten** — `01-brand-guidelines.md`, `02-philosophy.md`, `05-peer-references.md` purged of all Pacific Blue references. Categorical palette role mapping corrected: backhand = `--plum` (was incorrectly mapped to non-existent Pacific). 6-slot ladder cited against tokens.css.
5. **`<Sidebar>` component spec added** to `06-components.md`. Full spec for `<Sidebar>`, `<SidebarBrand>`, `<SidebarNav>`, `<SidebarLink>`, `<SidebarCollapse>`, `<SidebarFoot>`, `<UserMenu>` — fixed positioning, 200/72px collapse, body offset, localStorage persistence, a11y, foot order (avatar-LEFT then theme), UserMenu popover behavior.
6. **`<MatchRow>` → `<RecordingRow>`** spec rewritten with the actual 3-column grid (`110px 1fr 40px`), no score column, no win/loss pill. Dropzone copy updated to "recording".
7. **Player roster reordered** in dashboard.html to match players-list.html (Maya / Jordan / Aspen / Daniel / Sophie / Khalil).
8. **Dead CSS purged** — top-bar `<nav>` rules deleted from 8 internal mocks (sidebar replaced them); `.team-strip`, `.patterns-card`, `.match-row .score/.stat-mini/.pattern-mini/.result-pill` rules removed; orphan JS `.team-stat` reference cleaned.
9. **"Recording" terminology** finished the sweep — "hrs of film" → "hrs recorded" (13 strings), "Drop in your film" → "Drop in your recording", "Upload your first match" → "Upload your first recording", match-detail breadcrumb "Clips" → "Recordings", auth.html "film room" copy fixed.
10. **Missing CSS vars fixed** — auth.html now declares `--slate` in both light and dark, matches-list.html now declares `--court-surface` in both modes.
11. **`brand-style-reference.html` archived** — was 43KB of stale Pacific-Blue duplicate. Moved to `brand-drop/archives/` so the React port doesn't pick up the wrong palette.
12. **README.md table of contents corrected** — added `visuals.html`, removed deleted `patterns.html`.
13. **preview.html** dark tokens synced; `.force-dark` panel CSS uses new dark hex; new `07 · Color Palette` section added with both light + dark swatches (19 each, 4 groups: Surfaces / Ink / Brand · Court / Accents).
14. **Landing fixes** — feature-section h2 tightened from `clamp(40,5.6vw,76px)` → `clamp(30,3.6vw,48px)` and section padding `100px → 72px` so the panel + cards fit in one screen. Scroll-spy bug fixed (How It Works no longer stays highlighted after scrolling back above all sections).

---

## 30-second orientation

CourtCheck is a tennis analytics product for college coaches. The brand has been through ~20+ iteration rounds and is **locked**. The canonical source of truth is `docs/brand-drop/` — a self-contained bundle. The live preview at `docs/brand-drop/preview.html` shows every locked asset + every mock thumbnail.

---

## ⚠ READ FIRST — design philosophy is the bar

`docs/brand-drop/02-philosophy.md` is the operative design doc. Re-read before any new work. Key principles being actively enforced:

1. **One thing per screen** / **One insight per card** — Coaches first, always
2. **Visual-first, tables last** — Court overlays, big numbers, comparison cards. Never dense tables.
3. **Gentle palette, no panic colors** — Clay for "needs attention" (not red). Lime is the highlight only.
4. **Animation is feedback, not decoration**
5. **The court is the universal frame**
6. **Onboarding earns trust in 5 minutes**

**Brian's standing directive (re-stated multiple times)**: **always reference `mocks/visuals.html` when building or refactoring elements.** Port verbatim — hover patterns, viz components, `.card` lift, `.insight` lift, `.match-row` arrow-translate, `.legend-chip` dim, the `data-count-to` count-up pattern. Do not paraphrase.

---

## What's LOCKED (do not touch)

### Logo + motion assets (`docs/brand-drop/logo/`, `docs/brand-drop/motion/`)
- `logo/CourtCheckLogoLight.png` — transparent, tight-cropped, dark warm-grey ink + lime ball
- `logo/CourtCheckLogoDark.png` — transparent, cream + brighter lime ball
- `logo/CourtCheckAnimation.mp4` — original 5-second logo splash (full wordmark inks in)
- `logo/CourtCheckAnimation.webm` — transparent alpha WebM of full splash, 800×427, used for the page-load splash on landing.html. ffmpeg chroma-key (`colorkey=0xEFECDF:0.22:0.08`) + VP9 yuva420p encode.
- `logo/CourtCheckCheckmark.webm` — transparent alpha WebM of JUST the checkmark+ball animation (no wordmark), **500×500**, ~2.0s duration. Used for the upload-flow "Clip analyzed" celebration. Crop offset (320, 27) — keeps ball in frame across all motion + settled state geometrically centered.
- `logo/ucd-tennis.png` — UC Davis Tennis partnership mark, transparent alpha (chroma-keyed from original).
- `motion/Bounce_Animated.mp4` — original 6-second seamlessly looping hand-drawn court bounce
- `motion/Bounce_Animated.webm` — transparent alpha WebM of bounce loop, used in landing hero + upload-flow processing/uploading. Chroma-keyed `0xF8F1E3`.
- `motion/Bounce_Still.png` — source frame, also poster + reduced-motion fallback

### Brand laws
- Cream paper `#F4F0E6` light, near-black `#0A0E14` dark
- Newsreader (display + numerics), Inter Tight (body), JetBrains Mono UPPERCASE (system meta)
- **Italic = clay**. Locked exception: the `Check` half of CourtCheck wordmark stays italic-lime. The closing CTA "Start winning." is a second locked italic-lime exception per Brian.
- No red, ever — clay handles errors/destructive
- All numerics Newsreader 500 with tabular numerals
- Wordmark: `Court` upright + `Check` italic lime
- Lime is the ball moment — sparing

### Anti-patterns (do not re-introduce)
- Em dashes in user-facing copy (AI tell)
- "AI-powered insights" / "leverage" / "robust" / "elevate" / "transform" marketing slop
- Pure red, traffic-light coloring, neon green
- 3D court / before-after photos / fake school logos / fake testimonials / soft-glow chrome
- Helvetica wordmarks
- 14+ KPIs on a dashboard (philosophy violation)

---

## File map

```
docs/
├── HANDOFF.md                              ← this file
└── brand-drop/                             ← canonical brand bundle
    ├── 00-first-prompt.md                  ← Claude Design onboarding
    ├── 01-brand-guidelines.md
    ├── 02-philosophy.md                    ⚠ READ FIRST
    ├── 03-voice.md
    ├── 04-motion-spec.md
    ├── 05-peer-references.md
    ├── 06-components.md
    ├── tokens.css                          ← canonical CSS variables (synced 2026-05-12)
    ├── preview.html                        ← live preview of all locked assets + mock thumbnails + 07 · Color Palette
    ├── archives/
    │   └── brand-style-reference.html      ← stale Pacific Blue palette, archived 2026-05-12
    │
    ├── fonts/
    ├── logo/                               ← masters + animation WebMs + UCD partnership
    ├── motion/                             ← bounce loop assets
    │
    └── mocks/
        ├── landing.html                    ← public landing page (top-bar nav)
        ├── auth.html                       ← sign in / sign up / reset (no nav)
        ├── dashboard.html                  ← coach overview · player roster (sidebar)
        ├── dashboard-empty.html            ← first-session state (sidebar)
        ├── visuals.html                    ⚠ CANONICAL viz/hover reference (sidebar; Clips active)
        ├── upload-flow.html                ← 4 upload states (sidebar)
        ├── match-detail.html               ← clip-detail page (sidebar)
        ├── player-profile.html             ← (sidebar)
        ├── players-list.html               ← (sidebar)
        └── matches-list.html               ← (sidebar)
```

Patterns mock was removed 2026-05-12 — coaches drill into match-detail for tendencies; a separate Patterns page was redundant.

### Sidebar (in-product nav, 2026-05-12)

All 8 internal pages share an `<aside class="app-sidebar">` block (replaces the old top-bar `<nav>`). Spec:

- **200px wide** on desktop; collapses to 72px icon-only via a chevron toggle (state persists in `localStorage` under `cc-sidebar-collapsed`); auto-collapses below 1100px viewport.
- **Collapse toggle**: 26px circular button, absolute-positioned at the top-right corner of the sidebar (`top: 14px; right: 8px`). Stays in place when collapsed and rotates the chevron to indicate state.
- 4 nav items in this order: **Dashboard / Players / Upload video / Recordings** (Patterns removed; "Clips" renamed to "Recordings"; "Upload" → "Upload video").
- Active state uses background shade + bold weight + a 3px court-color left rail (per a11y `color-not-only` rule).
- Brand-mark in header re-uses the existing `.brand-mark` class so the per-page hover-video JS picks it up automatically. Logo is `width: 100%; max-height: 64px`, fills the sidebar width.
- Foot row holds: theme toggle + avatar (collapse moved out to top-right).
- `aria-label="Primary navigation"` on the aside, `aria-current="page"` on the active link, `title=` on every link for the icon-only collapsed state.
- Body uses `padding-left: 200px` (or 72px collapsed) so the centered `.container` flows in the remaining viewport.

Generation script lives at `/tmp/apply_sidebar.py` (not checked in) — idempotent: re-running strips any prior sidebar block and inserts the canonical version. Useful when the spec evolves.

### Dashboard splash (2026-05-12)

`dashboard.html` and `dashboard-empty.html` now play the locked CourtCheckAnimation as a full-screen splash on entry. Same recipe as landing.html but gated by a separate sessionStorage key `ccDashSplashSeen` so it fires once per session for dashboard entry independent of the landing splash. Auto-dismisses on video `ended`, with 4.5s fallback and 6s hard ceiling. Click to skip. Disabled under `prefers-reduced-motion`.

### Match-detail accuracy bar graph (2026-05-12)

Match-detail's "Shot mix" card became "Shot breakdown" — a 2-column grid with **Mix** (shot frequency %) on the left and **Accuracy** (in/won % per stroke) on the right. Same horizontal-bar style for both: 760ms ease-out-quart fill, 100ms stagger, count-up tick. `buildShotBars()` was generalized to take `(targetId, data)`; called twice with `SHOT_MIX_DATA` and `ACCURACY_DATA`. Stacks vertically below 880px.

---

## State of each mock

### landing.html
- **Page-load splash overlay** with the locked CourtCheckAnimation WebM. sessionStorage-gated so it plays once per session.
- **Hero**: text left, transparent bouncing court right. 3 floating glass cards over the hero (Shots / Coverage / Backhand tendency) with subtle drift. UC Davis Tennis partnership line below CTAs.
- **Headline**: hard `<br>` so wrap is deterministic — "See every / shot." + "Know every / move."
- **Closing CTA**: "Stop guessing. / Start winning." — `Start winning` is italic-lime (second locked italic-lime exception, beyond the wordmark).
- **Nav**: cream glass, 3-col grid (logo / centered links / actions), "See It Work" + "How It Works" links, Get started primary CTA + Sign in text link, scroll-spy active state.
- **Pinned scroll-through (.scrolly section)**: 4 stages.
  - Cross-fade between TWO stacked court SVGs (`buildCourtSVG('top')` for opponent, `buildCourtSVG('bottom')` for player). CSS: stage 1+2 → top court visible; stage 3+4 → bottom court visible. 480ms ease-in-out cross-fade.
  - **Stage 4 coverage heatmap now peaks at BOTTOM (player's baseline)** — `buildCoverage()` at line 1577 uses `lengthFrac = r / (ROWS - 1)` so r=0 (top of overlay) has low intensity and r=11 (bottom, player's baseline) has peak. ✅ shipped 2026-05-12.

### dashboard.html (NEW — was the old visuals.html role)
- Coach overview / player roster page
- Greeting + h1 "Your roster *today.*" + team-strip KPIs (Clips 14 / Players 6 / Patterns 23 / Hours ~80)
- **Watch list ABOVE roster** — 3 action callouts ("Park's first-serve in dropped 8 pts" / "Rivera's BH up 5" / "Walker trending up"). This is the "look here first" pattern. Watch list MUST stay above roster.
- **Player roster grid**: 6 player cards. Each card uses `.card` (visuals.html `.card:hover` lift). Each has avatar + name/year/last-clip + 5-metric row (FH acc, BH acc, Serve in, Baseline coverage, Clips season count).
- **Hover treatments**: visuals.html `.card:hover` (lift + shadow + border) plus per-metric and per-team-stat dim-siblings micro-hover (additive on top of visuals.html base patterns).
- **Count-up on stat values**: every stat value (team-strip 4 stats + 30 player-card metrics = 34 numbers) is wrapped in `<span data-count-to="N">0</span>`. JS at end of file ports the `data-count-to` IntersectionObserver pattern verbatim from visuals.html. Counts 0→target with ease-out-quart over 720ms on scroll-into-view, resets to 0 on scroll-out.
- **Lead-metric per card** ✅ shipped 2026-05-12: each player card has one `.metric.lead` (the metric with the biggest absolute delta). CSS rule `.metric:not(.lead) { opacity: 0.5 }` greys non-lead metrics at rest. Row-hover dim-siblings (0.35) still overrides, so coach can drill into any metric. Lead label gets `color: var(--ink-soft)` for a subtle weight bump. Current leads: M. Lin → FH (+4), J. Rivera → BH (+5), A. Tran → Serve (+3), S. Park → BH (-4), D. Walker → Serve (+4), K. Nguyen → BH (+3).

### visuals.html (was dashboard.html — RENAMED)
- Per-clip viz reference page. Canonical source for:
  - `.card:hover` lift + shadow pattern
  - `.insight:hover` prose-card lift
  - `.match-row:hover` row-bg + arrow translateX
  - `.legend-chip:hover` + `.muted:hover` opacity 1
  - Dim-siblings pattern (legend chips, etc.)
  - Court tile (`27/39` aspect, `--court-surface` bg, SVG overlays)
  - `buildBars()` shot mix horizontal bar chart (760ms ease-out-quart, 100ms stagger, count-up tick)
  - `buildShotMap` / `buildSpacing` / `buildCoverage` viz functions
  - `renderLegend` + `shotMapCounts` + `spacingCounts` + filter chip wiring
  - **`data-count-to` count-up animation** pattern
- Nav: links to real mock files (dashboard.html, matches-list.html, players-list.html, patterns.html). No active state since visuals is a contextual sub-page.

### match-detail.html
- **Video + notes side-by-side** (recently shipped): `.video-hero` is a 2-col grid (video 1.7fr / notes-card 1fr). Notes panel is its own `.card`, sticky-positioned at `top: 92px`. Mobile (<1000px) stacks.
- **Single viz panel** with 3-way selector (Shot map / Spacing / Coverage). 2-col grid: court tile left (max-height 56vh, aspect 27/39, shrinks width-wise to fit), side legend + coaching insight right. Toggle buttons at top.
- **Shot mix card** — horizontal bar chart `#shotBars` populated by `buildShotBars` (ported byte-identical from visuals.html `buildBars`). Now correctly renders with `width: 100%` on the container.
- **Scouting report card** (just reverted per Brian — coach Jackson liked the section headers): **6 section headers** restored:
  - Match Snapshot
  - Positioning Tendencies
  - Error Patterns
  - Strengths
  - Areas to Improve
  - One-Line Coaching Adjustment (final court-tinted left-rail block)
  - Each section is ONE short paragraph (~2-3 sentences). Headers in JetBrains Mono UPPERCASE court color.
- **Stats card** — 4 tiles only: Winners / Unforced errors / First serve in / Avg rally length. 4-col grid, collapses to 2×2 at <900px. Bigger value font, no stat-split sub-line.
- Patterns card REMOVED (was redundant with scouting's recurring-tactics section).
- **Stats hover** ✅ shipped 2026-05-12: dim-siblings pattern removed. `.stat-tile` now uses the visuals.html `.hero-stat-card` recipe verbatim — `transform: translateY(-2px)` + `box-shadow: var(--shadow-hover)` + `border-color: var(--ink-mute)` on hover. No sibling dimming. Reduced-motion media query updated to match.

### upload-flow.html
- Bigger bounce loader: 560px processing, 380px uploading. Uses transparent Bounce_Animated.webm.
- Uploading state animates progress 0→100 over 9s (`1 - (1-t)^2.4` ease-out curve, live "X.X MB / Y MB" sublabel).
- "What we analyze" panel: 4 coach-voice bullets, no jargon.
- **Done state celebration**: Locked CourtCheckCheckmark.webm (checkmark-only, no wordmark), 340×340 celebration container (consistent with bounce loader sizing). Confetti: 160 particles, brand palette only (lime/clay/court/plum — no red).
- Em dashes removed throughout, "match" → "clip" in user-facing copy.

### auth.html
- Headline simplified to "Create your *account.*" (italic clay on "account"). No longer "coach account" — both coaches and players sign up.
- Role selector: Coach / Player (was 4 options).
- "Email me match-ready notifications" checkbox removed.

### Other mocks
- `players-list.html`, `player-profile.html`, `patterns.html`, `matches-list.html`, `dashboard-empty.html`: brand-mark hover-video wiring, transparent WebM logo, splash CSS, italic-clay rule, mostly untouched in recent sessions.

---

## ⚠ Pending / in-flight at handoff time

### Shipped 2026-05-12
1. ✅ **landing.html stage 4 heatmap** flipped to peak at player's baseline (bottom of overlay). `buildCoverage()` line 1577.
2. ✅ **match-detail.html stat-tile hover** swapped to visuals.html `.hero-stat-card` pattern verbatim. Dim-siblings removed.
3. ✅ **Dashboard per-tile lead metric** — `.metric.lead` class + at-rest opacity dim on non-lead metrics. One lead per card; row-hover dim-siblings still works.
4. ✅ **Cross-mock nav consistency** — all internal mocks now use the new sidebar (no more `href="#"` placeholders).
5. ✅ **Patterns mock removed** — coaches drill into match-detail; redundant page deleted.
6. ✅ **In-product sidebar** shipped across all 8 internal pages (200px wide, top-right collapse toggle, 4 nav items: Dashboard / Clips / Players / Upload, brand-mark hover-video preserved).
7. ✅ **Dashboard splash overlay** — once-per-session full-screen CourtCheckAnimation on dashboard entry (`ccDashSplashSeen` sessionStorage key).
8. ✅ **Match-detail accuracy bar graph** — Shot mix card became 2-column "Shot breakdown" with Mix + Accuracy panels, identical horizontal-bar style.
9. ✅ **Recordings list simplified** — `matches-list.html` is now "Recordings". Stripped score, 1st-serve %, top-pattern, and win/loss pill columns. Each row is just date + player vs opponent + arrow. Filter bar preserved. Page title, h1 ("Every recording, scrubbable."), meta, results count, and footer all updated. File path stays `matches-list.html` for now (deferred rename).

### Still open

5. **🎯 NEXT SESSION — Production React codebase integration.** Brian's stated focus for 2026-05-13. Locked brand assets NOT yet in `frontend/web/`:
   - `components/brand/BrandMark.tsx` still uses old `public/courtcheck_ball_logo.png` — swap to locked Light/Dark PNGs + WebM video.
   - Copy locked PNGs and WebMs from `docs/brand-drop/logo/` and `docs/brand-drop/motion/` into `frontend/web/public/`.
   - `globals.css` italic rule needs to enforce brand law (italic = clay; locked exceptions: `Check` wordmark half + "Start winning" closing CTA).
   - Port landing hero (transparent bounce court video + 3 floating glass cards) and page-load splash overlay verbatim from `mocks/landing.html`.
   - Wire dashboard, match-detail, player-profile, matches-list, players-list to consume `docs/brand-drop/tokens.css` and the locked component patterns (especially the visuals.html viz components).

6. **Match-detail nav label vs file**: nav says "Clips" but the list page is `matches-list.html` and labels itself "Matches". Two options: (a) rename file to `clips-list.html` + update label to "Clips" everywhere, (b) revert match-detail nav to "Matches". Match-detail's `<a class="active">Clips</a>` currently links to `matches-list.html`. Pick a side next session.

7. **One UCD coach quote** — single highest-leverage credibility addition for the landing page. Hold until Brian grabs IRL.

8. **CTAs to real destinations** — currently `#`. Don't share landing URL publicly until "Get started" goes somewhere real.

---

## ffmpeg transparent WebM workflow (canonical recipe)

```bash
# Step 1: sample bg color of source MP4
ffmpeg -y -i source.mp4 -vf "crop=20:20:10:10,scale=1:1:flags=area" -frames:v 1 /tmp/bg.png
python3 -c "from PIL import Image; print(Image.open('/tmp/bg.png').convert('RGB').getpixel((0,0)))"

# Step 2: crop + chroma-key + encode VP9 alpha
ffmpeg -y [-t DURATION] -i source.mp4 \
  -vf "crop=W:H:X:Y,colorkey=0xRRGGBB:0.22:0.08,format=yuva420p" \
  -c:v libvpx-vp9 -pix_fmt yuva420p -b:v 0 -crf 28 \
  -metadata:s:v:0 alpha_mode=1 -an -row-mt 1 -threads 4 \
  output.webm
```

Required: `format=yuva420p` in filter chain, `-pix_fmt yuva420p` on encode, `alpha_mode=1` metadata.

Use in CSS:
```css
.target-video { /* no mix-blend-mode hack needed — alpha is real */ }
body.dark .target-video { filter: invert(1) hue-rotate(180deg); /* dark ink → cream, lime stays lime */ }
```

---

## visuals.html canonical patterns (port verbatim)

When building NEW components, port these from `mocks/visuals.html`:

| Pattern | What it does | visuals.html line range |
|---|---|---|
| `.card:hover` | translateY(-2px) + shadow-hover. Base card lift. | line 240 |
| `.insight:hover` | translateY(-1px) + border-color ink-mute. Subtle prose card lift. | line 432 |
| `.match-row:hover` | bg → shade + `.arrow` translateX(4px). List interactivity. | lines 398–422 |
| `.legend-chip:hover` | translateY(-1px) + border warm. Filter chip. | line 492 |
| `.hero-stat-card:hover` | translateY + shadow + border ink-mute. Stat tile lift. | lines 583–599 |
| Dim-siblings | Container hover → children opacity 0.10–0.45, hovered child opacity 1 + transform. Shot-mix bar rows. | line 522 (`.legend-chip.muted:hover`) |
| `.card-svg` court tile | aspect 27/39, `--court-surface` bg, SVG overlays | line 982 onward |
| `buildBars()` shot mix | 5-row horizontal bars, 760ms ease-out-quart, 100ms stagger, count-up tick | line 1500 |
| `buildShotMap` | individual shot dots with stroke colors + halo + drop-shadow | line 1167 |
| `buildSpacing` | player→ball line per shot, quality-color (court/clay/amber), lime ball endpoint | line 1290 |
| `buildCoverage` | gridded lime heatmap, length×width intensity formula | (post-rename) |
| `renderLegend` | filter chips with counts + active state + dim other strokes | (search legend) |
| `data-count-to` count-up | IntersectionObserver, ease-out-quart 720ms, replays on scroll-out | lines 947–980 |

---

## How to verify state on next session

1. Open `docs/brand-drop/preview.html` in a browser. Toggle theme. Click each mock thumbnail.
2. `cd docs/brand-drop && grep -rn "—" mocks/ --include="*.html" | grep -v "comment\|/\*\|//"` should return no em dashes in user-facing copy.
3. `mocks/visuals.html` is the canonical reference page.
4. Refresh landing.html twice — splash plays first time, sessionStorage gate skips it on the second.
5. Dashboard.html: scroll down past the team strip and the roster. All stat numbers should count up from 0 to target on scroll-in (data-count-to pattern from visuals.html).
6. Match-detail.html: video + notes side-by-side, court viz fits in one viewport, scouting has 6 section headers, stats has 4 tiles.
7. Landing.html scroll demo: at stage 2, court shows opponent's half with shot dots up top. At stage 3+4, court fades to player's half. Spacing lines visible on bottom of tile. **Stage 4 coverage heatmap should be concentrated at BOTTOM (player's baseline) — pending fix in `buildCoverage()`**.

---

*Brand is locked. Build forward. Coaches first. Always reference visuals.html.*
