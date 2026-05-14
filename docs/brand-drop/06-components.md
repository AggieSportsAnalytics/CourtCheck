# CourtCheck Component Library

> Catalog of every primitive in the brand mocks. Each entry: what it does, where it's used, the visual + behavior contract, and notes for the React port.
>
> Companion to [`brand-style.html`](./brand-style.html), [`dashboard.html`](./dashboard.html), [`dashboard-empty.html`](./dashboard-empty.html), [`upload-flow.html`](./upload-flow.html), [`ui-philosophy.md`](./ui-philosophy.md), and [`motion-and-data.md`](./motion-and-data.md).

---

## Conventions

- **Stack target**: React 19 + Next.js 16 App Router + TypeScript + Tailwind v4 + shadcn/ui (Radix primitives).
- **Live in**: `frontend/web/components/courtcheck/` — flat folder, one `.tsx` per primitive.
- **Theme**: All components read from CSS variables defined in `app/globals.css`. Light + dark via `body.dark`.
- **Motion**: All animations use the easing/duration tokens from `motion-and-data.md`. Honor `prefers-reduced-motion`.
- **Iconography**: Lucide only (already in `package.json`). Stroke `1.75`, round caps and joins. See brand-style.html § 05 for the spec.

---

## 1. Layout primitives

### `<Container>`
Max-width wrapper used by every page.
- **Width**: `1320px`, `padding: 0 56px` desktop, `0 28px` mobile
- **Variants**: `narrow` (720px, used for upload page), `wide` (1480px, rare)

### `<Sidebar>`
Primary in-product navigation. Used on every authenticated page (Dashboard, Players, Upload, Recordings, Player Profile, etc.). The public landing page uses `<Nav>` instead.
- **Element**: `<aside class="app-sidebar" aria-label="Primary navigation">`
- **Position**: `fixed; top: 0; left: 0; height: 100vh`, `z-index: 50`
- **Width**: `200px` (default), `72px` (collapsed). Toggled by user click on `<SidebarCollapse>` OR auto-collapsed below `1100px` viewport
- **Background**: `--paper`. Border-right: `1px solid var(--line-soft)`
- **Padding**: `22px 14px`
- **Body offset**: `body { padding-left: 200px }` (or `72px` collapsed) so centered `.container` flows in remaining viewport
- **Transition**: `width 220ms var(--ease-out)` (also applied to body padding)
- **Children** (top to bottom): `<SidebarBrand>`, `<SidebarCollapse>` (positioned absolute), `<SidebarNav>`, `<SidebarFoot>`

### `<SidebarBrand>`
Wraps the CourtCheck logo as a clickable home link at the top of the sidebar.
- **Classes**: `brand-mark sidebar-brand` (the global `.brand-mark` JS hook plays the locked CourtCheckAnimation.webm on hover)
- **Min-height**: `76px`
- **Border-bottom**: `1px solid var(--line-soft)`
- **Logo elements**: light + dark wordmark PNGs (`52px` tall, `max-width: 100%`) and the hover-video
- **Collapsed mode**: wordmark hides; the locked CourtCheckCheckmark.webm shows instead (paused at frame 0 by default, plays on hover)

### `<SidebarNav>`
Flex column container holding the four primary `<SidebarLink>` items.
- **Layout**: `display: flex; flex-direction: column; gap: 4px`
- **Order is fixed**: Dashboard, Players, Upload video, Recordings

### `<SidebarLink>`
Anchor with icon + label representing one nav destination.
- **Min-height**: `44px` (touch-target)
- **Padding**: `10px 12px`
- **Font**: `0.92rem`, weight `500`
- **Color**: `--ink-soft` default, `--ink` on hover with `--shade` background
- **Active state**: color `--ink`, weight `600`, background `--shade`, AND a `3px` court-color left rail via `::before` (supplements color so meaning is not color-only, for a11y)
- **A11y**: active link has `aria-current="page"`. Every link has a `title=` attribute so the icon-only collapsed mode still surfaces the destination
- **Icon**: Lucide, `18px`, stroke `1.75`, placed before label

### `<SidebarCollapse>`
Circular chevron button that toggles sidebar width.
- **Size**: `26px` circle
- **Position**: `absolute; top: 14px; right: 8px` (inside the sidebar)
- **Behavior**: toggles `body.sidebar-collapsed` class, persisted to `localStorage` under key `cc-sidebar-collapsed`
- **Icon**: Lucide chevron SVG; rotates `180deg` when sidebar is collapsed
- **A11y**: `aria-label="Collapse sidebar"` (or "Expand sidebar" when collapsed), `aria-expanded` reflects state

### `<SidebarFoot>`
Bottom row of the sidebar.
- **Layout**: `display: flex; justify-content: flex-start; gap: 10px`
- **Order (left to right)**: `<UserMenu>` first, then `<ThemeToggle>`
- **Padding**: `14px 6px 4px`
- **Border-top**: `1px solid var(--line-soft)`

### `<UserMenu>`
Avatar trigger that opens a popover menu (account, team, help, sign out).
- **Trigger**: `<button class="avatar">` showing user initials
- **Open**: click trigger; **Close**: click-outside or Escape
- **Popover position**: `absolute; bottom: calc(100% + 12px); left: 0` (pops UP from the avatar with left edge aligned to sidebar interior)
- **Popover width**: `248px`
- **Popover chrome**: `--paper` background, `12px` border-radius, `--shadow-hover`
- **Animation**: slide-in from `translateY(8px)` to `0`, opacity `0 → 1`
- **Contents**:
  - Header (avatar + name "Coach B. Le" + email in JetBrains Mono)
  - Divider
  - Items: Account settings, Team settings, Help & feedback
  - Divider
  - Sign out (clay color, visually separated)
- **A11y**: trigger has `aria-haspopup="menu"` and `aria-expanded`; popover has `role="menu"` and `aria-hidden` reflecting state; each item has `role="menuitem"`

### `<Nav>`
Public-facing top navigation bar. **Landing page only** (in-product pages use `<Sidebar>`).
- **Sticky**, `padding: 22px 0`, blurred backdrop (`backdrop-filter: blur(10px)`)
- **Children**: `<BrandMark>` (left), nav links + `<ThemeToggle>` + `<Avatar>` (right)
- **Active link**: bold `font-weight: 600`, ink color (vs `--ink-soft` default)

### `<BrandMark>`
Logo + wordmark.
- **Logo**: `<img>` referencing the chosen logo SVG (currently `logo-arc-loop.svg`)
- **Wordmark**: Newsreader 500, opsz 60, `1.25rem`, letter-spacing `-0.018em`
- **Sizes**: 32px (default), 28px (compact)

### `<ThemeToggle>`
Sun/moon button. Stores preference in `localStorage` under key `cc-theme`.
- **Size**: 36×36, circle border-radius
- **Hover**: rotates `20deg`, border becomes `--ink`
- **Icon**: shows moon in light mode, sun in dark — both Lucide

### `<Avatar>`
Circle with initials.
- **Default size**: 36×36
- **Background**: `--court` (light) / `--court-deep` (dark) by default; can be overridden per-instance with categorical colors (see Match Row use case where avatars rotate `--court` / `--clay` / `--plum`)
- **Type**: Newsreader 500, `0.95rem`
- **Image fallback**: Initials only — never use stock avatars

---

## 2. Typography & marks

### Display heading
- Class: `.display`
- Font: Newsreader 500, `font-variation-settings: 'opsz' 72`
- Letter-spacing: `-0.022em`
- Use for: page titles, hero headings, big stat numbers

### `.ital`
Editorial italic accent — applied to one to three words inside a display heading. Newsreader italic, weight 400.

### `.num`
Number rendering. Newsreader 500, opsz 72, `font-feature-settings: 'tnum'` (tabular numerals so digits align). Use for **all** numeric values across the product, including chart axis labels.

### Mono
Class: `.mono` (or inline). JetBrains Mono, `0.72rem`, `letter-spacing: 0.12em`, uppercase. Color `--ink-mute`. Use for: stage labels, status meta, scoreboard sets, axis labels.

### Eyebrow
Class: `.eyebrow`. JetBrains Mono `0.72rem`, letter-spacing `0.18em`, uppercase, color `--court`. Always preceded by a 6×6 clay dot via `::before`. Use for: section labels above headings, card category labels.

---

## 3. Buttons

### `<Button>` — primary
- Background: `--ink` (light) / `--court-deep` (dark)
- Color: `--cream`
- Padding: `12px 22px`, border-radius `100px`
- Hover: `transform: translateY(-1px)` with `--ease-spring`
- Use for: single most important action per screen

### `<Button>` — ghost
- Background: transparent
- Border: `1px solid var(--line)`
- Hover: border becomes `--ink`
- Use for: secondary actions

### Sizes
- Default: `12px 22px`, `0.95rem`
- Small: `8px 14px`, `0.85rem` (use sparingly — always pair with a primary)
- Large (CTA hero): `14px 24px`, `1rem`

### Icon usage
Lucide icon at `18px`, `gap: 10px` from text. Stroke 1.75. Place icon before text by default; `→` chevron after for "view all" / "next step" actions.

---

## 4. Cards

### `<Card>`
Base card primitive.
- Background: `--paper`
- Border: `1px solid var(--line)`
- Radius: `--radius-lg` (14px)
- Padding: `28px`
- Hover: `transform: translateY(-2px)`, shadow strengthens (`--shadow-hover`), `--dur-base var(--ease-out)`

### `<StatCard>`
Card with eyebrow, big number, label, and a comparison footer.
- **Eyebrow** (top-left): JetBrains Mono small caps
- **Tag** (top-right, optional): `<Tag>` showing state (Watch / On track / Pattern)
- **Big number**: Newsreader 500, ~`4.4rem` (regular) or `clamp(86px, 10vw, 144px)` (hero)
- **Unit**: 0.4× the number size, `--ink-mute`
- **Label**: `--ink-soft`, `0.95-1rem`
- **Compare** (footer): Bordered top, two-column flex with directional arrow + "vs season avg" text. Uses `.pos` (court) or `.neg` (clay).

### `<HeroStatCard>`
Larger variant for the dashboard's hero insight.
- Two-column grid: copy left, big number + compare right (or single column on mobile)
- Number animates with `<CountUp>` on scroll-into-view

### `<ScoreCard>`
Dark-themed scoreboard card (the one bright Pacific Blue moment in dark mode).
- Background: `--ink` (light) / linear gradient `#1E5FA8 → #0F3D75` (dark)
- Color: `--cream` text on top
- Player rows: `<Avatar>` + name + set scores in mono
- Won set: lime accent

### `<EmptyCard>`
First-session card — explains what'll appear after a match is uploaded.
- Same Card chrome, but content is **teaching copy** + a small illustrative element (court placeholder, faint preview)
- Footer mono: `Earned after first match`
- See `dashboard-empty.html` for examples

---

## 5. Tags

### `<Tag>`
Small chip indicator.
- Sizes: `5px 11px` padding, `0.78rem`, weight 500
- Border-radius: `100px`
- **Variants** (color + soft background tint):
  - `green`: `--court` text on `rgba(46,83,65,0.10)` bg → `Pattern`, `On track`, `In play`
  - `clay`: `--clay` text on `rgba(176,91,54,0.10)` bg → `Watch`, `Concern`, `Bounce in`
  - `lime`: `#6E7522` text on `rgba(191,200,70,0.18)` bg → `Winner`, `Live`
- Modifier `dot`: prepends a 6×6 dot in the same color (use for state indicators)

### Dark-mode shifts
Each variant has dark-mode-specific bg and text colors (already specified in `dashboard.html` CSS).

---

## 6. Court tiles

### `<CourtTile>`
The canonical court SVG, vertical orientation.
- **Aspect ratio**: `27/39` (half-court display) or `27/78` (full court)
- **Surface**: `--court-surface` (clay light, Pacific Blue dark)
- **Lines**: white, stroke widths from spec (`0.3` boundary, `0.5` net, `0.25` service lines)
- **Props**: `half: 'top' | 'bottom' | 'full'`, `data` (varies by mode), `mode: 'shot-map' | 'heatmap' | 'spacing' | 'empty'`

### `<ShotMap>`
Court tile with two-circle shot dots.
- Each shot: white halo (`r=1.15`) + colored fill (`r=0.85`) — see motion-and-data.md § shot map
- Filtered by stroke type via parent's filter state

### `<Heatmap>`
Court tile with intensity zones.
- **Resolution**: `8 × 12` cells per half-court (96 total) — fine enough for spacing analysis
- **Single color**: `--lime`, opacity = intensity
- **Stagger reveal**: by row, baseline → net, 35ms per row
- Scroll-triggered via IntersectionObserver

### `<SpacingLines>`
Court tile with player→ball lines per shot.
- **Line**: `stroke-width: 0.5`, color by quality (`--court` ideal, `--clay` squeezed, `--amber` long)
- **Player end**: white circle `r=0.55` with colored border
- **Ball end**: lime circle `r=0.45` with colored border
- **Filter**: by stroke type (lines for non-active strokes dim to 10% opacity)

### `<QualityLegend>`
Mini-line legend for spacing visualization. SVG mini versions of squeezed/ideal/long lines, with line lengths reinforcing the meaning. Static, not clickable.

### `<HalfLabel>`
Mono caption under a court tile. e.g. "Opponent's half · where shots land".
- JetBrains Mono `0.68rem`, letter-spacing `0.14em`, uppercase, `--ink-mute`
- Always immediately under the court tile, centered

---

## 7. Charts

### `<BarChart>`
Horizontal bars with label, fill, and value.
- **Track**: `--shade` background, `22px` height, `100px` border-radius
- **Fill**: animates `width: 0 → target%` over `760ms` with `cubic-bezier(0.22, 1, 0.36, 1)` (ease-out-quart)
- **Stagger**: 100ms between bars
- **Number count-up**: synchronized with fill, runs same duration with same easing curve
- **Scroll-triggered**: fires on IntersectionObserver entry, resets on exit
- Width breakdown: `110px` label / `1fr` track / `60px` value (Newsreader 500)

### `<CountUp>`
Animated number from 0 to target over 720ms (default).
- **Trigger**: IntersectionObserver on **nearest card** (not the number itself — fires when card meaningfully visible)
- **Threshold**: 0.35
- **Resets** on scroll-out so re-entry replays
- Cancels in-flight `requestAnimationFrame` before reset

---

## 8. Filters

### `<LegendChip>`
Pill-shaped chip with swatch dot, label, and optional count.
- **Inactive**: `--paper` bg, `--line` border, `--ink-soft` text
- **Active**: `--cream` bg, `--ink` border, `--ink` text
- **Muted** (sibling chips when one is active): `opacity: 0.45`, hover restores to 1
- Click toggles its category as the active filter; clicking again resets to "all"
- When any chip is active, a "Show all" reset chip appears in italic at the end of the row

### `<Legend>`
Container for `<LegendChip>` rows. `display: flex`, `gap: 8px`, `flex-wrap: wrap`.

### `<PeriodPills>`
Segmented control for time period (Today / This week / Season).
- Pill bag with `4px` padding around 4-8px-padded buttons
- Active button: `--ink` bg, `--cream` text
- Used in dashboard header

---

## 9. Forms

### `<Field>`
Label + input wrapper.
- **Label**: JetBrains Mono `0.7rem`, letter-spacing `0.14em`, uppercase, `--ink-mute`
- **Input/Select**: `padding: 12px 14px`, `--paper` bg, `1px --line` border, `10px` radius
- **Focus**: border `--ink`, bg `--surface` (white in light, elevated in dark)
- **Layout**: vertical label-above-input by default; `<FieldRow>` for two-up

### `<FieldRow>`
Two-column field group, `1fr 1fr`, `gap: 18px`. Stacks on mobile.

### `<Dropzone>`
Drag-drop file upload area.
- **Idle**: dashed border (`--line`), upload glyph 56×56 in shade-colored rounded square, "Drop your recording here" + "or browse files" link, file types in mono
- **Hover/Drag**: border becomes `--ink`, brief background lift
- **States**: idle, uploading (file pill + progress), processing (rally animation + status), done (check + view CTA) — see `upload-flow.html` for the full state machine

### `<ProgressBar>`
- Track: `4px` height, `--shade` bg, `100px` radius
- Fill: `--court`, animates width transition over `--dur-base` with `--ease-out`
- Meta row below: left text + right percentage in mono

---

## 10. Coaching elements

### `<CoachingInsight>`
Callout box under a chart with a one-sentence interpretation.
- Background: `--shade` (light) / `--paper` (dark)
- Padding: `14px 18px`, radius `12px`
- Display: flex, gap 12px
- **Arrow**: italic `→` in `--clay` (light) / `--clay-soft` (dark), Newsreader italic
- **Text**: numbers wrapped in `<strong>` render as Newsreader 500 with tabular numerals
- Used after court tiles to surface the "what does this mean for Wednesday's drill" answer

### `<RecordingRow>` (renamed from `<MatchRow>`)
Row in the Recordings index. Each row is a single recording (date + player vs opponent + arrow).
- **Grid**: `110px 1fr 40px` (date / player-row / arrow), `gap: 16px`
- **Padding**: `18px 24px`
- **Border-bottom**: `1px solid var(--line-soft)`
- **Cursor**: pointer
- **Hover**: background `--shade` (light) / `--surface` (dark); arrow translates `4px` right
- **Children**:
  - `.ts` — date in JetBrains Mono `0.78rem`, color `--ink-mute`
  - `.player-row` — `<Avatar>` (32px) + `.player-info` column (player name + meta line)
  - `.arrow` — Lucide chevron, color `--ink-mute`
- Avatars use categorical palette: court / clay / plum (rotate per-row to differentiate players)

### `<InsightItem>`
Card showing a single recurring tactical insight surfaced from a player's match history. Used inside grouped insight lists on `player-profile.html`.
- Same chrome as `<Card>` but lighter padding (`20px`), `12px` radius
- **Head**: eyebrow + tag
- **Text**: Newsreader 500 `1.1rem`, numbers in italic-clay accent

### `<PatternCard>`
Variant of `<Card>` showing a court visualization with a coaching narrative. Used in the season-patterns section of `player-profile.html`.
- Top section: eyebrow + tag
- Middle: court tile with annotation arrow showing the pattern (e.g., forehand inside-out trajectory)
- Bottom: heading + 1-2 sentence explanation

---

## 11. State indicators

### `<DevToggle>` (mock-only)
Segmented control used in `upload-flow.html` to demo state transitions.
- **Not for production** — would be removed before ship.
- Useful pattern for documentation / brand pages.

### `<Loading>` — bounce arc
The bounce arc SVG animation (`logo-arc-loop.svg`). Use as the primary loading indicator for any operation that takes >1s.
- Sizes: 24px (inline), 48px (default), 80px (full-page)
- Loops at `--dur-loop` (2400ms)
- Pauses on `prefers-reduced-motion`

### `<Skeleton>`
Shimmer placeholder for loading content.
- Background gradient: `--shade → --paper → --shade`, `200%` size
- Animation: `shimmer` keyframes, `--dur-loop` linear infinite
- Use for: card-shaped placeholders during data fetch
- See motion-and-data.md § Loading for the keyframes

### Toast (planned)
Sonner toasts (already installed). Conventions:
- Success: `--court` accent
- Error: `--clay` accent (never red)
- Position: bottom-center for primary, top-right for system

---

## Component dependencies

```
<Container>
└─ <Nav>            (landing page only)
   ├─ <BrandMark>
   ├─ <ThemeToggle>
   └─ <Avatar>

<Sidebar>           (all in-product pages)
├─ <SidebarBrand>
│  └─ <BrandMark>
├─ <SidebarCollapse>
├─ <SidebarNav>
│  └─ <SidebarLink> (×4: Dashboard, Players, Upload video, Recordings)
└─ <SidebarFoot>
   ├─ <UserMenu>
   │  └─ <Avatar>
   └─ <ThemeToggle>

<Card>
├─ <Tag>
├─ <CountUp>
└─ <CoachingInsight>

<CourtTile>
├─ <ShotMap>
├─ <Heatmap>
└─ <SpacingLines>
   └─ <QualityLegend>

<BarChart>
├─ <CountUp>
└─ <Legend>
   └─ <LegendChip>

<Dropzone>
├─ <ProgressBar>
└─ <Loading> (bounce arc)
```

---

## Build order recommendation

When porting to React, build in this order so dependencies resolve cleanly:

1. **Tokens + globals** — CSS variables, Tailwind theme, Newsreader/Inter Tight/JetBrains Mono fonts
2. **Foundational primitives** — `<Container>`, `<BrandMark>`, `<Button>`, `<Card>`, `<Eyebrow>`, `<Avatar>`, `<Sidebar>` and its children (`<SidebarBrand>`, `<SidebarNav>`, `<SidebarLink>`, `<SidebarCollapse>`, `<SidebarFoot>`, `<UserMenu>`), `<ThemeToggle>`. Note: `<Nav>` is the public landing-page top bar only (build alongside or after the in-product `<Sidebar>`).
3. **Atoms** — `<Tag>`, `<Field>`, `<FieldRow>`
4. **Display** — `<StatCard>`, `<CountUp>`
5. **Charts** — `<CourtTile>` first (the foundation), then `<ShotMap>`, `<Heatmap>`, `<SpacingLines>`, `<BarChart>`
6. **Filters** — `<LegendChip>`, `<Legend>`, `<QualityLegend>`, `<PeriodPills>`
7. **Coaching surfaces** — `<CoachingInsight>`, `<RecordingRow>`, `<InsightItem>`, `<PatternCard>`, `<HalfLabel>`
8. **Forms + flows** — `<Dropzone>`, `<ProgressBar>`, `<Loading>`, `<Skeleton>`
9. **Empty states** — `<EmptyCard>` (uses everything above)
10. **Page assemblies** — Dashboard, FirstSessionDashboard, UploadFlow

Each phase is independently shippable. Phase 1-3 unblocks the marketing/landing pages. Phase 4-5 unblocks the dashboard. Phase 6-8 unblocks interactivity. Phase 9-10 ties it all together.

---

*Living catalog. When a primitive's contract changes in the brand mocks, update this doc in the same commit.*
