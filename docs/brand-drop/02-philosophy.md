# CourtCheck UI Philosophy

> **Coaches first. Always.**
> Take the analytical depth of SwingVision and package it in the essence of Robinhood:
> radically simple, visually obvious, instantly responsive, and never intimidating.

---

## North Star

Most tennis analytics tools were built for engineers. They're spreadsheets in a sport context — dense tables, hidden filters, "advanced" settings, jargon-heavy dashboards. A coach finishes a match at 9pm and isn't going to spend 40 minutes navigating a Bloomberg terminal to find that their #2's first-serve percentage dropped 8% on the deuce side.

We do the opposite. **A coach should walk away with a real insight in under 5 minutes of opening the app for the first time.** That's the bar.

The reference: Robinhood didn't simplify investing because investing is simple. It simplified investing because the *interface* didn't need to be hard. The trades are still real, the data is still complex — but the user is shielded from that complexity until they ask for it.

CourtCheck is that, for college tennis film.

---

## Six Principles

### 1. One thing per screen
Every screen has exactly one primary action. The next step is never ambiguous. If a coach has to scan to figure out what to do, the screen has already failed.

> **Test:** Can a coach identify the next action in under 5 seconds without reading body copy?

### 2. Visual-first, tables last
Replace numbers with shapes. Replace tables with court overlays. Replace columns with comparisons. Tables exist for export, not for understanding. Every screen leads with a chart, a court map, or a single big number — not a grid.

> **Test:** If we removed every table on this screen, would the coach still get the message?

### 3. The gentle palette (no panic colors)
Robinhood uses muted pink instead of alarm-red so a 2% drop doesn't feel like a crisis. We do the same. Clay terracotta for "needs attention", never alarm-red. Court green for "in favor" (in both light and dark mode), never neon. Lime is the highlight color only (the ball, the moment, the win), used sparingly to mean *something*.

> **Rule:** No pure `#FF0000`, no pure `#00FF00`, no traffic-light dashboards. Ever.

### 4. Animation is feedback, not decoration
Every tap, swipe, hover, and load gets a visible response. Buttons depress. Cards lift. Numbers count up. Patterns draw themselves on the court. The bounce arc loops while a match is processing — so the user knows the system is *working*.

But: we don't animate ambient elements. We animate the moment that matters. A wiggling sidebar is noise. A scoreline that fills in left-to-right as the player advances is information.

> **Test:** Does this animation tell the user what just happened, or is it just movement?

### 5. The court is the universal frame
Whenever data can be shown on the court, it is. Heatmaps. Shot patterns. Bounce locations. Serve placement. The overhead court is the mental model coaches already use — never make them translate.

The court tile shows up everywhere: pattern cards, match summaries, player profiles. It's our equivalent of Robinhood's portfolio chart — the one persistent visual that anchors the brand.

### 6. Onboarding earns trust in five minutes
- Sign in (Google, ~5s)
- Add team (paste a roster, or skip, ~15s)
- Upload first match (drag/drop, ~15s to start)
- While processing: a 90-second guided tour of *what's coming* — animated previews of patterns, court maps, video clips — so they're sold before the data arrives
- First real insight delivered as a notification: *"Pattern detected — your #1 wins 71% when she opens forehand inside-out."*

If we can't deliver a real insight in the first session, the user will never come back.

---

## What We Borrow From SwingVision (the depth)

- Auto-detected shot tracking — type, placement, in/out, speed
- Player + ball tracking with court calibration
- Frame-accurate clip generation around any shot
- Statistical breakdowns: serve %, winner/UE ratios, point construction patterns
- Annotation, tagging, and sharing primitives

These are the *capabilities*. We don't reinvent them.

## What We Refuse To Copy (the execution)

- Dense stat tables with 30 columns
- Pro-grade jargon ("BP saved %", "kick spin variance")
- Power-user filtering UIs
- Phone-first MVP layouts crammed into a desktop view
- Self-service report builders that ask the coach to assemble their own answer

We don't ship "the kitchen sink, in a web app." We ship "the answer, on the page."

---

## UI Patterns That Carry The Brand

### Numbers in Newsreader serif
Every stat — `68%`, `4-3`, `+12.4` — renders in Newsreader 500. Not for prettiness: for **trust**. A serif number reads as observed, considered, signed off on. A sans-serif number reads as auto-generated, churned out, machine-output. Our brand promise is "we watched the film for you" — the typography reinforces that promise.

### One insight per card
A card holds **one fact** and **one comparison**. Not three facts. Not a grid of metrics. If you have four things to say, you have four cards.

✅ "First serve in: 68% · +4.2 vs season avg"
❌ "First serve: 68% in / 8% ace / 12% DF / 64% win — last 5 matches: ↑ ↓ ↑ ↑ ↓"

### Two-color logic for outcomes
- **Court green** (both light and dark): in your favor, working, won, on track
- **Clay terracotta**: needs attention, opponent strength, decision point
- **Lime**: highlight only — the ball, the apex, the moment of impact

Red is reserved for one thing only: a system error or destructive action confirmation. Never for performance.

### Categorical color (shot type, player ID, court zone — not outcomes)

Outcomes use the two-color logic above. **Categorical data — the kind where there's no "better" or "worse" — needs its own extended palette.** Forehand isn't good. Backhand isn't bad. They're just different, and the chart needs to tell them apart.

**The categorical palette (6 slots, locked). Canonical source: `tokens.css` lines 30-34 (light) and 109-113 (dark):**

| Slot | Token | Light hex | Dark hex | Default role |
|---|---|---|---|---|
| 1 | `--court` | `#2E5341` | `#6FA88B` | First category (e.g., Forehand) |
| 2 | `--plum` | `#7B4E6E` | `#B584A6` | Second (e.g., Backhand) |
| 3 | `--amber` | `#C8923A` | `#DDB166` | Third (e.g., Serve) |
| 4 | `--clay` | `#B05B36` | `#E07A52` | Fourth (e.g., Volley) |
| 5 | `--slate` | `#5C6B7A` | `#889DB3` | Neutral / "Other" |
| 6 | `--lime` | `#BFC846` | `#DDE970` | Ball / accent (sparing) |

**Rules:**

- **Categories lock product-wide.** Forehand is always Court green. Always. Same color, every chart, every screen, every player. Coaches build muscle memory around it.
- **Outcome semantics don't apply inside a categorical chart.** Green here means "forehand," not "good." Clay means "slice," not "needs attention." Within a categorical view, the legend defines meaning.
- **Cap at 6 categories per chart.** If you have more, group or filter. A chart with 12 colors is a chart with zero information.
- **Never combine outcome logic and categorical logic in the same view.** Pick one. If a chart is about shot types, don't *also* color by win/loss. Use shape, opacity, or a second small chart instead.

### Empty states teach the product
Don't show "No matches yet." Show what's coming, and how to get there:

> **Patterns will appear here after your first match is processed.**
> Upload a video → we'll surface 5–10 patterns within ~20 minutes.
> *[Upload first match →]*

Every empty state is a tiny ad for the product itself.

### Loading is a story
The bounce arc animation isn't filler — it tells the user *the system is reading the match the way you would*. The trail draws as the ball travels: descent, bounce, rebound. Same metaphor every time. Coaches recognize it within 2–3 sessions and it becomes our brand-as-loading-state.

---

## The Dashboard Vision

The first thing a coach sees after sign-in:

1. **One hero card** — the most important insight from the most recent match. Not a list. Not a grid. The single thing they need to know.
2. **A small row of three cards** — recent matches, players to watch, upcoming opponents.
3. **A scrollable feed** of patterns, clips, and observations — chronological, easy to skim, generated automatically.

That's it. No nav rails of 12 modules. No customizable dashboards. No "build your own view." The product opinionates *for* the coach.

A coach who wants the full breakdown taps deeper. A coach who's checking quickly between practices gets value in 30 seconds.

---

## The Robinhood Test (use before shipping)

For every screen, every feature, every flow:

1. **5-second test** — Can a coach identify the next action in under 5 seconds?
2. **Pulse test** — When they tap, does something visibly respond?
3. **One-step test** — Is there exactly one action that's clearly the next step?
4. **Picture test** — Did we replace a table with a visual?
5. **Coach voice test** — Does the copy sound like an assistant coach, not a SaaS product manager?

If any answer is no — redesign before ship.

---

## Anti-Patterns We Will Never Ship

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

---

## Voice & Tone

Speak like a thoughtful assistant coach who's already watched the film:

- **Specific**: "Loses the cross 64% on second-serve return" — not "AI-powered insights."
- **Calm**: No exclamation marks. No urgency theater. The product earns trust by being quiet and right.
- **Coaching, not selling**: We don't say "Crush your competition." We say "Here's what to work on Wednesday."

We're not a feed. We're not a hype tool. We're the film room, automated.

---

*Living document. Update when a principle proves wrong in practice — never water one down because a feature didn't fit it.*
