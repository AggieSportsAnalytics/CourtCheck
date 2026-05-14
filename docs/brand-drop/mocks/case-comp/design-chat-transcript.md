# I need more context to name the thread. Could you share the user's first message about what they're designing or asking about?

_Started 2026-05-12 23:02 UTC_

---

## User

<system-info comment="Only acknowledge these if relevant">
Project title is now "Case Comp"
Current date is now May 12, 2026
</system-info>

<attached_files>
- uploads/Copy of Intern Presentation - Branch Staffing Optimization Model (August 2025).pdf
</attached_files>

<pasted_text name="Pasted text (171 lines)">
I'm building a case-competition pitch deck for CourtCheck. Friday May 15 deadline. I want TWO deliverables in this session:

A. THE CASE COMP DECK — 6 main slides + 4 appendix, filled with the content below
B. A REUSABLE SLIDE TEMPLATE LIBRARY — 10 empty themed layouts I can copy from to build future decks

STYLE DNA — fuse two sources:

1. STRUCTURE comes from the AAA MWG deck I'm attaching (PDF). Specifically:
   - Action title at top of every content slide — the takeaway IS the title, one declarative sentence with an underlined keyword (AAA underlines key phrases in black: "all 101 branches", "live", "immediate use"). Underline color in our brand = court green.
   - Thin gradient accent bar across the top of every slide (4px). AAA uses red-purple-blue; ours uses clay → plum → court (#B05B36 → #7B4E6E → #2E5341 light mode).
   - Small slide number bottom-right in mono uppercase
   - Section divider slides — full dark slide with big centered title (AAA uses navy; we use #0A0E14)
   - Step-by-step progression slides (Step 1 / 2 / 3 / 4) with circled numbered badges
   - 3-column executive-summary frame (Challenge / Solution / Impact)
   - 2-column layouts: narrow text sidebar + wide chart or table on the right, divided by a dashed vertical line
   - Color-coded categorical entities (AAA uses blue/green/purple/orange for 4 clusters). We map to: court green / plum / amber / clay.
   - Tables with chevron-shaped action columns
   - Boxed sidebar callouts with takeaways
   - Big highlighted KPIs called out in larger type than body

2. EXECUTION comes from the locked CourtCheck brand (see attached bundle):
   - Cream paper #F4F0E6 base, near-black #0A0E14 dark mode
   - Newsreader serif for display + numerics, Inter Tight body, JetBrains Mono UPPERCASE for system meta and column headers
   - Italic always renders clay (#B05B36 light / #E07A52 dark). Action titles get ONE italic-clay word, max.
   - All numerics in Newsreader 500 with tabular-nums
   - No red. Clay handles pain/failure framing. Lime (#BFC846) is the ball moment, sparing — reserve for impact slide kicker
   - Locked logo PNGs at bottom-left chrome (24-28px tall)

The TENSION between AAA's bold sans-serif consulting register and CourtCheck's editorial Newsreader serif resolves as: keep AAA's action-title structure and information density, but typeset everything in Newsreader (display + numerics) and Inter Tight (body) with italic-clay accents. Use underline-on-keyword sparingly, the way AAA does.

ASPECT: 16:9, 1920x1080. Build as HTML slides (one file, arrow-key navigation, G for grid view, T for theme toggle, print to PDF).

================================================================
A. THE CASE COMP DECK — 6 MAIN + 4 APPENDIX
================================================================

ARC: Pain → Why existing fail → Relief (demo) → Proof → Expansion → Ask
DOCTRINE: defensive content (methodology, architecture) lives in appendix as Q&A weapons

SLIDE 1 — COLD OPEN
Layout: full-bleed dark mode (#0A0E14), no chrome, no slide number, no logo
Content: single Newsreader pull quote centered, ~84px, cream type, italic-clay on one evocative word
Attribution beneath in JetBrains Mono UPPERCASE 16px, ink-mute color
Optional: Bounce_Still.png at 6-8% opacity full-bleed behind quote
Placeholder quote: "[PULL QUOTE FROM SARA JACKSON — TO LOCK MAY 12]"
Attribution placeholder: "SARA JACKSON · HEAD COACH, UC DAVIS WOMEN'S TENNIS"
Hold 4s silent in narration after this appears.

SLIDE 2 — THE STRUCTURAL GAP
Layout: cream paper, full chrome, action title at top
Action title: "The footage exists. The *coaching* moments don't." (italic-clay on "coaching")
Body: 4 stacked stat rows, hairline separator between each. Each row = giant Newsreader 500 tabular-num value on the left (~72px) + JetBrains Mono UPPERCASE descriptor on the right.
Placeholder stats (LOCK MAY 12):
   [80 hrs] — FOOTAGE PER SEASON, PER PROGRAM
   [12%] — PERCENT ACTUALLY REVIEWED
   [3 days] — TYPICAL LAG FROM MATCH TO COACHING INSIGHT
   [TBD%] — PERCENT OF COACHING DECISIONS DATA-BACKED TODAY
Kicker strip at bottom in Newsreader italic 28px: "It's not a data problem. It's a *workflow* problem." (italic-clay on "workflow")

SLIDE 3 — WHY EXISTING TOOLS FAIL
Layout: cream paper, action title at top
Action title: "Every existing option was built for *someone else*." (italic-clay on "someone else")
Body: 4-row failure matrix, 3 columns. JetBrains Mono UPPERCASE column headers tracked 0.1em. Hairline row dividers, no full borders. Each row gets a 4px left rail in a categorical color.
Columns: OPTION / BUILT FOR / WHY COACHES DON'T USE IT
Rows:
  1. Manual review (clay rail) — Built for: coach with unlimited time — Why fails: review takes longer than next match arrives
  2. Hawk-Eye (court rail) — Built for: ATP/WTA broadcast — Why fails: $100K+ installation, no D-I college budget
  3. SwingVision (plum rail) — Built for: recreational singles player — Why fails: phone-mounted single court, no team workflow, no coach-facing insights
  4. PlaySight (amber rail) — Built for: tennis academies — Why fails: hardware contract, prohibitively expensive, generic dashboards

SLIDE 4 — DEMO (RELIEF)
Layout: full-bleed, minimal 16px chrome only
Top-left eyebrow: "DEMO" in JetBrains Mono UPPERCASE
Top-right: slide number
Body: empty content frame (placeholder rectangle in --paper with thin --line border, radius 18px). I'll drop a live product window or pre-recorded screen capture here.
Speaker notes embedded as HTML comments: 5-beat demo storyboard tied to Slide 2 stats (footage volume → ingest, lag → minutes-not-days, percent-reviewed → coverage heatmap, percent-decisions-data-backed → tendency callouts, the "one decision" anecdote)

SLIDE 5 — IMPACT (PROOF)
Layout: cream paper, action title at top
Action title: "The relief, in numbers." (italic-clay on "relief")
Primary contrast: two massive Newsreader 500 tabular-num numerals side by side, ~280px tall:
   [80 hrs] → [12 min]
   Italic-clay on "12". Mono uppercase captions beneath each ("PER MATCH, MANUAL" / "PER MATCH, COURTCHECK").
Below the pair: 3-tile micro-grid of secondary stats
   [N matches processed] — LIFETIME · BACKEND
   [N% top-1 stroke accuracy] — EVAL HARNESS · MAY '26
   [1 design partner] — UC DAVIS WOMEN'S TENNIS
Right-side inset card: design partner pull quote in Newsreader italic 24px, different from Slide 1 quote
Bottom kicker strip in Inter Tight 22px: "[Anchor anecdote — one concrete coaching decision Sara made because of CourtCheck. TO LOCK MAY 12]"
Reserve the single lime accent on this slide — use it on the "12 min" numeral or the kicker.

SLIDE 6 — WHERE THIS GOES + ASK
Layout: cream paper, action title at top
Action title: "Built for UC Davis. Same problem, *every program*." (italic-clay on "every program")
Body: 3-tier horizontal progression separated by 1px ink-mute vertical hairlines. Each column has italic Roman numeral in Newsreader italic 88px (clay), then Newsreader 32px sub-headline, then 2-line Inter Tight body.
  i. TODAY — Tennis · D-I college tier · UC Davis design partner
  ii. 12-24 MONTHS — Adjacent racket sports · 8-12 college programs
  iii. 3-5 YEARS — Institutional sports analytics platform · multi-sport
Footer ask strip (JetBrains Mono UPPERCASE 14px, court color): "[ONE-LINE ASK — TO LOCK MAY 12]"

APPENDIX DIVIDER — dark mode slide, no chrome, "Appendix" centered in Newsreader italic 120px (italic-clay)

A1 — METHODOLOGY + EVAL HARNESS
Action title: "How we measure ourselves."
Body: per-class precision/recall/F1 table (forehand / backhand / serve / volley rows; P / R / F1 columns), JetBrains Mono UPPERCASE headers, court-color cell highlights on the strongest metric per row. Below: confusion-matrix thumbnail + tolerance bands callout in a sidebar card.

A2 — TECHNICAL ARCHITECTURE
Action title: "Five-stage pipeline. Designed to ship weekly."
Body: 5 horizontal blocks with circled step numbers and 3px court left rails:
  1. INGEST — TrackTennis cloud portal, transcode to H.264
  2. DETECTION — YOLOv8 ball + player class
  3. TRACKING — BoT-SORT, identity persistence
  4. TRAJECTORY — homography to court coordinates, minimap render
  5. PATTERN DETECTION — stroke classification, tendency clustering, scouting report generation

A3 — UNIT ECONOMICS
Action title: "Pricing built for athletic department budgets."
Body: 4-tile grid (Target ACV / Gross margin / Payback / Implementation) + pricing thesis sidebar card. Numbers are placeholders to lock — keep them clearly marked.

A4 — ROADMAP + RISKS
Action title: "12 months, three risks, three mitigations."
Left half: horizontal 4-quarter roadmap timeline with milestone markers
Right half: 3-row risk table — Risk / Likelihood / Mitigation

================================================================
B. REUSABLE TEMPLATE LIBRARY — 10 LAYOUTS
================================================================

Build these as a separate `templates.html` file (or templates section of the same deck). Each template is an empty layout with [PLACEHOLDER] text I can replace. All use the same brand tokens + AAA structural DNA.

T1 — Cover slide — title-card with "2026 [EVENT NAME]" eyebrow, project name in Newsreader, presenter name italic-clay, date in mono uppercase. Optional photo collage band at the bottom (AAA puts a 6-photo band on its cover — make this optional, brand-appropriate would be subtle bounce-court motion or static image).

T2 — Executive summary 3-col — Challenge / Solution / Impact columns. AAA Slide 2 structure. Each column has bold header, body bullets, and the third column gets darker accent treatment for "impact." Map to court / plum / amber accents.

T3 — Action-title + 4-stat stack — for problem framing (matches my Slide 2 above)

T4 — Action-title + 4-row failure matrix — categorical comparison (matches my Slide 3 above)

T5 — Action-title + sidebar-and-chart 2-col — narrow text sidebar on left with key points + dashed vertical divider + wide chart/visualization on right. AAA Slide 3 structure. Brand the chart with court/plum/amber/clay categorical palette.

T6 — Step progression (1 / 2 / 3 / 4) — circled badges, headline per step, body beneath. AAA "Step 1/2/3/4" structure. For methodology, pipelines, frameworks.

T7 — Cluster-action table — AAA Slide 6 structure. Left column = entity/cluster pill (court / plum / amber / clay), middle column = state (Over / Under), right column = chevron-arrow action callout. For decision frameworks.

T8 — Big-number contrast — two massive Newsreader numerals side by side with italic-clay accent on one, mono captions beneath, optional 3-tile micro-grid below for secondary stats. (Matches my Slide 5.)

T9 — 3-tier horizontal progression — italic Roman numerals (i / ii / iii), tier headline, 2-line body. For roadmap / expansion / before-now-future arcs. (Matches my Slide 6.)

T10 — Section divider — full dark mode slide, big centered Newsreader title (italic-clay on one word), mono uppercase subtitle beneath. (Matches Appendix divider.)

================================================================
DELIVERY PROTOCOL
================================================================

1. First, confirm you've read the bundle AND the AAA MWG PDF. Summarize what you see in each in 3 lines so I know you got the right reference.
2. Generate Slide 1 first as a single HTML file. Wait for my approval.
3. Then build Slide 2. Wait for approval.
4. Continue slide-by-slide until the case comp deck is complete.
5. Then build the 10 templates as `templates.html`. Each template fully rendered with [PLACEHOLDER] text.
6. At the end, combine everything into a single `deck.html` with keyboard navigation (arrows, G for grid, T for theme).

DO NOT:
- Don't generate the whole deck in one shot
- Don't invent specific stat values where I've written [PLACEHOLDER] or [TBD] — keep them clearly marked
- Don't use em dashes anywhere
- Don't use red, pure black, neon green, or any color outside tokens.css
- Don't add transitions/animations beyond the locked brand motion
- Don't draw new logo variants — use the locked PNGs from the bundle

Start with the bundle summary, then propose Slide 1.

</pasted_text>

<!-- The user explicitly selected the following skills for this project, as attachments to their message. These are not optional context — they define how you work. Use them. -->
<attached-skill name="Make a deck">
Create a presentation deck as a single self-contained HTML page.

Assume this role: you are a presentation designer. You build slide decks for a speaker to present — HTML is your output medium, but your design thinking is the same as a consultant, analyst, or executive preparing material for a boardroom: clarity, narrative flow, and back-of-the-room readability. You are not building a website.

Every slide is an exercise in both layout design and copywriting. Write an outline before you start; a good outline is an exercise in storytelling and narrative structure.

If a user does not tell you how long they want a presentation to be, in minutes, ask them.

Build at 1920×1080 (16:9). Do NOT hand-roll the stage/scaling/nav scaffolding — start by calling `copy_starter_component` with `kind: "deck_stage.js"`, then write your deck HTML as `<deck-stage width="1920" height="1080">` with one `<section data-label="…">` child per slide. The component handles letterboxed scaling, keyboard + tap navigation, the slide-count overlay, the speaker-notes postMessage contract, `data-screen-label` / `data-om-validate` tagging, and print-to-PDF (one page per slide). Load it with a plain `<script src="deck-stage.js"></script>` — it is vanilla JS, not JSX. (For PPTX export later: pass `resetTransformSelector: "deck-stage"` to gen_pptx — the component honours a `noscale` attribute that disables its shadow-DOM scaling so the capture sees authored-size geometry.)

Write the slide content as static HTML, not React or script-generated DOM. When a slide's body is plain markup inside `<deck-stage>`, the user can click any heading or paragraph in edit mode and retype it directly — the editor splices their change into the source file immediately. When the same content is rendered by a `<script type="text/babel">` block, a React component, or a loop over a JS array, that direct path is lost: every tweak has to round-trip through a chat message to you, which is slower for the user and makes it harder for them to polish the deck themselves. So for anything a static page can express — text, layout, background, image — write the literal element in the HTML and style it with CSS. Reach for babel/React or an extra `<script>` only when the slide genuinely needs behaviour static markup can't deliver (an interactive chart, a live demo, real state). The same rendered result in static HTML is strongly preferred over a dynamic one, because the static version is directly editable. The Tweaks panel (`tweaks-panel.jsx`) is the standing exception: it's a control surface that sits alongside the slides, not slide content, so still include it — its `<script type="text/babel">` tag doesn't make the slides themselves any less directly editable, because the editor routes each static slide element to the splice path independently of the panel's script.

Two details keep static slides directly editable: each piece of text lives in its own leaf element (put "Revenue" in its own `<span>` inside the `<h2>` rather than writing `<h2>Revenue <span class="sub">2025</span></h2>` with text and a child mixed in the same parent), and repeated structure is written out, not generated — three bullet `<li>`s in the markup, not one `<li>` rendered three times from an array. The repetition is the point; it's what lets the user edit bullet two without touching bullet one.

Use large type sizes (at least 48px for titles). When the user asks for a specific font size, assume they mean **points** (the PowerPoint/Keynote unit), not pixels — convert with `px = pt × 1.333`. So "make titles 36pt" → set ~48px in your CSS.

Image usage: make sure to view images and decide how they can best be displayed. Full-bleed images can be aspect-filled; screenshots and diagrams must be aspect-fit and rarely overlaid upon; transparent or aspect-fit images should be set against a contrasting background color. When putting text on top of images, match how the brand typically does this: use cards, protection gradients or blurs depending on what you see elsewhere.

Use smooth transitions between slides. Style with a clean, professional look — generous whitespace, strong typography, and a cohesive color palette. Pull in graphical elements liberally -- prefer images given to you by the user, or any relevant brand assets or icons you can find.

Do not use emoji or self-drawn assets unless asked. Use icons from your design system / brand, or images provided by the user.

Aim for visual variety, with a mix of full-image slides, different background colors, large numbers or figures, quotes, tables and some textual slides. Aim for visual balance on slides; we don't want a ton of top-aligned text, or mostly-empty slides, but some is fine.

Critical: AVOID PUTTING TOO MUCH TEXT ON SLIDES! This is a common failure mode. In your plan or thinking, discuss which parts of the story would be best as tables, diagrams, quotes, or images.

Parallelism is important: section header slides should look the same; repeated textual elements should be in the same position; etc.

The deck-stage component absolutely positions every slotted child for you — do NOT set position/inset/width/height on the slide `<section>` elements yourself.

# Slide writing guidelines

In general, the titles of a slide deck alone should tell you the overall story/content of the deck (similar to ToC in a book)
There are generally a few types of title structures that are used in slide decks:
- Short textbook-title-style, all capitalized (e.g., Market Research, Engagement Overview, Team Structure)
- Action titles, which are more like short phrases (e.g., “Asia is our largest market….”, “...but Eastern Europe has the highest potential for growth”)
Pick the appropriate title structure and stick with it.

Avoid these common Claude-isms that gives away that the deck was AI-generated:
- Claude likes to write titles and takeaways that “deliver the verdict,” overdramatize/simplify, create tension for no real reason (the classic “It’s not X. It’s Y.”), use strong imperatives, engage in heavy-handed reframing, or be dramatically suspenseful or faux-insightful
- Titles like “The magic moment”
- Basically, Claude likes to write titles that sound like the speaker’s punchline, rather than being a TITLE that introduces the slide -- AVOID!

# Planning steps

In addition to your normal planning, make sure to do these things:

1. Ask questions if relevant
2. Write out the full title sequence. Choose ONE grammatical style (for example, short topic noun-phrases or brief declarative sentences) that is appropriate for the content, and write every title in that style. Read them back to yourself and determine if a person reading ONLY the titles could follow the flow of the presentation. The titles should be like chapters in a book - they orient the reader on what to expect with straightforward language. Review the titles and revise as needed.
3. Define your type scale and spacing as CSS custom properties in a `<style>` block in `<head>` before writing any slide — these commit you to projection-appropriate sizing and stop you defaulting to web density. At 1920×1080 a reasonable starting scale is `:root { --type-title: 64px; --type-subtitle: 44px; --type-body: 34px; --type-small: 28px; --pad-top: 100px; --pad-bottom: 80px; --pad-x: 100px; --gap-title: 52px; --gap-item: 28px; }`. At 1280×720, scale by ~0.67. Reference these everywhere — every font-size uses a `--type-*` variable, every padding/gap uses a `--pad-*` or `--gap-*` variable, via `var(…)` in inline styles or class rules. Keeping these as CSS (not JS constants) means the user can change one number — in the style block directly, or via a Tweaks slider bound to the same variable — to re-size the whole deck, and the slide markup stays static HTML with no script needed to compute sizes. The explicit `--pad-bottom` reserves breathing room at the base of every slide; that space is structural, not empty. Web defaults (14-16px body, 48-72px padding) are too small for slides; if the values don't feel generous, they aren't. Your validator will throw an error if you use a size smaller than 24px.
4. Build the slides, remembering that each slide is an exercise in both design and copywriting. Give each slide the attention it deserves in terms of the layout, the text content, and the tone. Follow the principles below and ensure that each slide can stand alone; a person looking at that slide should be able to understand its high-level meaning without other context.

# Verification tips for slide decks
During review, check your screenshots against slide composition rules — not web-layout instincts. `align-items: flex-start` with open space in the bottom third is correct slide composition, not a defect. If you see content sitting in the top 2/3 with breathing room below and feel the urge to change `flex-start` to `center` — that urge is the web-design reflex. Resist it. The open space is intentional. Also verify: font sizes match your `--type-*` scale (not web density), slide frame padding matches your `--pad-*` values (not web-tight), title parallelism across slides, no accent-border cards or takeaway boxes
</attached-skill>

<attached-skill name="CourtCheck Design System (design system)">
[Design System] This project uses the **CourtCheck Design System** design system. This is a binding choice for visual style — every visual must follow it. Don't invent colors, type, spacing, or components not grounded here.

Scope: the design system is a visual style reference only. Its guide may describe example products, brands, or people that are unrelated to the user and unrelated to the subject of this conversation. Never treat anything in the design system as a fact about the user, their work, or the topic they asked about.

    Explore it to find what you need:
    - Always copy out the fonts and colors you need
    - For prototypes and designs, always copy out any relevant components
    - If the design system contains existing mocks of products, and you were asked to design or prototype something similar, copy and fork those mocks to start your design. This helps you make high-quality designs.
    
    Explore it quickly to find relevant UI kits (e.g. mocks of existing products you can copy and fork)

Full system at `/projects/019e1e53-2366-7aac-b3a6-fa383915a730/`. Its guide is reproduced below so you don't need to fetch it:

<design-system-guide>
# CourtCheck Design System

> **Tennis analytics for college coaches.** This is the locked brand system,
> packaged for design work. Cream paper canvas, Newsreader serif numerics,
> editorial italic-clay accents. Built so the next surface you ship feels
> like it came out of the same studio as the last ten.

The brand has been through ~20 iteration rounds and is **locked**. Build
product surfaces against it. Do not propose palette / typography / logo
changes unless you have written-up reasoning first.

---

## Index

| File / folder | What |
|---|---|
| [`colors_and_type.css`](./colors_and_type.css) | Canonical CSS variables + base type primitives. Import this on every page. |
| [`fonts/`](./fonts/) | Newsreader, Inter Tight, JetBrains Mono — variable TTFs, both styles |
| [`assets/logo/`](./assets/logo/) | Locked logo masters (light + dark PNG), splash + checkmark hover WebMs, UCD partnership mark. **The light/dark masters also serve as the app icon.** |
| [`assets/motion/`](./assets/motion/) | Locked hero bounce loop (WebM + still poster) |
| [`preview/`](./preview/) | Design-system tab cards (Type, Colors, Spacing, Components, Brand) |
| [`SKILL.md`](./SKILL.md) | Agent-skill manifest — drop the folder into Claude Code or another agent and this skill becomes invocable |

**Sources** (read-only, mounted via the user's File System Access API at
project start):

- `brand-drop/` — the canonical bundle, 10 reference mocks plus all docs
  (`01-brand-guidelines.md`, `02-philosophy.md`, `03-voice.md`,
  `04-motion-spec.md`, `05-peer-references.md`, `06-components.md`,
  `tokens.css`, and `mocks/visuals.html` as the visualization reference)

---

## What CourtCheck is

CourtCheck is the film room, automated. A college tennis coach uploads a
match recording; computer vision turns it into shot maps, coverage
heatmaps, spacing lines and pattern callouts. The product's promise is
"we watched the film for you" — and the brand's job is to make that
promise feel **observed, considered, signed off on**, never auto-generated.

The DNA is **Robinhood meets SwingVision**: SwingVision's analytical
depth, packaged in Robinhood's quiet, opinionated, big-number, gentle-palette
interface. Coaches finishing a match at 9pm shouldn't navigate a Bloomberg
terminal to find what changed.

### Audience

- **Primary:** college tennis head coaches (D1/D2/D3) — mid-30s to early
  60s, often former players, time-poor, opinion-rich, allergic to SaaS
  jargon and 14-KPI dashboards
- **Secondary:** assistant coaches, players, performance analysts

### Surfaces in scope

1. **Marketing site** — public landing (nav, hero with bounce-arc video,
   problem, big-number contrast, how-it-works, pinned scroll-through,
   closing CTA, footer)
2. **Product app** — auth, dashboard (populated + empty), upload flow,
   match detail, player profile, players list, matches list

---

## CONTENT FUNDAMENTALS

### Voice in one sentence
> Speak like a thoughtful assistant coach who's already watched the film.

The product has a **two-voice system**, made visible through type:

- **Coach voice** (Newsreader serif, italic accent in clay) — headlines,
  body copy, callouts, anything a human would say
- **System voice** (JetBrains Mono UPPERCASE, 0.14–0.18em tracking) —
  anything the *product itself* is narrating: timestamps, axis labels,
  "TRACKING", "LIVE", version strings, eyebrows, zone names

When a coach reads a headline they hear a coach. When they read a status
indicator they hear the system. The split is the brand.

### Tone rules

- **Specific, not generic.** ✅ "Loses the cross 64% on second-serve
  return." ❌ "AI-powered insights tailored to your performance."
- **Calm, not urgent.** No exclamation marks. No emoji. No "🎉".
- **Coaching, not selling.** ✅ "Here's what to work on Wednesday." ❌
  "Crush your competition."
- **Plain English, not jargon.** ✅ "First serve in: 68% — up 4 points
  from your season." ❌ "Serve efficiency delta: +4.2 ppt vs trailing 30
  baseline."
- **Numbers are wrapped.** Every numeric value in body copy renders in
  Newsreader serif with tabular numerals — `<span class="num">68%</span>`.
- **I vs you:** the product never says "I". It addresses the coach as
  "you" implicitly, or speaks in third-person observational mode ("Lin
  wins 71% of openers on her slice serve").
- **Casing.** Sentence case for everything except mono labels (those are
  ALL CAPS, wide-tracked). Title case is reserved for the wordmark.

### Headlines: the italic-clay rule

Every section / page title has exactly **one italic word**, always in
`--clay` (light) / clay-soft (dark). The italic word is the **verb** or
the **action** of the sentence — it carries the meaning.

```html
<h1 class="display-1">See every shot.<br>
  <span class="ital">Know every move.</span></h1>

<h2 class="display-2">Tennis has always been played on
  <span class="ital">instinct.</span></h2>

<h2 class="display-2">Stop guessing.
  <span class="ital">Start winning.</span></h2>
```

**Two locked exceptions** where italic renders in **lime** instead of
clay:

1. The "Check" half of the **CourtCheck wordmark** — italic lime, always.
2. The closing CTA "**Start winning.**" — italic lime, always. This is
   the page's one lime moment, the apex of the page-long buildup.

Anywhere else: italic = clay. Period.

### Voice anti-patterns (will not ship)

| Don't say | Say |
|---|---|
| "Powered by AI" | (just say what it does) |
| "Revolutionize your coaching" | "Watch the film for you" |
| "Leverage" / "Robust" / "Elevate" / "Transform" | (cut the word entirely) |
| "Crush your competition" | "Here's what to work on Wednesday" |
| "🎉 Welcome aboard!" | "Signed in" |
| "An error occurred" | "Couldn't reach the server" |
| "Click here to learn more" | "View match" |
| Em dashes in user-facing copy | Periods, or "—" inside product copy only (signals a coach's pause) |

### Toasts (Sonner)

Four levels, **never red**. Lead with what happened, then what to do.

- **Success** — court accent — "Match analyzed. View Stanford on your dashboard."
- **Info** — ink-mute accent — "Reset link sent."
- **Warn** — clay — "Video is 480p. Analysis quality may suffer."
- **Error** — clay (persistent, never auto-dismiss) — "Couldn't reach the server. Check your connection and try again."

### Empty states

Empty states are tiny ads for the product. Show what's coming, not what's
missing.

> **Patterns will appear here after your first match is processed.**
> Upload a video → we'll surface 5–10 patterns within ~20 minutes.
> *[Upload first match →]*

---

## VISUAL FOUNDATIONS

### Surface color — the cream paper canvas

Light mode is a clay-court afternoon. Dark mode is the same court after
dusk: still warm, still green, just quieter.

| Role | Light | Dark |
|---|---|---|
| Page background | `#F4F0E6` cream | `#07080F` deep void |
| Card / elevated | `#FBF8F1` paper | `#1F2940` navy paper |
| Top elevation (popovers) | `#FFFFFF` surface | `#2B374F` navy surface |
| Inset (input bg, bar track) | `#ECE6D5` shade | `#11161F` shade |
| Border | `#DDD7C5` line | `#2F3B52` line |
| Subtle divider | `#EAE4D2` line-soft | `#1E2735` line-soft |

The body has a **subtle two-point radial wash**: a faint clay glow in the
top-right and a faint court glow in the bottom-left. Always present, never
animated, attached to `background-attachment: fixed` so it doesn't move
with scroll.

### Ink

| Role | Light | Dark |
|---|---|---|
| Primary type | `#1A1815` | `#F4F0E6` |
| Body | `#43403A` | `#A8B0C0` |
| Caption / system | `#837F77` | `#6B7388` |

### Brand color (the three notes)

- **Court green** — `#2E5341` / `#6FA88B` — primary, "in favor"
- **Clay terracotta** — `#B05B36` / `#E07A52` — italic accent, "needs
  attention", errors. Never red. Ever.
- **Lime** — `#BFC846` / `#DDE970` — "the ball", **once per screen** as
  the single moment of impact. Two locked exceptions where it co-occurs
  with italic: the wordmark "Check", and the closing CTA "Start winning."

### Categorical palette (locked product-wide, 6 slots max)

| Slot | Token | Default role |
|---|---|---|
| 1 | `--court` | Forehand |
| 2 | `--plum` | Backhand |
| 3 | `--amber` | Serve |
| 4 | `--clay` | Volley |
| 5 | `--slate` | Neutral / Other |
| 6 | `--lime` | Ball (sparing) |

Forehand is always Court green, every chart, every player, every screen.
Coaches build muscle memory around it. Cap at 6 categories per chart.

### Type system

Three faces. Locked. **Do not introduce a fourth.**

| Face | Role | Notes |
|---|---|---|
| **Newsreader** (variable serif) | Display + numerics | Optical-size axis used deliberately — `opsz 144` for hero h1, `opsz 96` for section h2, `opsz 72` for card titles and numbers. Always 500 weight for display, 400 italic for the clay accent. All numerics use `font-feature-settings: 'tnum'`. |
| **Inter Tight** (sans) | Body | 400 / 500 / 600 / 700. Letter-spacing -0.005em (body), -0.014em (labels). Line-height 1.5 body, 1.3 short. |
| **JetBrains Mono** (mono) | System voice | 0.66–0.72rem, 0.14–0.18em tracking, **always uppercase**. Court color for eyebrows, ink-mute for labels. |

A serif number reads as *observed, signed off on*. A sans number reads as
*auto-generated*. The brand promises "we watched the film for you" — the
typography reinforces the promise.

### Spacing

Built on an 8-pt scale (`--space-1`..`--space-20`), with two anchor sizes:

- **Container** — max-width 1320px, 56px horizontal padding desktop / 28px
  mobile. `narrow` variant at 720px (upload page).
- **Card padding** — 28px default, 24px compact, 36px hero.
- **Section** — 56px top + 32px bottom.
- **Stack gap** — 20px between cards in a grid, 4px between sidebar nav
  items, 12px between buttons.

### Radius (intentional hierarchy)

| Token | Value | Use |
|---|---|---|
| `--radius-sm` | 8px | Court tile inner |
| `--radius-md` | 12px | Inputs, dropdown items |
| `--radius-lg` | 14px | Cards |
| `--radius-xl` | 18px | Hero cards (rare) |
| `--radius-pill` | 100px | **All buttons**, tags, pills |

Pill = action. Card = content. Input = entry. Never blur the distinction.

### Shadow

```
--shadow-card:  0 1px 0 rgba(26,24,21,0.04), 0 18px 48px -24px rgba(26,24,21,0.16)
--shadow-pop:   0 1px 0 rgba(26,24,21,0.06), 0 24px 60px -28px rgba(26,24,21,0.22)
```

Two registers. `--shadow-card` is the resting state. `--shadow-pop` is
the popover / hover-lift state. No drop shadows on text. No glow effects.
No soft chrome. Dark mode shadows are heavier (0.4–0.7 alpha) because the
canvas is near-black and elevation has to fight harder to read.

### Backgrounds

- **No imagery as page background.** The cream paper canvas + radial
  wash is the brand's hero environment. Never replace it with a photo.
- **Hero video is contained**, not full-bleed. `motion/Bounce_Animated.webm`
  sits in a paper card at native 16:9, 18px radius, brand card shadow.
  Never `object-cover` crop.
- **No textures, no patterns, no grain, no newspaper overlays.** All of
  these were tested and rejected.
- **No gradients as decoration.** The only gradients in the system are
  the body radial wash (a token, not a flourish) and the scoreboard
  card's dark-mode background.

### Animation

Animation is **feedback, not decoration**. Ambient motion is forbidden.

- **Easing:** `--ease-out` is the default for reveals, fades, hovers.
  `--ease-spring` for hero moments only (big number reveals, primary CTA
  hover). `--ease-in-out` for toggle / drawer state changes.
- **Duration:** `--dur-quick 160ms` (hovers), `--dur-base 240ms`
  (transitions), `--dur-reveal 480ms` (reveals on load/scroll),
  `--dur-hero 800ms` (one big entrance per page).
- **What animates:** initial hero reveal (top-to-bottom, 80ms stagger);
  card hover lift (translate -2px, shadow deepens); number count-up
  (Newsreader 0 → value, 720ms linear); court overlay data (shot dots
  fade in with 30ms stagger); bar growth (0 → target width, 60ms stagger
  top-to-bottom).
- **Never:** body text after initial reveal, tooltip fades, ambient
  pulses, drifting glows, idle breathing, scroll-jacking, parallax.
- **Loading:** the bounce-arc loop. The brand's only loading metaphor.
- `prefers-reduced-motion: reduce` honored everywhere — animations
  downgrade to instant; the hero video pauses on `Bounce_Still.png`.

### Hover + press states

- **Cards** — translate -2px, shadow strengthens from `--shadow-card` to
  `--shadow-hover`, border-color shifts to `--ink-mute`. 240ms ease-out.
- **Primary button** — translate -1px (or -3px on hero size), shadow
  glow deepens, background flips to `--court-deep`. ease-spring.
- **Ghost button** — border-color flips from `--line` to `--ink`,
  translate -2px.
- **Nav link / sidebar link** — color flips from `--ink-soft` to `--ink`.
  Active sidebar link adds a 3px court-color left rail via `::before`
  (so meaning isn't color-only — a11y).
- **Match / recording row** — background flips to `--shade`, the trailing
  chevron icon translates 4px right.
- **Legend chip** — when one chip is active, sibling chips dim to
  opacity 0.45.
- **Press** — buttons translate-Y goes from -3px back to -1px (a small
  re-settle), no scale-down. No "shrink-to-tap" affordances.

### Borders & dividers

- **1px solid `--line`** on all cards, inputs, buttons (ghost), and
  outer chrome.
- **1px solid `--line-soft`** on internal section dividers (card
  separators, list rows, sidebar internal lines).
- **3px court rail** as a left-side accent on the active sidebar link
  (via `::before`).
- **No double borders.** No border + heavy shadow. Pick one register.

### Transparency & blur

- **Sticky nav** uses `backdrop-filter: blur(10px)` with
  `color-mix(in srgb, var(--cream) 86%, transparent)`. The only place
  blur appears. No glassmorphism elsewhere.
- **Dim opacity** is used as a state — sibling legend chips at 0.45,
  inactive shot dots at 0.12. Never as a visual effect.

### Imagery vibe

Editorial. Warm. Tennis culture *in cultural memory*, not in stock-photo
form. Clay courts in afternoon sun. The quiet locker room at night.
Hand-drawn ink illustrations (the bounce loop, the logo mark). No stock
action photography, no soft-glow chrome, no neon, no gradient soup.

### Cards (canonical contract)

```
background: var(--paper)
border:     1px solid var(--line)
radius:     var(--radius-lg)        /* 14px */
padding:    28px                    /* 24px compact, 36px hero */
transition: transform 240ms var(--ease-out),
            box-shadow 240ms var(--ease-out),
            border-color 240ms var(--ease-out)

hover:
  transform: translateY(-2px)
  box-shadow: var(--shadow-hover)
```

### Court tile (the universal frame)

Every dashboard, every player profile, every recording detail leads
with the court. Vertical singles court (78ft × 27ft as SVG units),
warm-clay surface (`--court-surface`), white lines (0.3 stroke
boundary / 0.5 net / 0.25 service lines). It's our Robinhood
portfolio-chart — the one persistent visual that anchors the brand.

### Layout rules

- **Fixed sidebar at 200px** (auto-collapsed to 72px below 1100px
  viewport). Body padding-left matches. Sticky nav 14–22px padding,
  blurred backdrop.
- **One thing per screen.** Every screen has exactly one primary action.
- **Visual-first.** Charts, court overlays, big numbers — tables exist
  for export only.
- **One hero card + three supporting** on the dashboard. Not a 14-KPI
  grid. Not a customizable view. The product opinionates *for* the
  coach.

---

## ICONOGRAPHY

The brand uses **Lucide** icons exclusively (already in the production
codebase per `06-components.md`). Settings are locked:

- **Stroke width:** 1.75
- **Caps:** round
- **Joins:** round
- **Default size:** 18px (inline / sidebar / button), 16px (theme toggle,
  small chips), 56px (dropzone glyph), 24px (inline status indicators)
- **Color:** inherits `currentColor` — defaults to `--ink-soft` in
  navigation, `--ink-mute` in mono captions, `--court` in eyebrows.

Lucide is loaded from CDN in the UI kits — `https://unpkg.com/lucide-static/icons/*.svg`
for inline SVG, or `https://unpkg.com/lucide@latest` for the runtime
script if a kit needs many.

**No icon font** is shipped. **No bespoke SVG icon set** is shipped. Don't
hand-draw replacements for missing icons — find the closest Lucide
equivalent and use it. If genuinely missing, flag as a request.

**Emoji are never used** in user-facing copy, headers, toasts, or empty
states. They violate the calm-coach voice.

**Unicode arrows** are fine as occasional decoration — `→` for "next
step", `↑` / `↓` for trend indicators (always followed by a serif
number). Set them in the body font, not Newsreader.

### Logo / brand-mark inventory

| File | Use |
|---|---|
| `assets/logo/CourtCheckLogoLight.png` | Light master, full lockup, cream canvas |
| `assets/logo/CourtCheckLogoDark.png` | Dark master, full lockup, deep-void canvas |
| `assets/logo/CourtCheckAnimation.webm` | Splash animation — 5s, plays once, ends on static lockup. Transparent alpha. |
| `assets/logo/CourtCheckCheckmark.webm` | Hover loop for the collapsed-sidebar mark. Transparent alpha. |
| `assets/logo/ucd-tennis.png` | UC Davis Tennis partnership mark — used once on the landing page as the only partner credit |
| `assets/motion/Bounce_Animated.webm` | Hero animation — 6s seamless loop |
| `assets/motion/Bounce_Still.png` | Hero poster + reduced-motion fallback |

**Do not redraw, recolor, scale-warp, or "improve" the masters.** No
monogram variant, no badge variant, no watermark variant exist — flag
as a request if a use case needs one.

### Iconography in the wild

- **Sidebar nav** uses 18px Lucide: `layout-dashboard`, `users`,
  `upload`, `clapperboard` (Dashboard / Players / Upload video /
  Recordings). Always preceded the label, never alone unless the
  sidebar is collapsed (where it gets a `title=` attr for the
  destination name).
- **Trend arrows** in stat compares use Lucide `arrow-up-right` /
  `arrow-down-right`, 14–16px, colored by trend (`--court` positive,
  `--clay` negative).
- **Card / row affordance** uses Lucide `chevron-right` 16px in
  `--ink-mute`, translates 4px right on hover.
- **Theme toggle** uses Lucide `moon` (light mode) / `sun` (dark mode),
  16px.

---

## Caveats / open questions

- **No bespoke icon assets** were shipped in `brand-drop/`. Use Lucide
  via CDN at stroke-1.75, round caps + joins. If production has a
  custom icon set, point me at it.
- **No monogram variant** exists. If a use case needs one (square hero
  card, social embed, etc.) — flag as a request rather than improvising.
- **Lockup minimum size** is ~80px wide. Below that, the wordmark
  becomes illegible. Crop to mark-only or use the splash WebM.

---

*Living document. Build forward. The brand is locked; surfaces are not.*

</design-system-guide>

CSS tokens: the guide may describe tokens in prose, but the exact `--*` names are defined in the design system's stylesheet(s) (`/projects/019e1e53-2366-7aac-b3a6-fa383915a730/colors_and_type.css`, `/projects/019e1e53-2366-7aac-b3a6-fa383915a730/preview/_card.css`). The 71 custom properties it defines are:

--amber, --bg-grad-1, --bg-grad-2, --clay, --clay-soft, --court, --court-deep, --court-light, --court-surface, --cream, --dur-base, --dur-hero, --dur-loop, --dur-quick, --dur-reveal, --ease-in-out, --ease-out, --ease-spring, --font-display, --font-mono, --font-sans, --ink, --ink-mute, --ink-soft, --lime, --lime-soft, --line, --line-soft, --paper, --plum, --radius-lg, --radius-md, --radius-pill, --radius-sm, --radius-xl, --shade, --shadow-card, --shadow-hover, --shadow-pop, --slate, --space-1, --space-10, --space-12, --space-14, --space-16, --space-2, --space-20, --space-3, --space-4, --space-5, --space-6, --space-7, --space-8, --stroke-backhand, --stroke-forehand, --stroke-serve, --stroke-volley, --surface, --tag-clay-bg, --tag-court-bg, --tag-lime-bg, --tag-lime-ink, --type-body, --type-body-lg, --type-body-sm, --type-display-1, --type-display-2, --type-display-3, --type-mono, --type-num-hero, --type-num-mid

Only use `var(--*)` names from that list. Never guess a token name — an unresolved `var()` silently falls back to the browser default.

For assets and UI kits beyond the guide: `read_file("/projects/019e1e53-2366-7aac-b3a6-fa383915a730/<path>")` and `copy_files` to bring them into the current project.
</attached-skill>

## Assistant

_[tool: list_files]_

_[tool: invoke_skill]_

_[tool: run_script]_

_[tool: read_file]_

_[tool: list_files]_

_[tool: run_script]_

_[tool: copy_files]_

_[tool: run_script]_

_[tool: run_script]_

_[tool: run_script]_

_[tool: run_script]_

_[tool: run_script]_

_[tool: list_files]_

_[tool: set_project_title]_

_[tool: copy_files]_

## Assistant

_[tool: copy_starter_component]_

_[tool: copy_files]_

Bundle and PDF read. Setting up the project now and writing Slide 1.

**CourtCheck brand bundle (locked):**
- Cream paper canvas `#F4F0E6` / deep void `#07080F` (with a `#0A0E14` near-black for divider slides per your spec); two-voice system: **Newsreader** display + numerics (tabular), **Inter Tight** body, **JetBrains Mono** uppercase for system/meta; **italic = clay** rule (one italic word per headline, clay in light/dark) with two lime exceptions (wordmark "Check" and "Start winning").
- Categorical palette is **locked**: court → plum → amber → clay → slate, with **lime** reserved as "the ball" (one moment per screen). Spacing is 8pt, radii are pill/sm/md/lg/xl with intentional meaning, animation is feedback-not-decoration (`--ease-out`/`--ease-spring`, durations 160/240/480/800ms).
- Assets I'll use from the bundle: `CourtCheckLogoLight.png` / `CourtCheckLogoDark.png` for the chrome lockup, `Bounce_Still.png` for the cold-open ghost background, fonts (Newsreader + Inter Tight + JetBrains Mono variable TTFs).

**AAA MWG Branch Staffing Optimization deck (14 pages, structural reference):**
- Action title on every content slide is a single declarative sentence with an underlined keyword ("all 101 branches", "live", "immediate use"). Thin gradient accent bar across top, slide number bottom-right. Section dividers are full-bleed navy with one big centered title ("Sales Agent Analysis").
- Information architecture: a 3-column Challenge / Solution / Impact frame; Step 1 → 2 → 3 → 4 progression with circled badges (regression → cluster groupings → cluster strategy → over/under x cluster); cluster entities color-coded across all downstream slides (4 categorical buckets — Benchmark / Growth / Regional / Lean).
- Tables use chevron-shaped action callouts in the right column, dashed vertical dividers split sidebar text from charts, boxed sidebar callouts surface takeaways, and KPIs get oversize type. Pain framing is concrete with numbers ("19 Frontline Over"), and the relief slide uses verbs of action ("live", "ready for immediate use").

Now bringing the deck shell together — copying fonts, locked CSS tokens, the bounce-still poster, and both logo masters into the project, then writing Slide 1 only.

_[tool: grep]_

_[tool: grep]_

_[tool: grep]_

_[tool: write_file]_

_[tool: write_file]_

Writing the deck shell + Slide 1.

_[tool: done]_

_[tool: str_replace_edit]_

Cleaning up the validator findings: making the quote and attribution properly editable static HTML, and bumping the attribution to 24px (slide-scale floor; the spec's "16px" reads as point-size at this projection scale).

_[tool: str_replace_edit]_

_[tool: done]_

_[tool: save_screenshot]_

Let me see what's actually rendering and fix the structure properly.

_[tool: show_html]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: done]_

Slide 1 is up — the cold open, no chrome, near-black surface with the bounce-still ghost behind the centered Newsreader pull quote, italic-clay on the demo word ("QUOTE" stands in for whatever evocative word lands once Sara's line is locked). Attribution sits in JetBrains Mono with clay-dot separators.

Keyboard chrome: **←/→** navigate · **T** toggles theme · **G** toggles the thumbnail grid · **P** prints to PDF.

A few notes before I move on:
- **Em-dash rule** — I'm respecting it in all visible slide content. I'm still using them inside dev comments / this chat (signaling pauses), but never on a slide.
- **Slide number on Slide 1** — suppressed, per your spec ("no chrome, no slide number, no logo"). I'll start the slide numbers and bottom-left logo lockup on Slide 2.
- **#0A0E14** — that's slightly off the brand's `--cream` dark token (`#07080F`), but you spec'd it explicitly for the AAA-style navy divider surfaces, so I added it as a deck-local `--night` token rather than overwriting the brand.
- **Speaker notes** — I haven't wired the `#speaker-notes` JSON yet (you didn't ask for it). The "Hold 4s silent" direction lives as an HTML comment on the slide for now. Say the word if you want a full speaker-notes script.

**Approve Slide 1, and I'll build Slide 2 (The Structural Gap — 4 stacked stat rows with [80 hrs] / [12%] / [3 days] / [TBD%] placeholders + kicker strip).**

## User

finish the rest

## Assistant

_[tool: update_todos]_

_[tool: str_replace_edit]_

Going to build all remaining slides + the 10 templates into the same deck so navigation/print/theme just work across everything. Laying down the supporting CSS first, then the slides.

_[tool: str_replace_edit]_

Now writing all the slides into deck.html in one pass.

_[tool: done]_

_[tool: save_screenshot]_

_[tool: show_html]_

_[tool: sleep]_

_[tool: save_screenshot]_

_[tool: get_webview_logs]_

_[tool: save_screenshot]_

_[tool: show_html]_

_[tool: save_screenshot]_

_[tool: sleep]_

_[tool: save_screenshot]_

_[tool: update_todos]_

_[tool: done]_

_[tool: fork_verifier_agent]_

Full deck is built — **22 slides** in `deck.html`, single file, keyboard nav (←/→), **T** theme toggle, **G** thumbnail grid, **P** print to PDF.

**Main deck (6):** cold open · structural gap (4-stat stack) · why existing tools fail (failure matrix w/ clay·court·plum·amber rails) · demo (empty frame, 5-beat storyboard in HTML comments) · impact (80hrs → 12min with lime on "12" as the single ball-moment, micro-grid, design-partner inset quote, anchor anecdote strip) · expansion + ask (3-tier i/ii/iii w/ italic-clay Romans).

**Appendix (4 + divider):** methodology table (precision/recall/F1 with court-cell highlights + confusion-matrix card) · 5-stage pipeline · unit economics bento + pricing thesis sidebar · roadmap timeline + risk table.

**Template library (10 + divider):** T1 cover, T2 exec summary 3-col, T3 4-stat stack, T4 failure matrix, T5 sidebar+chart with dashed divider and bar chart in court/plum/amber/clay, T6 step progression, T7 cluster-action table with chevron right column, T8 big-number contrast, T9 3-tier progression, T10 section divider.

Verifier is screenshotting each slide now and will ping if anything's clipping, misaligned, or off-pattern. Two things you'll want to lock alongside Sara's quotes on May 12: the **[TBD%]** stat on slide 2 and the **anchor-anecdote** sentence + **one-line ask** on slides 5 and 6. The other `[N]` / `[$N]` placeholders in the appendix are pricing/eval-harness numbers also marked to lock.

## User

Verifier agent check completed

## Assistant

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: done]_

_[tool: fork_verifier_agent]_

Tightened slide 2 + T3 — pulled stat-row padding from 36/30 down to 24/22, dropped the giant value from 132px to 116px, and trimmed the kicker. That's ~160px of headroom recovered against a 41px overflow, so they should clear cleanly. Verifier is confirming the scrollHeights now.

## User

Verifier agent check completed

## Assistant

All 22 slides now fit cleanly at 1920×1080. Deck is complete — main 6, appendix divider + A1–A4, template-library divider + T1–T10, all in `deck.html`. ←/→ to navigate, **T** for theme, **G** for the thumbnail grid, **P** to print.

## User

<pasted_text name="Pasted text (60 lines)">
The Slide 2 you produced is brand-correct but structurally too editorial. It reads as a magazine layout, not a consulting deck. I'm sending it back for revision against the AAA MWG structural conventions, which take precedence over editorial flourish on content slides.

PROBLEMS WITH THE CURRENT DRAFT:

1. The slide has zero structural containers. No boxes, no panels, no color blocks. AAA slides anchor every data point inside a visible structural element. The current slide is type on a flat background.

2. The action title is too light. Newsreader 500 reads as editorial headline weight. Execs need it to read like a takeaway, not an article. Bump to Newsreader 600 (semibold), 60-64px, and add a court-color underline on the leading phrase ("The footage exists") the way AAA underlines "all 101 branches" or "live" on its action titles. Keep italic-clay on "coaching."

3. The stat descriptors are too small and the italic clay sub-line is noise. Remove the italic sub-line entirely. Replace with a single sans-serif descriptor in Inter Tight 500 at 18px, ink-soft color. Substantial, not decorative.

4. The thesis kicker is floating. Put "It's not a data problem. It's a workflow problem." inside a court-rail callout box at the bottom — 4px solid court-color left border, paper background, padding 24px, italic-clay on "workflow." It should look like a contained statement, not editorial filigree.

5. Add data structure. Each of the 4 stats goes in its own card box. The cards sit in a 4-column grid. Each card has:
   - A 3-4px top accent rail in a categorical color (court / plum / amber / clay, left to right)
   - White or paper background (--paper token)
   - 1px ink-mute border, 14px radius
   - The big Newsreader 600 tabular-num value (~96px) centered top
   - The unit in Newsreader 400 italic-clay (~32px) inline beside the value
   - JetBrains Mono UPPERCASE descriptor at the bottom, 14px tracked 0.12em
   - The descriptor is BOLD enough to read at executive distance

6. Make the top accent bar more substantial. Bump to 14-16px, NOT 4px. Add a small court-color logo chip hanging off the right edge if it fits the brand, the way AAA hangs the red AAA logo chip off its gradient bar.

7. The bottom chrome (logo + slide number) needs to be visible. Bump logo height to 36-40px, slide number to Mono 18px ink-mute. Both should read clearly at executive distance.

REVISED LAYOUT (slide 2):

Top: 16px gradient accent bar (clay → plum → court). Small CourtCheck checkmark chip aligned to right edge.

Title block (top-left, 64px from edges):
   "The footage exists. The *coaching* moments don't."
   Newsreader 600, 64px, line-height 1.1
   Court underline on "footage exists" (text-decoration-thickness 3px, offset 8px)
   Italic-clay on "coaching"
   Below title: 1px line-soft hairline separator at full slide width

4-column card grid (below hairline, generous padding):
   Card 1 — court top rail — "80 hrs" + "FOOTAGE PER SEASON, PER PROGRAM"
   Card 2 — plum top rail — "12%" + "ACTUALLY REVIEWED"
   Card 3 — amber top rail — "3 days" + "LAG FROM MATCH TO COACHING INSIGHT"
   Card 4 — clay top rail — "[TBD%]" + "COACHING DECISIONS DATA-BACKED TODAY"

   Each card has: paper bg, 1px ink-mute border, 14px radius, top rail 4px solid (categorical color), padding 32px.
   Numeric centered top. Inline unit (hrs / % / days / %) in Newsreader italic-clay 32px right of the numeric.
   Descriptor at bottom in JetBrains Mono UPPERCASE 14px, tracked 0.12em, ink-soft color, max 3 lines.
   Cards sit on a 24px gap grid.

Kicker callout (below card grid, full slide width minus chrome margins):
   Court-rail callout box: 4px solid court left border, paper bg, padding 28px.
   Text: "It's not a data problem. It's a *workflow* problem."
   Inter Tight semibold 26px, italic-clay on "workflow."

Chrome:
   Bottom-left: CourtCheckLogoLight.png at 36px height
   Bottom-right: "02" in JetBrains Mono 18px ink-mute, tracked 0.1em

REGENERATE SLIDE 2 with these structural changes. Same brand laws apply (Newsreader display, Inter Tight body, JetBrains Mono UPPERCASE meta, italic-clay rule, no red, lime sparing). The fix is structural, not chromatic.

Once Slide 2 is approved, apply the same structural discipline going forward: every content slide gets visible containers (cards, callouts, color rails, panels), substantial action titles with optional court underline on key phrases, sans-serif body descriptors (not italic decorative type), and a contained thesis callout at the bottom where applicable.

</pasted_text>

## Assistant

_[tool: str_replace_edit]_

Understood — structural-container discipline takes precedence over editorial flourish. Reworking Slide 2 to the AAA pattern and bumping global chrome (accent bar, logo, slide-num) so the rest of the deck inherits the change once Slide 2 is approved.Now adding the new card-grid + callout pattern and rewriting Slide 2:

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: done]_

_[tool: fork_verifier_agent]_

Slide 2 reworked to the AAA structural pattern:

- **14px** gradient accent bar with a small court chip hanging off the right edge
- Action title bumped to **Newsreader 600 / 64px**, court-color underline on "The footage exists", italic-clay on "coaching", hairline rule beneath
- **4 stat cards** in a row — each with a 4px categorical top rail (court → plum → amber → clay), paper bg, ink-mute border, big Newsreader 600 tabular numeral up top with italic-clay unit beside, bold JetBrains Mono descriptor at the bottom. No more italic-clay sub-line noise.
- **Court-rail callout** for the thesis: 4px solid court left border, paper bg, semibold Inter Tight, italic-clay on "workflow"
- Chrome bumped: logo at 38px, slide-num at 24px mono

T3 in the template library updated to match (since it's the explicit template for this pattern).

Verifier is screenshotting now to confirm rails, chip, underline, and clipping all check out. **Approve Slide 2 and I'll apply the same discipline (visible containers, substantial action titles with optional court underlines, sans-serif body descriptors, contained thesis callouts) across slides 3, 5, 6, A1–A4, and the affected templates.**

## User

Verifier agent check completed

## Assistant

Slide 2 is approved on the verifier side. Awaiting your sign-off before I apply the same structural discipline (visible containers, substantial action titles, contained thesis callouts) across slides 3, 5, 6, A1–A4 and the rest of the templates.

## User

yes do the rest of the slides:
ADDITIONAL RULE — APPLY UNIVERSALLY:

Every list, grid, or breakdown on every content slide gets numbered or lettered markers, the way AAA puts circled badges (A/B/C/D for metric categories, 1/2/3/4 for cluster types) on every multi-item slide.

Marker spec:
- Circular badge, 40-44px diameter, positioned top-left of each card or row
- Background: the categorical color of that item (court / plum / amber / clay, etc.)
- Foreground: cream text
- Font: JetBrains Mono 500, 16px, no letter-spacing
- Format: "01" / "02" / "03" / "04" with leading zero, OR "A" / "B" / "C" / "D", OR Roman numerals
- Choose the format per slide:
   - Letters for categorical facets (different angles on one thing) → AAA Slide 4 pattern
   - Numbers (with leading zero) for ordered/ranked/cluster items → AAA Slides 5, 6, 11
   - Roman numerals for tier or temporal progressions

APPLY TO THE REVISED SLIDE 2:
Each of the 4 stat cards gets a circular numbered badge at top-left:
   Card 1 (court rail): "01" badge in court green
   Card 2 (plum rail): "02" badge in plum
   Card 3 (amber rail): "03" badge in amber
   Card 4 (clay rail): "04" badge in clay
Badge sits in the top-left corner of the card, 24px from top and left edges. Numeric stat shifts down accordingly.

APPLY TO ALL SUBSEQUENT SLIDES:
- Slide 3 failure matrix: A / B / C / D badges on each row (categorical — they're 4 different alternative tools)
- Slide 5 impact 3-tile micro-grid: 01 / 02 / 03 numbered badges
- Slide 6 expansion tiers: keep Roman numerals i / ii / iii (already specified)
- Appendix A1 P/R/F1 table: A / B / C / D on each stroke class row
- Appendix A2 pipeline: keep Step 01 through 05 (already specified, just enforce the leading zero)
- Appendix A4 risks: 01 / 02 / 03 on each risk row

The marker is BOTH a visual anchor AND a verbal cue. The presenter should be able to say "If you look at point 2..." or "Category B here..." and the eye finds it instantly. AAA's consulting register depends on this.

## Assistant

_[tool: update_todos]_

_[tool: str_replace_edit]_

Adding the badge system + restructuring slides 3/5/6 and A1/A2/A3/A4 + mirrored templates. Big batch.Now restructuring slide 3 (failure matrix → cards with A/B/C/D badges) and slide 5 (micro-grid as cards with badges):

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

Now updating all slide markups to add badges and keyword underlines:Now Slide 3 — full restructure to failure-cards with A/B/C/D badges:

_[tool: str_replace_edit]_

Now slide 5 micro-tiles + badge, slide 6 keyword underline, A1 badges, A2 leading zeros, A3 underline, A4 badges:

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

Now updating templates T4, T6, T7, T8 to match the new patterns:

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: done]_

_[tool: update_todos]_

_[tool: fork_verifier_agent]_

Universal badge rule + structural discipline applied across the deck:

- **Slide 2** — 01/02/03/04 in court/plum/amber/clay on each stat card
- **Slide 3** — failure matrix rebuilt as 4 paper cards with **A/B/C/D** badges (categorical facets), court-underlined "Every existing option"
- **Slide 5** — micro-grid converted to paper cards with **01/02/03** badges, anchor-anecdote now sits in a court-rail callout
- **Slide 6** — court-underline on "Built for UC Davis"; Roman numerals kept
- **A1** — A/B/C/D badges on stroke-class rows (forehand→volley = court→clay)
- **A2** — pipeline stages relabeled **01–05** with leading zeros
- **A3** — court-underline on "Pricing built"
- **A4** — risk table gains a badge column with **01/02/03** badges in court/plum/amber
- **Templates** — T2 gets A/B/C on the exec-summary columns; T4 rebuilt to match Slide 3; T6 enforces 01–04; T7 adds 01–05 on each cluster row; T8 micro-grid gains badges; all template action titles gain `<span class="keyword">` + title-rule for parallelism

Verifier is sweeping all 22 slides for the badges, scrollHeights, and the title-rule hairlines now.

## User

Verifier agent check completed

## Assistant

_[tool: str_replace_edit]_

_[tool: done]_

## Assistant

Fixed — T3 now carries the 01/02/03/04 badges in court/plum/amber/clay to match slide 2's structure. Trivial tweak, ending turn.

## User

<pasted_text name="Pasted text (45 lines)">
The deck-so-far has been brand-faithful but audience-wrong. I'm reissuing the brief with corrected priorities. The case comp deck is a CONSULTING ARTIFACT that wears CourtCheck colors. It is NOT a brand-faithful editorial surface. The brand laws relax for this deliverable.

NEW PRIMARY DIRECTIVE — AUDIENCE FIRST:

The audience is case-comp judges sitting 15 feet from a projected screen. They scan, they don't read. Each slide gets 30 seconds. They are evaluating strategic clarity and execution rigor. They expect McKinsey/BCG/AAA register.

REVISED TYPOGRAPHY (overrides earlier brand instructions for this deck only):

- ACTION TITLES: Inter Tight 700 (Bold), 40-48px, ink color. NOT Newsreader. Sans-serif because it scans at distance. Apply court-color underline on the key phrase. ONE italic-clay word allowed per title, otherwise straight bold sans.
- BODY + LABELS + DESCRIPTORS: Inter Tight 500-600, 18-22px, ink-soft color. Substantial enough to read at 10 feet.
- COLUMN HEADERS, EYEBROWS, BADGES: JetBrains Mono UPPERCASE 14px, tracked 0.12em.
- BIG KPI NUMERALS ONLY: Newsreader 600 with tabular-nums. This is the ONE place Newsreader still earns its keep — serif tabular figures read as authoritative in big-number callouts. Everywhere else, sans-serif.
- ITALIC-CLAY: SPARING. Once per slide max, only on the most semantically loaded word. Some content slides should have NO italic-clay at all. Section dividers are the exception — italic-clay headlines fine there.
- NO italic decorative sub-lines under descriptors. Cut all of that.

REVISED LAYOUT GRAMMAR — AAA STRUCTURAL CONVENTIONS:

Every content slide includes ALL of these elements:

1. Thick gradient accent bar (14-16px) at the top, clay → plum → court
2. Action title in Inter Tight 700, with court-color underline on key phrase, italic-clay on AT MOST one word
3. 1px ink-mute hairline separator below the title
4. STRUCTURAL CONTAINERS for every data point: filled cards, dark callout boxes, color-coded panels. No floating type on flat background.
5. Circled numbered/lettered badges on every grid item (01/02/03 with leading zero, OR A/B/C/D, OR i/ii/iii)
6. Information density 2-3x higher than a marketing page. Each slide should prove something with multiple supporting elements.
7. Bottom kicker thesis in a court-rail callout box (NOT floating italic)
8. Visible chrome: logo bottom-left at 36-40px, slide number bottom-right in mono uppercase at 16-18px

COLOR USAGE:

- Cream paper as canvas (--cream)
- Paper white for cards (--paper) on 1px ink-mute border, 14px radius
- Categorical entity colors (court / plum / amber / clay) used FUNCTIONALLY: each card or row in a multi-item grid gets ONE assigned color via a 4px top rail or left rail. Used for visual chunking, not decoration.
- Dark filled callout boxes (near-black #1A1815 bg, cream text) for "Impact" panels or quote inserts, mirroring AAA's navy filled boxes
- Clay = pain/failure framing. Court = solution/win framing. Lime = THE single ball-moment, once per deck max (reserve for the impact slide's most important numeral or the close).
- High contrast everywhere. Ink for primary, ink-soft for secondary, ink-mute only for chrome metadata.

REGENERATE SLIDE 2 with these revised priorities:

- Action title: Inter Tight 700 at 44px: "The footage exists. The coaching moments don't." with court-color underline on "footage exists" and italic-clay on "coaching" ONLY
- Below hairline: 4-card grid as before with circled 01/02/03/04 badges in categorical colors. Big Newsreader 600 numeric centered (the ONE place serif stays), Inter Tight 500 descriptor below.
- Kicker callout box at the bottom: court left rail 4px, paper bg, Inter Tight 600 at 24px: "It's not a data problem. It's a workflow problem." Italic-clay on "workflow" only.

Apply the new priorities to every subsequent slide. The deck is a consulting artifact that wears CourtCheck colors. Build accordingly.

</pasted_text>

## Assistant

_[tool: str_replace_edit]_

Big consulting-register pivot understood. Refitting typography across the whole deck: Inter Tight 700 action titles with sans body/labels, Newsreader retained only for the big KPI numerals, italic-clay drastically reduced to one word per slide max, chrome micro-typography pulled in.

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

Now table headers, pipeline, contrast, tier, exec-grid headers, and the rest of the tertiary chrome:

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: done]_

_[tool: fork_verifier_agent]_

Consulting register pivot landed across the deck:

- **Action titles** → Inter Tight 700 / 44px on every content slide. Italic-clay reduced to one word per title; "built for" descriptors on slide 3 now sans ink-soft italic (no clay).
- **Body / labels / descriptors** → Inter Tight 500–600 at 18–22px. Mono caps reserved for column headers, eyebrows, badges at 14px.
- **Big KPI numerals** → Newsreader 600 tabular retained on stat cards, the 80→12 contrast pair, micro-tiles, bento, metric table.
- **Dark callout** added: slide 5's design-partner insert now near-black `var(--ink)` filled with cream text + a lime "Design partner" eyebrow (the deck's one lime moment).
- **Badges** → 38px with mono 14px (small chrome). Slide 6 + T9 tier markers converted from big italic Romans into court-color badge circles holding "I / II / III".
- **Chrome** → logo 36px, slide-num mono 16px, tpl-eyebrow 14px.

Verifier is sweeping for stray serif body/labels, italic-clay overuse, and any clipping introduced by the type swap.

## User

Verifier agent check completed

## Assistant

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: done]_

Fixed:

- **T5** point markers converted from italic-clay Romans → 38px court-color badge circles with mono Roman characters (I/II/III/IV, no period).
- **T7** state arrows changed from italic-clay → upright ink-mute, dropping all 5 italic-clay leaves.

Both templates now respect the one-italic-clay-per-slide rule. Trivial tweak, ending turn.

## User

<pasted_text name="Pasted text (59 lines)">
Every template you've produced has the same problem: top-weighted content, 40-70% empty vertical space, editorial sparseness instead of consulting density. The structural bones are right (action title, gradient bar, badges, color rails, chrome). The content density is wrong. Densify EVERY template you've produced — not just a subset.

UNIVERSAL DENSITY RULE — APPLY TO EVERY TEMPLATE WITHOUT EXCEPTION:

A template fails the density test if any of these is true:
1. More than 20% of the slide body (below title hairline, above chrome) is empty
2. Any card, row, column, or content block has only 1-2 lines of content with empty space below
3. Any chart or visual element has no axis labels, no data values, no legend, no caption, no source footnote
4. Any list element has fewer than 4 items, or 3 items where 4+ would fit
5. Any cell has 1 data point where 2-3 layered data points would fit

If a template fails any of these, regenerate it with more content layers, more structural elements, and tighter vertical compression.

UNIVERSAL DENSITY FIXES — APPLY EVERYWHERE THEY MAKE SENSE:

A. EVERY CARD becomes a 3-zone layout:
   - Top zone: badge + headline value (Newsreader display for numerics, Inter Tight 700 for non-numeric headlines)
   - Middle zone: primary descriptor in mono uppercase tracked 0.12em
   - Bottom zone: secondary context line in Inter Tight 16px ink-soft (source / comparison / sub-detail / proof point)

B. EVERY ROW IN A TABLE/MATRIX becomes 4+ columns with 2-line cells. Single-line cells are banned. Add columns until the row earns its width (option / built-for / why-fails / cost-or-signal — or equivalent layered structure for the content).

C. EVERY CHART gets the full consulting kit:
   - Caption (mono uppercase) above
   - Y-axis label + X-axis label (mono uppercase 11px)
   - Data values labeled directly on bars/lines (Newsreader tabular-num)
   - Legend below with counts or units
   - 1-line insight callout in Inter Tight italic 18px below the chart
   - Source footnote in mono uppercase 11px ink-mute at the very bottom of the chart card

D. EVERY LIST ELEMENT (sidebar bullets, step descriptions, column bullets) gets a 2-line sub-detail beneath it in Inter Tight 15px ink-mute. Tools, outputs, sources, examples, or sub-points — pick what's appropriate for the content type.

E. EVERY TEMPLATE gets a bottom band — pick one based on slide purpose:
   - Kicker thesis callout box (court-rail left, paper bg, italic-clay accent)
   - Insight strip (mono uppercase one-liner spanning full width)
   - Source/method footnote band
   - Summary callout (synthesizes what's above)
   Banned: empty vertical space at the bottom of any content slide.

F. EVERY STEP / TIER / CLUSTER element gets a sub-element below the description:
   - "TOOL:" or "OUTPUT:" or "METRIC:" or "OWNER:" or "TIMING:" label in mono uppercase 11px
   - Value in Inter Tight 500 ink-soft 14px
   - And horizontal connector line/arrow BETWEEN sequential elements where it shows flow

G. COVER SLIDES get a hero element zone above the project name (large stat, pull quote, OR prominent logo). 50% empty cream above the title is banned.

H. SECTION DIVIDER SLIDES are exempt from density rules — they're meant to be sparse. Everything else must fill 85%+ of the slide body.

REGENERATE EVERY TEMPLATE WITH THESE DENSITY FIXES. Don't ask me which templates to fix — fix all of them. Same brand laws apply (italic-clay sparing, no em dashes, no red, no lime beyond the single ball moment per deck, brand chrome).

VERIFICATION:
After regenerating, run each template through this check:
1. Body fill ratio: is the content area 85%+ filled? If no, add elements.
2. Layered cells: does every card/row/column have 3+ layers of information? If no, add layers.
3. Bottom band: does the slide have a kicker, footnote, or insight strip at the bottom? If no, add one.
4. Executive distance test: would a judge sitting 15 feet from a projector be able to scan the slide and extract the takeaway in under 10 seconds? If no, restructure.

If a template still has >20% empty space after applying these rules, the template is wrong for the content type — propose a different layout entirely for that slide.

</pasted_text>

## Assistant

_[tool: str_replace_edit]_

Big densify pass. Adding universal density patterns to CSS, then rebuilding T1–T9 with layered cards, full chart kit, connectors, and bottom bands.

_[tool: str_replace_edit]_

Updating T1 cover for density:Now updating T1 HTML and T2 with 4-bullet columns + sub-details + bottom callout:

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

Now adding signal cells to slide 3 failure cards + insight strip:

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: grep]_

Good, still intact. Now updating T8 + slide 5 (impact) markups with delta strip and micro-tile sub-details:

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: done]_

_[tool: fork_verifier_agent]_

Density pass complete:

- **Universal patterns** added: `card-meta` (3rd zone on every card), `subdetail` (sub-line under list items), `insight-strip` (bottom-band mono callout), `kpi-sub` (in-card KPI tile), `delta-strip`, `sub-label` (TOOL: / OUTPUT: per step), `chart-x-axis` / `chart-legend` / `source-footnote` (full chart kit), `cover-hero` + `cover-meta-strip` (denser cover).
- **Templates rebuilt:** T1 now leads with a pull-quote hero and a 4-cell meta strip above the photo band. T2's three columns each carry 4 bullets with subdetails + a KPI sub-tile + bottom kicker callout. T3/T4 stat & failure cards both gained third-zone meta (source/prior/signal) + insight strip. T5 chart card has caption, bars, axis ticks, 4-color legend with counts, source footnote, and a slide-level insight strip. T6 steps gained TOOL/OUTPUT sub-labels + synthesis callout. T7 cluster table is now 4 columns with examples + decision-rule callout. T8 has a delta strip + card-meta on every micro-tile + insight. T9 tiers gained Proof/Owner sub-labels + synthesis callout.
- **Main deck inherited:** Slide 2 stat cards now carry "Source" meta. Slide 3 failure cards gained a signal column ("$100K+ per court", "1 court at a time" etc.) + insight strip. Slide 5 micro-tiles carry source/prior/since meta. Slide 6 tiers gained Proof/Owner. A1 + A3 + A4 each gained an insight strip; A1 added a source footnote on the confusion card; A2 stages carry TOOL/OUTPUT.

Verifier is sweeping all paper slides for clipping, fill ratio, layered density, and bottom-band coverage.

