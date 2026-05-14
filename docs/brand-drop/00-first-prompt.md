# First-prompt.md — fresh Claude Design onboarding

> Paste the message below into your first Claude Design conversation, after uploading the bundle.
> This is calibrated as a FRESH ONBOARDING for a new Claude Design project, NOT a refinement of prior work.

---

## Copy-paste this verbatim into Claude Design

```
Hi. This is the CourtCheck brand — fully locked. I'm onboarding you fresh against the canonical version. Your job is to build product surfaces against this brand, not to iterate on the brand itself.

Read the bundle in this order:
1. 01-brand-guidelines.md  ← THE primary spec with YAML front matter, full system
2. logo/concept-notes.md   ← The logo is locked; this doc confirms the rules
3. logo/current-system.md  ← Inventory of locked logo assets
4. 02-philosophy.md        ← Brand DNA (Robinhood meets SwingVision)
5. 03-voice.md             ← Coach voice + microcopy + toast voice
6. 04-motion-spec.md       ← Motion DNA + locked hero + splash animations
7. 05-peer-references.md   ← The editorial register we're after
8. tokens.css              ← The implemented globals.css (current state in code)
9. mocks/                  ← Brand-applied product surfaces — 10 reference pages

LOCKED ASSETS (use as-is, do not regenerate):

- logo/CourtCheckLogoLight.png — primary lockup, light mode, cream canvas
- logo/CourtCheckLogoDark.png — primary lockup, dark mode, near-black canvas
- logo/CourtCheckAnimation.mp4 — splash animation, 5s, plays once
- motion/Bounce_Animated.mp4 — hero background animation, 6s, seamless loop
- motion/Bounce_Still.png — hero animation source frame (also poster + reduced-motion fallback)

These are the brand. Do not redraw, regenerate, or "improve" them. If a use case feels like it needs a variant that doesn't exist (e.g., a square-format hero asset, a monogram-only mark), flag it as a request — do not silently invent.

WHAT I'M ASKING YOU TO DO:

Build product surfaces against the locked brand. The first surface is the public landing page. Subsequent surfaces will be: auth flow, dashboard (empty + populated), upload flow, match detail, player profile.

For the landing page, the structure I want is:

1. NAV — sticky top, locked brand lockup at left, links + theme toggle + Sign in pill at right
2. HERO — full-viewport, two-column layout. Left: eyebrow + headline + lede + CTAs + trust line. Right: motion/Bounce_Animated.mp4 in a contained card at native 16:9, NOT full-bleed background. The video sits in a paper-colored card with a 1px line border, 18px radius, brand card shadow. No object-cover crop.
3. PROBLEM — two-column, eyebrow + headline on left, supporting paragraphs on right ("Tennis has always been played on instinct.")
4. BIG-NUMBER CONTRAST PANEL — two massive Newsreader numbers side by side ("80 hrs" + "12 min") with mono captions beneath, on warm paper bg
5. HOW IT WORKS — 4-column grid with italic Roman numerals (i. ii. iii. iv.), each step a short heading + body
6. PINNED SCROLL-THROUGH FEATURE SECTION — left side sticky court tile, right side scrolls through 5 stages of feature copy. Court tile transforms: empty → shot map → heatmap → spacing lines → pattern callout. Smooth scroll behavior. (See specifics below.)
7. CLOSING CTA — centered Newsreader headline "Stop guessing. Start winning." with italic-clay on the verb phrase
8. FOOTER — 4-column grid + brand mark + tagline

LOCKED BRAND RULES YOU MUST FOLLOW:

- Cream paper background (#F4F0E6) is the base canvas across the whole page
- Newsreader serif for all display type, Inter Tight for body, JetBrains Mono uppercase for system meta
- italic-clay rule: every section headline has exactly ONE italic word, rendered in clay terracotta (#B05B36 light / #E07A52 dark)
- Lime (#BFC846) only appears once per section as "the ball moment". The CourtCheck wordmark renders `Check` in italic lime — this is the SINGLE locked exception where italic + lime co-occur, and it counts as the page's lime moment when the wordmark is present
- No red, ever — clay for errors and accents only
- All numerics render in Newsreader 500 with tabular numerals
- The locked brand assets (logo lockups, hero video, splash) are immovable

WHAT NOT TO INCLUDE (all previously rejected, do not regenerate):

- NO fake school logo strip ("In film rooms at..." with UC Davis / Stanford / Cal)
- NO fake testimonials
- NO 3D court rendering or Three.js scene as the hero — use motion/Bounce_Animated.mp4
- NO before/after photo transformation pattern
- NO vintage newspaper background overlay
- NO "powered by AI" badges, "revolutionize" copy, exclamation marks

START BY:

1. Acknowledging the brief and confirming you've read the bundle
2. Walking me through your hero treatment FIRST — show me the hero section before generating the rest of the page
3. Ask any clarifying questions about copy or layout BEFORE producing the design — do not invent copy not specified above

Don't generate the whole landing page in one shot. We'll review section by section.
```

---

## Why this prompt works

It's calibrated to:

- **Frame as fresh onboarding, not refinement** — sets the right scope, prevents Claude Design from re-litigating decisions we've already made
- **List the locked assets explicitly** — Claude Design treats them as immovable inputs, not editable mocks
- **Order the bundle reading** — front-loads the locked spec before the supporting context
- **Specify the page structure** — gives Claude Design a clear deliverable, not "design a landing page"
- **List "what not to include"** explicitly — prevents regression to all the rejected directions we already tried
- **End with "acknowledge before generating"** — forces alignment before assets are produced

If Claude Design wants to push back on any of the locked items, it should propose the change in writing first. Never as a fait accompli in a generated asset.

---

## After Claude Design ingests the bundle

Run a test before generating: ask it to summarize what's locked and what's open. If it can correctly list the locked logo assets, the locked hero animation, and the locked brand laws, you're good. If it tries to "iterate" on any of them, re-emphasize that those are immovable inputs and re-paste the relevant constraints.

---

## Follow-up prompts (use after each section is approved)

### After hero is approved
> Now generate the Problem section, sitting beneath the hero. Two-column layout, eyebrow + headline on left ("Tennis has always been played on instinct."), three supporting paragraphs on right. italic-clay accent on "instinct."

### After Problem is approved
> Now the Big-Number panel. Two massive Newsreader serifs side by side: "80 hrs" + "12 min" with the brand's signature italic-clay treatment applied to "12" specifically. Mono captions beneath each. Warm paper bg.

### After Big-Number panel is approved
> Now How It Works — 4-column grid with italic Roman numerals as the numeric markers. Each step has a Newsreader sub-headline and a 2-line body. Use the copy in 01-brand-guidelines.md section [reference].

### After How It Works is approved
> Now the Pinned Scroll-Through. Sticky court tile on left (using my Web UI Kit components — pull from there, don't generate new), scrolling feature copy on right, 5 stages: empty / shot map / heatmap / spacing / pattern callout chip. Smooth scroll behavior, 100vh per stage.

### After everything is approved
> Now the Closing CTA + Footer. Centered "Stop guessing. Start winning." headline. Footer 4-column grid.

---

*The bundle reflects the locked brand as of the current state. Use it. Don't iterate.*
