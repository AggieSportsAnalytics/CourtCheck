# Voice & tone

> The coach voice. Same register everywhere — headlines, body, toasts, errors, empty states.
> Specific. Calm. Coaching. Plain English. No exclamation marks. No emoji.

---

## Voice in one sentence

Speak like a thoughtful assistant coach who's already watched the film.

We are not a feed. We are not a hype tool. We are quiet and right.

---

## The two-voice system

The brand has two voices. They live in different typefaces:

| Voice | Type | Used for |
|---|---|---|
| **Coach voice** (the human) | Newsreader (italic accent in Clay) | Headlines, callouts, body copy, the human-spoken word |
| **System voice** (the machine) | JetBrains Mono UPPERCASE 0.18em | Anything *the product itself* says — timestamps, status, version, axis labels, "tracking", "live", "cam · baseline", zone names |

When the coach reads a headline, they hear a coach. When they read a status indicator, they hear the system. The split is the brand.

---

## Five voice tests

For every string of copy:

### 1. Specific test
Does it tell the coach something they couldn't write themselves?

✅ "Loses the cross 64% on second-serve return."
❌ "AI-powered insights tailored to your performance."

### 2. Calm test
Does it use exclamation marks, emoji, or urgency theater?

✅ "Pattern detected. Lin's slice serve wins 71% of openers."
❌ "🎉 Amazing pattern unlocked!"

### 3. Coaching test
Is it telling the coach what to do, or selling the product?

✅ "Here's what to work on Wednesday."
❌ "Crush your competition with CourtCheck."

### 4. Plain English test
Could a coach who's never used analytics tools read it?

✅ "First serve in: 68% — up 4 points from your season."
❌ "Serve efficiency delta: +4.2 ppt vs trailing 30 baseline."

### 5. Numbers test
Are numbers wrapped so they render in Newsreader serif?

✅ `First serve in: <span class="num">68%</span>`
❌ `First serve in: 68%` (raw — falls back to body sans, breaks the brand promise)

---

## Headline rules

### One italic word per headline

Every page title has exactly **one italic word**, always rendered in `--clay`.

The italic word is:
- The **verb** of the sentence ("automated", "lives", "opens"), or
- The **action** being claimed ("Know every move", "Get back in")

```html
<!-- Hero -->
<h1 class="display">
  See every shot.<br>
  <span class="ital clay">Know every move.</span>
</h1>

<!-- Section -->
<h2>
  Tennis has always been played on
  <span class="ital clay">instinct.</span>
</h2>

<!-- Closing CTA -->
<h2 class="display">
  Stop guessing. <span class="ital clay">Start winning.</span>
</h2>

<!-- Auth -->
<h1 class="display auth-title">
  Get back <span class="ital clay">in.</span>
</h1>
```

This is the brand's signature. Coaches will start to recognize the pattern unconsciously.

### Newsreader for everything display

- Hero h1: opsz 144, weight 500
- Section h2: opsz 96, weight 500
- Card title: opsz 72, weight 500
- Italic accent: opsz 18, weight 400 italic

Never use Newsreader at body size. Body is Inter Tight.

---

## Body copy rules

- Lead with the most useful information
- One idea per paragraph
- Numbers in serif: `<span class="num">71%</span>`
- Em dashes are fine in product copy (they signal a pause, like a coach speaking)
- Never use exclamation marks. The product earns trust by being quiet.

---

## Notification / toast voice

Same register, applied to in-product feedback. Lead with **what happened**, then **what to do** about it.

### Success
- ✅ "Match analyzed. View Stanford on your dashboard."
- ❌ "🎉 Your match has been successfully analyzed!"

### Info
- ✅ "Pattern detected · Lin's slice serve wins 71% of openers."
- ❌ "New insight available!"

### Warning
- ✅ "Video is 480p. Analysis quality may suffer — for best results, upload 720p or higher."
- ❌ "⚠️ Low quality video!"

### Error
- ✅ "Couldn't reach the server. Check your connection and try again."
- ❌ "An error occurred. Error code: NET_FAIL_1138."

**Errors NEVER auto-dismiss silently.** They persist until the user dismisses or the underlying state resolves.

**Errors NEVER use red.** Clay only.

---

## Empty state voice

Don't show "No matches yet." Show what's coming, and how to get there. Empty states are tiny ads for the product itself.

✅
> **Patterns will appear here after your first match is processed.**
> Upload a video → we'll surface 5–10 patterns within ~20 minutes.
> [Upload first match →]

❌
> No data available.

Every empty state teaches the product.

---

## Microcopy rules

### Buttons
- Verb-first when the action is obvious from context: "Upload", "Sign in", "Continue"
- Verb + object when it's not: "Upload your first match", "Send reset link"
- Never "Click here", "Learn more", "Get started" — too vague

### Links
- Tell them where they're going: "View match", "See all patterns"
- Never "Read more" or "Click here"

### Labels (in forms, axes, captions)
- JetBrains Mono uppercase, 0.18em letter-spacing
- Short, dense: "EMAIL", "PASSWORD", "ROLE", "TEAM"
- Never ask twice: if the field is "Email", the placeholder isn't "Enter your email"

### Placeholders
- Show example data, not instructions
- "coach@school.edu", not "Enter your email here"
- "UC Davis Tennis", not "Enter your team name"

### Helper text
- Use sparingly. If you need helper text, the field probably isn't clear enough.
- When used: under the field, 0.85rem, `--ink-soft`, plain English

---

## Anti-patterns (will never ship)

| Don't say | Say |
|---|---|
| "Powered by AI" | (just say what it does) |
| "Revolutionize your coaching" | "Watch the film for you" |
| "Crush your competition" | "Here's what to work on Wednesday" |
| "🎉 Welcome aboard!" | "Signed in" |
| "An error occurred" | "Couldn't reach the server" |
| "Click here to learn more" | "View match" |
| "Get started for free!" | "Upload your first match" |
| "Loading..." | (a bounce-arc animation, no text) |

---

## Reference brands for voice

We sound like:
- The Atlantic news writing
- Bloomberg's daily ticker copy
- Linear's product UI
- Stripe's documentation
- The New York Times' running-stat banners

We do NOT sound like:
- Hudl (overloaded, urgent)
- SwingVision (jargon-heavy)
- Generic AI startups (hype, exclamation marks)
- Sports betting apps (exclamatory, urgency theater)

---

*Living spec. As real product copy ships, capture the wording so the coach voice stays consistent.*
