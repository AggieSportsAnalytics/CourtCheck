# CourtCheck — May 15 Case Competition PRD

**Owner:** Brian Le (PM)
**Deadline:** May 15, 2026 (case competition demo)
**Codebase:** `/Users/Brian/PycharmProjects/CourtCheck`
**Stack:** Python 3.10 backend (Modal/GPU), Next.js 16 frontend (Vercel), Supabase

---

## Context for New Sessions

This PRD was authored on 2026-04-22. CourtCheck is a tennis video analytics platform built for UC Davis Women's Tennis coaches. It uses computer vision (YOLOv8, TrackNet) to track the ball, detect bounces, classify strokes, and render overlays + heatmaps on match footage.

The system is live. The pipeline runs on Modal (A10G GPU). Frontend is deployed on Vercel. Supabase handles storage and database.

**Before touching any feature:** read `CLAUDE.md` in the project root for architecture, key file locations, and always-active skills (verification-before-completion, TDD, systematic-debugging).

---

## Priority Stack (P0 → P4)

| # | Feature | Priority | Owner | Blocks |
|---|---------|----------|-------|--------|
| 1 | Stroke classifier rebuild | P0 | Kenny + Anik | Everything |
| 2 | Heatmap improvements | P1 | Anik | Demo visuals |
| 3 | Player page + video assignment | P2 | Sienna + Anik | Coach workflow |
| 4 | UI overhaul (landing + dashboard) | P3 | Sienna + Brian | Presentation |
| 5 | Cuts / sets / side-switching | P4 | Kenny + Brian | Optional for demo |

---

## P0 — Stroke Classifier Rebuild

### Problem
Current `stroke_detector.py` uses InceptionV3 + LSTM, is slow and inaccurate on UC Davis footage. The TCN architecture (`stroke_classifier_tcn.py`) is built but untrained on real data.

### Decision
- **Abandon** `stroke_detector.py` (legacy, InceptionV3-based). It is dead code after this sprint.
- **Train** `stroke_classifier_tcn.py` — pose keypoint TCN (34 input dims = 17 COCO keypoints × 2 coords).
- **Pretrain on THETIS** dataset first, then fine-tune on UC Davis annotations. THETIS has ~1,415 clips across 8 stroke types. Since input is pose keypoints (not raw video), domain shift is minimal.
- **Swing gate stays:** `vision/swing_detector.py` already gates classifier invocation on wrist velocity + ball proximity. Do not remove this — it's what makes inference tractable on long videos.

### Scope: Near-side player only
- Only classify strokes for the near-side player (lower half of court, Y > net_Y in court coordinates).
- Far-side player: render as gray dots on heatmaps, no stroke classification attempt.
- This is a deliberate product decision — scope reduction for accuracy, not a limitation.

### Annotation Pipeline (already built)
```bash
# Step 1 — extract swing clips from raw footage
modal run backend/app.py::extract_swings --local-dir data/raw_videos --output-dir data/annotation

# Step 2 — label clips (run this for each annotator)
streamlit run backend/tools/label_swings.py -- --manifest data/annotation/manifest.csv --annotator <name>
```
- Drop TrackTennis `.mp4` files in `data/raw_videos/`
- Activity gate skips YOLO during dead time (no ball for 5s)
- Target: **200-400 labeled clips** across 4 classes (Forehand / Backhand / Serve/Overhead / Slice)
- All 6 team members annotate together. Joseph, Michelle, Sienna are the annotation workforce.

### THETIS Pretraining Steps
1. Download THETIS dataset (publicly available, TU Berlin)
2. Run YOLOv8-Pose inference on all THETIS clips to extract pose keypoint sequences
3. Normalize keypoints: torso-relative coordinates (translate to hip-center, divide by shoulder-hip distance) — this removes position/scale variance across different cameras and court setups
4. Pretrain `StrokeTCN` on THETIS keypoints (8-class output for THETIS, swap final layer for UC Davis fine-tune)
5. Fine-tune on UC Davis annotations with frozen early layers

### Key Files
- `backend/models/stroke_classifier_tcn.py` — TCN architecture, `StrokeTCN` class, `STROKE_LABELS`
- `backend/vision/swing_detector.py` — `SwingDetector` class, wrist velocity gate
- `backend/models/stroke_detector.py` — legacy LSTM model, do not modify, mark deprecated
- `backend/pipeline/run.py` — where stroke classifier is invoked in the two-pass pipeline

### Acceptance Criteria
- [ ] THETIS pretraining script exists and runs
- [ ] UC Davis fine-tuning script exists with torso-relative normalization
- [ ] Model achieves >70% accuracy on held-out UC Davis clips (stretch: 80%)
- [ ] Far-side player strokes NOT classified — gray dots only on heatmap
- [ ] SwingDetector gate still active — no full-video frame-by-frame classification
- [ ] `stroke_detector.py` marked deprecated in docstring, not removed yet

---

## P1 — Heatmap Improvements

### Problem
Current heatmaps (`vision/heatmaps.py`) use gaussian blur for both shot and player heatmaps. Coaches want per-stroke-type breakdown and player positioning analysis, not blurry density maps.

### Changes

#### Shot Heatmap
- Color-code bounce dots by stroke type using `STROKE_COLORS_BGR` (already defined in `vision/drawing.py`)
- Render in/out markers: in-bounds = filled circle with white ring; out-of-bounds = red X (both already implemented in `_draw_bounce_dot`)
- Wire `frame_stroke_labels` through the pipeline so bounce dots reflect actual stroke labels
- Far-side player shots: gray dot (`(128, 128, 128)` BGR), no stroke color

#### Player Position Map
- **Replace gaussian blur heatmap** with individual position dots per shot
- One dot per shot hit, plotted at player's court position at moment of contact
- Color: same stroke-type color scheme as shot heatmap
- This enables: positioning error analysis, spacing patterns, court coverage gaps

#### Far Player
- Single config change: when player is far-side (`court_y < net_Y`), use gray for all dot renders
- No stroke classification for far player (feeds back to P0 decision)

### Key Files
- `backend/vision/heatmaps.py` — primary file for all changes
- `backend/vision/drawing.py` — `STROKE_COLORS_BGR` color map
- `backend/pipeline/run.py` — passes `frame_stroke_labels` dict to heatmap functions

### Acceptance Criteria
- [ ] Bounce dots colored by stroke type (FH/BH/Serve/Slice have distinct colors)
- [ ] In/out markers correctly rendered
- [ ] Far-side player dots are gray regardless of stroke label
- [ ] Player position map shows individual dots (no gaussian blur)
- [ ] No regression on existing heatmap generation for videos without stroke labels

---

## P2 — Player Page + Video Assignment

### Problem
No way to associate uploaded videos with specific players. No per-player stats aggregation. Coaches need to track individual player progress across matches.

### Database Changes

New table: `players`
```sql
CREATE TABLE players (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  position TEXT,           -- e.g. "Singles #1", "Doubles Specialist"
  year TEXT,               -- e.g. "Junior", "Senior"
  photo_url TEXT,
  team_id UUID,            -- for future multi-team support
  created_at TIMESTAMPTZ DEFAULT now()
);
```

Modify table: `recordings`
```sql
ALTER TABLE recordings ADD COLUMN player_id UUID REFERENCES players(id);
```

Migration file: `supabase/migrations/YYYYMMDD_add_players.sql`

### Upload Flow Change
- After video upload form, add step: "Who is the near-side player in this video?"
- Dropdown populated from `players` table
- `player_id` saved to recording on upload
- This is the only required change to the upload flow — no other fields needed

### New Page: `/players`

Route: `frontend/web/app/players/page.tsx`

**Layout:**
- Page header: "UC Davis Women's Tennis"
- Grid of player cards (roster view)
- Each card: player name, year/position, photo (or initials avatar), aggregated stats:
  - Total videos analyzed
  - Most frequent stroke (e.g. "Forehand dominant")
  - Error rate (out-of-bounds / total shots)
  - Court coverage score (% of court zones reached)

**Player detail:** clicking a card → `/players/[id]` with full stat breakdown + list of their recordings

### Data Seeding
- Scrape UC Davis Athletics women's tennis roster page for names, years, positions
- Write a one-time seed script: `backend/tools/seed_players.py`
- Store scraped data in `players` table via Supabase

### Key Files to Create/Modify
- `supabase/migrations/` — new migration
- `frontend/web/app/players/page.tsx` — new page
- `frontend/web/app/players/[id]/page.tsx` — player detail page
- `frontend/web/app/upload/page.tsx` — add player assignment step
- `frontend/web/app/api/players/route.ts` — new API route (GET all, POST new)
- `backend/tools/seed_players.py` — one-time roster scraper

### Acceptance Criteria
- [ ] `players` table exists in Supabase with UC Davis roster seeded
- [ ] Upload flow includes player assignment dropdown
- [ ] `/players` page renders roster grid with per-player stats
- [ ] `/players/[id]` renders individual stats + recording history
- [ ] `player_id` FK correctly set on new recordings

---

## P3 — UI Overhaul (Landing + Dashboard)

### Context
Brian has locked a brand direction. Do NOT deviate from the design system below.

### Brand System

**Colors (CSS variables):**
```css
--bg:          #080B10;   /* near-black, cold blue tint */
--surface:     #0F1318;   /* cards */
--surface-2:   #171C24;   /* elevated / hover */
--border:      #1E2530;   /* dividers */
--text:        #E8EDF5;   /* primary text */
--text-muted:  #5A6478;   /* labels, metadata */
--accent:      #B8FA4A;   /* signature chartreuse */
--accent-dim:  #4A6318;   /* accent backgrounds, pressed states */
--red:         #FF3D3D;   /* out-of-bounds, errors */
--court-fill:  #0D1F38;   /* court diagram fill */
--court-line:  #C8D8F0;   /* court lines */
```

**Typography:**
- Display / headings: `Barlow Condensed 700`, all-caps, `letter-spacing: 0.06em`
- UI labels / nav: `Outfit 500`
- Stats / numbers: `JetBrains Mono`
- Body: `DM Sans 400`

**Logo:** TBD — placeholder for now. Use "COURTCHECK" wordmark in Barlow Condensed until logo is finalized.

**Design principles:**
- Dark mode only
- Data density > white space — coaches want information at a glance
- Court diagram / minimap is the hero visual in the app
- Sharp corners on interactive elements (no heavy border-radius except cards: 8px)
- Use 21st.dev components where available: `npx shadcn@latest add "https://21st.dev/r/[component]"`

### Landing Page (`/landing`)

Structure:
1. **Hero** — full viewport dark background. CSS 3D perspective: court minimap diagram floating at ~12deg tilt with subtle glow. Headline: existing headline copy (Brian to confirm). Single CTA button in `--accent`.
2. **Social proof strip** — UC Davis Tennis logo + "Built for D1 Coaches"
3. **Feature cards** (3): Stroke Analysis / Court Heatmaps / Player Profiles — each with product screenshot
4. **Footer** — minimal, dark

3D hero implementation: CSS `perspective` + `rotateX` + `translateZ` on the court SVG element. No Three.js — CSS only is sufficient and faster.

### Dashboard

The dashboard is the recordings/stats view coaches use daily. Current implementation is functional but visually dated.

Changes:
- Apply brand system (colors, fonts) globally via `globals.css`
- Replace default shadcn card styles with custom surface cards from the brand system
- Stats figures: JetBrains Mono, large, primary color
- Navigation: apply Barlow Condensed to nav items, `--accent` active state indicator
- Court minimap in analysis views: increase prominence, make it the centerpiece not a sidebar widget

### Key Files
- `frontend/web/styles/globals.css` — add CSS variables and font imports
- `frontend/web/app/landing/page.tsx` — overhaul hero + structure
- `frontend/web/components/` — update card, nav, stat components
- `frontend/web/app/layout.tsx` — apply font classes globally

### Acceptance Criteria
- [ ] CSS variables applied globally
- [ ] Barlow Condensed + Outfit + JetBrains Mono loaded via Google Fonts
- [ ] Landing page hero renders 3D court diagram with CSS perspective
- [ ] Dashboard uses brand color system (no default shadcn gray/white)
- [ ] Court minimap is visually prominent in analysis views
- [ ] No regression on existing functionality

---

## P4 — Cuts / Sets / Side Switching

### Problem
Pipeline processes video linearly. Real match footage has cuts, set breaks, and players switching sides. Without handling this, heatmaps and stats are wrong for multi-set footage.

### Approach: Coach-Annotated Keypoints

Skip full algorithmic detection. Add a keypoint annotation UI in video playback. Coaches mark timestamps manually — they have context the model doesn't.

**Keypoint types:**
- `set_start` — beginning of a new set
- `side_switch` — players switch sides (odd games)
- `cut` — camera cut / jump in footage

**Frontend:** add a timeline annotation bar below the video player on the recording detail page (`/recordings/[id]`). Clicking adds a keypoint at the current timestamp. Keypoints stored in Supabase on the `recordings` record (JSON column or separate `keypoints` table).

**Backend:** `run.py` receives a `segments` array derived from keypoints. Each segment is processed independently for heatmaps and stats. Final output merges per-segment results, respecting side-switch flips (mirror court coordinates for switched-side segments).

### Acceptance Criteria
- [ ] Annotation timeline UI in video player
- [ ] Keypoints saved to Supabase
- [ ] Backend receives segments array and processes per-segment
- [ ] Side-switch flag correctly mirrors court coordinates for that segment
- [ ] Stats page shows per-set breakdown when keypoints exist

---

## Case Competition Video Concept

**Primary concept: "Ghost Player" + Coach Insight Reel**

Two deliverables for the demo video:

**1. Ghost Player Overlay**
- Show the statistically ideal court position (derived from shot distribution analysis) vs. where the player actually stood
- Render as: actual position = solid dot, ideal position = ghost/outline dot
- Gap between them = the coaching moment
- This is the most visually compelling concept for a competition audience

**2. Coach Insight Reel**
- Auto-generate a 2-minute highlight clip from match footage
- Pull: 5 biggest positioning errors + 3 best-executed strokes
- Add text overlays: "Error: 2.3m off optimal position" / "Clean forehand winner"
- Demonstrates automated coaching intelligence, not just data visualization

**Implementation notes:**
- Ghost position = centroid of top-20% shot outcomes from that court zone
- Use existing `storage.py` ffmpeg integration for clip generation
- Text overlays via OpenCV `putText` in `drawing.py`

---

## Timeline

| Date | Milestone |
|------|-----------|
| Apr 22 (today) | Annotation sprint kickoff — Streamlit app sent to team at meeting |
| Apr 22-27 | All 6 team members annotating; THETIS download + preprocessing |
| Apr 28 | First TCN training run; heatmap improvements begin |
| May 1 | TCN v1 integrated into pipeline; player page DB migration |
| May 5 | Player page frontend live; UI brand system applied |
| May 8 | Full pipeline demo internally — end-to-end on real footage |
| May 11 | Case comp video concept shot/rendered |
| May 12-14 | Buffer: bug fixes, polish, coach pilot if possible |
| **May 15** | **Case competition** |

---

## What's Already Built (Do Not Rebuild)

- `backend/vision/swing_detector.py` — swing gate, works
- `backend/models/stroke_classifier_tcn.py` — architecture, needs training only
- `backend/vision/heatmaps.py` — scaffolding exists, needs changes per P1
- `backend/vision/drawing.py` — `STROKE_COLORS_BGR` color map
- Annotation pipeline — `extract_swings` Modal function + Streamlit labeler
- Court calibration — complete for all UC Davis courts
- Supabase integration — `pipeline/storage.py`
- Auth, upload flow, recordings page — functional, needs styling only

---

## Team

| Person | Role | Assigned To |
|--------|------|-------------|
| Brian | PM + frontend direction | P3 (UI), P4 (product design), case comp video |
| Kenny | Backend | P0 (TCN training pipeline), P4 (segment logic) |
| Anik | Backend | P0 (pipeline integration), P1 (heatmaps) |
| Joseph | Data Scientist | Annotation (labeling) |
| Michelle | Data Scientist | Annotation (labeling) |
| Sienna | Frontend | P2 (player page UI), P3 (UI components) |

---

## Open Decisions (Resolved)

- **THETIS pretraining:** YES — use it. Pose keypoints are camera-invariant, pretraining is free signal.
- **Racket tracking as TCN input:** SKIP for now. YOLOv8-Pose doesn't natively track racket head.
- **Player roster:** SCRAPE UC Davis Athletics site. Write `seed_players.py`.
- **UI framework for new components:** 21st.dev + existing shadcn. Install: `npx shadcn@latest add "https://21st.dev/r/[component-slug]"`.
- **Landing page 3D:** CSS perspective only, no Three.js.
- **Cuts/sets detection:** coach-annotated keypoints, not algorithmic.

---

## Out of Scope for May 15

- Opponent analysis (explicitly deferred)
- Far-side player stroke classification
- Multi-team support (players table has `team_id` stub for future)
- Algorithmic cut/set detection
- Mobile app
- Real-time (live match) processing