# CourtCheck Triage — 2026-07-05

Full-repo triage: open-issues inventory, bounded fixes (on branch `triage/2026-07-05`), and escalation writeups for deep issues. No pushes to main. No Modal deploys.

**Branch layout after this session:**
- `backup/working-tree-2026-07-05` — exact snapshot of the ~620 uncommitted lines found on main (P1 tracking fix, minimap net-crossing gate, ball-tracker experiments, reprocess-button fix). Created via `git stash create`; the working tree itself was not touched.
- `triage/2026-07-05` — 3 new commits with this session's bounded fixes (`b5b5e8a`, `87bbed5`, `bafcb7d`). The uncommitted working-tree changes still sit on top, unmodified.

---

## (a) Open-Issues Inventory

| # | Issue | Area | Severity | Class | Notes |
|---|---|---|---|---|---|
| 1 | ~620 lines of validated fixes uncommitted on main; Modal deployed from working tree, not git | process | **Critical** | Escalate (E1) | P1 fix is live on Modal but absent from git — any redeploy from a clean main silently rolls it back. Snapshotted to `backup/working-tree-2026-07-05`. |
| 2 | Shotmap click → video seek mismatch (open issue #3) | pipeline + frontend | High | **Deep** (E2) | Seeks to the bounce-landing frame, not the stroke; nominal-fps drift on VFR sources amplifies. Bounded hardening shipped (see fix log #3). |
| 3 | Scouting report shows fabricated placeholder prose when generation fails | frontend | High (pilot trust) | Bounded | **Fixed** (fix log #1). Coach could previously read fake "inside-out forehand, 80% accuracy" analysis as real. |
| 4 | Scouting report section parser assigns blocks by position; shifts under omitted sections/preambles; header regex ate short prose lines | frontend | Med | Bounded | **Fixed** (fix log #2). |
| 5 | `/players/[id]` shows "Player not found" for players the by-id API returns | frontend | Med | Bounded | **Fixed** (fix log #4). Page did list-and-find against the ownership-filtered list endpoint. |
| 6 | Reprocess button cancels itself ~30s in | frontend API route | High | Fixed in working tree (pre-existing, not mine) | AbortController probe fix in `trigger-process/route.ts` (uncommitted batch). Does not repro locally — needs Vercel **preview** deploy + manual Reprocess test before prod. |
| 7 | Minimap phantom/orphan bounces in-video (open issue #2) | pipeline | High | Addressed in working tree (pre-existing) | Net-crossing gate validated on court2_pick03 (34/49 shown). Awaiting Brian's visual confirm, then commit + `modal deploy`. |
| 8 | P1 tracking dropout after walk-off (open issue #1) | pipeline | High | Resolved, unconfirmed | Two root causes fixed + deployed to Modal 2026-06-25. Awaiting Brian's visual confirm on a fresh process. In the uncommitted batch. |
| 9 | Bounce accuracy | model | — | **Resolved** | Retrained on 307 UC Davis bounces, val recall 0.955, threshold 0.35. Committed (`66dd0a8`, `2e51a06`) + deployed. |
| 10 | Bounce model validated only on Court 2 clips | model | Med | Monitor (E5) | Courts 4/6 contributed train data only. Risk of geometry overfit. |
| 11 | "Missed returns" conflates opponent winners + detector noise into P1 errors | pipeline | Med | **Deep** (E3) | Skews the error summary and the scouting report's Error Patterns section. |
| 12 | Handedness silently defaults to right-handed | pipeline | Med | **Deep** (E4) | Unbound/unset players get FH/BH swapped everywhere if left-handed. |
| 13 | `/match-stats` 404 | frontend | Low | No action needed | **Nothing links to it** — no nav/link site exists in source; the audit hit was a typed URL. Dead demo route `app/match-stats/[gameId]` + unused `'Match Stats'` type member (`types/index.ts:12`) flagged for cleanup, not deleted. |
| 14 | fastapi pinned to 0.136.1 over withdrawn advisory | deps | Low | Bounded | **Fixed** (fix log #5). MAL-2026-4750 was withdrawn as a false positive 2026-05-26. |
| 15 | Diagnostic prints in production pipeline (`[Diag]`, `near_votes`, `[Recovery]`, `[Minimap]`) | pipeline | Low | Defer | Intentionally left in until Brian's visual confirms land. Strip when committing the working-tree batch. |
| 16 | "Scouting report notes" (Brian's 2026-05-21 concern, never specified) | product | ? | Needs Brian | Most plausible reading is the AI report itself — #3/#4/#11 above cover its main trust gaps. The separate manual "Timed notes" panel (`NotesPanel.tsx`) is self-contained and looks healthy. Confirm what Brian meant. |

---

## (b) Fix Log — branch `triage/2026-07-05`

All frontend fixes verified with `tsc --noEmit` (clean) and a production `next build` (clean). Live authed-page verification was blocked: the Playwright audit session (`scripts/audit/storageState.json`) has expired — re-run `node scripts/audit/sign-in.mjs` to re-arm it.

### 1. Scouting report: honest empty state (commit `b5b5e8a`)
`frontend/web/app/recordings/[id]/page.tsx`
- `parseScoutingReport` returned six paragraphs of **fabricated placeholder analysis** whenever `scouting_report` was null — which happens on any backend generation failure (`generate_scouting_report` returns `None` on exception). A coach had no way to tell it was fake.
- Now returns `null`; the page renders an explicit "No scouting report was generated for this recording" card in the same brand shell.
- Tested: parser unit tests (T5) + build. No demo-mode coupling on this page (demo roster/recordings live elsewhere).

### 2. Scouting report: label-matched section parsing (commit `b5b5e8a`)
`frontend/web/app/recordings/[id]/page.tsx`
- Old parser mapped text blocks to the six sections **by position**. The GPT prompt tells the model to omit N/A sections, so one omission shifted every later section under the wrong heading; a preamble line ("Here is the report:") did the same.
- Also found while testing: the header regex accepted **any capitalized line under 60 chars**, including prose sentences — short sentences fragmented sections. Tightened to exclude sentence punctuation from header text.
- New parser captures each header's text and assigns blocks by keyword (`snapshot`, `position`, `error`, `strength`, `improve`, `adjustment|coaching`); unlabeled leftovers fall back to positional fill, flat prose keeps the old blank-line behavior.
- Tested: extracted the real function from the page source and ran 7 unit cases (full 6-section, omitted section, bold headers, preamble, empty→null, structured JSON, flat prose) — all pass. The omitted-section and preamble cases fail on the old code.

### 3. Shotmap seek: `time_s` null no longer seeks to 0:00 (commit `b5b5e8a`)
`frontend/web/components/recordings/VizPanel.tsx`, `frontend/web/app/recordings/[id]/page.tsx`
- `BouncePanel`'s "Play from here" did `shot.time_s ?? 0` — any shot exported while backend `fps` was falsy seeks to the start of the video. Now falls back to `frame / fps` (same `fpsSafe` pattern `RallyTable` already uses), with `fps` threaded from `recording.fps` through `VizPanel`.
- This is hardening only — the deeper semantic/VFR mismatch is escalation E2, not fixed here.
- Tested: `tsc` + build; logic mirrors the proven RallyTable path.

### 4. Player detail: fetch by id (commit `87bbed5`)
`frontend/web/app/players/[id]/page.tsx`
- Page fetched `/api/players` (the **list**) and `.find()`-ed the id client-side. The list endpoint hides demo/template rows (`user_id IS NULL`) from onboarded users, so players the by-id endpoint (`/api/players/[id]`, which allows null-owner rows) happily returns rendered as "Player not found" — the Kaia Wolfe bug from the 2026-05-24 audit.
- Now fetches `/api/players/${id}`; 404 maps to the not-found state. All fields the page uses (name, position, year, photo_url, handedness) are in the by-id select. Demo-mode (`demo-*` ids) path untouched.
- Residual: if Kaia's row is owned by a *different* user_id, both endpoints 404 correctly and it becomes a data/ownership question — check with `select id, name, user_id from players where name ilike 'kaia%'`.

### 5. fastapi unpin (commit `bafcb7d`)
`requirements.txt`
- `fastapi==0.136.1` → `fastapi>=0.137.0`. MAL-2026-4750 (the "malicious" 0.136.3) was **withdrawn as a false positive on 2026-05-26** — `fastar` is a FastAPI-team package. 0.137.0–0.139.0 are clean releases requiring Python ≥3.10; the Modal image is 3.10 and CI is 3.11, so the constraint resolves in both target environments (local system Python 3.9 does not matter — nothing backend runs there).
- Note for next `modal deploy`: image rebuild picks up a new fastapi — smoke the webhook endpoint once.

### Also updated (not in this repo)
- JarvisEA `projects/CourtCheck/README.md`: replaced the frozen "Q2 2026 Goals" block (which still said "PoseConv3D fine-tuned on THETIS" — reversed 2026-05-02) with a current status section: UC-Davis-only TCN strategy, June bounce retrain, P1 fix, minimap gate, reprocess fix, open pre-pilot issues.

---

## (c) Escalations — diagnosis only, no patches

### E1 — Uncommitted working tree is the deployed reality (process, CRITICAL)
**Observed:** ~620 lines across `ball_tracker.py`, `player_tracker.py`, `config.py`, `run.py`, `drawing.py`, `trigger-process/route.ts` are uncommitted on main. The P1 fix among them is already **live on Modal** (deployed 2026-06-25 from the working tree). The batch also contains ball-tracker changes memory doesn't record as validated: velocity-predicted candidate gating, Otsu adaptive threshold on the far-ROI pass, ROI-preferred far-court fusion, `enable_inpaint_net` flipped back to **True** (it was disabled 2026-04-29 for erratic extrapolations — the new comment argues bounces read the raw track so it's safe, but that's exactly the class of change the eval harness should confirm).
**Risk of acting blind:** committing + deploying untested parts ships unvalidated tracking behavior; NOT committing means any teammate push + redeploy silently rolls back the live P1 fix. Both failure modes are invisible until a coach sees bad output.
**Directions:** (1) Brian eyeballs court2_pick01 (P1) + court2_pick03 (minimap) on the current Modal deploy; if good, run `backend/eval/run_eval.py` on the StMarys clip as the regression gate, strip the `[Diag]` prints, commit the batch, deploy from git. (2) If the ball-tracker experiments (velocity gate/Otsu/inpaint flip) weren't part of what Modal is running, split them out and eval them separately before they ride along.

### E2 — Shotmap seek mismatch (open issue #3, deep)
**Observed:** "Play from here" on a bounce dot lands in the wrong place. Full trace:
- Backend stores `frame = bounce frame` and `time_s = bounce_frame / nominal_fps` on each shot (`run.py:888-889`). The refined `contact_frame` (±6 around wrist-velocity peak, ≤7 court-unit gate) is computed but used only for spacing coords, then discarded. So a click seeks to the **ball landing**, ~0.3–1.5s after the stroke the coach expects — consistently late.
- Amplifier: `fps` comes from `CAP_PROP_FPS` (`run.py:2094`) — a nominal average. On variable-frame-rate sources (phone recordings), `frame / nominal_fps` drifts from true presentation time, growing through the clip — matches "don't match up at all."
- Rendered video preserves frame count/fps (writer at source fps, ffmpeg remux without `-r`), so indexing itself is consistent; the 30→ fps bounce-detection subsample maps indices back correctly.
**Why not patch blind:** the right fix depends on intended UX (seek to the stroke? the bounce? a 2s pre-roll?) and on whether pilot footage is VFR — a hardcoded lead offset would mask the semantic question and still drift on VFR.
**Directions:** (1) Persist `peak_frame`/`contact_frame` (+ derived time) on shot records and let the frontend seek to stroke-with-pre-roll — schema addition, backfill-safe since old rows fall back to bounce time. (2) For VFR: normalize renders to CFR (`-r <fps> -vsync cfr` in `make_streamable_mp4`) or export per-frame PTS. Cheap probe first: pull one real recording's `shots[]` and compare `time_s` against the visible bounce in the video to measure constant-lag vs growing-drift — that decides which fix matters.

### E3 — "Missed returns" error attribution (deep)
**Observed:** `build_error_summary` (`run.py:1385-1423`) books any in-bounds near-half bounce P1 didn't swing near as a P1 "missed return." Opponent clean winners and phantom detector bounces both land in that bucket, inflating P1's error count in the Coach Insights tile and the scouting report.
**Why not patch blind:** distinguishing "winner against you" from "you missed a makeable return" needs either rally-context heuristics (ball speed, P1 proximity/movement) or a product decision to rename/drop the bucket. A wrong heuristic misleads coaches differently.
**Directions:** (1) Rename the bucket to "opponent placements / unreturned" so the claim matches the data. (2) Gate it on P1-within-N-court-units of the bounce to keep only plausibly-makeable balls as "missed returns."

### E4 — Handedness silently defaults to right (deep-ish)
**Observed:** `near_player_handedness` falls back to `"right"` when the recording has no bound roster player or the player has no handedness set (`run.py:1791`, `2152-2153`). For an unbound lefty, every FH/BH label — video overlays, shot map colors, stroke counts, scouting report — is silently swapped.
**Why not patch blind:** the fix spans product flow (require player binding at upload? prompt for handedness?) and possibly a pose-based handedness classifier; a backend-only default flip helps nobody.
**Directions:** (1) Surface it: when handedness is defaulted, tag the recording/report "assuming right-handed — set handedness on the player profile" (bounded, one-line UI + one flag in export). (2) Longer term: infer handedness from wrist-keypoint dominance across swings and flag disagreement.

### E5 — Bounce model cross-court validation (monitor)
**Observed:** the 0.955-recall retrain validated on Court 2 clips only; Courts 4/6 were train-only. Geometry overfit risk is real but unmeasured.
**Direction:** cut + annotate one held-out clip each from Courts 4 and 6 (`cut_bounce_clips.py` → `annotate.py` → `train_bounce.py --eval-only`, ~1 evening) before rolling the pilot beyond Court 2 footage.

---

## Verification gaps to close before pilot

1. Re-arm Playwright auth (`node scripts/audit/sign-in.mjs`) and spot-check `/recordings/[id]` + `/players/[id]` with real data (this session's fixes were verified by typecheck + build + parser unit tests; authed visual pass was blocked by the expired session).
2. Reprocess-button fix: Vercel **preview** deploy, then a manual Reprocess on a real recording (memory: `npm run build` does not catch `use server` runtime 500s — never re-point prod without a runtime smoke).
3. Brian's visual confirms for P1 (court2_pick01) and minimap gate (court2_pick03), then E1 commit + deploy-from-git.
