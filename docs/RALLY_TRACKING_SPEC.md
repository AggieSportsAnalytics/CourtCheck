# Rally Tracking — Implementation Spec

*Self-contained scope for a separate Claude session. Read this top-to-bottom before touching code.*

## Goal

Turn the unordered streams of `bounces` + `swing_events` + `in_bounds_set` into a structured list of **rallies**, each with:
- Length (number of shots)
- Who served
- Who won
- Why the rally ended (winner / your-error / opponent-error / missed return)
- Sequence of strokes (so we can mine patterns: "you lose 65% of rallies past 4 shots")

Surface in the Coach Insights panel and the recording-detail page.

## Why now

`build_error_summary` (in `backend/pipeline/run.py`) already detects "missed returns" — that's a one-event subset of rally tracking. The full state machine reuses the same primitives and unlocks:
- Avg rally length (replaces the placeholder `avgRally` calc in the match-detail page)
- Pattern detection ("you keep losing crosscourt-to-crosscourt exchanges")
- Serve effectiveness
- Per-rally jump-to-video timestamps

## Inputs (already produced by the pipeline)

All available in `run_pipeline()` at the point where `build_error_summary` is called:

| Variable | Shape | Source |
|---|---|---|
| `bounces_all` | `set[int]` of frame indices | `backend.vision.bounce_detector` |
| `ball_track` | `list[(x, y) \| None]` per frame | TrackNet |
| `swing_events` | `list[{peak_frame: int, track_id: int, …}]` | `swing_detector` / pose-based |
| `frame_stroke_labels` | `dict[int, dict[int, str]]` — `frame → {track_id → stroke}` | post-TCN classifier |
| `homography_matrices` | `list[np.ndarray \| None]` per frame | court calibration |
| `in_bounds_set` | `set[int]` of frames where bounce was in | `count_in_out_bounces` |
| `court_ref` | `CourtReference` instance | calibration |
| `fps` | float | OpenCV |

**Player attribution:** `track_id > 0` = near player (P1, our coached player). `track_id < 0` = far player (P2, opponent — synthetic ID from the temporal stabilizer).

**Court coords:** project ball position through homography, normalize to `svg_x ∈ [0, 27]`, `svg_y ∈ [0, 78]`. Net at `y=39`. P1 territory = `y > 39`, P2 territory = `y < 39`.

## Output

A new pipeline output `rallies` — list of structured rally objects:

```python
[
  {
    "rally_idx": 0,
    "start_frame": 245,
    "end_frame": 612,
    "duration_s": 12.23,
    "shot_count": 5,
    "server": 1,                       # 1 = P1 served, 2 = P2 served, None = ambiguous
    "winner": 2,                       # 1 = P1 won, 2 = P2 won, None = unknown
    "end_reason": "p1_long",           # see end_reason enum below
    "shots": [
      {
        "frame": 250,
        "time_s": 8.33,
        "player": 1,
        "stroke": "serve",
        "bounce_frame": 268,
        "bounce_x": 18.4,
        "bounce_y": 22.1,
        "in": true
      },
      …
    ]
  },
  …
]
```

### `end_reason` enum
- `"p1_winner"` — P1 hit a clean winner P2 didn't reach (in-bounds bounce on P2 side + no P2 swing within window)
- `"p2_winner"` — symmetric
- `"p1_long"` / `"p1_wide"` / `"p1_net"` — P1's shot went OOB / hit net
- `"p2_long"` / `"p2_wide"` / `"p2_net"` — P2's shot went OOB
- `"p1_missed_return"` — opponent's ball landed in P1 territory and P1 didn't swing within ~1.5s
- `"p2_missed_return"` — symmetric (interesting if we ever surface opponent stats)
- `"unknown"` — fall-through; rally segmentation found a sequence but couldn't infer cause (rare; log + drop from UX)

## Algorithm

### Step 1: Build the time-ordered event stream

Merge bounces and swings into one list, sorted by frame.

```python
events: list[Event] = []
for f in sorted(bounces_all):
    events.append({"kind": "bounce", "frame": f, "in_bounds": f in in_bounds_set})
for s in swing_events:
    pf = int(s.get("peak_frame", -1))
    if pf < 0: continue
    tid = int(s.get("track_id", 0))
    label = frame_stroke_labels.get(pf, {}).get(tid, "unknown")
    events.append({"kind": "swing", "frame": pf, "player": 1 if tid > 0 else 2, "stroke": label})
events.sort(key=lambda e: e["frame"])
```

### Step 2: Segment into rallies by time gap

A gap of > **4 seconds** between consecutive events starts a new rally. Reuse the threshold from `calculate_rally_count()` — keep the constant in one place (`RALLY_GAP_SECONDS = 4`).

### Step 3: Per-rally state machine

For each rally:

1. **Find the serve.** First swing event whose stroke is `"serve"` (or any stroke if no serve detected). Record `server = event["player"]`.
2. **Walk events in order.** For each event:
   - `swing` → record the shot, remember `last_hitter`, project the bounce (if any follows) to court coords
   - `bounce` → pair it with the most recent unpaired swing within ±2s (reuse the strict-sequential pattern from `build_shots()`)
3. **Detect rally end** at the last event:
   - If last bounce is OOB → end_reason = `f"{last_hitter}_long"` / `_wide` / `_net`
   - If last bounce is in-bounds AND it's on the other player's side AND no subsequent swing from that player → end_reason = `f"{last_hitter}_winner"` OR `f"{other_player}_missed_return"`
   - If last bounce is in-bounds on hitter's own side → mishit; `f"{last_hitter}_{long|wide|net}"` based on coords
4. **Set winner** from `end_reason`:
   - `*_winner` or other player `*_long`/`*_wide`/`*_net`/`*_missed_return` → P1 / P2 won (logic table)

### Step 4: Pattern mining (optional, post-MVP)

Aggregate across all rallies:
- Avg / median rally length
- Stroke-sequence n-grams (e.g. "FH-FH-FH-OOB" frequency)
- Win rate by rally length bucket

This goes in a separate function (`build_rally_summary`) that consumes `rallies`. Don't lump into the state machine.

### Step 5: Persist

Two new columns on `matches`:
- `rallies` — JSONB array (full per-rally objects)
- `rally_summary` — JSONB (aggregate stats: avg length, win rates, patterns)

Migration: `supabase/migrations/YYYYMMDD_add_rallies.sql` adding both with `IF NOT EXISTS`.

### Step 6: Surface in the frontend

1. **Match-detail stats tile** — replace placeholder `avgRally` with `rally_summary.avg_length`. Add a new tile: "Rallies won" (`rally_summary.win_count`/`total`).
2. **New "Rallies" tab** under the Coach Insights panel (or its own section). Table:
   - `#` | Server | Length | Pattern (compact stroke sequence) | Outcome (with color) | Timestamp (jump-to-video chip)
3. **Rally drill-down** — click a row → side panel like the existing `BouncePanel`, showing the full shot sequence with bounce dots highlighted on a mini court diagram + "Play from here" → seek video to `start_frame / fps`.

## Acceptance criteria

For an MVP, the implementation passes if:

1. Re-processing the existing UC Davis test clip produces a `rallies` array with `len(rallies) >= 3`.
2. At least one rally has `end_reason == "p1_long"` (we already know there are P1 long misses post-noise-filter — those should bucket into a real rally).
3. The "missed return" count from the existing `build_error_summary` matches the count of rallies with `end_reason == "p1_missed_return"` ± 1 (small drift from rally-segmentation edge cases is OK).
4. Sum of `end_reason ∈ {p1_long, p1_wide, p1_net, p1_missed_return}` ≤ `error_summary.total` (rally-ended errors are a subset of total P1 errors; some errors won't have rally context).
5. Frontend Rallies tab renders without errors, click-to-seek works.

## Edge cases & gotchas

- **Empty `swing_events`.** If the pose-based swing detector returns nothing (low confidence frame), all rally-end attributions default to "unknown". Drop from UX rather than fabricate.
- **Bounce noise.** The plausibility gate I added to `build_error_summary` (`svg_y ∈ [-6, 84]`, `svg_x ∈ [-3, 30]`) should be applied identically here — reuse it as a helper or copy the logic.
- **Far-player synthetic IDs.** `track_id < 0` means "the temporal stabilizer believes this is the far player." Treat all negative IDs as P2 for state-machine purposes.
- **Doubles.** Out of scope. The `track_id > 0` near-player logic collapses doubles into "P1." Note in README; don't try to disambiguate.
- **Serve mis-detection.** TCN sometimes labels a backhand as a serve at frame-edge. If the first swing of a rally is a "serve" by P2 → fine, attribute serve to P2. If the first swing is from a player at non-baseline position, it's probably a mid-rally classification glitch — fall back to no `server`.
- **Rally segmentation at clip boundaries.** First/last rallies in a clip may be truncated. Mark `truncated: bool` in the rally object so the UI can grey them out or hide them.
- **The 4-second gap threshold.** Calibrated empirically. Some long rallies have 5+ second exchanges (drop shots, lobs). Watch for over-segmentation. If it's an issue, tune the constant.

## Estimated effort

- Backend state machine + tests: 3-4 hours
- Migration + DB plumbing: 30 min
- Frontend tile + table + drill-down: 2-3 hours
- Pattern mining (post-MVP): 1-2 hours

**Total v1: ~6 hours.** Don't bundle pattern mining into v1 — ship the rally objects first, then layer aggregates.

## File map

Backend:
- `backend/pipeline/run.py` — add `build_rallies(events, ...)` and `build_rally_summary(rallies)` next to existing `build_error_summary`. Call them inline after the existing error / shots builders.
- (optional) `backend/pipeline/rallies.py` — pull `build_rallies` out into its own module if `run.py` gets too crowded.

Frontend:
- `frontend/web/components/recordings/RallyTable.tsx` — new component for the table
- `frontend/web/components/recordings/CoachInsights.tsx` — add a `Rallies` tab or a `RallyTile`
- `frontend/web/app/api/recordings/[id]/route.ts` — pass `rallies` + `rally_summary` to the client
- `frontend/web/app/recordings/[id]/page.tsx` — wire the new data into the page

Storage:
- `supabase/migrations/YYYYMMDD_add_rallies.sql`

## Out of scope (don't get tempted)

- ❌ Pose-conditioned end-reason refinement (e.g. "she was off-balance when she missed") — needs new model
- ❌ Cross-rally trend analysis ("you missed 4 in a row to the FH side") — post-MVP
- ❌ Doubles support
- ❌ Live (streaming) rally tracking — batch-only is fine

## How to verify when done

Re-process the existing test clip (`02c40ec3-...` or `cf66f1ea-...`), then:

```sql
select id, jsonb_array_length(rallies) as n_rallies,
       rally_summary->>'avg_length' as avg_len,
       rally_summary->>'p1_win_rate' as win_rate
from matches where id = '<recording_id>';
```

Expect `n_rallies >= 3` and reasonable values. Eyeball one rally's `shots[]` against the actual video — frames should be sequential, strokes should match what you see.

---

*Spec author: Claude (Brian's session, 2026-05-14). Read this top-to-bottom before writing a single line.*
