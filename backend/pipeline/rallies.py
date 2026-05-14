"""Rally state machine.

Reconstructs structured rallies from the unordered streams of bounces +
swing events + in-bounds set produced earlier in the pipeline. Output
shape and segmentation rules are documented in
`docs/RALLY_TRACKING_SPEC.md`.
"""
from __future__ import annotations

import numpy as np
import cv2

# 4-second gap between consecutive events starts a new rally.
# Shared with `calculate_rally_count` in run.py — keep in one place.
RALLY_GAP_SECONDS = 4

# Plausibility gate copied from build_error_summary. Bounces outside this
# box are detector noise (projection blew up past the stands).
_PLAUSIBLE_SVG_X = (-3.0, 30.0)
_PLAUSIBLE_SVG_Y = (-6.0, 84.0)

# Net zone — bounces at y in [37, 41] are side-ambiguous and labeled "net".
_NET_Y_MIN = 37.0
_NET_Y_MAX = 41.0

# In-bounds court box (matches count_in_out_bounces semantics).
_COURT_X_MIN = 1.0
_COURT_X_MAX = 26.0
_COURT_Y_MIN = 0.0
_COURT_Y_MAX = 78.0

# Stroke label normalization (raw classifier label -> frontend key).
_STROKE_MAP = {
    "Forehand": "forehand",
    "Backhand": "backhand",
    "Serve/Overhead": "serve",
}

# Window (seconds) after a bounce in which the receiver could swing
# without it counting as "no follow-up".
_FOLLOWUP_WINDOW_S = 1.5


def _build_event_stream(
    bounces: set,
    swing_events: list[dict],
    in_bounds_set: set,
) -> list[dict]:
    """Merge bounces and swings into one frame-sorted event list.

    Each event has:
      - kind: "bounce" | "swing"
      - frame: int
      - bounce extras: in_bounds (bool)
      - swing extras: player (1 | 2), stroke (lowercase frontend key)
    """
    events: list[dict] = []
    for f in sorted(bounces):
        events.append({"kind": "bounce", "frame": int(f), "in_bounds": f in in_bounds_set})
    for s in swing_events:
        try:
            pf = int(s.get("peak_frame", -1))
        except (TypeError, ValueError):
            continue
        if pf < 0:
            continue
        try:
            tid = int(s.get("track_id", 0))
        except (TypeError, ValueError):
            continue
        raw_label = s.get("label", "")
        stroke = _STROKE_MAP.get(raw_label, "unknown")
        events.append({
            "kind": "swing",
            "frame": pf,
            "player": 1 if tid > 0 else 2,
            "stroke": stroke,
        })
    events.sort(key=lambda e: e["frame"])
    return events


def _segment_events_into_rallies(events: list[dict], fps: float) -> list[list[dict]]:
    """Split a sorted event list into rally groups by RALLY_GAP_SECONDS.

    A gap > gap_frames between consecutive events starts a new rally.
    """
    if not events:
        return []
    gap_frames = float(fps) * RALLY_GAP_SECONDS
    groups: list[list[dict]] = [[events[0]]]
    for ev in events[1:]:
        if ev["frame"] - groups[-1][-1]["frame"] > gap_frames:
            groups.append([ev])
        else:
            groups[-1].append(ev)
    return groups


def _is_plausible(svg_x: float, svg_y: float) -> bool:
    return (
        _PLAUSIBLE_SVG_X[0] <= svg_x <= _PLAUSIBLE_SVG_X[1]
        and _PLAUSIBLE_SVG_Y[0] <= svg_y <= _PLAUSIBLE_SVG_Y[1]
    )


def _project_bounce_to_svg(
    frame: int,
    ball_track: list,
    homography_matrices: list,
    court_ref,
) -> "tuple[float, float] | None":
    """Project a bounce's ball pixel position through homography into svg
    units. Returns None if any input is missing or projection fails.
    """
    if frame < 0 or frame >= len(ball_track) or frame >= len(homography_matrices):
        return None
    bp = ball_track[frame]
    if bp is None or bp[0] is None or bp[1] is None:
        return None
    H = homography_matrices[frame]
    if H is None:
        return None
    pt = np.array([[[float(bp[0]), float(bp[1])]]], dtype=np.float32)
    try:
        mapped = cv2.perspectiveTransform(pt, H)
    except cv2.error:
        return None
    cx, cy = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
    left_x = court_ref.left_court_line[0][0]
    right_x = court_ref.right_court_line[0][0]
    top_y = court_ref.baseline_top[0][1]
    bottom_y = court_ref.baseline_bottom[0][1]
    span_x = max(1.0, right_x - left_x)
    span_y = max(1.0, bottom_y - top_y)
    svg_x = (cx - left_x) / span_x * 27.0
    svg_y = (cy - top_y) / span_y * 78.0
    return svg_x, svg_y


def _direction_from_coords(svg_x: float, svg_y: float) -> str:
    """Bucket a bounce into long / wide / net.

    Order matters: net zone wins over long/wide because a bounce right at
    the net is ambiguous in pure-coords terms and "net" is the most
    informative label for coaching.
    """
    if _NET_Y_MIN <= svg_y <= _NET_Y_MAX:
        return "net"
    if svg_y < _COURT_Y_MIN or svg_y > _COURT_Y_MAX:
        return "long"
    if svg_x < _COURT_X_MIN or svg_x > _COURT_X_MAX:
        return "wide"
    # Inside the box but flagged OOB upstream — rare rounding case.
    return "wide"


_WINNER_TABLE: dict = {
    "p1_winner": 1,
    "p2_winner": 2,
    "p1_long": 2,
    "p1_wide": 2,
    "p1_net": 2,
    "p2_long": 1,
    "p2_wide": 1,
    "p2_net": 1,
    "p1_missed_return": 2,
    "p2_missed_return": 1,
}


def _winner_from_end_reason(end_reason: str) -> "int | None":
    return _WINNER_TABLE.get(end_reason)


def _last_bounce_in_group(group: list[dict]) -> "dict | None":
    for ev in reversed(group):
        if ev["kind"] == "bounce":
            return ev
    return None


def _last_swing_before(group: list[dict], frame: int) -> "dict | None":
    last = None
    for ev in group:
        if ev["kind"] == "swing" and ev["frame"] <= frame:
            last = ev
    return last


def _has_swing_after(
    group: list[dict],
    frame: int,
    player: int,
    window_frames: int,
) -> bool:
    for ev in group:
        if ev["kind"] != "swing":
            continue
        if ev["player"] != player:
            continue
        if frame < ev["frame"] <= frame + window_frames:
            return True
    return False


def _classify_rally_end(
    group: list[dict],
    ball_track: list,
    homography_matrices: list,
    in_bounds_set: set,
    court_ref,
    fps: float,
) -> str:
    """Apply the end-reason rules from the spec to a rally event group."""
    last_bounce = _last_bounce_in_group(group)
    if last_bounce is None:
        return "unknown"
    bframe = last_bounce["frame"]
    last_swing = _last_swing_before(group, bframe)
    if last_swing is None:
        return "unknown"
    hitter = last_swing["player"]
    other = 2 if hitter == 1 else 1
    proj = _project_bounce_to_svg(bframe, ball_track, homography_matrices, court_ref)
    if proj is None:
        return "unknown"
    svg_x, svg_y = proj
    if not _is_plausible(svg_x, svg_y):
        return "unknown"

    bounce_on_p1_side = svg_y > 39.0
    bounce_on_p2_side = svg_y < 39.0

    # OOB last bounce -> hitter erred.
    if bframe not in in_bounds_set:
        direction = _direction_from_coords(svg_x, svg_y)
        return f"p{hitter}_{direction}"

    # In-bounds + on hitter's own side -> they mishit into their own court.
    if (hitter == 1 and bounce_on_p1_side) or (hitter == 2 and bounce_on_p2_side):
        direction = _direction_from_coords(svg_x, svg_y)
        return f"p{hitter}_{direction}"

    # In-bounds on the other player's side -> did they swing?
    follow_window = int(round(_FOLLOWUP_WINDOW_S * float(fps)))
    receiver_swung = _has_swing_after(group, bframe, other, follow_window)
    if receiver_swung:
        return "unknown"
    # Receiver missed it. Spec convention:
    #   hitter == P1 -> "p1_winner" (P1 placed it well)
    #   hitter == P2 -> "p1_missed_return" (P1 failed to return)
    # This makes acceptance-criterion-3 hold: build_error_summary's
    # missed_return count is the rallies where hitter==P2 and the
    # in-bounds bounce landed on P1's side with no P1 swing follow-up.
    if hitter == 1:
        return "p1_winner"
    return "p1_missed_return"


def _find_server(group: list[dict]) -> "int | None":
    """Return the player whose first swing was a serve, else the player of
    the first swing in the group (best-effort), else None."""
    first_swing = next((e for e in group if e["kind"] == "swing"), None)
    if first_swing is None:
        return None
    explicit_serve = next(
        (e for e in group if e["kind"] == "swing" and e["stroke"] == "serve"),
        None,
    )
    if explicit_serve is not None:
        return explicit_serve["player"]
    return first_swing["player"]


def _build_shot_records(
    group: list[dict],
    ball_track: list,
    homography_matrices: list,
    in_bounds_set: set,
    court_ref,
    fps: float,
) -> list[dict]:
    """For each swing in the group, pair it with the next bounce in the
    group and emit a shot record with court-projected bounce coords."""
    fps_f = float(fps) if fps else 30.0
    shots: list[dict] = []
    swing_indices = [i for i, e in enumerate(group) if e["kind"] == "swing"]
    for idx_pos, idx in enumerate(swing_indices):
        swing = group[idx]
        next_swing_frame = (
            group[swing_indices[idx_pos + 1]]["frame"]
            if idx_pos + 1 < len(swing_indices)
            else None
        )
        bounce_event = None
        for j in range(idx + 1, len(group)):
            cand = group[j]
            if cand["kind"] != "bounce":
                continue
            if next_swing_frame is not None and cand["frame"] >= next_swing_frame:
                break
            bounce_event = cand
            break
        bounce_frame = None
        bounce_x = None
        bounce_y = None
        in_flag = None
        if bounce_event is not None:
            bounce_frame = bounce_event["frame"]
            in_flag = bounce_frame in in_bounds_set
            proj = _project_bounce_to_svg(
                bounce_frame, ball_track, homography_matrices, court_ref
            )
            if proj is not None and _is_plausible(*proj):
                bounce_x = round(proj[0], 2)
                bounce_y = round(proj[1], 2)
        shots.append({
            "frame": swing["frame"],
            "time_s": round(swing["frame"] / fps_f, 2),
            "player": swing["player"],
            "stroke": swing["stroke"],
            "bounce_frame": bounce_frame,
            "bounce_x": bounce_x,
            "bounce_y": bounce_y,
            "in": in_flag,
        })
    return shots


def build_rallies(
    bounces: set,
    ball_track: list,
    homography_matrices: list,
    swing_events: list[dict],
    in_bounds_set: set,
    court_ref,
    fps: float,
) -> list[dict]:
    """Build structured rallies from raw pipeline streams.

    Output shape documented in docs/RALLY_TRACKING_SPEC.md.
    """
    fps_f = float(fps) if fps else 30.0
    events = _build_event_stream(bounces, swing_events, in_bounds_set)
    groups = _segment_events_into_rallies(events, fps_f)
    if not groups:
        return []
    total_frames = max(len(ball_track), len(homography_matrices))
    rallies: list[dict] = []
    for idx, group in enumerate(groups):
        start_frame = group[0]["frame"]
        end_frame = group[-1]["frame"]
        end_reason = _classify_rally_end(
            group, ball_track, homography_matrices, in_bounds_set, court_ref, fps_f
        )
        winner = _winner_from_end_reason(end_reason)
        server = _find_server(group)
        shots = _build_shot_records(
            group, ball_track, homography_matrices, in_bounds_set, court_ref, fps_f
        )
        edge = int(round(fps_f))
        truncated = start_frame < edge or (
            total_frames > 0 and end_frame > total_frames - edge
        )
        rallies.append({
            "rally_idx": idx,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "duration_s": round((end_frame - start_frame) / fps_f, 2),
            "shot_count": len(shots),
            "server": server,
            "winner": winner,
            "end_reason": end_reason,
            "truncated": bool(truncated),
            "shots": shots,
        })
    return rallies


def build_rally_summary(rallies: list[dict]) -> dict:
    """Aggregate rally stats for the match-detail tile rail and Coach card.

    Truncated rallies (clipped by the recording's start/end) keep their
    length in the average/median (still useful for cadence stats) but drop
    from winner counts where the rally end is structurally ambiguous.
    """
    if not rallies:
        return {
            "total": 0,
            "avg_length": 0.0,
            "median_length": 0,
            "p1_wins": 0,
            "p2_wins": 0,
            "p1_win_rate": 0.0,
            "end_reasons": {},
        }
    lengths = sorted(int(r.get("shot_count", 0)) for r in rallies)
    total = len(rallies)
    avg = sum(lengths) / total
    mid = total // 2
    median = lengths[mid] if total % 2 == 1 else (lengths[mid - 1] + lengths[mid]) // 2
    end_reasons: dict = {}
    p1_wins = 0
    p2_wins = 0
    decisive = 0
    for r in rallies:
        reason = r.get("end_reason", "unknown")
        end_reasons[reason] = end_reasons.get(reason, 0) + 1
        if r.get("truncated"):
            continue
        winner = r.get("winner")
        if winner == 1:
            p1_wins += 1
            decisive += 1
        elif winner == 2:
            p2_wins += 1
            decisive += 1
    win_rate = (p1_wins / decisive * 100.0) if decisive else 0.0
    return {
        "total": total,
        "avg_length": round(avg, 1),
        "median_length": int(median),
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "p1_win_rate": round(win_rate, 1),
        "end_reasons": end_reasons,
    }
