"""Unit tests for rally state machine (backend/pipeline/rallies.py)."""
import numpy as np
import pytest

from backend.pipeline.rallies import (
    RALLY_GAP_SECONDS,
    build_rallies,
    build_rally_summary,
    _build_event_stream,
    _segment_events_into_rallies,
    _project_bounce_to_svg,
    _is_plausible,
    _direction_from_coords,
    _winner_from_end_reason,
)


def _make_swing(peak_frame: int, track_id: int, label: str = "Forehand") -> dict:
    return {"peak_frame": peak_frame, "track_id": track_id, "label": label}


def _make_fake_court_ref():
    """Court reference with the same pixel bounds as CourtReference uses.

    left_x = 286, right_x = 1379 (span 1093)
    top_y  = 561, bottom_y = 2935 (span 2374)
    """
    class _FakeCourtRef:
        left_court_line = [(286, 0)]
        right_court_line = [(1379, 0)]
        baseline_top = [(0, 561)]
        baseline_bottom = [(0, 2935)]
    return _FakeCourtRef()


def _ball_at(frame_to_pos: dict, n_frames: int) -> list:
    """Build a ball_track list of length n_frames with positions at given frames."""
    track: list = [None] * n_frames
    for f, pos in frame_to_pos.items():
        if 0 <= f < n_frames:
            track[f] = pos
    return track


class TestEventStream:
    def test_empty_inputs_returns_empty_list(self):
        events = _build_event_stream(bounces=set(), swing_events=[], in_bounds_set=set())
        assert events == []

    def test_bounces_and_swings_sorted_by_frame(self):
        bounces = {50, 200}
        in_bounds = {50}
        swings = [_make_swing(100, 1, "Serve/Overhead"), _make_swing(180, -1, "Forehand")]
        events = _build_event_stream(bounces=bounces, swing_events=swings, in_bounds_set=in_bounds)
        assert [e["frame"] for e in events] == [50, 100, 180, 200]
        assert events[0] == {"kind": "bounce", "frame": 50, "in_bounds": True}
        assert events[3] == {"kind": "bounce", "frame": 200, "in_bounds": False}
        assert events[1] == {"kind": "swing", "frame": 100, "player": 1, "stroke": "serve"}
        assert events[2] == {"kind": "swing", "frame": 180, "player": 2, "stroke": "forehand"}

    def test_skips_swings_with_negative_peak_frame(self):
        events = _build_event_stream(
            bounces=set(),
            swing_events=[_make_swing(-1, 1, "Forehand"), _make_swing(50, 1, "Forehand")],
            in_bounds_set=set(),
        )
        assert len(events) == 1
        assert events[0]["frame"] == 50

    def test_unknown_stroke_label_falls_through(self):
        events = _build_event_stream(
            bounces=set(),
            swing_events=[_make_swing(10, 1, "")],
            in_bounds_set=set(),
        )
        assert events[0]["stroke"] == "unknown"


class TestSegmentation:
    def test_single_rally_when_no_gap_exceeds_threshold(self):
        events = [
            {"kind": "swing", "frame": 0, "player": 1, "stroke": "serve"},
            {"kind": "bounce", "frame": 30, "in_bounds": True},
            {"kind": "swing", "frame": 60, "player": 2, "stroke": "forehand"},
            {"kind": "bounce", "frame": 90, "in_bounds": True},
        ]
        groups = _segment_events_into_rallies(events, fps=30.0)
        assert len(groups) == 1
        assert len(groups[0]) == 4

    def test_new_rally_when_gap_exceeds_threshold(self):
        # 5-second gap (150 frames @ 30fps) between event 1 and 2 -> 2 rallies.
        events = [
            {"kind": "swing", "frame": 0, "player": 1, "stroke": "serve"},
            {"kind": "swing", "frame": 200, "player": 1, "stroke": "serve"},
            {"kind": "bounce", "frame": 230, "in_bounds": True},
        ]
        groups = _segment_events_into_rallies(events, fps=30.0)
        assert len(groups) == 2
        assert [len(g) for g in groups] == [1, 2]

    def test_empty_events_returns_empty_list(self):
        assert _segment_events_into_rallies([], fps=30.0) == []


class TestProjectBounce:
    def test_returns_none_when_homography_missing(self):
        court_ref = _make_fake_court_ref()
        result = _project_bounce_to_svg(
            frame=50,
            ball_track=[(100.0, 200.0)] * 60,
            homography_matrices=[None] * 60,
            court_ref=court_ref,
        )
        assert result is None

    def test_returns_none_when_ball_missing(self):
        court_ref = _make_fake_court_ref()
        H = np.eye(3, dtype=np.float32)
        result = _project_bounce_to_svg(
            frame=10,
            ball_track=[None] * 20,
            homography_matrices=[H] * 20,
            court_ref=court_ref,
        )
        assert result is None

    def test_returns_svg_coords_when_inputs_valid(self):
        court_ref = _make_fake_court_ref()
        H = np.eye(3, dtype=np.float32)
        # identity homography -> projects pixel (500, 800) to (500, 800).
        # svg_x = (500 - 286) / 1093 * 27 ≈ 5.29
        # svg_y = (800 - 561) / 2374 * 78 ≈ 7.85
        ball_track = _ball_at({2: (500.0, 800.0)}, 5)
        result = _project_bounce_to_svg(
            frame=2,
            ball_track=ball_track,
            homography_matrices=[H] * 5,
            court_ref=court_ref,
        )
        assert result is not None
        svg_x, svg_y = result
        assert svg_x == pytest.approx(5.29, abs=0.1)
        assert svg_y == pytest.approx(7.85, abs=0.1)


class TestPlausibilityGate:
    def test_in_range_accepted(self):
        assert _is_plausible(15.0, 40.0) is True

    def test_out_of_range_rejected(self):
        assert _is_plausible(-10.0, 40.0) is False
        assert _is_plausible(15.0, 90.0) is False


class TestDirectionFromCoords:
    def test_long_when_past_baseline(self):
        assert _direction_from_coords(svg_x=15.0, svg_y=-3.0) == "long"
        assert _direction_from_coords(svg_x=15.0, svg_y=82.0) == "long"

    def test_wide_when_past_sideline(self):
        assert _direction_from_coords(svg_x=0.5, svg_y=20.0) == "wide"
        assert _direction_from_coords(svg_x=29.0, svg_y=20.0) == "wide"

    def test_net_when_in_net_zone(self):
        assert _direction_from_coords(svg_x=15.0, svg_y=39.0) == "net"
        assert _direction_from_coords(svg_x=15.0, svg_y=37.5) == "net"
        assert _direction_from_coords(svg_x=15.0, svg_y=40.5) == "net"


class TestWinnerFromEndReason:
    def test_p1_wins_when_p2_errs(self):
        assert _winner_from_end_reason("p2_long") == 1
        assert _winner_from_end_reason("p2_wide") == 1
        assert _winner_from_end_reason("p2_net") == 1
        assert _winner_from_end_reason("p2_missed_return") == 1
        assert _winner_from_end_reason("p1_winner") == 1

    def test_p2_wins_when_p1_errs(self):
        assert _winner_from_end_reason("p1_long") == 2
        assert _winner_from_end_reason("p1_missed_return") == 2
        assert _winner_from_end_reason("p2_winner") == 2

    def test_unknown_returns_none(self):
        assert _winner_from_end_reason("unknown") is None


class TestBuildRallies:
    def test_empty_inputs_returns_empty_list(self):
        rallies = build_rallies(
            bounces=set(),
            ball_track=[],
            homography_matrices=[],
            swing_events=[],
            in_bounds_set=set(),
            court_ref=_make_fake_court_ref(),
            fps=30.0,
        )
        assert rallies == []

    def test_p1_long_rally_attributed_correctly(self):
        # P1 serves at f=10, bounces in on P2 side at f=40.
        # P2 returns at f=70, bounces in on P1 side at f=100.
        # P1 hits at f=130 and the ball lands LONG (past far baseline) at f=160.
        # Expected: 1 rally, shot_count=3, end_reason=p1_long, winner=2.
        court_ref = _make_fake_court_ref()
        H = np.eye(3, dtype=np.float32)
        n = 200
        # py=420 -> svg_y = (420-561)/2374*78 ≈ -4.6 -> PASSES gate as "long".
        ball_track = _ball_at({40: (700.0, 1000.0), 100: (700.0, 2000.0), 160: (700.0, 420.0)}, n)
        bounces = {40, 100, 160}
        in_bounds = {40, 100}  # last bounce OOB
        swings = [
            _make_swing(10, 1, "Serve/Overhead"),
            _make_swing(70, -1, "Forehand"),
            _make_swing(130, 1, "Forehand"),
        ]
        rallies = build_rallies(
            bounces=bounces,
            ball_track=ball_track,
            homography_matrices=[H] * n,
            swing_events=swings,
            in_bounds_set=in_bounds,
            court_ref=court_ref,
            fps=30.0,
        )
        assert len(rallies) == 1
        r = rallies[0]
        assert r["shot_count"] == 3
        assert r["server"] == 1
        assert r["end_reason"] == "p1_long"
        assert r["winner"] == 2
        assert r["rally_idx"] == 0
        assert "truncated" in r
        strokes = [s["stroke"] for s in r["shots"]]
        assert strokes == ["serve", "forehand", "forehand"]

    def test_two_rallies_when_5_second_gap(self):
        court_ref = _make_fake_court_ref()
        H = np.eye(3, dtype=np.float32)
        n = 600
        ball_track = _ball_at({40: (700.0, 1000.0), 230: (700.0, 1000.0)}, n)
        bounces = {40, 230}
        in_bounds = {40, 230}
        swings = [_make_swing(10, 1, "Serve/Overhead"), _make_swing(200, 1, "Serve/Overhead")]
        rallies = build_rallies(
            bounces=bounces,
            ball_track=ball_track,
            homography_matrices=[H] * n,
            swing_events=swings,
            in_bounds_set=in_bounds,
            court_ref=court_ref,
            fps=30.0,
        )
        assert len(rallies) == 2
        assert rallies[0]["rally_idx"] == 0
        assert rallies[1]["rally_idx"] == 1

    def test_missed_return_when_in_bounds_no_followup_swing(self):
        # P2 serves at f=10, bounces in on P1 side at f=40.
        # No P1 swing follows -> p1_missed_return -> winner=2.
        court_ref = _make_fake_court_ref()
        H = np.eye(3, dtype=np.float32)
        n = 200
        # P1 side -> svg_y > 41 -> py > ~1810. py=2000 -> svg_y ≈ 48.5
        ball_track = _ball_at({40: (700.0, 2000.0)}, n)
        rallies = build_rallies(
            bounces={40},
            ball_track=ball_track,
            homography_matrices=[H] * n,
            swing_events=[_make_swing(10, -1, "Serve/Overhead")],
            in_bounds_set={40},
            court_ref=court_ref,
            fps=30.0,
        )
        assert len(rallies) == 1
        r = rallies[0]
        assert r["end_reason"] == "p1_missed_return"
        assert r["winner"] == 2
        assert r["server"] == 2

    def test_p1_winner_when_p1_hits_in_bounds_no_p2_swing(self):
        # P1 hits at f=10, bounces in on P2 side at f=40, no P2 swing follows.
        court_ref = _make_fake_court_ref()
        H = np.eye(3, dtype=np.float32)
        n = 200
        # P2 side -> svg_y < 39 -> py < ~1748. py=1000 -> svg_y ≈ 18.5
        ball_track = _ball_at({40: (700.0, 1000.0)}, n)
        rallies = build_rallies(
            bounces={40},
            ball_track=ball_track,
            homography_matrices=[H] * n,
            swing_events=[_make_swing(10, 1, "Forehand")],
            in_bounds_set={40},
            court_ref=court_ref,
            fps=30.0,
        )
        assert len(rallies) == 1
        r = rallies[0]
        assert r["end_reason"] == "p1_winner"
        assert r["winner"] == 1

    def test_no_swings_yields_unknown_end_reason(self):
        court_ref = _make_fake_court_ref()
        H = np.eye(3, dtype=np.float32)
        n = 200
        ball_track = _ball_at({40: (700.0, 1000.0)}, n)
        rallies = build_rallies(
            bounces={40},
            ball_track=ball_track,
            homography_matrices=[H] * n,
            swing_events=[],
            in_bounds_set={40},
            court_ref=court_ref,
            fps=30.0,
        )
        assert len(rallies) == 1
        assert rallies[0]["server"] is None
        assert rallies[0]["end_reason"] == "unknown"
        assert rallies[0]["winner"] is None


class TestBuildRallySummary:
    def test_empty_returns_zeroed_summary(self):
        summary = build_rally_summary([])
        assert summary == {
            "total": 0,
            "avg_length": 0.0,
            "median_length": 0,
            "p1_wins": 0,
            "p2_wins": 0,
            "p1_win_rate": 0.0,
            "end_reasons": {},
        }

    def test_aggregates_counts_and_averages(self):
        rallies = [
            {"shot_count": 4, "winner": 1, "end_reason": "p2_long", "truncated": False},
            {"shot_count": 6, "winner": 2, "end_reason": "p1_missed_return", "truncated": False},
            {"shot_count": 2, "winner": 1, "end_reason": "p1_winner", "truncated": False},
            {"shot_count": 10, "winner": 2, "end_reason": "p1_long", "truncated": False},
        ]
        summary = build_rally_summary(rallies)
        assert summary["total"] == 4
        assert summary["avg_length"] == 5.5  # (4+6+2+10)/4
        assert summary["median_length"] == 5  # sorted: 2,4,6,10 -> (4+6)//2 = 5
        assert summary["p1_wins"] == 2
        assert summary["p2_wins"] == 2
        assert summary["p1_win_rate"] == 50.0
        assert summary["end_reasons"] == {
            "p2_long": 1,
            "p1_missed_return": 1,
            "p1_winner": 1,
            "p1_long": 1,
        }

    def test_excludes_truncated_rallies_from_winner_rate(self):
        rallies = [
            {"shot_count": 3, "winner": 1, "end_reason": "p2_long", "truncated": True},
            {"shot_count": 5, "winner": 1, "end_reason": "p1_winner", "truncated": False},
            {"shot_count": 5, "winner": 2, "end_reason": "p1_long", "truncated": False},
        ]
        summary = build_rally_summary(rallies)
        assert summary["total"] == 3
        assert summary["p1_wins"] == 1
        assert summary["p2_wins"] == 1
        assert summary["p1_win_rate"] == 50.0


class TestRallyGapConstant:
    def test_constant_is_four_seconds(self):
        assert RALLY_GAP_SECONDS == 4
