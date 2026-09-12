"""Unit tests for backend/tools/reap_stuck_processing.py.

Focus: H2 — the stuck-'processing' reaper's decision logic. Only rows older than
the age window are reaped; a missing/blank/unparseable timestamp is never reaped;
Supabase 'Z'-suffixed and offset timestamps parse correctly.
"""
from datetime import datetime, timedelta, timezone

from backend.tools.reap_stuck_processing import is_stuck, _parse_ts


NOW = datetime(2026, 9, 11, 12, 0, 0, tzinfo=timezone.utc)


def _ago(minutes: int) -> str:
    return (NOW - timedelta(minutes=minutes)).isoformat()


def test_fresh_row_is_not_stuck():
    assert is_stuck(_ago(5), NOW, max_age_minutes=45) is False


def test_old_row_is_stuck():
    assert is_stuck(_ago(60), NOW, max_age_minutes=45) is True


def test_boundary_is_not_stuck():
    # Exactly at the window is not "older than" the window.
    assert is_stuck(_ago(45), NOW, max_age_minutes=45) is False


def test_missing_timestamp_is_not_stuck():
    assert is_stuck(None, NOW, max_age_minutes=45) is False
    assert is_stuck("", NOW, max_age_minutes=45) is False


def test_unparseable_timestamp_is_not_stuck():
    assert is_stuck("not-a-timestamp", NOW, max_age_minutes=45) is False


def test_parse_ts_handles_z_suffix():
    assert _parse_ts("2026-09-11T11:00:00Z") == datetime(
        2026, 9, 11, 11, 0, 0, tzinfo=timezone.utc
    )


def test_parse_ts_handles_offset_and_micros():
    dt = _parse_ts("2026-09-11T11:00:00.123456+00:00")
    assert dt is not None
    assert dt.tzinfo is not None


def test_naive_timestamp_assumed_utc():
    assert _parse_ts("2026-09-11T11:00:00") == datetime(
        2026, 9, 11, 11, 0, 0, tzinfo=timezone.utc
    )
