"""Unit tests for backend/tools/reap_stuck_processing.py.

Focus: H2 — the stuck-'processing' reaper's decision logic. Only rows older than
the age window are reaped; a missing/blank/unparseable timestamp is never reaped;
Supabase 'Z'-suffixed and offset timestamps parse correctly.
"""
from datetime import datetime, timedelta, timezone

import backend.tools.reap_stuck_processing as reap_module
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


# --- reap() coverage with a mocked Supabase client -------------------------


def _row(mid, minutes_old, results_path=None, name="clip"):
    created = (datetime.now(timezone.utc) - timedelta(minutes=minutes_old)).isoformat()
    return {
        "id": mid,
        "name": name,
        "status": "processing",
        "created_at": created,
        "progress": 0.5,
        "results_path": results_path,
    }


class _FakeResp:
    def __init__(self, data):
        self.data = data


class _FakeSelect:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def in_(self, *a, **k):
        return self

    def execute(self):
        return _FakeResp(list(self._rows))


class _FakeUpdate:
    def __init__(self, table, payload):
        self._table = table
        self._payload = payload
        self._filters = {}

    def eq(self, col, val):
        self._filters[col] = val
        return self

    def execute(self):
        self._table.updates.append(
            {"payload": self._payload, "filters": dict(self._filters)}
        )
        return _FakeResp(self._table.update_result)


class _FakeTable:
    def __init__(self, rows, update_result):
        self._rows = rows
        self.update_result = update_result
        self.updates = []

    def select(self, *a, **k):
        return _FakeSelect(self._rows)

    def update(self, payload):
        return _FakeUpdate(self, payload)


class _FakeSupabase:
    def __init__(self, rows, update_result):
        self._table = _FakeTable(rows, update_result)

    def table(self, name):
        return self._table


def _patch(monkeypatch, rows, update_result=None):
    # update_result defaults to one affected row (a successful guarded update).
    fake = _FakeSupabase(rows, [{"id": "x"}] if update_result is None else update_result)
    monkeypatch.setattr(reap_module, "_get_supabase", lambda: fake)
    return fake


def test_reap_dry_run_does_not_write(monkeypatch):
    fake = _patch(monkeypatch, [_row("a", 90)])
    reap_module.reap(match_ids=None, apply=False, max_age_minutes=45)
    assert fake._table.updates == []


def test_reap_apply_marks_failed_with_status_guard(monkeypatch):
    fake = _patch(monkeypatch, [_row("a", 90)])
    reap_module.reap(match_ids=None, apply=True, max_age_minutes=45)
    assert len(fake._table.updates) == 1
    u = fake._table.updates[0]
    assert u["payload"]["status"] == "failed"
    assert u["filters"] == {"id": "a", "status": "processing"}


def test_reap_skips_fresh_rows(monkeypatch):
    fake = _patch(monkeypatch, [_row("a", 5)])
    reap_module.reap(match_ids=None, apply=True, max_age_minutes=45)
    assert fake._table.updates == []


def test_reap_skips_reprocess_rows_with_results_path(monkeypatch):
    fake = _patch(monkeypatch, [_row("a", 90, results_path="a/processed.mp4")])
    reap_module.reap(match_ids=None, apply=True, max_age_minutes=45)
    assert fake._table.updates == []


def test_reap_match_id_still_respects_age(monkeypatch):
    fake = _patch(monkeypatch, [_row("a", 5)])
    reap_module.reap(match_ids=["a"], apply=True, max_age_minutes=45)
    assert fake._table.updates == []


def test_reap_zero_row_race_counts_as_skipped(monkeypatch, capsys):
    fake = _patch(monkeypatch, [_row("a", 90)], update_result=[])
    reap_module.reap(match_ids=None, apply=True, max_age_minutes=45)
    out = capsys.readouterr().out
    assert "skipped (no longer processing)" in out
    assert "Reaped 0" in out
    assert len(fake._table.updates) == 1
