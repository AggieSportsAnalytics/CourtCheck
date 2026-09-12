"""Unit tests for backend/pipeline/storage.py upload orchestration.

Focus: H1 — a failed processed-video upload must raise (so the pipeline marks the
match failed) instead of silently returning results_path=None, which previously let
a match be marked "done" with no playable video. Heatmap uploads stay optional.
"""
import pytest

from backend.pipeline import storage


def test_video_upload_failure_raises(monkeypatch):
    """If the processed video fails to upload, the call must raise."""

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated video upload failure")

    monkeypatch.setattr(storage, "upload_processed_video", _boom)

    with pytest.raises(RuntimeError):
        storage.upload_results_parallel(
            local_video_path="/tmp/does_not_matter.mp4",
            match_id="match-123",
        )


def test_heatmap_failure_is_tolerated(monkeypatch, tmp_path):
    """A failed heatmap upload is non-fatal: video path returned, heatmap key None."""
    monkeypatch.setattr(
        storage, "upload_processed_video", lambda *a, **k: "match-123/processed.mp4"
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated heatmap upload failure")

    monkeypatch.setattr(storage, "upload_heatmap_png", _boom)

    bounce = tmp_path / "bounce.png"
    bounce.write_bytes(b"x")

    result = storage.upload_results_parallel(
        local_video_path="/tmp/does_not_matter.mp4",
        match_id="match-123",
        local_bounce_path=str(bounce),
    )

    assert result["results_path"] == "match-123/processed.mp4"
    assert result["bounce_heatmap_path"] is None


def test_all_uploads_succeed(monkeypatch, tmp_path):
    """Happy path: video + heatmaps all return their remote paths."""
    monkeypatch.setattr(
        storage, "upload_processed_video", lambda *a, **k: "match-123/processed.mp4"
    )
    monkeypatch.setattr(
        storage,
        "upload_heatmap_png",
        lambda local_path, match_id, filename, bucket="results": f"{match_id}/{filename}",
    )

    bounce = tmp_path / "bounce.png"
    bounce.write_bytes(b"x")
    player = tmp_path / "player.png"
    player.write_bytes(b"x")

    result = storage.upload_results_parallel(
        local_video_path="/tmp/does_not_matter.mp4",
        match_id="match-123",
        local_bounce_path=str(bounce),
        local_player_path=str(player),
    )

    assert result["results_path"] == "match-123/processed.mp4"
    assert result["bounce_heatmap_path"] == "match-123/bounce_heatmap.png"
    assert result["player_heatmap_path"] == "match-123/player_heatmap.png"


def test_video_failure_sweeps_orphaned_heatmaps(monkeypatch, tmp_path):
    """#5: a video-upload failure after a heatmap uploaded must sweep the orphan."""

    def _fail_video(*args, **kwargs):
        raise RuntimeError("simulated video upload failure")

    monkeypatch.setattr(storage, "upload_processed_video", _fail_video)
    monkeypatch.setattr(
        storage,
        "upload_heatmap_png",
        lambda local_path, match_id, filename, bucket="results": f"{match_id}/{filename}",
    )

    swept = {}

    class _FakeBucket:
        def remove(self, paths):
            swept["paths"] = list(paths)

    class _FakeStorage:
        def from_(self, bucket):
            swept["bucket"] = bucket
            return _FakeBucket()

    class _FakeClient:
        storage = _FakeStorage()

    monkeypatch.setattr(storage, "get_supabase", lambda: _FakeClient())

    bounce = tmp_path / "bounce.png"
    bounce.write_bytes(b"x")

    with pytest.raises(RuntimeError):
        storage.upload_results_parallel(
            local_video_path="/tmp/does_not_matter.mp4",
            match_id="match-123",
            local_bounce_path=str(bounce),
        )

    assert swept["bucket"] == "results"
    assert swept["paths"] == ["match-123/bounce_heatmap.png"]
