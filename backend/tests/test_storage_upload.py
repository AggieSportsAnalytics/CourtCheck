"""Unit tests for backend/pipeline/storage.py upload orchestration.

Focus: H1 — a failed processed-video upload must raise (so the pipeline marks the
match failed) instead of silently returning results_path=None, which previously let
a match be marked "done" with no playable video. Heatmap uploads stay optional.
"""
import pytest
import httpx

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


def test_upload_retries_transient_then_succeeds(monkeypatch):
    """A transient error (timeout/connection blip) is retried, not fatal."""
    monkeypatch.setattr(storage.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("transient blip")
        return "done"

    assert storage._upload_with_retry(_flaky, "test") == "done"
    assert calls["n"] == 3


def test_upload_does_not_retry_permanent_4xx(monkeypatch):
    """A 413 (file over the project size limit) is permanent — fail on first try."""
    monkeypatch.setattr(storage.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def _too_big():
        calls["n"] += 1
        request = httpx.Request("POST", "http://example/storage")
        response = httpx.Response(413, request=request)
        raise httpx.HTTPStatusError("payload too large", request=request, response=response)

    with pytest.raises(httpx.HTTPStatusError):
        storage._upload_with_retry(_too_big, "test")
    assert calls["n"] == 1
