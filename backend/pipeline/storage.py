from supabase import create_client
import base64
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

# The processed video is uploaded via Supabase's resumable (TUS) endpoint rather
# than the standard /object endpoint, which rejects bodies over ~50MB (413) and is
# subject to storage3's 20s-per-request timeout — a larger clip under load hits one
# or the other and the whole job is marked failed. TUS sends 6MB chunks (each well
# under both limits). Uploads also gained retry, which the download path already had.
_TUS_CHUNK_SIZE = 6 * 1024 * 1024  # Supabase requires 6MB TUS chunks (last may be smaller)
_UPLOAD_MAX_ATTEMPTS = 4
_STORAGE_TIMEOUT_SEC = 120.0  # per-request timeout; also capped by the overall upload budget
_UPLOAD_BUDGET_SEC = 600.0  # overall wall-clock per artifact across retries (< the 1800s job cap)
_FFMPEG_BUDGET_SEC = 900  # overall encode budget shared across codec attempts (< the 1800s job cap)

PROCESSED_VIDEO_UPLOAD_ERROR = "The analyzed video could not be saved. Press Reprocess to try again."


class ProcessedVideoUploadError(RuntimeError):
    """The required video artifact could not be stored."""


def make_streamable_mp4(input_path: str, source_audio_path: str | None = None) -> str:
    """
    Re-encode and remux to a browser-streamable MP4 (moov atom first).

    When ``source_audio_path`` is provided, the audio track from that file is
    muxed into the output so the annotated video keeps the original recording's
    sound. The OpenCV writer used to produce ``input_path`` strips audio, so
    without this the result is silent. Audio mapping uses ``?`` so files with
    no audio track still encode cleanly (video-only output).

    Tries h264_nvenc (GPU, ~2-3 ms/frame) first for maximum throughput on A10G.
    Falls back to libx264 (CPU) when NVENC is unavailable (local dev, CPU instances).
    Falls back to the original file if ffmpeg is not found at all.
    """
    input_path = Path(input_path)
    output_path = input_path.with_suffix("").with_name(input_path.stem + "_web.mp4")
    # One deadline shared across both codec attempts: a hung nvenc followed by a
    # hung libx264 could otherwise each burn the full budget and take the whole job.
    encode_deadline = time.monotonic() + _FFMPEG_BUDGET_SEC

    def _run_ffmpeg(codec: str) -> bool:
        remaining = encode_deadline - time.monotonic()
        if remaining <= 0:
            # Budget spent by a prior attempt: fail rather than fall back to the
            # unencoded original, which would upload as a non-playable processed.mp4.
            raise RuntimeError(f"encode budget exhausted before {codec}")
        cmd = ["ffmpeg", "-y", "-i", str(input_path)]
        if source_audio_path:
            cmd += ["-i", str(source_audio_path)]
        cmd += ["-c:v", codec, "-preset", "fast", "-crf", "23"]
        if source_audio_path:
            # Take video from the annotated input, audio from the source.
            # The ``?`` makes the audio stream optional so source files
            # without an audio track still encode (silent output).
            cmd += [
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:a", "aac",
                "-b:a", "128k",
                "-shortest",
            ]
        else:
            cmd += ["-c:a", "copy"]
        cmd += ["-movflags", "+faststart", str(output_path)]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=remaining,
            )
            return True
        except subprocess.CalledProcessError:
            return False
        except subprocess.TimeoutExpired as exc:
            # A hung/too-slow encode must fail the run, not ship the unencoded
            # original as processed.mp4 (which would be marked done but unplayable).
            raise RuntimeError(
                f"ffmpeg ({codec}) exceeded the {_FFMPEG_BUDGET_SEC}s encode budget"
            ) from exc

    try:
        if _run_ffmpeg("h264_nvenc"):
            print("[Storage] Encoded with h264_nvenc (GPU)")
            return str(output_path)
        print("[Storage] h264_nvenc unavailable — falling back to libx264")
        if _run_ffmpeg("libx264"):
            print("[Storage] Encoded with libx264 (CPU fallback)")
            return str(output_path)
        print("⚠️  Both h264_nvenc and libx264 failed — returning original file.")
        return str(input_path)
    except FileNotFoundError:
        print("⚠️  ffmpeg not found — skipping streamable remux (video still usable locally).")
        return str(input_path)

def get_supabase():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

    if not url or not key:
        raise RuntimeError("Supabase URL or Service Role Key not set.")
    return create_client(url, key)


def _b64(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


def _permanent_status(exc: Exception) -> bool:
    """True for a non-retryable 4xx, across httpx and the storage SDK.

    The video (TUS) path raises ``httpx.HTTPStatusError``; the heatmap path raises
    storage3's ``StorageApiError`` (status on ``.status``). 408/429 stay retryable.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
    else:
        status = getattr(exc, "status", None)
    try:
        status = int(status)
    except (TypeError, ValueError):
        return False
    return 400 <= status < 500 and status not in (408, 429)


def _upload_with_retry(upload_fn, label: str, deadline: float):
    """Run an upload with bounded exponential backoff and an overall time budget.

    Supabase storage occasionally returns transient 5xx/504 under load; the video
    upload had no retry, so a single blip failed the whole run. Retries stop once
    ``deadline`` (monotonic) is reached so a degraded endpoint can't run out the
    Modal job wall-clock. Permanent 4xx (e.g. 413) are not retried.
    """
    last_exc = None
    for attempt in range(1, _UPLOAD_MAX_ATTEMPTS + 1):
        try:
            return upload_fn()
        except Exception as exc:
            if _permanent_status(exc):
                raise
            last_exc = exc
            backoff = min(30, 2 ** attempt)
            if attempt == _UPLOAD_MAX_ATTEMPTS or time.monotonic() + backoff >= deadline:
                break
            print(
                f"[Storage] {label} upload attempt {attempt}/{_UPLOAD_MAX_ATTEMPTS} "
                f"failed ({type(exc).__name__}: {exc}); retrying in {backoff}s"
            )
            time.sleep(backoff)
    raise last_exc


def _tus_offset(client: httpx.Client, location: str, auth: dict, timeout: float) -> int:
    """Return the server's current durable offset for a TUS upload (to resume)."""
    resp = client.head(location, headers={**auth, "Tus-Resumable": "1.0.0", "x-upsert": "true"}, timeout=timeout)
    resp.raise_for_status()
    return int(resp.headers["Upload-Offset"])


def _upload_resumable(
    local_path: str,
    bucket: str,
    remote_path: str,
    content_type: str,
    deadline: float,
    cache_control: str = "3600",
) -> None:
    """Upload a file via Supabase's resumable (TUS) endpoint in 6MB chunks.

    Each chunk is its own request, so no single request approaches the standard
    endpoint's ~50MB cap or storage3's 20s timeout. A transient chunk failure
    resumes from the server's offset instead of restarting the session, and the
    whole upload (create + chunks + retries) is bounded by ``deadline``.
    """
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    size = os.path.getsize(local_path)
    auth = {"Authorization": f"Bearer {key}", "apikey": key}
    metadata = ",".join(
        [
            f"bucketName {_b64(bucket)}",
            f"objectName {_b64(remote_path)}",
            f"contentType {_b64(content_type)}",
            f"cacheControl {_b64(cache_control)}",
        ]
    )

    def _timeout() -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"upload budget exhausted for {remote_path}")
        return min(_STORAGE_TIMEOUT_SEC, remaining)

    with httpx.Client() as client:
        def _create() -> str:
            resp = client.post(
                f"{url}/storage/v1/upload/resumable",
                headers={
                    **auth,
                    "Tus-Resumable": "1.0.0",
                    "Upload-Length": str(size),
                    "Upload-Metadata": metadata,
                    "x-upsert": "true",
                },
                timeout=_timeout(),
            )
            resp.raise_for_status()
            loc = resp.headers["Location"]
            return f"{url}{loc}" if loc.startswith("/") else loc

        # Creating the session is stateless, so a failed create can restart freely.
        location = _upload_with_retry(_create, f"TUS create {remote_path}", deadline)

        offset = 0
        failures = 0
        with open(local_path, "rb") as f:
            while offset < size:
                f.seek(offset)
                chunk = f.read(_TUS_CHUNK_SIZE)
                try:
                    patched = client.patch(
                        location,
                        headers={
                            **auth,
                            "Tus-Resumable": "1.0.0",
                            "Upload-Offset": str(offset),
                            "Content-Type": "application/offset+octet-stream",
                            # Supabase evaluates upsert when the object is finalized, i.e. on the
                            # PATCH that completes the upload. Without it here, reprocessing a
                            # recording whose processed.mp4 already exists fails with 409 Conflict
                            # (observed on production 2026-09-14, match e6ba4268).
                            "x-upsert": "true",
                        },
                        content=chunk,
                        timeout=_timeout(),
                    )
                    patched.raise_for_status()
                    offset = int(patched.headers["Upload-Offset"])
                    failures = 0
                except Exception as exc:
                    if _permanent_status(exc):
                        raise
                    failures += 1
                    backoff = min(30, 2 ** failures)
                    if failures >= _UPLOAD_MAX_ATTEMPTS or time.monotonic() + backoff >= deadline:
                        raise
                    print(f"[Storage] TUS chunk @ {offset} failed ({type(exc).__name__}: {exc}); resuming in {backoff}s")
                    time.sleep(backoff)
                    offset = _tus_offset(client, location, auth, _timeout())


def upload_processed_video(
    local_path: str,
    match_id: str,
    bucket: str = "results"
) -> str:
    remote_path = f"{match_id}/processed.mp4"
    # _upload_resumable retries and resumes internally within this budget.
    _upload_resumable(
        local_path, bucket, remote_path, "video/mp4",
        deadline=time.monotonic() + _UPLOAD_BUDGET_SEC,
    )
    return remote_path


def upload_heatmap_png(
    local_path: str,
    match_id: str,
    filename: str,
    bucket: str = "results"
) -> str:
    """Upload a heatmap PNG to Supabase storage."""
    remote_path = f"{match_id}/{filename}"

    def _do() -> None:
        supabase = get_supabase()
        with open(local_path, "rb") as f:
            supabase.storage.from_(bucket).upload(
                remote_path,
                f,
                file_options={
                    "content-type": "image/png",
                    "cacheControl": "3600",
                    "x-upsert": "true",
                },
            )

    _upload_with_retry(_do, label=f"{filename} {match_id}", deadline=time.monotonic() + _UPLOAD_BUDGET_SEC)
    return remote_path


def upload_results_parallel(
    local_video_path: str,
    match_id: str,
    local_bounce_path: str | None = None,
    local_player_path: str | None = None,
    local_shot_map_path: str | None = None,
) -> dict:
    """
    Upload processed video and heatmaps to Supabase in parallel.
    Returns dict with keys: results_path, bounce_heatmap_path, player_heatmap_path, player_shot_map_path.
    """
    executor = ThreadPoolExecutor(max_workers=4)
    try:
        video_future = executor.submit(upload_processed_video, local_video_path, match_id)
        heatmap_futures: dict = {}
        if local_bounce_path and os.path.exists(local_bounce_path):
            heatmap_futures["bounce_heatmap_path"] = executor.submit(
                upload_heatmap_png, local_bounce_path, match_id, "bounce_heatmap.png"
            )
        if local_player_path and os.path.exists(local_player_path):
            heatmap_futures["player_heatmap_path"] = executor.submit(
                upload_heatmap_png, local_player_path, match_id, "player_heatmap.png"
            )
        if local_shot_map_path and os.path.exists(local_shot_map_path):
            heatmap_futures["player_shot_map_path"] = executor.submit(
                upload_heatmap_png, local_shot_map_path, match_id, "player_shot_map.png"
            )

        # The processed video is the one mandatory artifact. Block on it FIRST so a
        # video failure fails fast (marking the match failed) instead of waiting for
        # optional heatmap retries to burn their budget. Heatmaps upload concurrently;
        # collect them only once the video is safely stored.
        # We deliberately do NOT sweep optional uploads on failure: heatmap keys are
        # deterministic (<match_id>/*.png) with x-upsert, so on a reprocess they may
        # belong to a prior successful run whose row still references them.
        try:
            results = {"results_path": video_future.result()}
        except Exception as e:
            # Coach-readable terminal error; run_pipeline writes it to the row.
            raise ProcessedVideoUploadError(PROCESSED_VIDEO_UPLOAD_ERROR) from e
        if not results["results_path"]:
            raise ProcessedVideoUploadError(PROCESSED_VIDEO_UPLOAD_ERROR)
        for key, future in heatmap_futures.items():
            try:
                results[key] = future.result()
            except Exception as e:
                print(f"[Storage] Upload failed for {key}: {e}")
                results[key] = None
        return results
    finally:
        # Don't block on still-running optional uploads (esp. after a video failure).
        executor.shutdown(wait=False)
