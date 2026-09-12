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
            print(f"[Storage] encode budget exhausted before {codec}")
            return False
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
        except subprocess.TimeoutExpired:
            print(f"[Storage] ffmpeg ({codec}) hit the {_FFMPEG_BUDGET_SEC}s encode budget")
            return False

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
    endpoint's ~50MB cap or storage3's 20s timeout. Every request's timeout is
    capped by ``deadline`` (monotonic) so the whole upload stays within budget.
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
        created = client.post(
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
        created.raise_for_status()
        location = created.headers["Location"]
        if location.startswith("/"):
            location = f"{url}{location}"

        offset = 0
        with open(local_path, "rb") as f:
            while offset < size:
                chunk = f.read(_TUS_CHUNK_SIZE)
                patched = client.patch(
                    location,
                    headers={
                        **auth,
                        "Tus-Resumable": "1.0.0",
                        "Upload-Offset": str(offset),
                        "Content-Type": "application/offset+octet-stream",
                    },
                    content=chunk,
                    timeout=_timeout(),
                )
                patched.raise_for_status()
                offset = int(patched.headers["Upload-Offset"])


def upload_processed_video(
    local_path: str,
    match_id: str,
    bucket: str = "results"
) -> str:
    remote_path = f"{match_id}/processed.mp4"
    deadline = time.monotonic() + _UPLOAD_BUDGET_SEC
    _upload_with_retry(
        lambda: _upload_resumable(local_path, bucket, remote_path, "video/mp4", deadline),
        label=f"processed video {match_id}",
        deadline=deadline,
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
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        tasks["results_path"] = executor.submit(upload_processed_video, local_video_path, match_id)
        if local_bounce_path and os.path.exists(local_bounce_path):
            tasks["bounce_heatmap_path"] = executor.submit(
                upload_heatmap_png, local_bounce_path, match_id, "bounce_heatmap.png"
            )
        if local_player_path and os.path.exists(local_player_path):
            tasks["player_heatmap_path"] = executor.submit(
                upload_heatmap_png, local_player_path, match_id, "player_heatmap.png"
            )
        if local_shot_map_path and os.path.exists(local_shot_map_path):
            tasks["player_shot_map_path"] = executor.submit(
                upload_heatmap_png, local_shot_map_path, match_id, "player_shot_map.png"
            )

    results = {}
    for key, future in tasks.items():
        try:
            results[key] = future.result()
        except Exception as e:
            print(f"[Storage] Upload failed for {key}: {e}")
            results[key] = None

    # The processed video is the one mandatory artifact — raise rather than let the
    # caller mark the match "done" with no playable video (heatmaps stay optional).
    # We deliberately do NOT sweep the optional uploads here: heatmap keys are
    # deterministic (<match_id>/*.png) with x-upsert, so on a reprocess they may
    # belong to a prior successful run whose row still references them. A first-time
    # failure only leaves small, unreferenced PNGs; safe cleanup needs attempt-
    # specific keys (future work).
    if results.get("results_path") is None:
        raise RuntimeError(f"processed video upload failed for match {match_id}")

    return results
