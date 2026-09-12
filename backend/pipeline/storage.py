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
_STORAGE_TIMEOUT_SEC = 120.0
_FFMPEG_TIMEOUT_SEC = 900  # cap a hung encode instead of running to the Modal wall-clock

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

    def _run_ffmpeg(codec: str) -> bool:
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
                timeout=_FFMPEG_TIMEOUT_SEC,
            )
            return True
        except subprocess.CalledProcessError:
            return False
        except subprocess.TimeoutExpired:
            print(f"[Storage] ffmpeg ({codec}) timed out after {_FFMPEG_TIMEOUT_SEC}s — trying fallback")
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


def _upload_with_retry(upload_fn, label: str):
    """Run an upload with bounded exponential backoff.

    Supabase storage occasionally returns transient 5xx/504 under load; the video
    upload had no retry, so a single blip failed the whole run. Permanent 4xx
    (e.g. 413 when a file exceeds the project's global size limit) are not retried.
    """
    last_exc = None
    for attempt in range(1, _UPLOAD_MAX_ATTEMPTS + 1):
        try:
            return upload_fn()
        except Exception as exc:
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                if 400 <= status < 500 and status not in (408, 429):
                    raise
            last_exc = exc
            if attempt == _UPLOAD_MAX_ATTEMPTS:
                break
            backoff = min(30, 2 ** attempt)
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
    cache_control: str = "3600",
) -> None:
    """Upload a file via Supabase's resumable (TUS) endpoint in 6MB chunks.

    Each chunk is its own request, so no single request approaches the standard
    endpoint's ~50MB cap or storage3's 20s timeout.
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

    with httpx.Client(timeout=httpx.Timeout(_STORAGE_TIMEOUT_SEC)) as client:
        created = client.post(
            f"{url}/storage/v1/upload/resumable",
            headers={
                **auth,
                "Tus-Resumable": "1.0.0",
                "Upload-Length": str(size),
                "Upload-Metadata": metadata,
                "x-upsert": "true",
            },
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
                )
                patched.raise_for_status()
                offset = int(patched.headers["Upload-Offset"])


def upload_processed_video(
    local_path: str,
    match_id: str,
    bucket: str = "results"
) -> str:
    remote_path = f"{match_id}/processed.mp4"
    _upload_with_retry(
        lambda: _upload_resumable(local_path, bucket, remote_path, "video/mp4"),
        label=f"processed video {match_id}",
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

    _upload_with_retry(_do, label=f"{filename} {match_id}")
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
