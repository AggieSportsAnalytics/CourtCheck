from supabase import create_client
import os
import subprocess
from pathlib import Path

def make_streamable_mp4(input_path: str) -> str:
    """
    Re-encode and remux to a browser-streamable MP4 (moov atom first).

    Tries h264_nvenc (GPU, ~2-3 ms/frame) first for maximum throughput on A10G.
    Falls back to libx264 (CPU) when NVENC is unavailable (local dev, CPU instances).
    Falls back to the original file if ffmpeg is not found at all.
    """
    input_path = Path(input_path)
    output_path = input_path.with_suffix("").with_name(input_path.stem + "_web.mp4")

    def _run_ffmpeg(codec: str) -> bool:
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(input_path),
                    "-c:v", codec,
                    "-preset", "fast",
                    "-crf", "23",
                    "-c:a", "copy",
                    "-movflags", "+faststart",
                    str(output_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except subprocess.CalledProcessError:
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


def upload_processed_video(
    local_path: str,
    match_id: str,
    bucket: str = "results"
) -> str:
    supabase = get_supabase()
    remote_path = f"{match_id}/processed.mp4"

    with open(local_path, "rb") as f:
        supabase.storage.from_(bucket).upload(
            remote_path,
            f,
            file_options={
                "content-type": "video/mp4",
                "cacheControl": "3600",
                "x-upsert": "true"
            }
        )

    return remote_path


def upload_heatmap_png(
    local_path: str,
    match_id: str,
    filename: str,
    bucket: str = "results"
) -> str:
    """Upload a heatmap PNG to Supabase storage."""
    supabase = get_supabase()
    remote_path = f"{match_id}/{filename}"

    with open(local_path, "rb") as f:
        supabase.storage.from_(bucket).upload(
            remote_path,
            f,
            file_options={
                "content-type": "image/png",
                "cacheControl": "3600",
                "x-upsert": "true"
            }
        )

    return remote_path
