from supabase import create_client
import os
import subprocess
from pathlib import Path

def make_streamable_mp4(input_path: str) -> str:
    """
    Remux MP4 so browsers can stream it (moov atom first).
    Does NOT re-encode frames.
    Falls back to the original file if ffmpeg is not available (e.g. local Windows testing).

    :param input_path: Path to the input mp4 video file.
    """

    input_path = Path(input_path)
    output_path = input_path.with_name(input_path.stem + "_web.mp4")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(input_path),
                "-movflags", "+faststart",
                str(output_path)
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return str(output_path)
    except FileNotFoundError:
        print("⚠️  ffmpeg not found — skipping streamable remux (video still usable locally).")
        return str(input_path)
    except subprocess.CalledProcessError as e:
        print(f"⚠️  ffmpeg remux failed ({e}) — returning original file.")
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
