import os
import tempfile
import modal
import requests
from typing import Dict
from fastapi import Request, HTTPException
app = modal.App("tennis-modal")

def download_model_weights():
    """Pre-download pretrained weights during image build so they're cached."""
    import torchvision
    torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
    torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libgl1",       # Required for OpenCV
        "libglib2.0-0", # often needed with OpenCV
        "ffmpeg",       # needed for video writing
        "libgomp1",     # OpenMP runtime (required by catboost)
    )
    .pip_install_from_requirements("requirements.txt")
    .run_function(download_model_weights)
    .add_local_python_source("backend")
    .add_local_dir(
        "backend/weights",
        remote_path="/root/backend/weights"
    )
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    secrets=[
        modal.Secret.from_name("supabase-secrets"),
        modal.Secret.from_name("openai-secrets"),
        modal.Secret.from_name("webhook-secret"),
    ]
)
@modal.fastapi_endpoint(method="POST")
async def process_video(request: Request):
    auth_header = request.headers.get("Authorization", "")
    expected_secret = os.environ.get("MODAL_WEBHOOK_SECRET", "")
    if not expected_secret or auth_header != f"Bearer {expected_secret}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    payload = await request.json()
    file_key = payload["file_key"]
    match_id = payload["match_id"]

    from supabase import create_client

    supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )

    signed = supabase.storage.from_("raw-videos").create_signed_url(
        file_key,
        expires_in=3600
    )

    signed_url = signed["signedUrl"]

    # Download video — preserve original extension so OpenCV picks the right decoder
    ext = os.path.splitext(file_key)[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        r = requests.get(signed_url, stream=True)
        r.raise_for_status()
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
        video_path = f.name

    from backend.pipeline.run import run_pipeline
    result = run_pipeline(video_path, match_id)

    return result


@app.local_entrypoint()
def test_local(video: str = "", output: str = "output.mp4"):
    """
    Local test entrypoint — runs the pipeline in local_mode (no Supabase upload).

    Usage:
        modal run backend/app.py --video path/to/video.mp4
        modal run backend/app.py --video path/to/video.mp4 --output result.mp4
    """
    import uuid
    import shutil
    from backend.pipeline.run import run_pipeline

    if not video:
        print("ERROR: Provide a video path with --video path/to/video.mp4")
        return

    match_id = str(uuid.uuid4())
    print(f"\n🎾 Running pipeline locally (local_mode=True)")
    print(f"   Video:    {video}")
    print(f"   Output:   {output}")
    print(f"   Match ID: {match_id}\n")

    result = run_pipeline(video_path=video, match_id=match_id, local_mode=True)

    out_file = result.get("output_file")
    if out_file and __import__("os").path.exists(out_file):
        shutil.copy(out_file, output)
        print(f"\n✅ Done! Output saved to: {output}")
    else:
        print(f"\n⚠️  Pipeline finished but output file not found.")