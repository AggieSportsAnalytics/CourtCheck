"""One-shot Modal function: run the FULL pipeline on a clip in local_mode
(no Supabase) on an A10G and return the annotated mp4 bytes. Used to eyeball
rendering changes (e.g. minimap bounce filtering) before deploying.

Usage:
    modal run backend/training/process_clip_modal.py --clip data/bounce_train/clips/court2_pick01.mp4

The clip filename must contain a court token (e.g. "court2") so the pipeline
resolves the UC Davis calibration and runs the calibrated path.
"""
from __future__ import annotations

from pathlib import Path
import sys

import modal

# Reuse the production image so every weight/dep matches the deployed pipeline.
from backend.app import image

app = modal.App("tennis-debug-process")


@app.function(image=image, gpu="A10G", timeout=1800)
def process(clip_bytes: bytes, filename: str) -> bytes:
    import os
    import tempfile
    import uuid
    from backend.pipeline.run import run_pipeline

    # Preserve the court token in the temp filename so _resolve_camera_id works.
    with tempfile.NamedTemporaryFile(suffix=f"_{filename}", delete=False) as tmp:
        tmp.write(clip_bytes)
        path = tmp.name

    print(f"[debug] running full pipeline on {filename} (local_mode)")
    result = run_pipeline(video_path=path, match_id=str(uuid.uuid4()), local_mode=True)

    out = result.get("output_file")
    if not out or not os.path.exists(out):
        raise RuntimeError(f"pipeline produced no output_file (result keys: {list(result)})")
    with open(out, "rb") as f:
        data = f.read()
    print(f"[debug] annotated video {len(data)} bytes | bounces shown filtered to paired set")
    return data


@app.local_entrypoint()
def main(clip: str):
    p = Path(clip).resolve()
    if not p.exists():
        print(f"not found: {p}", file=sys.stderr)
        sys.exit(1)
    data = process.remote(p.read_bytes(), p.name)
    out_path = f"/tmp/{p.stem}_annotated.mp4"
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"[local] wrote {out_path} ({len(data)} bytes)")
