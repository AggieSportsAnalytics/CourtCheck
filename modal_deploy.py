"""
Modal deployment configuration for CourtCheck video processing.

This module sets up Modal functions for processing tennis videos with:
- Court detection using Detectron2
- Ball tracking using TrackNet
- Player detection
- Bounce detection
"""

import modal
import os
from pathlib import Path

# Define the Modal image with all dependencies
# Start from your Docker image with weights, then add Python dependencies
base_image = modal.Image.from_registry(
    "ghcr.io/anikmajumdar/courtcheck-weights:latest",
    add_python="3.10",
)

image = (
    base_image
    .apt_install(
        "git",  # Required for pip install from git repositories
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libglib2.0-0",
        "libgomp1",
        "python3-pip",
        "python3-dev",
        "build-essential",  # Required for building Detectron2
        "clang",  # C++ compiler required by detectron2
    )
    # Install torch and dependencies (required by detectron2)
    .pip_install(
        "wheel",  # Required for --no-build-isolation
        "ninja",  # Optional but speeds up detectron2 build
        "torch==2.1.0",
        "torchvision==0.16.0",
        "torchaudio==2.1.0",
        "opencv-python==4.8.1.78",
        "numpy==1.24.3",
        "matplotlib==3.7.2",
        "tqdm==4.66.1",
        "scipy==1.11.2",
        "pandas==2.0.3",
        "Pillow==10.0.0",
        "scenedetect[opencv]",
    )
    # Install detectron2 using run_commands with --no-build-isolation to use already-installed torch
    .run_commands(
        "python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'",
    )
    # Copy project files to image (copy=True to embed in image for subsequent build steps)
    .add_local_file("court_detection_module.py", "/root/court_detection_module.py", copy=True)
    .add_local_file("ball_detection.py", "/root/ball_detection.py", copy=True)
    .add_local_file("video_processor.py", "/root/video_processor.py", copy=True)
    # Copy CourtCheck directory for TrackNet model
    .add_local_dir("CourtCheck", "/root/CourtCheck", copy=True)
    # Copy weights from Docker image location to expected location
    .run_commands(
        "mkdir -p /models/weights /videos",
        "cp -r /opt/models/* /models/weights/ 2>/dev/null || true",
        "ls -lah /models/weights/",
    )
)

# Create a Modal volume for persistent model storage
models_volume = modal.Volume.from_name("courtcheck-models", create_if_missing=True)

# Create a Modal volume for input/output videos
videos_volume = modal.Volume.from_name("courtcheck-videos", create_if_missing=True)

app = modal.App("courtcheck")


@app.function(
    image=image,
    gpu="A10G",  # Use A10G GPU for inference
    volumes={
        "/models": models_volume,
        "/videos": videos_volume,
    },
    timeout=3600,  # 1 hour timeout
    # secrets=[modal.Secret.from_name("courtcheck-secrets")],  # Uncomment if you need secrets
)
def process_video(
    video_path: str,
    output_path: str,
    model_weights_path: str = "/models/weights",
):
    """
    Process a tennis video and generate analysis output.
    
    Args:
        video_path: Path to input video file (on Modal volume)
        output_path: Path to save output video (on Modal volume)
        model_weights_path: Path to model weights directory
    """
    import sys
    import cv2
    import numpy as np
    import torch
    from pathlib import Path
    
    # Add current directory to path for imports
    sys.path.insert(0, "/root")
    
    # Add CourtCheck to path for TrackNet import
    import os
    courtcheck_path = "/root/CourtCheck"
    if os.path.exists(courtcheck_path):
        sys.path.insert(0, "/root")
    
    from court_detection_module import CourtDetectionProcessor
    from ball_detection import BallDetector
    from video_processor import VideoProcessor
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Model weights path: {model_weights_path}")
    
    # Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize processors
    # Check for court detection model (Detectron2) - optional if using TrackNet-based court detection
    court_model_path = f"{model_weights_path}/court_detection_model.pth"
    if not os.path.exists(court_model_path):
        print(f"Warning: Court detection model not found at {court_model_path}")
        print("Will use TrackNet-based court detection if available")
        court_processor = None
    else:
        court_processor = CourtDetectionProcessor(
            model_path=court_model_path,
            config_path=f"{model_weights_path}/court_detection_config.yaml",
            device=device,
        )
    
    # Ball detector uses TrackNet - should be in /opt/models from Docker image
    tracknet_path = f"{model_weights_path}/tracknet_weights.pt"
    if not os.path.exists(tracknet_path):
        # Try the original Docker image location
        tracknet_path = "/opt/models/tracknet_weights.pt"
    
    ball_detector = BallDetector(
        model_path=tracknet_path,
        device=device,
    )
    
    # Process video
    print("Processing with ball detection" + (" and court detection" if court_processor is not None else " only"))
    processor = VideoProcessor(
        court_processor=court_processor,  # Can be None - VideoProcessor handles this
        ball_detector=ball_detector,
    )
    
    result = processor.process(
        input_path=video_path,
        output_path=output_path,
    )
    
    return result


@app.function(
    image=image,
    volumes={"/models": models_volume},
)
def upload_models(local_models_dir: str):
    """
    Upload model weights to Modal volume.
    
    Args:
        local_models_dir: Local directory containing model weights
    """
    import shutil
    from pathlib import Path
    
    local_path = Path(local_models_dir)
    if not local_path.exists():
        raise ValueError(f"Local models directory not found: {local_models_dir}")
    
    # Copy models to volume
    dest_path = Path("/models/weights")
    dest_path.mkdir(parents=True, exist_ok=True)
    
    for model_file in local_path.rglob("*"):
        if model_file.is_file():
            rel_path = model_file.relative_to(local_path)
            dest_file = dest_path / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_file, dest_file)
            print(f"Copied {model_file} -> {dest_file}")
    
    # Commit the volume
    models_volume.commit()
    print("Models uploaded successfully!")


@app.function(
    image=image,
    volumes={"/videos": videos_volume},
)
def upload_video(local_video_path: str, remote_video_name: str):
    """
    Upload a video file to Modal volume.
    
    Args:
        local_video_path: Local path to video file
        remote_video_name: Name to save video as on Modal volume
    """
    import shutil
    from pathlib import Path
    
    local_path = Path(local_video_path)
    if not local_path.exists():
        raise ValueError(f"Local video file not found: {local_video_path}")
    
    dest_path = Path("/videos") / remote_video_name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(local_path, dest_path)
    print(f"Copied {local_path} -> {dest_path}")
    
    # Commit the volume
    videos_volume.commit()
    print(f"Video uploaded successfully as {remote_video_name}!")
    
    return str(dest_path)


@app.function(
    image=image,
    volumes={"/videos": videos_volume},
)
def download_result(remote_video_path: str, local_output_path: str):
    """
    Download processed video from Modal volume.
    
    Args:
        remote_video_path: Path to video on Modal volume
        local_output_path: Local path to save the video
    """
    import shutil
    from pathlib import Path
    
    remote_path = Path(remote_video_path)
    if not remote_path.exists():
        raise ValueError(f"Remote video file not found: {remote_video_path}")
    
    local_path = Path(local_output_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(remote_path, local_path)
    print(f"Downloaded {remote_path} -> {local_path}")
    
    return str(local_path)


@app.local_entrypoint()
def main():
    """
    Main entry point for local testing.
    """
    print("CourtCheck Modal Deployment")
    print("=" * 50)
    print("\nAvailable functions:")
    print("1. process_video - Process a tennis video")
    print("2. upload_models - Upload model weights to Modal")
    print("3. upload_video - Upload a video file to Modal")
    print("4. download_result - Download processed video")
    print("\nUse: modal run modal_deploy.py::function_name")
