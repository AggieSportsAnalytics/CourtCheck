"""
Helper script to upload videos from local machine to Modal volumes.
Run this locally (not on Modal) to upload videos for processing.
"""

import modal
from pathlib import Path
import sys

# Get the app and volume from modal_deploy
app = modal.App.lookup("courtcheck", create_if_missing=False)
videos_volume = modal.Volume.from_name("courtcheck-videos", create_if_missing=True)


def upload_video_to_modal(local_path: str, remote_name: str):
    """
    Upload a video file from local machine to Modal volume.
    
    Args:
        local_path: Path to video file on your local machine
        remote_name: Name to save the video as on Modal (e.g., "my_video.mp4")
    """
    local_path = Path(local_path)
    
    if not local_path.exists():
        raise ValueError(f"Local video file not found: {local_path}")
    
    print(f"Uploading {local_path} to Modal...")
    print(f"File size: {local_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Upload to Modal volume
    with videos_volume.batch_upload() as batch:
        batch.put_file(local_path, f"/{remote_name}")
    
    print(f"✓ Video uploaded successfully as: /videos/{remote_name}")
    print(f"\nNext step - Process the video:")
    print(f'  modal run modal_deploy.py::process_video --video-path "/videos/{remote_name}" --output-path "/videos/{remote_name.replace(".mp4", "_output.mp4")}"')
    
    return f"/videos/{remote_name}"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python upload_video_local.py <local_video_path> <remote_name>")
        print('Example: python upload_video_local.py "C:\\Users\\amaju\\Downloads\\test_video.mp4" "test_video.mp4"')
        sys.exit(1)
    
    local_path = sys.argv[1]
    remote_name = sys.argv[2]
    
    upload_video_to_modal(local_path, remote_name)
