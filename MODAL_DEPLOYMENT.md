# Modal Deployment Guide for CourtCheck

This guide walks you through deploying CourtCheck to Modal for cloud-based video processing.

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install with `pip install modal`
3. **Model Weights**: You'll need the trained model weights:
   - Court detection model (Detectron2): `court_detection_model.pth`
   - Ball tracking model (TrackNet): `tracknet_weights.pt`
   - Config file: `court_detection_config.yaml` (optional)

## Project Structure

```
courtCheck/
├── modal_deploy.py              # Main Modal deployment file
├── court_detection_module.py     # Court detection using Detectron2
├── ball_detection.py            # Ball tracking using TrackNet
├── video_processor.py            # Main video processing pipeline
├── requirements_modal.txt        # Python dependencies
└── MODAL_DEPLOYMENT.md          # This file
```

## Step 1: Install Modal

```bash
pip install modal
```

## Step 2: Authenticate with Modal

```bash
modal token new
```

This will open a browser window for authentication.

## Step 3: Prepare Model Weights

You mentioned you'll include the models/weights in an image. For now, you can upload them to Modal volumes:

### Option A: Upload via Modal Function (Recommended)

Create a local directory with your model weights:

```
models/
├── weights/
│   ├── court_detection_model.pth
│   ├── tracknet_weights.pt
│   └── court_detection_config.yaml
```

Then upload them:

```bash
modal run modal_deploy.py::upload_models --local-models-dir ./models
```

### Option B: Use Modal Image with Pre-loaded Models

If you have a Docker image with models, you can modify `modal_deploy.py` to use it:

```python
image = (
    modal.Image.from_dockerhub("your-registry/courtcheck-models:latest")
    .pip_install(...)
)
```

## Step 4: Upload a Video

Upload a video file to Modal:

```bash
modal run modal_deploy.py::upload_video \
    --local-video-path ./input_video.mp4 \
    --remote-video-name input_video.mp4
```

## Step 5: Process the Video

Process the uploaded video:

```python
import modal

# Connect to your app
app = modal.App.lookup("courtcheck")

# Process video
result = app.process_video.remote(
    video_path="/videos/input_video.mp4",
    output_path="/videos/output_video.mp4",
    model_weights_path="/models/weights",
)
```

Or use the CLI:

```bash
modal run modal_deploy.py::process_video \
    --video-path /videos/input_video.mp4 \
    --output-path /videos/output_video.mp4 \
    --model-weights-path /models/weights
```

## Step 6: Download Results

Download the processed video:

```bash
modal run modal_deploy.py::download_result \
    --remote-video-path /videos/output_video.mp4 \
    --local-output-path ./output_video.mp4
```

## Using Modal Volumes

Modal volumes provide persistent storage for models and videos:

### List volumes:
```bash
modal volume list
```

### Inspect volume contents:
```bash
modal volume show courtcheck-models
modal volume show courtcheck-videos
```

## Configuration

### GPU Selection

The deployment uses `A10G` GPU by default. You can change this in `modal_deploy.py`:

```python
@app.function(
    image=image,
    gpu="T4",  # or "A10G", "A100", etc.
    ...
)
```

### Timeout

The default timeout is 3600 seconds (1 hour). Adjust if needed:

```python
@app.function(
    timeout=7200,  # 2 hours
    ...
)
```

## Local Testing

You can test the functions locally before deploying:

```python
# Test court detection
from court_detection_module import CourtDetectionProcessor
processor = CourtDetectionProcessor(
    model_path="./models/weights/court_detection_model.pth",
    device="cpu"  # or "cuda" if you have GPU locally
)
```

## Troubleshooting

### Import Errors

If you get import errors, make sure:
1. All dependencies are listed in `requirements_modal.txt`
2. Detectron2 is properly installed (it's installed via git in the Modal image)

### Model Loading Errors

- Check that model paths are correct
- Ensure models are uploaded to Modal volumes
- Verify model file formats match expected formats

### GPU Issues

- Check GPU availability: `modal gpu list`
- Try a different GPU type if A10G is unavailable
- Use CPU for testing: remove `gpu="A10G"` parameter

## Next Steps

1. **Deploy the app**: `modal deploy modal_deploy.py`
2. **Set up API endpoints** (optional): Add web endpoints for video upload/processing
3. **Monitor usage**: Check Modal dashboard for usage and costs
4. **Optimize**: Adjust GPU types, batch sizes, etc. based on performance

## Cost Considerations

- GPU time is billed per second
- A10G: ~$1.10/hour
- T4: ~$0.40/hour
- Storage (volumes): ~$0.10/GB/month

## Support

For issues:
- Modal docs: https://modal.com/docs
- Modal Discord: https://discord.gg/modal
- CourtCheck issues: Check the main README
