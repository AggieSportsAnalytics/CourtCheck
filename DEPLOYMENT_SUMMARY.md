# CourtCheck Modal Deployment - Summary

## What Has Been Done

### 1. Refactored Court Detection Module
- **File**: `court_detection_module.py`
- Extracted and cleaned up the Detectron2-based court detection from the Colab notebook
- Created a proper Python class `CourtDetectionProcessor` with:
  - Model loading and initialization
  - Keypoint detection
  - Keypoint stabilization (smoothing over frames)
  - Homography matrix computation for 2D transformation
  - Visualization functions

### 2. Created Ball Detection Module
- **File**: `ball_detection.py`
- Wraps TrackNet model for ball tracking
- Handles frame buffering for temporal context
- Post-processing for stable ball detection

### 3. Created Video Processor
- **File**: `video_processor.py`
- Main pipeline that combines court detection and ball tracking
- Handles video I/O
- Generates output videos with visualizations

### 4. Modal Deployment Configuration
- **File**: `modal_deploy.py`
- Complete Modal setup with:
  - Docker image with all dependencies (PyTorch, Detectron2, OpenCV, etc.)
  - GPU support (A10G by default)
  - Persistent volumes for models and videos
  - Functions for:
    - Processing videos
    - Uploading models
    - Uploading videos
    - Downloading results

### 5. Documentation
- **File**: `MODAL_DEPLOYMENT.md`
- Complete walkthrough for deployment
- Troubleshooting guide
- Cost considerations

## Project Structure

```
courtCheck/
├── modal_deploy.py              # Modal deployment configuration
├── court_detection_module.py    # Court detection (Detectron2)
├── ball_detection.py            # Ball tracking (TrackNet)
├── video_processor.py           # Main processing pipeline
├── requirements_modal.txt       # Python dependencies
├── MODAL_DEPLOYMENT.md         # Deployment guide
└── DEPLOYMENT_SUMMARY.md       # This file
```

## Key Changes from Original Code

### Path Refactoring
- **Before**: Hardcoded Google Drive paths (`/content/drive/MyDrive/...`)
- **After**: Configurable paths that work with Modal volumes (`/models`, `/videos`)

### Import Refactoring
- **Before**: Colab-specific imports (`google.colab`, `cv2_imshow`)
- **After**: Standard Python imports with proper error handling

### Model Loading
- **Before**: Assumed models in specific Google Drive locations
- **After**: Flexible model paths via Modal volumes or image-mounted models

### Storage
- **Before**: Output to Google Drive
- **After**: Output to Modal volumes, downloadable via function

## Next Steps for Deployment

### 1. Prepare Model Weights
You mentioned you'll include models in an image. You have two options:

**Option A: Modal Image with Models**
```python
# In modal_deploy.py, modify the image definition:
image = (
    modal.Image.from_dockerhub("your-registry/courtcheck-models:latest")
    .pip_install(...)
)
```

**Option B: Modal Volumes**
```bash
# Upload models to volume
modal run modal_deploy.py::upload_models --local-models-dir ./models
```

### 2. Test Locally (Optional)
Before deploying to Modal, you can test locally:

```python
from court_detection_module import CourtDetectionProcessor
import cv2

processor = CourtDetectionProcessor(
    model_path="./models/court_detection_model.pth",
    device="cpu"  # or "cuda" if available
)

img = cv2.imread("test_frame.jpg")
result = processor.process_frame(img)
```

### 3. Deploy to Modal
```bash
# Install Modal
pip install modal

# Authenticate
modal token new

# Deploy
modal deploy modal_deploy.py
```

### 4. Process a Video
```python
import modal

app = modal.App.lookup("courtcheck")

result = app.process_video.remote(
    video_path="/videos/input.mp4",
    output_path="/videos/output.mp4",
    model_weights_path="/models/weights",
)
```

## Model Requirements

You'll need these model files:

1. **Court Detection Model**
   - File: `court_detection_model.pth` (Detectron2 weights)
   - Optional: `court_detection_config.yaml` (if custom config)

2. **Ball Tracking Model**
   - File: `tracknet_weights.pt` (TrackNet weights)

## Configuration Options

### GPU Selection
In `modal_deploy.py`, you can change the GPU:
```python
gpu="T4"  # Cheaper option
gpu="A10G"  # Default, better performance
gpu="A100"  # Most powerful, most expensive
```

### Timeout
Adjust timeout for long videos:
```python
timeout=7200  # 2 hours
```

## Important Notes

1. **Import Paths**: The code tries multiple import paths for TrackNet to work in both local and Modal environments
2. **Model Paths**: All model paths are configurable via function parameters
3. **Volumes**: Models and videos are stored in Modal volumes for persistence
4. **Dependencies**: All dependencies are specified in the Modal image definition

## Testing Checklist

- [ ] Install Modal CLI
- [ ] Authenticate with Modal
- [ ] Upload model weights
- [ ] Upload test video
- [ ] Process video
- [ ] Download results
- [ ] Verify output quality

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure CourtCheck directory is copied to Modal image
2. **Model Not Found**: Verify model paths and that models are uploaded to volumes
3. **GPU Not Available**: Try different GPU type or remove GPU requirement for testing
4. **Timeout**: Increase timeout for long videos

See `MODAL_DEPLOYMENT.md` for detailed troubleshooting.

## Cost Estimate

- **GPU (A10G)**: ~$1.10/hour
- **Storage**: ~$0.10/GB/month
- **Example**: Processing a 10-minute video might take ~5-10 minutes on GPU = ~$0.10-0.20

## Support

For issues or questions:
1. Check `MODAL_DEPLOYMENT.md` for detailed guide
2. Review Modal documentation: https://modal.com/docs
3. Check Modal Discord: https://discord.gg/modal
