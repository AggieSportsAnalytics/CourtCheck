# Modal Deployment Steps - Walkthrough

## ✅ Completed Steps

1. ✅ Created Docker image with model weights
2. ✅ Pushed Docker image to `ghcr.io/anikmajumdar/courtcheck-weights:latest`

## 📋 Next Steps

### Step 3: Install Modal CLI

```bash
pip install modal
```

### Step 4: Authenticate with Modal

```bash
modal token new
```

This will:
- Open a browser window
- Ask you to sign in to Modal
- Generate an authentication token

### Step 5: Test Modal Connection

```bash
modal app list
```

You should see an empty list or any existing apps.

### Step 6: Deploy the App

Deploy your CourtCheck app to Modal:

```bash
modal deploy modal_deploy.py
```

This will:
- Build the Modal image (starting from your Docker image)
- Install all Python dependencies
- Copy your code files
- Create the Modal app named "courtcheck"

**Expected output:**
```
✓ Created objects.
  ├── 🔨 Created image courtcheck
  ├── 🔨 Created function process_video
  ├── 🔨 Created function upload_models
  ├── 🔨 Created function upload_video
  └── 🔨 Created function download_result

✓ App deployed to courtcheck
```

### Step 7: Verify Deployment

Check that your app is deployed:

```bash
modal app list
```

You should see `courtcheck` in the list.

### Step 8: Test Video Upload

Upload a test video to Modal:

```bash
modal run modal_deploy.py::upload_video --local-video-path "path/to/your/video.mp4" --remote-video-name "test_video.mp4"
```

**Note:** Replace `path/to/your/video.mp4` with the actual path to your video file.

### Step 9: Process the Video

Process the uploaded video:

```bash
modal run modal_deploy.py::process_video --video-path "/videos/test_video.mp4" --output-path "/videos/output_video.mp4" --model-weights-path "/models/weights"
```

This will:
- Load the models from your Docker image
- Process the video frame by frame
- Generate output with court detection and ball tracking
- Save the result to the videos volume

**Expected output:**
```
Processing video: /videos/test_video.mp4
Output will be saved to: /videos/output_video.mp4
Model weights path: /models/weights
Using device: cuda
Ball detection model loaded from /models/weights/tracknet_weights.pt
Processing complete. Output saved to: /videos/output_video.mp4
```

### Step 10: Download the Result

Download the processed video:

```bash
modal run modal_deploy.py::download_result --remote-video-path "/videos/output_video.mp4" --local-output-path "./output_video.mp4"
```

## 🔧 Troubleshooting

### Issue: "Image build failed"

**Solution:**
- Check that your Docker image is accessible: `docker pull ghcr.io/anikmajumdar/courtcheck-weights:latest`
- Verify the image name is correct in `modal_deploy.py`

### Issue: "Model not found"

**Solution:**
- The weights should be in `/opt/models/` in your Docker image
- They will be copied to `/models/weights/` during image build
- Check the build logs to see if the copy command succeeded

### Issue: "Import errors"

**Solution:**
- Make sure `CourtCheck` directory is copied to the image
- Check that all Python files are in the correct locations
- Review the image build logs

### Issue: "GPU not available"

**Solution:**
- Check GPU availability: `modal gpu list`
- Try a different GPU type (T4 is cheaper): Change `gpu="A10G"` to `gpu="T4"` in `modal_deploy.py`
- Or remove GPU requirement for testing

### Issue: "Timeout"

**Solution:**
- Increase timeout in `modal_deploy.py`: `timeout=7200` (2 hours)
- Process shorter videos first to test

## 📊 Monitoring

### View App Status

```bash
modal app show courtcheck
```

### View Function Logs

```bash
modal app logs courtcheck
```

### Check Volume Contents

```bash
modal volume show courtcheck-videos
modal volume show courtcheck-models
```

## 🎯 Quick Reference

### Common Commands

```bash
# Deploy app
modal deploy modal_deploy.py

# Upload video
modal run modal_deploy.py::upload_video --local-video-path "video.mp4" --remote-video-name "video.mp4"

# Process video
modal run modal_deploy.py::process_video --video-path "/videos/video.mp4" --output-path "/videos/output.mp4"

# Download result
modal run modal_deploy.py::download_result --remote-video-path "/videos/output.mp4" --local-output-path "./output.mp4"

# View logs
modal app logs courtcheck

# List apps
modal app list
```

## 💰 Cost Estimation

- **GPU (A10G)**: ~$1.10/hour
- **Processing time**: ~5-10 minutes for a 10-minute video
- **Estimated cost per video**: ~$0.10-0.20

## 🚀 Next Steps After Deployment

1. **Create API endpoints** (optional): Add web endpoints for easier access
2. **Set up monitoring**: Monitor processing times and costs
3. **Optimize**: Adjust GPU types, batch sizes, etc.
4. **Scale**: Process multiple videos in parallel

## 📝 Notes

- The Docker image contains `tracknet_weights.pt` at `/opt/models/tracknet_weights.pt`
- This will be copied to `/models/weights/` during Modal image build
- If you have a court detection model (Detectron2), you can upload it separately or add it to the Docker image
- Videos are stored in Modal volumes for persistence
