# Quick Start Guide - Modal Deployment

## 🚀 Fast Track Deployment

### 1. Install Modal
```bash
pip install modal
```

### 2. Authenticate
```bash
modal token new
```

### 3. Deploy
```bash
modal deploy modal_deploy.py
```

### 4. Upload Video
```bash
modal run modal_deploy.py::upload_video --local-video-path "your_video.mp4" --remote-video-name "input.mp4"
```

### 5. Process Video
```bash
modal run modal_deploy.py::process_video --video-path "/videos/input.mp4" --output-path "/videos/output.mp4"
```

### 6. Download Result
```bash
modal run modal_deploy.py::download_result --remote-video-path "/videos/output.mp4" --local-output-path "./result.mp4"
```

## ✅ That's it!

Your processed video will be in `./result.mp4`

For detailed instructions, see `DEPLOYMENT_STEPS.md`
