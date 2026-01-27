# CourtCheck - Complete Deployment Guide

A comprehensive guide for deploying CourtCheck's tennis video analysis platform using Modal.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Backend Deployment (Modal)](#backend-deployment-modal)
5. [Frontend Setup](#frontend-setup)
6. [Usage](#usage)
7. [Troubleshooting](#troubleshooting)
8. [Cost Considerations](#cost-considerations)

---

## Overview

CourtCheck uses:
- **Backend**: Modal (serverless GPU compute) for video processing
- **Frontend**: React web app for user interface
- **Models**: TrackNet (ball detection) pre-packaged in Docker image
- **Storage**: Modal volumes for video input/output

**Architecture**:
```
Frontend (React) → Modal Web API → GPU Processing → Results
```

---

## Prerequisites

### Required
- Python 3.10+
- Node.js 14+ and npm
- Modal account (free tier available at [modal.com](https://modal.com))
- Docker (for model weights image)
- GitHub Container Registry access (for model weights)

### Install Dependencies

```bash
# Backend (Python)
pip install modal

# Frontend (Node.js)
cd frontend
npm install
```

---

## Quick Start

### 1. Setup Modal

```bash
# Install Modal
pip install modal

# Authenticate (opens browser)
modal token new

# Verify setup
modal token set --token-id <your-token-id> --token-secret <your-secret>
```

### 2. Deploy Backend

```bash
# Deploy main processing app
modal deploy modal_deploy.py

# Deploy web API for frontend
modal deploy modal_web_api.py
```

**Expected Output:**
```
✓ App deployed! 🎉
View Deployment: https://modal.com/apps/[your-username]/main/deployed/courtcheck-web
```

Save the deployment URL - you'll need it for the frontend.

### 3. Setup Frontend

```bash
cd frontend

# Create .env file
echo "REACT_APP_API_URL=https://[your-modal-url].modal.run" > .env

# Start development server
npm start
```

The app will open at `http://localhost:3000`.

---

## Backend Deployment (Modal)

### Architecture

The backend consists of two Modal apps:

1. **`courtcheck`** (`modal_deploy.py`): Core video processing
2. **`courtcheck-web`** (`modal_web_api.py`): Web API for frontend

### Deployment Steps

#### Step 1: Prepare Model Weights

Your TrackNet weights are pre-packaged in a Docker image at:
```
ghcr.io/anikmajumdar/courtcheck-weights:latest
```

Verify the image is public or authenticate Docker:
```bash
docker login ghcr.io
```

#### Step 2: Deploy Processing Backend

```bash
modal deploy modal_deploy.py
```

This creates:
- ✅ GPU-accelerated video processing function
- ✅ Modal volumes for video storage
- ✅ Detectron2 + TrackNet models loaded

#### Step 3: Deploy Web API

```bash
modal deploy modal_web_api.py
```

This creates:
- ✅ `/api/upload` - Upload videos
- ✅ `/api/process/{video_id}` - Start processing
- ✅ `/api/status/{video_id}` - Check status
- ✅ `/api/download/{video_id}` - Download results
- ✅ `/api/health` - Health check

#### Step 4: Get API URL

After deployment, Modal provides a URL like:
```
https://[your-username]--courtcheck-web-fastapi-app.modal.run
```

Save this URL for frontend configuration.

### Testing Backend

Test the API with curl:

```bash
# Health check
curl https://[your-modal-url].modal.run/api/health

# Upload video
curl -X POST -F "file=@video.mp4" https://[your-modal-url].modal.run/api/upload

# Check status (use video_id from upload response)
curl https://[your-modal-url].modal.run/api/status/[video-id]
```

---

## Frontend Setup

### Configuration

1. **Set API URL** in `frontend/.env`:
   ```env
   REACT_APP_API_URL=https://[your-modal-url].modal.run
   ```

2. **Install Dependencies**:
   ```bash
   cd frontend
   npm install
   ```

3. **Development Mode**:
   ```bash
   npm start
   ```
   Opens at `http://localhost:3000`

4. **Production Build**:
   ```bash
   npm run build
   ```
   Creates optimized build in `frontend/build/`

### Frontend Features

- **Drag & Drop Upload**: Upload tennis match videos
- **Real-time Progress**: Live upload and processing status
- **Dashboard**: View processed videos and statistics
- **Analytics**: Heat maps, shot analysis, player tracking

### Deployment Options

#### Option 1: Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy from frontend directory
cd frontend
vercel

# Set environment variable
vercel env add REACT_APP_API_URL
```

#### Option 2: Netlify

```bash
# Install Netlify CLI
npm i -g netlify-cli

# Deploy
cd frontend
netlify deploy --prod

# Set environment
netlify env:set REACT_APP_API_URL https://[your-modal-url].modal.run
```

#### Option 3: Modal (Static Hosting)

Deploy frontend as static files on Modal:

```python
# Add to modal_web_api.py
from modal import Mount

frontend_mount = Mount.from_local_dir("frontend/build", remote_path="/frontend")

@app.function(mounts=[frontend_mount])
@modal.asgi_app()
def serve_frontend():
    from fastapi.staticfiles import StaticFiles
    web_app.mount("/", StaticFiles(directory="/frontend", html=True))
    return web_app
```

---

## Usage

### Via Web Interface

1. Open `http://localhost:3000` (or your deployed URL)
2. Drag & drop a tennis match video (MP4, MOV, AVI)
3. Wait for processing (progress shown)
4. View results and download processed video

### Via CLI (Advanced)

#### Upload Video:
```bash
python upload_video_local.py "path/to/video.mp4" "my_video.mp4"
```

#### Process Video:
```bash
modal run modal_deploy.py::process_video \
  --video-path "/videos/my_video.mp4" \
  --output-path "/videos/my_video_output.mp4"
```

#### Download Result:
```bash
modal run modal_deploy.py::download_result \
  --remote-video-path "/videos/my_video_output.mp4" \
  --local-output-path "output_video.mp4"
```

---

## Troubleshooting

### Common Issues

#### 1. **"cannot mount volume on non-empty path"**
**Fix**: Models volume removed (weights are in Docker image)
```python
# Correct (already fixed):
volumes={"/videos": videos_volume}  # No /models mount
```

#### 2. **"Model architecture mismatch"**
**Fix**: Already updated to match checkpoint:
```python
self.model = BallTrackerNet(input_channels=3, out_channels=15)
```

#### 3. **"CORS error" in frontend**
**Fix**: CORS is enabled in `modal_web_api.py`:
```python
allow_origins=["*"]  # For development
```
For production, specify your domain:
```python
allow_origins=["https://yourapp.com"]
```

#### 4. **Frontend can't connect to API**
Check:
- ✅ `.env` file has correct `REACT_APP_API_URL`
- ✅ Modal deployment is running: `modal app list`
- ✅ Health check works: `curl [api-url]/api/health`

#### 5. **Video processing fails**
Check Modal logs:
```bash
modal app logs courtcheck-web
```

Look for:
- Model loading errors
- Out of memory issues (upgrade GPU)
- Video format compatibility

### Debug Commands

```bash
# List deployed apps
modal app list

# View logs
modal app logs courtcheck-web

# Check volumes
modal volume list

# Restart app
modal app stop courtcheck-web
modal deploy modal_web_api.py
```

---

## Cost Considerations

### Modal Pricing

**Compute**:
- A10G GPU: ~$1.10/hour
- Processing time: ~2-5 minutes per video
- **Estimated cost**: $0.04-$0.10 per video

**Storage** (Modal Volumes):
- $0.10/GB/month
- Videos compressed after processing

**Free Tier**:
- $30/month free credits
- ~300-750 video processing runs/month

### Optimization Tips

1. **Delete processed videos**: 
   ```python
   os.remove(f"/videos/{video_id}_output.mp4")
   ```

2. **Use smaller GPU for shorter videos**:
   ```python
   @app.function(gpu="T4")  # Cheaper option
   ```

3. **Compress videos before upload**

4. **Batch process multiple videos**

---

## Next Steps

### Add Features

1. **Real-time notifications**: Add websockets for live progress
2. **Database**: Store video metadata in PostgreSQL/MongoDB
3. **User authentication**: Add login/signup
4. **Multiple court types**: Support different court configurations
5. **Advanced analytics**: Shot classification, player heatmaps

### Scale

1. **CDN**: Use Cloudflare/CloudFront for video delivery
2. **Caching**: Cache processed results
3. **Queue system**: Use Redis for job queue
4. **Multi-region**: Deploy Modal functions in multiple regions

---

## Support

- **Modal Docs**: https://modal.com/docs
- **GitHub Issues**: [Your repo URL]
- **Email**: [Your support email]

---

## Summary

✅ Backend deployed on Modal with GPU processing  
✅ Frontend React app with video upload  
✅ Real-time processing status  
✅ Download processed videos  
✅ Cost-effective serverless architecture

**You're ready to process tennis videos! 🎾**
