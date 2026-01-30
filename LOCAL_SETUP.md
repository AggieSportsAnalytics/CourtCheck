# CourtCheck - Local Development Setup

Complete guide for running CourtCheck locally (no Modal required).

---

## 📋 Prerequisites

- Python 3.10+
- Node.js 14+
- GPU recommended (CUDA) but CPU works
- 8GB+ RAM
- 10GB+ disk space

---

## 🚀 Quick Start

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This will take 10-15 minutes (includes PyTorch and Detectron2).

### 2. Verify Model Weights

Ensure these files are in the root directory:
- ✅ `tracknet_weights.pt` (Ball detection)
- ✅ `model_tennis_court_det.pt` (Court detection)
- ✅ `stroke_classifier_weights.pth` (Stroke classification)
- ✅ `bounce_detection_weights.cbm` (Bounce detection)
- ✅ `coco_instances_results.json` (COCO metadata)

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Start Everything

```bash
python run_local.py
```

This starts:
- ✅ Backend API at `http://localhost:8000`
- ✅ Frontend at `http://localhost:3000`

### 5. Use the App

1. Open browser to `http://localhost:3000`
2. Drag & drop a tennis match video
3. Wait for processing (~2-5 minutes)
4. View results and download

---

## 🔧 Manual Setup (Alternative)

### Start Backend Only

```bash
python local_backend.py
```

- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

### Start Frontend Only

```bash
cd frontend
npm start
```

- App: `http://localhost:3000`

---

## 📦 Model Details

### 1. TrackNet (Ball Detection)
- **File**: `tracknet_weights.pt`
- **Input**: Single RGB frame (3 channels)
- **Output**: 15-channel heatmap
- **Purpose**: Detect tennis ball position in each frame

### 2. Court Detection (Detectron2)
- **File**: `model_tennis_court_det.pt`
- **Input**: Video frame
- **Output**: Court keypoints (14 points)
- **Purpose**: Detect court boundaries and lines

### 3. Stroke Classifier
- **File**: `stroke_classifier_weights.pth`
- **Classes**: Forehand, Backhand, Serve, Volley, Smash
- **Purpose**: Classify shot types

### 4. Bounce Detector
- **File**: `bounce_detection_weights.cbm`
- **Algorithm**: CatBoost classifier
- **Purpose**: Detect when ball bounces on court

### 5. COCO Results
- **File**: `coco_instances_results.json`
- **Content**: Pre-computed detections for reference
- **Purpose**: Metadata for model validation

---

## 🎯 API Endpoints

Once `local_backend.py` is running, you can use these endpoints:

### Upload Video
```bash
curl -X POST -F "file=@video.mp4" http://localhost:8000/api/upload
```

**Response:**
```json
{
  "video_id": "abc-123-def",
  "filename": "video.mp4",
  "status": "uploaded"
}
```

### Start Processing
```bash
curl -X POST "http://localhost:8000/api/process/{video_id}?filename=video.mp4"
```

### Check Status
```bash
curl http://localhost:8000/api/status/{video_id}
```

**Response (completed):**
```json
{
  "video_id": "abc-123-def",
  "status": "completed",
  "progress": 100,
  "download_url": "/api/download/abc-123-def",
  "result": {
    "total_bounces": 15,
    "ball_detection_rate": 0.92,
    "stroke_statistics": {
      "Forehand": 8,
      "Backhand": 5,
      "Serve": 2
    }
  }
}
```

### Download Result
```bash
curl http://localhost:8000/api/download/{video_id} --output result.mp4
```

Or just visit in browser: `http://localhost:8000/api/download/{video_id}`

---

## 📁 Directory Structure

```
courtCheck/
├── local_backend.py         # ✅ Local FastAPI server
├── run_local.py            # ✅ Start backend + frontend
├── ball_detection.py       # ✅ TrackNet integration
├── court_detection_module.py # ✅ Court detection
├── stroke_classifier.py    # ✅ NEW - Stroke classification
├── bounce_detection.py     # ✅ NEW - Bounce detection
├── video_processor.py      # ✅ Updated - Uses all 5 models
├── requirements.txt        # ✅ Updated - All dependencies
├── uploads/                # Created automatically - Uploaded videos
├── outputs/                # Created automatically - Processed videos
└── frontend/
    ├── src/
    │   └── components/
    │       └── VideoUpload.js  # ✅ Updated - Real API calls
    └── .env                # ✅ Points to localhost:8000
```

---

## 🎮 Processing Pipeline

When you upload a video, here's what happens:

```
1. Frontend Upload
   ├─> POST /api/upload
   └─> Save to ./uploads/

2. Start Processing
   ├─> POST /api/process/{video_id}
   └─> Background task starts

3. Processing Steps (in video_processor.py)
   ├─> Pass 1: Ball Detection (TrackNet)
   │   └─> Detect ball in each frame
   ├─> Bounce Detection (CatBoost)
   │   └─> Analyze ball trajectory
   ├─> Pass 2: Annotation
   │   ├─> Court Detection (Detectron2)
   │   ├─> Stroke Classification (CNN)
   │   ├─> Draw ball trace
   │   ├─> Draw bounce indicators
   │   └─> Draw stroke labels
   └─> Save to ./outputs/

4. Check Status
   ├─> GET /api/status/{video_id}
   └─> Returns progress and results

5. Download Result
   ├─> GET /api/download/{video_id}
   └─> Streams processed video
```

---

## 🧪 Testing

### Test Backend

```bash
# Start backend
python local_backend.py

# In another terminal - health check
curl http://localhost:8000/api/health
```

**Expected:**
```json
{
  "status": "healthy",
  "service": "CourtCheck Local API",
  "device": "cuda"
}
```

### Test Frontend

```bash
cd frontend
npm start
```

Visit `http://localhost:3000` and upload a short test video.

---

## 🐛 Troubleshooting

### Backend won't start

**Issue**: Port 8000 already in use

**Fix**:
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (replace PID)
taskkill /PID <pid> /F
```

### Frontend can't connect

**Fix**: Check `frontend/.env`:
```env
REACT_APP_API_URL=http://localhost:8000
```

### CUDA out of memory

**Fix**: Reduce video resolution or use CPU:
```python
# In local_backend.py
device = "cpu"  # Force CPU mode
```

### Model loading errors

**Fix**: Ensure all weight files are in root directory:
```bash
# List weight files
dir *.pt
dir *.pth
dir *.cbm
```

### Missing catboost

**Fix**:
```bash
pip install catboost
```

---

## 📊 Performance

### Processing Time (Local)

**GPU (CUDA)**:
- Short clip (< 1 min): ~30-60 seconds
- Medium video (1-5 min): ~2-5 minutes
- Long match (> 10 min): ~10-20 minutes

**CPU**:
- 3-5x slower than GPU
- Not recommended for videos > 2 minutes

### Memory Usage
- **Minimum**: 4GB RAM
- **Recommended**: 8GB+ RAM
- **GPU**: 4GB+ VRAM

---

## 🔄 Development Workflow

### Modify Processing

1. Edit `video_processor.py`
2. Restart backend: `CTRL+C` then `python local_backend.py`
3. Upload test video
4. Check results

### Modify Frontend

1. Edit files in `frontend/src/`
2. Webpack auto-reloads (no restart needed)
3. Refresh browser

### Add New Models

1. Add model file to root directory
2. Create module (e.g., `new_model.py`)
3. Update `local_backend.py` to load model
4. Update `video_processor.py` to use model
5. Restart backend

---

## 📈 Optimization Tips

### Faster Processing

1. **Use GPU**: Much faster than CPU
2. **Lower resolution**: Reduce frame size
3. **Skip frames**: Process every 2nd frame
4. **Smaller models**: Use lighter weight models

### Better Accuracy

1. **Higher resolution**: Keep original size
2. **All models**: Enable all 5 models
3. **Fine-tune**: Retrain on your specific videos
4. **Ensemble**: Combine multiple models

---

## 🎯 Next Steps

After local testing works:

1. ✅ Test with various videos
2. ✅ Verify all models working
3. ✅ Check analytics accuracy
4. 🚀 Deploy to Modal (see `DEPLOYMENT.md`)
5. 🌐 Host frontend on Vercel/Netlify

---

## 📝 Summary

**To run locally:**
```bash
# One command to start everything
python run_local.py
```

**Access:**
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

**Features:**
- ✅ Upload videos via web interface
- ✅ Ball detection with TrackNet
- ✅ Court detection with Detectron2
- ✅ Stroke classification
- ✅ Bounce detection
- ✅ Download processed videos
- ✅ View analytics and statistics

**You're ready to test locally! 🎾**
