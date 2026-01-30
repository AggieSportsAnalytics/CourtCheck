# 🎾 START HERE - CourtCheck Local Setup

## ✅ Your System is Ready!

All 5 models have been integrated and the system is configured for **local development**.

---

## 🚀 To Start Testing (3 Steps)

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**⏱️ Takes ~10-15 minutes** (PyTorch, Detectron2, CatBoost, etc.)

### Step 2: Start the Application

```bash
python run_local.py
```

**OR** on Windows, double-click: `start.bat`

**Wait for:**
```
✅ CourtCheck is running!
📌 Frontend:  http://localhost:3000
📌 Backend:   http://localhost:8000
```

### Step 3: Test with Your Video

1. Browser auto-opens to `http://localhost:3000`
2. Drag & drop a tennis video
3. Wait 2-5 minutes
4. Download result + view analytics!

---

## 📦 Model Integration Status

All 5 weight files have been integrated:

| # | Model | File | Module | Status |
|---|-------|------|--------|--------|
| 1 | Ball Detection | `tracknet_weights.pt` | `ball_detection.py` | ✅ Ready |
| 2 | Court Detection | `model_tennis_court_det.pt` | `court_detection_module.py` | ✅ Ready |
| 3 | Stroke Classifier | `stroke_classifier_weights.pth` | `stroke_classifier.py` | ✅ NEW |
| 4 | Bounce Detector | `bounce_detection_weights.cbm` | `bounce_detection.py` | ✅ NEW |
| 5 | COCO Metadata | `coco_instances_results.json` | Reference data | ✅ Ready |

---

## 🎯 What Each Model Does (You'll See in Output Video)

### 1. Ball Detection (Yellow Dots)
- Tracks ball position frame-by-frame
- Shows trajectory with yellow trace
- Current position highlighted

### 2. Court Detection (Green/White Lines)
- Identifies court boundaries
- Draws court lines overlay
- 14 keypoints detected

### 3. Stroke Classifier (Magenta Labels)
- Labels appear in top-left corner
- Shows: "Forehand (0.87)" etc.
- Updates each frame

### 4. Bounce Detection (Yellow Circles + "BOUNCE")
- Large yellow circle when ball bounces
- "BOUNCE" text label
- Helps count rallies

### 5. COCO Results
- Metadata for validation
- Not visible in output (reference only)

---

## 📊 Expected Output

### Visual (in processed video):
```
┌────────────────────────────────────────────┐
│  Forehand (0.89)                           │  ← Stroke label
│                                             │
│         ╱─────╲                            │
│        │       │  Court lines              │  ← Court overlay
│        │   •━━━━━━━━○  Ball trace          │  ← Ball trajectory
│        │       │                           │
│         ╲_____╱                            │
│                ⭕ BOUNCE                    │  ← Bounce indicator
└────────────────────────────────────────────┘
```

### Analytics (JSON):
```json
{
  "total_frames": 9000,
  "duration_seconds": 300,
  "fps": 30,
  "ball_detected_frames": 8235,
  "ball_detection_rate": 0.915,
  "total_bounces": 18,
  "stroke_statistics": {
    "Forehand": 9,
    "Backhand": 6,
    "Serve": 2,
    "Volley": 1
  }
}
```

---

## 🎮 Processing Flow

```
Upload Video (Frontend)
    ↓
Save to ./uploads/ (Backend)
    ↓
Start Processing (Background Task)
    │
    ├─→ Pass 1: Ball Detection
    │   └─→ TrackNet processes each frame
    │
    ├─→ Bounce Analysis
    │   └─→ CatBoost analyzes ball trajectory
    │
    └─→ Pass 2: Annotation
        ├─→ Court Detection (per frame)
        ├─→ Stroke Classification (per frame)
        ├─→ Draw ball + trace
        ├─→ Draw bounce indicators
        ├─→ Draw stroke labels
        └─→ Save to ./outputs/
            ↓
Return Analytics to Frontend
    ↓
User Downloads Result
```

---

## 🔧 Quick Commands

### Start Everything
```bash
python run_local.py
```

### Start Backend Only
```bash
python local_backend.py
```

### Test API
```bash
# Health check
curl http://localhost:8000/api/health

# Upload video
curl -X POST -F "file=@test.mp4" http://localhost:8000/api/upload
```

### Check Logs
Backend logs will show:
- Model loading status
- Processing progress
- Any errors

---

## ⚠️ Important Notes

### Model Compatibility

1. **TrackNet**: Updated to match your weights
   - Input: 3 channels (single frame RGB)
   - Output: 15 channels

2. **Court Detection**: Uses your new `model_tennis_court_det.pt`
   - Based on similar architecture to TrackNet
   - Adapted for court keypoint detection

3. **Stroke Classifier**: Loads from checkpoint
   - Expects `model_state` key
   - 6 stroke classes

4. **Bounce Detector**: CatBoost model
   - Requires `catboost` package
   - Falls back to heuristics if not available

### Performance

- **GPU highly recommended** for faster processing
- **CPU mode works** but 3-5x slower
- **Memory**: 4-8GB RAM recommended

---

## 🐛 Troubleshooting

### If run_local.py fails:

**Check Python version:**
```bash
python --version  # Should be 3.10+
```

**Check Node.js:**
```bash
node --version  # Should be 14+
```

**Install missing dependencies:**
```bash
pip install -r requirements.txt
cd frontend && npm install
```

### If models don't load:

**Verify files exist:**
```bash
dir tracknet_weights.pt
dir model_tennis_court_det.pt
dir stroke_classifier_weights.pth
dir bounce_detection_weights.cbm
```

All should be in the root directory (`c:\courtCheck\`).

### If processing is slow:

**Check device:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If False, you're using CPU (slower).

---

## 📚 Documentation

- **START_HERE.md** ← You are here
- **LOCAL_SETUP.md** - Detailed local development guide
- **README.md** - Project overview
- **DEPLOYMENT.md** - Cloud deployment (for later)

---

## 🎯 Test Checklist

Before reporting issues, verify:

- [ ] Python 3.10+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] All 5 weight files in root directory
- [ ] Frontend dependencies installed (`cd frontend && npm install`)
- [ ] Backend starts without errors
- [ ] Can access http://localhost:8000/api/health
- [ ] Frontend opens at http://localhost:3000
- [ ] Can upload a video
- [ ] Processing completes
- [ ] Can download result

---

## 🎊 You're Ready!

Just run:

```bash
python run_local.py
```

Then upload a tennis video at `http://localhost:3000` and watch the magic happen! ✨

---

**Questions?** Check `LOCAL_SETUP.md` for detailed troubleshooting.

**Ready for cloud?** See `DEPLOYMENT.md` when you're ready to deploy to Modal.

<div align="center">
  <h2>🚀 Let's Process Some Tennis Videos! 🎾</h2>
</div>
