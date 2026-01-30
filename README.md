# CourtCheck

<div align="center">
  <h1>🎾 CourtCheck</h1>
  <h3>AI-Powered Tennis Match Analysis</h3>
  
  <p>
    <strong>Upload a tennis video → Get ball tracking, bounce detection, stroke classification, and more!</strong>
  </p>
</div>

---

## 🚀 Quick Start (Local)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start the application
python run_local.py
```

**That's it!** The app will open at `http://localhost:3000`.

Upload a tennis video and watch the AI analyze it in real-time! 🎾

---

## ✨ Features

### 🎯 Core Detection
- **Ball Tracking** - TrackNet deep learning model tracks ball position frame-by-frame
- **Court Detection** - Detectron2 identifies court boundaries and keypoints
- **Bounce Detection** - CatBoost ML model detects when ball bounces
- **Stroke Classification** - CNN classifies shot types (forehand, backhand, serve, etc.)

### 📊 Analytics
- Ball trajectory visualization
- Bounce points highlighted
- Stroke type labels
- Court overlay with lines
- Match statistics

### 🖥️ Web Interface
- Drag & drop video upload
- Real-time processing progress
- Download processed videos
- View analytics dashboard

---

## 📋 Prerequisites

- **Python 3.10+**
- **Node.js 14+**
- **GPU with CUDA** (recommended, CPU works but slower)
- **8GB+ RAM**
- **10GB+ disk space**

---

## 🔧 Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This includes PyTorch, Detectron2, CatBoost, and other CV libraries. Takes ~10-15 minutes.

### Step 2: Verify Model Weights

Ensure these weight files are in the root directory:

| Model | File | Size | Purpose |
|-------|------|------|---------|
| TrackNet | `tracknet_weights.pt` | ~50MB | Ball detection |
| Court Detector | `model_tennis_court_det.pt` | ~100MB | Court lines |
| Stroke Classifier | `stroke_classifier_weights.pth` | ~20MB | Shot types |
| Bounce Detector | `bounce_detection_weights.cbm` | ~1MB | Bounce events |
| COCO Metadata | `coco_instances_results.json` | ~2MB | Reference data |

### Step 3: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## 🏃 Running the Application

### Option 1: Run Everything (Recommended)

```bash
python run_local.py
```

**OR** (Windows):
```bash
start.bat
```

This starts:
- ✅ Backend API at `http://localhost:8000`
- ✅ Frontend at `http://localhost:3000`

**Browser will auto-open to `http://localhost:3000`**

### Option 2: Run Backend Only

```bash
python local_backend.py
```

**OR**:
```bash
start_backend_only.bat
```

- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/health`

### Option 3: Run Frontend Only

```bash
cd frontend
npm start
```

---

## 📖 Usage

### Web Interface

1. Open `http://localhost:3000` in your browser
2. Drag & drop a tennis match video (MP4, MOV, AVI)
3. Wait for processing (progress bar shows status)
4. View results with:
   - ✅ Ball tracking overlay
   - ✅ Court lines
   - ✅ Bounce indicators  
   - ✅ Stroke labels
5. Download processed video
6. View analytics dashboard

### API (Programmatic)

```python
import requests

# Upload
with open('video.mp4', 'rb') as f:
    response = requests.post('http://localhost:8000/api/upload', files={'file': f})
video_id = response.json()['video_id']

# Process
requests.post(f'http://localhost:8000/api/process/{video_id}?filename=video.mp4')

# Check status
while True:
    status = requests.get(f'http://localhost:8000/api/status/{video_id}').json()
    if status['status'] == 'completed':
        print(f"Analytics: {status['result']}")
        break
    time.sleep(5)

# Download
response = requests.get(f'http://localhost:8000/api/download/{video_id}')
with open('output.mp4', 'wb') as f:
    f.write(response.content)
```

---

## 🏗️ Architecture

### System Overview

```
┌─────────────┐      HTTP/REST      ┌──────────────┐
│   Frontend  │ ───────────────────▶ │   Backend    │
│   (React)   │ ◀─────────────────── │  (FastAPI)   │
└─────────────┘                      └──────────────┘
                                            │
                                            ▼
                                     ┌──────────────┐
                                     │  Processing  │
                                     │   Pipeline   │
                                     └──────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │                                               │
            ┌───────▼────────┐                              ┌──────▼──────┐
            │  Ball Detector │                              │   Court     │
            │   (TrackNet)   │                              │  Detector   │
            └───────┬────────┘                              └──────┬──────┘
                    │                                               │
            ┌───────▼────────┐                              ┌──────▼──────┐
            │     Bounce     │                              │   Stroke    │
            │   Detector     │                              │ Classifier  │
            └────────────────┘                              └─────────────┘
```

### Processing Pipeline

```python
# video_processor.py
1. Load video frames
2. Pass 1: Ball detection (TrackNet)
   └─> Detect ball in each frame
3. Bounce analysis (CatBoost)
   └─> Find bounce points from trajectory
4. Pass 2: Annotation
   ├─> Court detection (Detectron2)
   ├─> Stroke classification (CNN)
   ├─> Draw overlays
   └─> Add labels
5. Save output video
6. Return analytics
```

---

## 🧰 Technology Stack

### Backend
| Component | Technology |
|-----------|------------|
| Web Framework | FastAPI |
| Deep Learning | PyTorch 2.1.0 |
| Ball Detection | TrackNet (Custom CNN) |
| Court Detection | Detectron2 (Facebook AI) |
| Bounce Detection | CatBoost |
| Stroke Classification | Custom CNN |
| Computer Vision | OpenCV 4.8 |

### Frontend
| Component | Technology |
|-----------|------------|
| UI Framework | React 18.2 |
| Styling | Tailwind CSS |
| Charts | Chart.js |
| Build Tool | Webpack 5 |

---

## 📁 Project Structure

```
courtCheck/
├── Backend (Python)
│   ├── local_backend.py            # FastAPI server
│   ├── ball_detection.py           # TrackNet
│   ├── court_detection_module.py   # Detectron2
│   ├── stroke_classifier.py        # Stroke CNN
│   ├── bounce_detection.py         # CatBoost
│   └── video_processor.py          # Main pipeline
│
├── Frontend (React)
│   ├── src/
│   │   ├── App.js
│   │   └── components/
│   │       ├── VideoUpload.js      # Upload interface
│   │       ├── Dashboard.js        # Analytics display
│   │       └── ...
│   └── .env                        # Config (localhost:8000)
│
├── Models
│   ├── tracknet_weights.pt
│   ├── model_tennis_court_det.pt
│   ├── stroke_classifier_weights.pth
│   ├── bounce_detection_weights.cbm
│   └── coco_instances_results.json
│
├── Scripts
│   ├── run_local.py                # Start both servers
│   ├── start.bat                   # Windows launcher
│   └── start_backend_only.bat      # Backend only
│
└── Documentation
    ├── README.md                   # This file
    ├── LOCAL_SETUP.md              # Detailed local setup
    └── DEPLOYMENT.md               # Cloud deployment (Modal)
```

---

## 🎓 How It Works

### 1. Ball Detection (TrackNet)
- **Model**: Deep CNN trained on tennis ball images
- **Input**: Single frame (640x360)
- **Output**: 15-channel heatmap
- **Accuracy**: ~90-95% detection rate

### 2. Court Detection (Detectron2)
- **Model**: Keypoint R-CNN
- **Input**: Full resolution frame
- **Output**: 14 court keypoints
- **Use**: Map 3D court to 2D for analytics

### 3. Bounce Detection (CatBoost)
- **Model**: Gradient boosting classifier
- **Features**: Ball velocity, acceleration, position
- **Output**: Bounce/no-bounce classification
- **Accuracy**: ~85-90%

### 4. Stroke Classification (CNN)
- **Model**: Custom CNN
- **Classes**: Forehand, Backhand, Serve, Volley, Smash
- **Input**: Frame crops around player
- **Accuracy**: ~75-85%

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'X'"
```bash
pip install -r requirements.txt
```

### "Port 8000 already in use"
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <pid> /F

# Linux/Mac
lsof -ti:8000 | xargs kill
```

### "CUDA out of memory"
Edit `local_backend.py`:
```python
device = "cpu"  # Force CPU mode
```

### Models not loading
Check files exist in root directory:
```bash
dir *.pt *.pth *.cbm
```

### Frontend won't start
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

See [LOCAL_SETUP.md](LOCAL_SETUP.md) for detailed troubleshooting.

---

## 🌐 Cloud Deployment

Want to deploy to production? See [DEPLOYMENT.md](DEPLOYMENT.md) for:
- Modal serverless deployment
- Frontend hosting (Vercel/Netlify)
- Cost optimization
- Scaling strategies

---

## 📊 Performance

### Local Performance
- **GPU (CUDA)**: 2-5 minutes for 5-minute video
- **CPU**: 10-20 minutes for 5-minute video
- **Memory**: 4-8GB RAM

### Model Accuracy
- Ball Detection: ~90-95%
- Court Detection: ~90-98%
- Bounce Detection: ~85-90%
- Stroke Classification: ~75-85%

---

## 🤝 Contributing

Contributions welcome! To contribute:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **TrackNet**: Huang et al. (2019) - Tennis ball tracking
- **Detectron2**: Facebook AI Research - Object detection
- **CatBoost**: Yandex - Gradient boosting
- **React**: Meta - UI framework

---

## 📞 Support

- 📖 Docs: [LOCAL_SETUP.md](LOCAL_SETUP.md), [DEPLOYMENT.md](DEPLOYMENT.md)
- 🐛 Issues: GitHub Issues
- 💬 Questions: GitHub Discussions

---

<div align="center">
  <p><strong>Ready to analyze tennis matches? 🎾</strong></p>
  <p>
    <a href="#-quick-start-local">Quick Start</a> •
    <a href="LOCAL_SETUP.md">Setup Guide</a> •
    <a href="#-usage">Usage</a> •
    <a href="DEPLOYMENT.md">Deploy to Cloud</a>
  </p>
  <p>Made with ❤️ for tennis players and coaches</p>
</div>
