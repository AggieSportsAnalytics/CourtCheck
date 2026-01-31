# CourtCheck Model Troubleshooting Guide

## Overview
This guide helps you identify and fix common issues preventing the CourtCheck tennis analysis model from running.

## Quick Diagnostic

### Step 1: Run the Diagnostic Script
```bash
cd /path/to/CourtCheck
python diagnose_courtcheck.py
```

This will check:
- ✅ Python version (3.10+ required)
- ✅ Required files presence
- ✅ Python dependencies
- ✅ Model weight files
- ✅ Common error patterns
- ✅ Port availability

---

## Common Issues & Solutions

### Issue 1: Missing Model Weight Files

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'tracknet_weights.pt'
```

**Required Files:**
- `tracknet_weights.pt` (~50MB) - Ball detection
- `model_tennis_court_det.pt` (~100MB) - Court detection
- `stroke_classifier_weights.pth` (~20MB) - Stroke classification
- `bounce_detection_weights.cbm` (~1MB) - Bounce detection
- `coco_instances_results.json` (~2MB) - COCO metadata

**Solution:**
1. Check if files exist in the root directory:
   ```bash
   ls -lh *.pt *.pth *.cbm *.json
   ```

2. If missing, you need to:
   - Download from the project's release page or drive
   - Or train the models yourself (see docs)
   - Place them in the root directory

3. Verify file sizes match expected values

---

### Issue 2: CUDA/GPU Device Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
AssertionError: Torch not compiled with CUDA enabled
```

**Solution A - Force CPU Mode:**

Edit `local_backend.py`:
```python
# Find this line:
device = "cuda" if torch.cuda.is_available() else "cpu"

# Change to force CPU:
device = "cpu"
```

Edit `ball_detection.py`:
```python
# In the load_model function, add map_location:
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
```

Edit `video_processor.py`:
```python
# Wherever models are loaded, add:
model = torch.load('path/to/weights.pt', map_location='cpu')
```

**Solution B - Fix CUDA Memory:**

If you have a GPU but getting OOM errors:

```python
# Add at the beginning of processing functions:
torch.cuda.empty_cache()

# Reduce batch size in ball_detection.py:
batch_size = 1  # Instead of 4 or 8
```

---

### Issue 3: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'detectron2'
ModuleNotFoundError: No module named 'catboost'
```

**Solution:**

1. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. For Detectron2 specifically (often problematic):
   ```bash
   # For CPU:
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/index.html
   
   # For CUDA 11.8:
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1/index.html
   ```

3. Verify installation:
   ```python
   python -c "import torch; import detectron2; import catboost; import cv2; print('All imports OK')"
   ```

---

### Issue 4: Path/Directory Errors

**Symptoms:**
```
FileNotFoundError: Video file not found
PermissionError: [Errno 13] Permission denied
```

**Solution:**

1. Check file paths in your code:
   ```python
   # Instead of hardcoded paths:
   weights_path = "tracknet_weights.pt"
   
   # Use os.path:
   import os
   BASE_DIR = os.path.dirname(os.path.abspath(__file__))
   weights_path = os.path.join(BASE_DIR, "tracknet_weights.pt")
   ```

2. Ensure upload/output directories exist:
   ```python
   os.makedirs("uploads", exist_ok=True)
   os.makedirs("outputs", exist_ok=True)
   ```

3. Check file permissions:
   ```bash
   chmod +r *.pt *.pth *.cbm
   chmod +w uploads/ outputs/
   ```

---

### Issue 5: OpenCV Video Codec Errors

**Symptoms:**
```
cv2.error: OpenCV(4.x) Error in VideoWriter
Could not open video file
```

**Solution:**

1. Install additional codec support:
   ```bash
   # Ubuntu/Debian:
   sudo apt-get install ffmpeg
   
   # macOS:
   brew install ffmpeg
   
   # Windows: Download from ffmpeg.org
   ```

2. Try different fourcc codes in `video_processor.py`:
   ```python
   # Instead of:
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   
   # Try:
   fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
   # or
   fourcc = cv2.VideoWriter_fourcc(*'XVID')
   # or
   fourcc = cv2.VideoWriter_fourcc(*'H264')
   ```

---

### Issue 6: Model Loading Errors

**Symptoms:**
```
RuntimeError: Error(s) in loading state_dict
KeyError: Unexpected key(s) in state_dict
```

**Solution:**

1. Add error handling when loading models:
   ```python
   try:
       model.load_state_dict(torch.load(weights_path))
   except RuntimeError as e:
       print(f"Loading weights with strict=False due to: {e}")
       model.load_state_dict(torch.load(weights_path), strict=False)
   ```

2. Ensure model architecture matches weights:
   ```python
   # Check model architecture in ball_detection.py, stroke_classifier.py
   # Ensure it matches the architecture used during training
   ```

---

### Issue 7: NumPy/Array Compatibility

**Symptoms:**
```
AttributeError: 'numpy.ndarray' object has no attribute 'numpy'
ValueError: setting an array element with a sequence
```

**Solution:**

1. Check NumPy version compatibility:
   ```bash
   pip install numpy==1.24.3
   ```

2. Fix array conversions:
   ```python
   # Instead of:
   x = torch.tensor(array)
   
   # Use:
   x = torch.tensor(array, dtype=torch.float32)
   
   # For numpy conversion:
   x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
   ```

---

### Issue 8: Port Already in Use

**Symptoms:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**

1. Find and kill the process using the port:
   ```bash
   # Linux/Mac:
   lsof -ti:8000 | xargs kill -9
   
   # Windows:
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   ```

2. Or change the port in `local_backend.py`:
   ```python
   if __name__ == "__main__":
       import uvicorn
       uvicorn.run(app, host="0.0.0.0", port=8001)  # Changed from 8000
   ```

---

### Issue 9: Frontend Can't Connect to Backend

**Symptoms:**
```
Network Error: Failed to fetch
CORS policy error
```

**Solution:**

1. Ensure backend is running:
   ```bash
   curl http://localhost:8000/api/health
   ```

2. Add CORS middleware in `local_backend.py`:
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["http://localhost:3000"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

3. Update frontend API URL in `frontend/.env`:
   ```
   REACT_APP_API_URL=http://localhost:8000
   ```

---

### Issue 10: Out of Memory Errors

**Symptoms:**
```
MemoryError
Killed (OOM)
```

**Solution:**

1. Process video in smaller chunks:
   ```python
   # In video_processor.py:
   chunk_size = 100  # Process 100 frames at a time
   for i in range(0, total_frames, chunk_size):
       frames = read_frames(i, i+chunk_size)
       process_frames(frames)
       del frames  # Free memory
       gc.collect()
   ```

2. Reduce video resolution:
   ```python
   # Resize frames before processing:
   frame = cv2.resize(frame, (640, 360))
   ```

3. Use CPU instead of GPU for large videos

---

## Step-by-Step Verification Process

### 1. Environment Setup
```bash
# Check Python version
python --version  # Should be 3.10+

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Files
```bash
# Check all files exist
ls -l local_backend.py
ls -l video_processor.py
ls -l ball_detection.py
ls -l court_detection_module.py
ls -l stroke_classifier.py
ls -l bounce_detection.py

# Check model weights
ls -lh tracknet_weights.pt
ls -lh model_tennis_court_det.pt
ls -lh stroke_classifier_weights.pth
ls -lh bounce_detection_weights.cbm
ls -lh coco_instances_results.json
```

### 3. Test Imports
```python
python << EOF
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import cv2
print(f"OpenCV: {cv2.__version__}")

import detectron2
print(f"Detectron2: OK")

import catboost
print(f"CatBoost: OK")

import fastapi
print(f"FastAPI: OK")

print("\nAll imports successful!")
EOF
```

### 4. Test Backend
```bash
# Start backend
python local_backend.py

# In another terminal, test:
curl http://localhost:8000/api/health
```

### 5. Test Frontend
```bash
cd frontend
npm install
npm start

# Should open http://localhost:3000
```

---

## Model-Specific Fixes

### Ball Detection (TrackNet)

**Common Issue:** Model not detecting balls

**Fix:**
```python
# In ball_detection.py, adjust confidence threshold:
confidence_threshold = 0.5  # Lower if not detecting
# or
confidence_threshold = 0.8  # Higher to reduce false positives
```

### Court Detection (Detectron2)

**Common Issue:** Court not detected

**Fix:**
```python
# In court_detection_module.py, check predictor initialization:
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

cfg = get_cfg()
cfg.merge_from_file("path/to/config.yaml")
cfg.MODEL.WEIGHTS = "model_tennis_court_det.pt"
cfg.MODEL.DEVICE = "cpu"  # Force CPU if needed
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust threshold

predictor = DefaultPredictor(cfg)
```

### Bounce Detection (CatBoost)

**Common Issue:** Bounces not detected

**Fix:**
```python
# In bounce_detection.py, check model loading:
import catboost

model = catboost.CatBoostClassifier()
model.load_model('bounce_detection_weights.cbm')

# Verify features match training:
features = ['ball_y_velocity', 'ball_y_accel', 'ball_y_pos']
```

---

## Emergency Quick Fix Script

Create `quick_fix.py`:

```python
#!/usr/bin/env python3
import os
import sys

print("Applying emergency fixes to CourtCheck...")

# Fix 1: Force CPU mode
files_to_fix = ['local_backend.py', 'ball_detection.py', 'video_processor.py']

for file in files_to_fix:
    if os.path.exists(file):
        with open(file, 'r') as f:
            content = f.read()
        
        # Force CPU
        content = content.replace(
            'device = "cuda" if torch.cuda.is_available() else "cpu"',
            'device = "cpu"'
        )
        
        # Add map_location to torch.load
        content = content.replace(
            'torch.load(',
            'torch.load(map_location="cpu", '
        )
        
        with open(file, 'w') as f:
            f.write(content)
        
        print(f"✓ Fixed {file}")

# Fix 2: Create required directories
for dir in ['uploads', 'outputs', 'temp']:
    os.makedirs(dir, exist_ok=True)
    print(f"✓ Created {dir}/ directory")

print("\nQuick fixes applied! Try running the application again.")
```

Run with: `python quick_fix.py`

---

## Getting Help

If you're still experiencing issues:

1. **Run the diagnostic script:**
   ```bash
   python diagnose_courtcheck.py
   ```

2. **Check error logs:**
   - Look at the terminal output
   - Check browser console (F12) for frontend errors

3. **Provide this information when asking for help:**
   - Python version
   - Operating system
   - GPU/CUDA availability
   - Complete error message
   - Output from diagnostic script

---

## Additional Resources

- Project README: See main README.md
- Dependencies: requirements.txt
- Model documentation: Check docs/ folder
- GitHub Issues: Report bugs on GitHub

---

**Last Updated:** January 2026
