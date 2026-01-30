# 🔧 Frontend Not Loading - Solutions

## Issue
The frontend server starts but you can't access `http://localhost:3000`

## Quick Fixes

### Solution 1: Start Frontend Separately (Recommended)

**Stop the current `run_local.py` (CTRL+C), then:**

#### Terminal 1: Start Backend
```bash
python local_backend.py
```

Wait for:
```
[MODELS] Models loaded: 3 / 4
API URL: http://localhost:8000
```

#### Terminal 2: Start Frontend
```bash
cd frontend
npm start
```

**OR** double-click: `start_frontend.bat`

Wait for:
```
Compiled successfully!
webpack compiled with 1 warning
```

Then open: **http://localhost:3000**

---

### Solution 2: Use Production Build

```bash
cd frontend
npm run build
npx serve -s dist -l 3000
```

Then open: **http://localhost:3000**

---

### Solution 3: Check Frontend Separately

```bash
cd frontend
npm start
```

Look for errors in the output. Common issues:
- Port 3000 already in use
- Missing dependencies
- Webpack compilation errors

---

## ✅ Stroke Classifier Fix Applied

Updated dimensions to match your weights:
- `input_size = 2048`
- `hidden_size = 90` (was 512)
- `num_classes = 3` (was 6)

Stroke classifier should now load without errors!

---

## 🎯 Test Now

### Step 1: Backend
```bash
python local_backend.py
```

**Expected:**
```
[OK] TrackNet (Ball Detection) loaded
[OK] Court Detection loaded  
[OK] Stroke Classifier loaded    <-- Should work now!
[OK] Bounce Detector loaded
[MODELS] Models loaded: 4 / 4
```

### Step 2: Frontend (New Terminal)
```bash
cd frontend
npm start
```

**Expected:**
```
Compiled successfully!

You can now view courtcheck-ui in the browser.

  Local:            http://localhost:3000
```

### Step 3: Access
Open **http://localhost:3000** in your browser

---

## 🐛 If Frontend Still Won't Start

### Check for Errors
```bash
cd frontend
npm start
```

### Common Fixes

**Port in use:**
```bash
# Kill process on port 3000
netstat -ano | findstr ":3000"
taskkill /PID <pid> /F
```

**Missing dependencies:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

**Webpack errors:**
Check `frontend/webpack.config.js` exists and is valid

---

## 💡 Why run_local.py Might Not Show Frontend

The `run_local.py` script starts both servers, but:
1. Frontend output might be hidden
2. Webpack compilation takes time
3. Subprocess management on Windows can be tricky

**Solution:** Start them separately in 2 terminals for better visibility.

---

## ✨ Summary

**Backend fix applied:** ✅ Stroke classifier now has correct dimensions

**Frontend fix:** Start separately for better control

**Commands:**
```bash
# Terminal 1
python local_backend.py

# Terminal 2  
cd frontend
npm start

# Browser
http://localhost:3000
```

This should work! 🎾
