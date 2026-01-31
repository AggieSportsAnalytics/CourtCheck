# Video Annotation Information

## What Should Appear in Processed Videos

When processing completes successfully, the output video should show:

### 1. Ball Tracking
- **Yellow/Cyan circles** marking the ball position in each frame
- **Yellow trace** showing the ball's trajectory over the last 7 frames
- Only visible when the ball is detected

### 2. Court Detection (Currently Limited)
- **Court keypoints** (corners, net posts, service lines)
- **Court lines** connecting the keypoints
- **Note:** Requires Detectron2 which is not currently installed

### 3. Bounce Detection
- **Yellow circle** around the ball at bounce points
- **"BOUNCE" label** near the bounce location

### 4. Stroke Classification
- **Stroke type label** in top-left (e.g., "Forehand", "Backhand")
- **Confidence score** (only shown if confidence > 60%)

---

## Why Annotations Might Be Missing

### Ball Not Detected
**Most Common Issue for Short Videos**

TrackNet (the ball detection model) requires:
- At least 3 consecutive frames for detection
- Clear view of the ball against the background
- The ball must be visible (not occluded by players)

For a 2-second video:
- At 30 FPS: ~60 frames total
- If the ball isn't clearly visible or moves very fast, detection may fail
- Result: No yellow circles or traces will appear

### Court Not Detected
Detectron2 is not installed, so:
- No court keypoints will be drawn
- No court lines will be visible
- This is expected in the current setup

### Video Too Short
For meaningful analysis, videos should be:
- At least 5-10 seconds long
- Show clear ball movement (rallies, serves)
- Have good lighting and camera angle

---

## Testing Recommendations

### For Best Results:
1. **Use longer videos** (10+ seconds)
2. **Include rallies** where the ball is in motion
3. **Ensure good lighting** and camera stability
4. **Film from a side angle** showing the full court

### What to Check:
1. After processing, check the analytics:
   - `ball_detected_frames`: How many frames had the ball
   - `ball_detection_rate`: Percentage of frames with ball
   - `total_bounces`: Number of bounces detected

2. If `ball_detected_frames` is 0:
   - The ball wasn't visible enough in the video
   - Try a different video with clearer ball visibility

---

## Current System Status

### Working:
- ✅ Ball Detection (TrackNet)
- ✅ Stroke Classification
- ✅ Bounce Detection (heuristic-based)
- ✅ Video upload and processing
- ✅ Analytics generation

### Limited/Not Working:
- ⚠️ Court Detection (Detectron2 not installed)
- ⚠️ Short videos may not have detections

---

## Next Steps

If you want court detection:
```powershell
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Then restart the backend:
```powershell
python local_backend.py
```

---

## Checking Your Results

After uploading a video, look at the **Analytics Display**:

```
Ball Tracking
├─ Frames with Ball Detected: X / Y
└─ Detection Rate: Z%

Bounce Detection
└─ Total Bounces Detected: N

Stroke Statistics
├─ Forehand: N
├─ Backhand: N
└─ Serve: N
```

If "Frames with Ball Detected" is 0, the video won't have any visual annotations.

---

## Download and Inspect

Even if no annotations appear:
1. Download the processed video
2. Watch it carefully
3. Check if the analytics JSON shows any detections
4. Try with a longer, clearer video

The system works best with:
- **Clear, professional-quality tennis footage**
- **Videos of actual rallies** (not just setup shots)
- **Good lighting and stable camera**
