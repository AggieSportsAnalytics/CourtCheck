# 🏸 CourtCheck

CourtCheck is an end-to-end tennis match analysis tool that uses computer vision and machine learning to automatically generate visual and statistical insights from raw match footage. Built in collaboration with the **UC Davis Men’s and Women’s Tennis Teams**, the goal of this project is to make post-match film review faster, smarter, and more objective — without requiring manual tagging or expensive equipment.

---

## 🎯 Project Goals

- Automate the analysis of full-length tennis match videos
- Provide coaches and players with **immediate visualizations** (e.g. heatmaps, bounce locations, player movement)
- Reduce human error and subjective judgment in ball calls or movement assessment
- Offer a **scalable, cloud-based workflow** that can handle long videos efficiently (Google Colab, A100 GPU)

---

## ⚠️ Current Challenges & Optimization Needs

CourtCheck currently performs well on short clips, but hits **scalability and memory limits** when processing longer videos (>5–10 minutes). Specific issues include:

- ❌ Loads all frames into memory before processing → causes **OOM crashes** on Colab
- ❌ Draws overlays (court lines, labels, etc.) on every frame → slows down processing unnecessarily
- ❌ Processes every frame at full FPS → redundant for analytics-focused use cases

We're actively working to:

- Stream frames one at a time (no full memory loading)
- Add flags to disable visualization for faster analytics-only runs
- Downsample FPS during processing
- Improve GPU efficiency via better tensor handling and batching

---

## 🧠 How CourtCheck Works

CourtCheck integrates several specialized models to process and annotate match footage:

| Component           | Model / Technique           | Function                                      |
| ------------------- | --------------------------- | --------------------------------------------- |
| 🎾 Court Detection  | Detectron2 (keypoint R-CNN) | Detects court boundaries and reference points |
| 🟡 Ball Tracking    | Custom CNN (TrackNet-style) | Predicts ball coordinates across frames       |
| ⛳ Bounce Detection | CatBoost ML model           | Identifies when and where the ball bounces    |
| 🧍 Player Detection | Faster R-CNN (PyTorch)      | Locates and tracks players on both sides      |
| 🔜 Stroke Detection | _(Planned)_                 | Will classify serve, forehand, backhand, etc. |

These outputs are then used to generate:

- Minimap overlays
- Trajectory visualizations
- Heatmaps (bounce, player movement)
- Analytics outputs for coaching and feedback

---

## 📦 Technologies Used

- Python, PyTorch, OpenCV, NumPy, SciPy, CatBoost
- Detectron2 (court keypoints), custom CNN (TrackNet), torchvision models (Faster R-CNN)
- Google Colab (A100 GPU) for GPU-accelerated inference and post-processing
- Scene detection via PySceneDetect

---

## 🚀 Getting Started

1. Set paths to your input video (MP4) and model weights in the script
2. Run `process_video.py` on **Google Colab with GPU enabled**
3. Optionally adjust:
   - `ENABLE_VISUALIZATION = False` to speed up runs
   - Frame rate downsampling
   - Input resolution (default: 640×360)

---

## 📊 Output Formats

- `video.mp4` with overlays (if enabled)
- `minimap.mp4` showing real-time positions
- `bounce_heatmap.png`, `player_heatmap.png`
- (Planned) `analytics.csv` with summary stats per match

---

## 📥 Collaboration & Feedback

We welcome collaborators in the following areas:

- Optimizing inference and video pipelines
- Stroke classification modeling
- Frontend dashboard or web portal development
- Integration with other sports tech platforms

If you're interested or have ideas, contact:  
**📧 corypham1@gmail.com**
