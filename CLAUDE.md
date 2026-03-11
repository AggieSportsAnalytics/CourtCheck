# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Always-Active Skills

These skills must be followed at all times, without exception:

- **`verification-before-completion`** — Never claim work is done, fixed, or passing without running verification commands first. Evidence before assertions, always.
- **`test-driven-development`** — Write the test before the implementation for any new feature or bug fix.
- **`systematic-debugging`** — Always find the root cause before proposing a fix. No patches without investigation first.

## Project Overview

CourtCheck is a tennis video analysis application that uses computer vision and machine learning to analyze tennis match recordings. The system detects balls, court boundaries, bounces, and strokes, then overlays tracking data and a minimap on the processed video.

## Architecture

This is a full-stack monorepo with two main components:

### Backend (`/backend`)
- **Language**: Python 3.10
- **Deployment**: Modal serverless platform with GPU (A10G)
- **Entry Point**: `backend/app.py` - Modal function that downloads video from Supabase, runs pipeline, uploads results
- **Core Pipeline**: `backend/pipeline/run.py` - Two-pass video processing (tracking, then drawing)
- **Dependencies**: PyTorch, Ultralytics (YOLO), OpenCV, CatBoost, FastAPI

**Key Modules**:
- `backend/models/` - ML model wrappers:
  - `ball_tracker.py` - YOLO-based ball detection (TrackNet)
  - `court_line_detector.py` - Court keypoint detection network
  - `bounce_detector.py` - CatBoost-based bounce detection
  - `stroke_detector.py` - Stroke/action recognition
  - `player_tracker.py` - YOLOv8x player detection and tracking with player filtering
- `backend/vision/` - Computer vision utilities:
  - `homography.py` - Court perspective transformation
  - `court_reference.py` - Reference court diagram generation
  - `drawing.py` - Overlay rendering (ball traces, court lines, minimap)
  - `heatmaps.py` - Heatmap generation (bounce + player position heatmaps using histogram2d + gaussian blur)
  - `postprocess.py` - Post-processing utilities
- `backend/pipeline/` - Pipeline orchestration:
  - `run.py` - Main pipeline execution
  - `storage.py` - Supabase integration, video upload/download, ffmpeg streamable conversion
  - `config.py` - Configuration

**Model Weights**: Stored in `backend/weights/` (not checked into git). Required files:
- `tracknet_weights.pt` - Ball detection YOLO model
- `keypoints_model.pth` - Court keypoint detection model
- `bounce_detection_weights.cbm` - CatBoost bounce model
- `stroke_classifier_weights.pth` - Stroke recognition model
- `yolov8x.pt` - YOLOv8x player detection model (auto-downloads on first run)

### Frontend (`/frontend/web`)
- **Framework**: Next.js 16 (React 19) with TypeScript
- **UI**: shadcn/ui components with Tailwind CSS
- **Routing**: App Router (Next.js 13+ pattern)
- **State**: React Hook Form with Zod validation
- **Data**: Supabase client for database and storage

**Page Routes**:
- `/` - Dashboard (main page)
- `/upload` - Video upload interface
- `/recordings` - View processed recordings list
- `/recordings/[id]` - Watch processed video with heatmaps
- `/match-stats` - Match statistics
- `/overall-stats` - Overall player statistics
- `/opponents` - Opponent analysis
- `/settings` - User settings
- `/api/recordings` - Recordings list API
- `/api/recordings/[id]` - Single recording API (returns signed URLs for video + heatmaps)

## Development Commands

### Backend

**Install dependencies**:
```bash
cd backend
pip install -r ../requirements.txt
```

**Run pipeline locally** (for testing, requires weights):
```bash
# From project root
python -m backend.pipeline.run --video /path/to/video.mp4 --output output.mp4

# Note: This runs in local_mode (skips Supabase upload)
```

**Deploy to Modal**:
```bash
modal deploy backend/app.py
```

**Test Modal function locally**:
```bash
modal run backend/app.py
```

**Environment variables** (create `backend/.env`):
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Supabase service role key

### Frontend

**Install dependencies**:
```bash
cd frontend/web
npm install
```

**Development server** (runs on http://localhost:3000):
```bash
cd frontend/web
npm run dev
```

**Production build**:
```bash
cd frontend/web
npm run build
npm start
```

**Lint**:
```bash
cd frontend/web
npm run lint
```

**Environment variables** (create `frontend/web/.env.local`):
- `NEXT_PUBLIC_SUPABASE_URL` - Supabase project URL
- `NEXT_PUBLIC_SUPABASE_ANON_KEY` - Supabase anonymous key

## Video Processing Pipeline

The pipeline processes tennis videos in two passes using a **streaming approach** (never loads all frames into memory):

1. **Pass 1: Ball + Player Tracking** (frames 0-N)
   - Ball detection on every frame using YOLO (TrackNet)
   - Player detection and tracking using YOLOv8x
   - Results stored in `ball_track` list and `player_detections` list
   - Only detection results stored (not frame pixels) - memory efficient!
   - Player filtering: Keeps only 2 main players closest to court
   - Bounce detection using CatBoost on ball trajectory
   - Progress: 5% → 50%

2. **Pass 2: Drawing** (frames 0-N)
   - Court detection every 5th frame (court_detection_interval)
   - Homography estimation for perspective transform
   - Draw overlays: ball trace, court keypoints/lines, player bboxes, minimap
   - Minimap shows bird's-eye view of court with ball position and bounces
   - Homography matrices collected per-frame for heatmap generation
   - Progress: 50% → 95%

3. **Heatmap Generation** (post Pass 2, ~<1s overhead):
   - Uses homography matrices from Pass 2 to project positions to court-space
   - Rally filtering via `detect_shot_frames()` — only counts frames during active play
   - **Bounce heatmap**: Direct circle drawing on court reference for each detected bounce
   - **Player position heatmap**: `np.histogram2d` binning + `scipy.ndimage.gaussian_filter` for smooth density, INFERNO colormap with 99th percentile contrast clipping
   - PNGs saved locally, uploaded to Supabase `results` bucket
   - Controlled by `config.generate_heatmaps` flag (default: True)

4. **Post-processing**:
   - Convert to browser-streamable MP4 (ffmpeg with `+faststart`)
   - Upload to Supabase storage
   - Update database with results_path, heatmap paths, status, metadata

**Video Requirements**:
- Format: MP4
- Resolution: Automatically resized to 1280x720 if needed
- Input videos are downloaded from Supabase `raw-videos` bucket
- Output videos are uploaded to Supabase `results` bucket

## Supabase Schema

**Tables** (inferred from code):
- `matches` - Video processing jobs
  - `id` - Match identifier (UUID)
  - `user_id` - User identifier (UUID, FK to auth.users)
  - `status` - 'processing', 'done', or 'failed'
  - `progress` - Float 0.0-1.0
  - `results_path` - Path to processed video in storage
  - `input_path` - Path to raw uploaded video in storage
  - `bounce_heatmap_path` - Path to bounce heatmap PNG in storage (nullable)
  - `player_heatmap_path` - Path to player position heatmap PNG in storage (nullable)
  - `fps` - Frames per second
  - `num_frames` - Total frame count
  - `error` - Error message if failed
  - `created_at` - Timestamp

**Storage Buckets**:
- `raw-videos` - Uploaded input videos
- `results` - Processed output videos and heatmap PNGs (stored as `{match_id}/processed.mp4`, `{match_id}/bounce_heatmap.png`, `{match_id}/player_heatmap.png`)

## Common Patterns

### Adding a new detection model
1. Create model wrapper class in `backend/models/` following existing patterns
2. Initialize model in `run_pipeline()` with weights path
3. Add inference calls in appropriate pipeline pass
4. Update drawing functions in `backend/vision/drawing.py` if visualization needed

### Adding a new frontend page
1. Create route directory in `frontend/web/app/`
2. Add `page.tsx` with default export
3. Use shadcn/ui components from `components/ui/`
4. Follow existing patterns for Supabase data fetching

### Modal deployment notes
- Modal requires all dependencies in `requirements.txt`
- Local code is synced via `.add_local_python_source("backend")`
- Weights must be added via `.add_local_dir()` during image build
- GPU is required for inference (currently using A10G)
- Secrets managed via Modal dashboard (supabase-secrets)
