# CourtCheck <img src="https://github.com/AggieSportsAnalytics/CourtCheck/blob/cory/images/courtcheck_ball_logo.png" alt="CourtCheck Logo" style="width: 80px; vertical-align: middle;"> 

**AI-powered tennis analytics by [Aggie Sports Analytics](https://aggiesportsanalytics.com)**

🎾 **[courtcheck-rho.vercel.app](https://courtcheck-rho.vercel.app)**

---

Every game day, UC Davis tennis coaches face 6 singles matches worth of footage to review. A thorough analysis of each video takes hours, time that coaches and players simply don't have.

**CourtCheck turns 20+ hours of weekly film review into a 5-minute read.**

Upload a match recording and CourtCheck's computer vision pipeline automatically tracks every ball, player, shot, and bounce so coaches and players can instantly understand what happened and what to fix, without sitting through hours of tape.

---

## What It Does

**Ball & Bounce Tracking**
Frame-by-frame ball trajectory with precise bounce event detection across the full court.

**Player Heatmaps**
Visual court coverage maps that reveal positioning patterns, movement tendencies, and court dominance.

**Stroke & Shot Analysis**
Automatic shot classification with percentages and breakdowns across a full match.

**AI Scouting Reports**
GPT-powered match summaries that synthesize player positioning and ball data into actionable coaching insights.

**Match & Opponent Stats**
Aggregated performance trends across multiple matches with head-to-head opponent breakdowns.

**Recordings Library**
Upload, manage, and replay processed match videos, all in one place.

---

## How It Works

1. **Upload** a match recording through the dashboard
2. **CourtCheck processes** the video using GPU-accelerated computer vision models
3. **View your analytics** in minutes: heatmaps, shot charts, stats, and scouting report

---

## Built With

- Computer Vision:YOLOv8, OpenCV, PyTorch, CatBoost
- AI:OpenAI GPT-4o-mini
- Frontend:Next.js, TypeScript, Tailwind CSS
- Backend:Python, FastAPI, Modal (A10G GPU)
- Infrastructure:Vercel, Supabase

---

Built by Aggie Sports Analytics at UC Davis.
