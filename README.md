# CourtCheck <img src="https://github.com/AggieSportsAnalytics/CourtCheck/blob/cory/images/courtcheck_ball_logo.png" alt="CourtCheck Logo" style="width: 80px; vertical-align: middle;">

**AI-powered tennis analytics by [Aggie Sports Analytics](https://aggiesportsanalytics.com)**

*See every shot. Know every move.*

🎾 **[courtcheck-rho.vercel.app](https://courtcheck-rho.vercel.app)**

---

Tennis has always been played on instinct. CourtCheck changes that.

Every game day, UC Davis tennis coaches face 6 singles matches worth of footage to review. A thorough analysis of each video takes hours, time that coaches and players simply don't have.

| ⏱️ Before CourtCheck | ⚡ After CourtCheck |
|---|---|
| 20+ hours of film review per week | 5-minute read per match |
| Manually scrubbing through footage | Instant heatmaps, stats & AI reports |
| Easy to miss key patterns | Every shot, bounce & position tracked |

No sensors, no special equipment, no setup. Just upload your video.

---

## 🎯 What It Does

🎾 **Ball & Bounce Tracking**
Frame-by-frame ball trajectory with precise bounce event detection across the full court.

🗺️ **Player Heatmaps**
Visual court coverage maps that reveal positioning patterns, movement tendencies, and court dominance.

📊 **Stroke & Shot Analysis**
Automatic shot classification with percentages and breakdowns across a full match.

🤖 **AI Scouting Reports**
GPT-powered match summaries that synthesize player positioning and ball data into actionable coaching insights.

📈 **Match & Opponent Stats**
Aggregated performance trends across multiple matches with head-to-head opponent breakdowns.

🎥 **Recordings Library**
Upload, manage, and replay processed match videos, all in one place.

---

## ⚙️ How It Works

1. 📤 **Upload** a match recording through the dashboard
2. 🧠 **CourtCheck processes** the video using GPU-accelerated computer vision models
3. 📋 **View your analytics** in minutes: heatmaps, shot charts, stats, and scouting report

---

## 🛠️ Built With

- 👁️ Computer Vision: YOLOv8, OpenCV, PyTorch, CatBoost
- 🤖 AI: OpenAI GPT-4o-mini
- 🌐 Frontend: Next.js, TypeScript, Tailwind CSS
- ⚙️ Backend: Python, FastAPI, Modal (A10G GPU)
- ☁️ Infrastructure: Vercel, Supabase

---

Built by Aggie Sports Analytics at UC Davis. 🐮
