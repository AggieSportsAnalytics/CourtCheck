# Swing extraction runs via the main Modal app to reuse the cached image.
# See backend/app.py — extract_swings_from_video() and extract_swings().
#
# Usage:
#   modal run backend/app.py::extract_swings --local-dir /path/to/videos
#   modal run backend/app.py::extract_swings  # process Volume contents
