## Camera identification
Camera selection follows: explicit config → auto-identify → filename hint → detected homography.
Automatic matching uses median NCC of normalized 96×54 grayscale thumbnails across five startup frames: at least 0.75 correlation and a 0.12 lead over the runner-up, configured in `PipelineConfig`.
Detector-vs-detector keypoints (at least 10 shared points) are a recorded cross-check only. Only courts 2, 4, and 6 have fingerprints; both court-1 setups require explicit config or a recognized filename hint.
Add a court with `python3 -m backend.tools.calibrate_court --image <frame> --camera-id <id>`, save its reference frame under `calibration_frames/`, and add its camera ID and frame filename to `camera_fingerprints.json`.
Validate locally with `python3 -m backend.tools.identify_camera <video-or-image> [more...] --frames 5 --json` (CPU; requires model weights).
