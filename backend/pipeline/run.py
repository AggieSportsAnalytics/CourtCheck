# pipeline/run.py
import cv2
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

from backend.models import BallDetector, CourtLineDetector, PlayerTracker, BounceDetector, ActionRecognition, PoseStrokeClassifier, TrajectoryRectifier
from backend.vision import HomographyEstimator, CourtReference, draw_ball_trace, draw_court_keypoints_and_lines, draw_minimap_ball_and_bounces, draw_minimap_players, draw_player_bboxes, draw_stroke_labels
from backend.vision import SwingDetector, extract_pose_sequence
from backend.vision.calibration import load_calibration
from backend.vision.heatmaps import generate_minimap_heatmaps, generate_player_shot_dot_map
from backend.vision.postprocess import detect_shot_frames

from backend.pipeline.storage import upload_processed_video, upload_heatmap_png, get_supabase, make_streamable_mp4, upload_results_parallel
from backend.pipeline.config import PipelineConfig
from backend.pipeline.rallies import (
    RALLY_GAP_SECONDS,
    build_rallies,
    build_rally_summary,
)


def ensure_720p(input_path, intermediate_path, target_width=1280, target_height=720):
    """Resize video to target resolution using ffmpeg (GPU NVENC when available)."""
    import subprocess
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Original input: {width}x{height}, fps={fps:.2f}")
    if (width == target_width) and (height == target_height):
        print(f"Video is already {target_width}x{target_height} @ {fps:.0f}fps; using input directly.")
        return input_path

    print(f"Resizing from ({width}x{height} @ {fps:.0f}fps) to ({target_width}x{target_height} @ {fps:.0f}fps) -> {intermediate_path}")
    scale_filter = f"scale={target_width}:{target_height}"

    def _run(codec):
        cmd = ["ffmpeg", "-y", "-i", input_path, "-vf", scale_filter]
        cmd += ["-c:v", codec, "-preset", "fast", "-c:a", "copy", intermediate_path]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0

    if _run("h264_nvenc"):
        print(f"Finished writing intermediate (h264_nvenc): {intermediate_path}")
        return intermediate_path
    if _run("libx264"):
        print(f"Finished writing intermediate (libx264): {intermediate_path}")
        return intermediate_path

    # Last resort: copy streams + scale filter (no re-encode, may not work for all inputs)
    print("ffmpeg h264_nvenc and libx264 failed — falling back to OpenCV resize")
    cap_in = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(intermediate_path, fourcc, fps, (target_width, target_height))
    while True:
        ret, frame = cap_in.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        out.write(frame)
    cap_in.release()
    out.release()
    print(f"Finished writing intermediate (OpenCV): {intermediate_path}")
    return intermediate_path


def calculate_rally_count(shot_frames, fps=30.0, gap_seconds=RALLY_GAP_SECONDS):
    """
    Group shot frames into rallies. A gap > gap_seconds between consecutive
    shots is treated as the start of a new rally.
    """
    if not shot_frames:
        return 0
    sorted_frames = sorted(shot_frames)
    gap_frames = fps * gap_seconds
    rallies = 1
    for i in range(1, len(sorted_frames)):
        if sorted_frames[i] - sorted_frames[i - 1] > gap_frames:
            rallies += 1
    return rallies


def _interpolate_player_detections(detections: list[dict]) -> list[dict]:
    """
    Fill empty player-detection frames by linearly interpolating between the
    nearest non-empty frames on either side.

    Empty frames arise when player_detection_interval > 1: YOLO is only called
    every Nth frame, so intermediate frames are stored as {}.

    For each track_id present in the surrounding anchor frames, the four bbox
    coordinates (x1, y1, x2, y2) are linearly interpolated. Frames before the
    first anchor are back-filled; frames after the last anchor are forward-filled.
    """
    n = len(detections)
    result = [dict(d) for d in detections]  # shallow-copy each frame dict

    # Collect anchor indices (frames where YOLO actually ran)
    anchors = [i for i, d in enumerate(detections) if d]

    if not anchors:
        return result

    # Back-fill frames before the first anchor
    for i in range(anchors[0]):
        result[i] = dict(detections[anchors[0]])

    # Forward-fill frames after the last anchor
    for i in range(anchors[-1] + 1, n):
        result[i] = dict(detections[anchors[-1]])

    # Interpolate between consecutive anchors
    for a_idx in range(len(anchors) - 1):
        start = anchors[a_idx]
        end = anchors[a_idx + 1]
        if end - start <= 1:
            continue  # adjacent anchors, nothing to fill
        start_bboxes = detections[start]
        end_bboxes = detections[end]
        # Only interpolate stable YOLO track IDs (positive integers).
        # Synthetic ROI IDs (< 0) are re-assigned each frame — interpolating them
        # creates bogus paths between unrelated detections (e.g. ball in frame N,
        # player in frame N+3) that confuse the temporal stabilizer.
        shared_ids = {tid for tid in set(start_bboxes) & set(end_bboxes) if tid >= 0}
        for frame_i in range(start + 1, end):
            t = (frame_i - start) / (end - start)
            for tid in shared_ids:
                s = start_bboxes[tid]
                e = end_bboxes[tid]
                result[frame_i][tid] = [
                    s[0] + t * (e[0] - s[0]),
                    s[1] + t * (e[1] - s[1]),
                    s[2] + t * (e[2] - s[2]),
                    s[3] + t * (e[3] - s[3]),
                ]

    # Forward-fill fallback: if a frame is still empty after interpolation
    # (e.g., no shared track_ids between adjacent anchors), propagate last known bbox.
    _last_frame: dict = {}
    for i in range(len(result)):
        if result[i]:
            _last_frame = result[i]
        elif _last_frame:
            result[i] = dict(_last_frame)

    return result


def _interpolate_ball_track(ball_track: list, max_gap: int = 3) -> list:
    """
    Fill None entries in ball_track by linear interpolation between nearby
    detected positions. Only gaps of <= max_gap frames are filled; longer gaps
    stay as None so the bounce detector sees the ball as genuinely absent
    (between serves, player holding ball, etc.).

    max_gap=3 covers ball_detection_interval=2 skips (gap of 1) and normal
    short tracker misses (2-3 frames), without synthesizing trajectories across
    long ball-absent stretches that would create false bounce inflections.
    """
    n = len(ball_track)
    result = list(ball_track)

    anchors = [
        i for i, p in enumerate(ball_track)
        if p is not None and p[0] is not None and p[1] is not None
    ]
    if not anchors:
        return result

    for a_idx in range(len(anchors) - 1):
        start, end = anchors[a_idx], anchors[a_idx + 1]
        if end - start <= 1 or end - start > max_gap + 1:
            continue
        x0, y0 = result[start]
        x1, y1 = result[end]
        for frame_i in range(start + 1, end):
            t = (frame_i - start) / (end - start)
            result[frame_i] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))

    return result


def _detect_court_once(
    frames: list,
    court_detector,
    homography_estimator,
) -> tuple:
    """
    Run court detection on the provided frames and return the homography
    matrix + keypoints from the frame with the most valid detected keypoints.

    Args:
        frames: List of BGR frames to try (typically first N frames of video).
        court_detector: CourtLineDetector instance.
        homography_estimator: HomographyEstimator instance.

    Returns:
        (H_ref, keypoints) — best result, or (None, None) if all frames fail.
    """
    best_H_ref = None
    best_kps = None
    best_kp_count = 0

    for frame in frames:
        kps = court_detector.infer_single(frame)
        if kps is None:
            continue
        valid_count = sum(1 for k in kps if k is not None)
        if valid_count <= best_kp_count:
            continue
        H_ref, _ = homography_estimator.estimate(kps)
        if H_ref is None:
            continue
        best_H_ref = H_ref
        best_kps = kps
        best_kp_count = valid_count

    return best_H_ref, best_kps


def build_shots(
    bounces,
    ball_track,
    homography_matrices,
    swing_events,
    player_detections,
    court_ref,
    in_bounds_set,
    fps,
):
    """Build the per-shot record used by the frontend courtmaps.

    For each bounce, pair it with the most recent preceding swing event (within
    1.5s @ fps) to recover the stroke type + hitter. Project bounce, ball-at-
    contact, and player-at-contact into 0-27 x 0-78 court units (net at y=39)
    matching the locked viz SVG geometry from `docs/brand-drop/mocks/visuals.html`.
    """
    left_x   = court_ref.left_court_line[0][0]    # 286 px
    right_x  = court_ref.right_court_line[0][0]   # 1379 px
    top_y    = court_ref.baseline_top[0][1]        # 561 px
    bottom_y = court_ref.baseline_bottom[0][1]     # 2935 px

    span_x = max(1.0, right_x - left_x)
    span_y = max(1.0, bottom_y - top_y)

    def to_court(px, py):
        """pixel court coords -> 0-27 x 0-78 SVG units."""
        if px is None or py is None:
            return None, None
        return (
            float((px - left_x) / span_x) * 27.0,
            float((py - top_y) / span_y) * 78.0,
        )

    def project(frame_idx, x, y):
        if x is None or y is None:
            return None, None
        if frame_idx >= len(homography_matrices):
            return None, None
        H = homography_matrices[frame_idx]
        if H is None:
            return None, None
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        try:
            mapped = cv2.perspectiveTransform(pt, H)
        except cv2.error:
            return None, None
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

    def _is_swing_a_return(swing_event):
        """Ball-approach gate: a swing only counts as a real return if the ball
        is moving toward the hitter's baseline in the 8 frames leading up to
        peak_frame.

        - Player 1 (track_id > 0, near side): ball court_y must be INCREASING
          (heading toward y=bottom_y / svg y=78).
        - Player 2 (track_id < 0, far side): ball court_y must be DECREASING
          (heading toward y=top_y / svg y=0).

        Operates on pixel court coords (projected via homography) to avoid
        redundant SVG-unit math. ~5 px/frame slope threshold is roughly
        equivalent to ~0.16 SVG units/frame given span_y ~2374 px; tightened
        below to 8 px/frame to suppress noise from idle wrist jitter.
        """
        try:
            pf = int(swing_event.get("peak_frame", -1))
        except (TypeError, ValueError):
            return False
        if pf < 0:
            return False
        try:
            tid = int(swing_event.get("track_id", 1))
        except (TypeError, ValueError):
            return False

        start = max(0, pf - 8)
        end = min(len(ball_track), pf + 1)
        if end - start < 4:
            return False

        ys = []
        for f in range(start, end):
            if f >= len(ball_track) or f >= len(homography_matrices):
                continue
            bp = ball_track[f]
            if bp is None or bp[0] is None or bp[1] is None:
                continue
            H = homography_matrices[f]
            if H is None:
                continue
            pt = np.array([[[float(bp[0]), float(bp[1])]]], dtype=np.float32)
            try:
                mapped = cv2.perspectiveTransform(pt, H)
            except cv2.error:
                continue
            ys.append(float(mapped[0, 0, 1]))

        if len(ys) < 4:
            return False

        # mean(last 3) - mean(first 3): positive = descending toward near
        # player, negative = descending toward far player. Avg over ~6 frames
        # span so a threshold of ~8 px (cumulative) is roughly ~1.3 px/frame
        # in instantaneous terms, well above ball-at-rest noise.
        head = sum(ys[:3]) / 3.0
        tail = sum(ys[-3:]) / 3.0
        delta = tail - head
        threshold_px = 8.0

        if tid > 0:
            return delta > threshold_px
        return delta < -threshold_px

    # Sort swings by peak_frame; gate logged but doesn't filter.
    all_swings_sorted = sorted(swing_events, key=lambda e: int(e.get("peak_frame", 0)))
    gated_swings = [e for e in all_swings_sorted if _is_swing_a_return(e)]
    total_swings = len(all_swings_sorted)
    print(f"[Shots] {len(gated_swings)}/{total_swings} swings passed ball-approach gate (informational)")

    # Mapped to lowercase frontend stroke keys.
    STROKE_MAP = {
        "Forehand": "forehand",
        "Backhand": "backhand",
        "Serve/Overhead": "serve",
    }

    # Two-pass pairing:
    #   PASS 1 (strict, sequential) — each swing claims the first unclaimed
    #   bounce between its peak_frame and the next swing's peak_frame. This is
    #   the highest-confidence pairing (mirrors rally structure: swing → ball
    #   travels → bounce → next swing).
    #
    #   PASS 2 (fallback, nearest within window) — any bounce still unpaired
    #   gets the label of the NEAREST swing whose peak_frame is within
    #   `FALLBACK_WINDOW` seconds, in either direction. This catches:
    #     - bounces past the LAST swing (post-rally rolls, end-of-clip)
    #     - second bounces of OOB shots that the sequential pass orphaned
    #     - bounces between two swings that were too clustered for strict claim
    #
    # Bounces with NO swing inside the fallback window stay "unknown" (true
    # warmup balls, dead time).
    sorted_bounces = sorted(bounces)
    bounce_to_swing: dict[int, dict] = {}

    # ----- PASS 1: strict sequential -----
    bounces_for_search = list(sorted_bounces)
    claim_cursor = 0
    for i, swing in enumerate(all_swings_sorted):
        pf = int(swing.get("peak_frame", -1))
        if pf < 0:
            continue
        next_pf = (
            int(all_swings_sorted[i + 1].get("peak_frame", -1))
            if i + 1 < len(all_swings_sorted)
            else None
        )
        while claim_cursor < len(bounces_for_search) and bounces_for_search[claim_cursor] <= pf:
            claim_cursor += 1
        if claim_cursor >= len(bounces_for_search):
            break
        candidate = bounces_for_search[claim_cursor]
        if next_pf is not None and candidate >= next_pf:
            continue
        bounce_to_swing[candidate] = swing
        claim_cursor += 1
    strict_paired = len(bounce_to_swing)

    # ----- PASS 2: nearest-within-window fallback -----
    FALLBACK_WINDOW_FRAMES = int(round(2.5 * (fps or 30.0)))
    swing_peaks_with_event = [
        (int(s.get("peak_frame", -1)), s)
        for s in all_swings_sorted
        if int(s.get("peak_frame", -1)) >= 0
    ]
    fallback_paired = 0
    for bframe in sorted_bounces:
        if bframe in bounce_to_swing:
            continue
        best_swing = None
        best_dist = FALLBACK_WINDOW_FRAMES + 1
        for pf, swing in swing_peaks_with_event:
            d = abs(bframe - pf)
            if d < best_dist:
                best_dist = d
                best_swing = swing
        if best_swing is not None and best_dist <= FALLBACK_WINDOW_FRAMES:
            bounce_to_swing[bframe] = best_swing
            fallback_paired += 1

    print(
        f"[Shots] Paired {len(bounce_to_swing)}/{len(sorted_bounces)} bounces "
        f"(strict={strict_paired}, fallback={fallback_paired}, "
        f"orphan={len(sorted_bounces) - len(bounce_to_swing)})",
        flush=True,
    )

    shots = []
    # Diagnostic counters — surfaced in logs so we can tell from a Modal log
    # exactly why bounces aren't producing shot-map dots.
    skip_oob = 0
    skip_no_ball = 0
    skip_no_proj = 0
    ok_count = 0

    for bframe in sorted(bounces):
        if bframe >= len(ball_track):
            skip_oob += 1
            continue
        bpos = ball_track[bframe]
        # The bounce detector predicts a bounce moment from CatBoost trajectory
        # analysis over surrounding frames. The ball is often occluded AT the
        # bounce frame itself (player body, shadow, hard contrast with court
        # surface), so the interpolated ball_track may have None there even
        # though the bounce really happened. Scan nearby frames for the
        # closest valid detection so we still place the dot.
        if bpos is None or bpos[0] is None:
            BOUNCE_BALL_SCAN = 5
            bpos = None
            for delta in range(1, BOUNCE_BALL_SCAN + 1):
                for f in (bframe - delta, bframe + delta):
                    if 0 <= f < len(ball_track):
                        cand = ball_track[f]
                        if cand is not None and cand[0] is not None:
                            bpos = cand
                            break
                if bpos is not None:
                    break
            if bpos is None:
                skip_no_ball += 1
                continue
        bx, by = bpos
        court_bx, court_by = project(bframe, bx, by)
        if court_bx is None:
            skip_no_proj += 1
            continue
        svg_x, svg_y = to_court(court_bx, court_by)
        if svg_x is None:
            skip_no_proj += 1
            continue
        ok_count += 1

        ev = bounce_to_swing.get(bframe)
        stroke = "unknown"
        # When no stroke event is paired, infer the hitter from which half the
        # ball landed in: bounce on far half (svg_y < 39) -> the near player
        # (P1) hit it; bounce on near half -> the far player (P2) hit it. This
        # is what lets the "every bounce" shot map render all bounces, not just
        # the ones whose strokes were classified.
        player = 1 if svg_y < 39.0 else 2
        ball_svg_x, ball_svg_y = None, None
        player_svg_x, player_svg_y = None, None
        contact_frame = None

        if ev is not None:
            stroke = STROKE_MAP.get(ev.get("label", ""), "unknown")
            tid = int(ev.get("track_id", 1))
            player = 1 if tid > 0 else 2
            peak_frame = int(ev["peak_frame"])

            # Refine contact frame. peak_frame is peak wrist velocity, which is
            # close to (but not exactly) racquet-meets-ball. We pick the frame
            # in a *tight* window around peak where ball is closest to the
            # player's bbox CENTER (not feet — racquet is up around chest
            # height, projecting that to court coords would just add noise).
            #
            # Important: the search has to stay tight, because the ball tracker
            # often loses the ball during contact (body occlusion) and
            # re-acquires it mid-flight several frames later. A wide scan would
            # pick up that mid-flight position and plot the ball halfway across
            # the court — what the spacing viz was doing before the fix.
            SCAN_HALF = 6  # frames; ~0.2s at 30fps — wider window catches the
            # occlusion case where the ball tracker briefly loses the ball
            # through racquet contact and re-acquires it 4-5 frames later.
            # ~7 court units ≈ ~7 ft. Loose enough to admit a typical extended
            # forehand contact, tight enough to reject "ball is now mid-rally"
            # detections. With the chest-anchor change, the real-world
            # player-to-ball distance at contact rarely exceeds 4 ft, so 7 ft
            # leaves significant headroom for tracker noise.
            CONTACT_MAX_COURT_UNITS = 7.0
            contact_frame: int | None = None
            contact_ball_court: tuple[float, float] | None = None
            contact_player_court: tuple[float, float] | None = None

            scan_start = max(0, peak_frame - SCAN_HALF)
            scan_end = min(
                len(ball_track),
                len(player_detections),
                peak_frame + SCAN_HALF + 1,
            )
            best_d2_court = float("inf")
            for f in range(scan_start, scan_end):
                bp = ball_track[f]
                if bp is None or bp[0] is None:
                    continue
                dets_at_f = player_detections[f] or {}
                bbox = dets_at_f.get(tid)
                if bbox is None:
                    continue
                try:
                    x1, y1, x2, y2 = bbox[:4]
                except (TypeError, ValueError):
                    continue

                # Project both into court space so the distance check is in
                # real-world units, not pixels (one pixel at the far baseline
                # is several feet on court).
                bx_court, by_court = project(f, float(bp[0]), float(bp[1]))
                if bx_court is None:
                    continue
                # Anchor the player at CHEST height (~60% down the bbox) rather
                # than feet for spacing. The ball at contact is ~2.5 ft off the
                # ground and 2D homography projects it as if it were on the
                # court plane, biasing it deeper into the court. Anchoring the
                # player at the same approximate altitude as the ball cancels
                # most of the resulting bias so the spacing distance reflects
                # actual player↔ball reach, not camera-angle artifact.
                anchor_px = (float(x1) + float(x2)) / 2.0
                anchor_py = float(y1) + (float(y2) - float(y1)) * 0.6
                px_court, py_court = project(f, anchor_px, anchor_py)
                if px_court is None:
                    continue
                ball_svg = to_court(bx_court, by_court)
                player_svg = to_court(px_court, py_court)
                if ball_svg[0] is None or player_svg[0] is None:
                    continue
                d2_court = (ball_svg[0] - player_svg[0]) ** 2 + (ball_svg[1] - player_svg[1]) ** 2
                if d2_court < best_d2_court:
                    best_d2_court = d2_court
                    contact_frame = f
                    contact_ball_court = (ball_svg[0], ball_svg[1])
                    contact_player_court = (player_svg[0], player_svg[1])

            # Only accept the contact pair if the closest ball-player distance
            # in the window is within the plausible-racquet-reach threshold.
            if contact_frame is not None and best_d2_court <= CONTACT_MAX_COURT_UNITS ** 2:
                ball_svg_x, ball_svg_y = contact_ball_court  # type: ignore[misc]
                player_svg_x, player_svg_y = contact_player_court  # type: ignore[misc]
            else:
                # Couldn't find a frame where the ball was anywhere near the
                # player — spacing viz will skip this shot, shot map keeps the
                # bounce dot since that's a separate code path.
                contact_frame = None
                ball_svg_x = ball_svg_y = None
                player_svg_x = player_svg_y = None

        shots.append({
            "frame": int(bframe),
            "time_s": float(bframe / fps) if fps else None,
            "stroke": stroke,
            "player": int(player),
            "court_x": round(svg_x, 3),
            "court_y": round(svg_y, 3),
            "in": bool(bframe in in_bounds_set),
            "ball_court_x": round(ball_svg_x, 3) if ball_svg_x is not None else None,
            "ball_court_y": round(ball_svg_y, 3) if ball_svg_y is not None else None,
            "player_court_x": round(player_svg_x, 3) if player_svg_x is not None else None,
            "player_court_y": round(player_svg_y, 3) if player_svg_y is not None else None,
        })

    print(
        f"[Shots] Built {ok_count} shots from {len(sorted_bounces)} bounces "
        f"(skipped: {skip_oob} out-of-range, {skip_no_ball} no ball detection in ±5 frames, "
        f"{skip_no_proj} projection failed)",
        flush=True,
    )
    return shots


def build_coverage_grid(
    player_detections,
    homography_matrices,
    court_ref,
    rows: int = 12,
    cols: int = 8,
):
    """Build the player-coverage occupancy grid for the bottom half-court.

    For every frame, project each detected player's foot position into court
    units (0-27 x 0-78, net at y=39). Bin all positions falling on the bottom
    half (y in 39..78) into a `rows x cols` grid covering x=1..26, y=39..77,
    matching the locked viz geometry in `components/viz/Coverage.tsx`. Normalize
    so the hottest cell is 1.0.

    Returns a `list[list[float]]` of shape (rows, cols), all values in [0, 1].
    """
    left_x   = court_ref.left_court_line[0][0]
    right_x  = court_ref.right_court_line[0][0]
    top_y    = court_ref.baseline_top[0][1]
    bottom_y = court_ref.baseline_bottom[0][1]
    span_x = max(1.0, right_x - left_x)
    span_y = max(1.0, bottom_y - top_y)

    # Grid lives in SVG coords: x in [1, 26], y in [39, 85]. Mirrors
    # Coverage.tsx (halfH=46, ROWS=12, yOffset=39). Extending y beyond the
    # near baseline (y=78) by ~7 court units captures the typical 1-5 ft
    # behind-baseline play that earlier clamped every position into row 11
    # and made the heatmap look uniform.
    x0, x1 = 1.0, 26.0
    y0, y1 = 39.0, 85.0
    cell_w = (x1 - x0) / cols
    cell_h = (y1 - y0) / rows

    counts = [[0 for _ in range(cols)] for _ in range(rows)]
    n_frames = min(len(player_detections), len(homography_matrices))

    for frame_idx in range(n_frames):
        dets = player_detections[frame_idx]
        if not dets:
            continue
        H = homography_matrices[frame_idx]
        if H is None:
            continue
        for _tid, bbox in dets.items():
            if bbox is None:
                continue
            try:
                x1_px, _y1, x2_px, y2_px = bbox[:4]
            except (TypeError, ValueError):
                continue
            foot_px = (float(x1_px) + float(x2_px)) / 2.0
            foot_py = float(y2_px)
            pt = np.array([[[foot_px, foot_py]]], dtype=np.float32)
            try:
                mapped = cv2.perspectiveTransform(pt, H)
            except cv2.error:
                continue
            cx, cy = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
            # Pixel court -> SVG units (0..27, 0..78)
            svg_x = (cx - left_x) / span_x * 27.0
            svg_y = (cy - top_y) / span_y * 78.0
            # Drop far-half positions (not our player) but clamp behind-baseline
            # and off-sideline into the nearest edge cell — players spend most
            # of a rally 1-5 ft behind the baseline, which projects to svg_y > 77.
            # Coverage.tsx renders an `extendBehind=10` strip for exactly this.
            if svg_y < y0:
                continue
            if svg_x < x0:
                c = 0
            elif svg_x >= x1:
                c = cols - 1
            else:
                c = int((svg_x - x0) / cell_w)
            if svg_y >= y1:
                r = rows - 1
            else:
                r = int((svg_y - y0) / cell_h)
            if 0 <= r < rows and 0 <= c < cols:
                counts[r][c] += 1

    peak = max((counts[r][c] for r in range(rows) for c in range(cols)), default=0)
    if peak <= 0:
        return [[0.0] * cols for _ in range(rows)]
    return [
        [round(counts[r][c] / peak, 4) for c in range(cols)]
        for r in range(rows)
    ]


def build_position_summary(player_detections, homography_matrices, court_ref):
    """4-zone court-position breakdown for the near player (P1), per the
    `coach-insights-spec.md`: inside baseline / on baseline / 5-10 ft behind /
    10+ ft behind.

    Court coords: near baseline svg_y = 78, ~1 court unit ≈ 1 ft. We work in
    the same SVG units the frontend uses (0..27 width, 0..78 length).

    Returns:
        {
            "inside_pct": float,
            "on_pct": float,
            "behind_5_10_pct": float,
            "behind_10_plus_pct": float,
            "n_frames": int,
        }
    All percentages are 0..100, rounded to 1 decimal place.
    """
    left_x   = court_ref.left_court_line[0][0]
    right_x  = court_ref.right_court_line[0][0]
    top_y    = court_ref.baseline_top[0][1]
    bottom_y = court_ref.baseline_bottom[0][1]
    span_x = max(1.0, right_x - left_x)
    span_y = max(1.0, bottom_y - top_y)

    BASELINE_SVG = 78.0
    counts = {"inside": 0, "on": 0, "behind_5_10": 0, "behind_10_plus": 0}
    n_frames = 0

    for frame_idx in range(min(len(player_detections), len(homography_matrices))):
        dets = player_detections[frame_idx]
        H = homography_matrices[frame_idx]
        if not dets or H is None:
            continue
        # Near player has positive track_id (set by YOLO); far player synthetic IDs
        # are negative. Pick the largest-positive tid in the frame.
        near_bbox = None
        for tid, bbox in dets.items():
            if bbox is None:
                continue
            try:
                tid_int = int(tid)
            except (TypeError, ValueError):
                continue
            if tid_int > 0:
                near_bbox = bbox
                break
        if near_bbox is None:
            continue
        try:
            x1, _y1, x2, y2 = near_bbox[:4]
        except (TypeError, ValueError):
            continue
        foot_px = (float(x1) + float(x2)) / 2.0
        foot_py = float(y2)
        pt = np.array([[[foot_px, foot_py]]], dtype=np.float32)
        try:
            mapped = cv2.perspectiveTransform(pt, H)
        except cv2.error:
            continue
        cy = float(mapped[0, 0, 1])
        svg_y = (cy - top_y) / span_y * 78.0

        # Distance behind baseline in court units (~ft).
        behind = svg_y - BASELINE_SVG
        if behind < -1.0:
            counts["inside"] += 1
        elif behind < 1.0:
            counts["on"] += 1
        elif behind < 10.0:
            counts["behind_5_10"] += 1
        else:
            counts["behind_10_plus"] += 1
        n_frames += 1

    if n_frames == 0:
        return {
            "inside_pct": 0.0,
            "on_pct": 0.0,
            "behind_5_10_pct": 0.0,
            "behind_10_plus_pct": 0.0,
            "n_frames": 0,
        }
    return {
        "inside_pct": round(counts["inside"] / n_frames * 100, 1),
        "on_pct": round(counts["on"] / n_frames * 100, 1),
        "behind_5_10_pct": round(counts["behind_5_10"] / n_frames * 100, 1),
        "behind_10_plus_pct": round(counts["behind_10_plus"] / n_frames * 100, 1),
        "n_frames": n_frames,
    }


def build_net_approach_summary(
    player_detections,
    homography_matrices,
    court_ref,
    bounces,
    in_bounds_set,
    fps,
):
    """Net-approach detector + heuristic outcome.

    Approach: near player's foot svg_y crosses the service line (svg_y = 60)
    and stays in front of it for >= 5 consecutive frames. Outcome heuristic:
    rally ends (no bounce within 3s after the approach event) AND the most
    recent bounce was OUT (= opponent's error) -> "won"; otherwise "lost".

    This is intentionally simple; matches the spec's v1 pragmatic version.

    Returns:
        {
            "approaches": int,
            "wins": int,
            "win_pct": float (0..100),
            "events": [{"frame": int, "time_s": float, "outcome": "won"|"lost"}],
        }
    """
    top_y = court_ref.baseline_top[0][1]
    bottom_y = court_ref.baseline_bottom[0][1]
    span_y = max(1.0, bottom_y - top_y)
    SERVICE_LINE_SVG = 60.0
    MIN_FRAMES_IN_FRONT = 5

    n_frames = min(len(player_detections), len(homography_matrices))
    in_front: list[bool] = []
    for frame_idx in range(n_frames):
        dets = player_detections[frame_idx]
        H = homography_matrices[frame_idx]
        if not dets or H is None:
            in_front.append(False)
            continue
        near_bbox = None
        for tid, bbox in dets.items():
            if bbox is None:
                continue
            try:
                tid_int = int(tid)
            except (TypeError, ValueError):
                continue
            if tid_int > 0:
                near_bbox = bbox
                break
        if near_bbox is None:
            in_front.append(False)
            continue
        try:
            x1, _y1, x2, y2 = near_bbox[:4]
        except (TypeError, ValueError):
            in_front.append(False)
            continue
        foot_px = (float(x1) + float(x2)) / 2.0
        foot_py = float(y2)
        pt = np.array([[[foot_px, foot_py]]], dtype=np.float32)
        try:
            mapped = cv2.perspectiveTransform(pt, H)
        except cv2.error:
            in_front.append(False)
            continue
        svg_y = (float(mapped[0, 0, 1]) - top_y) / span_y * 78.0
        # In front = on net side of service line (smaller svg_y on near half).
        in_front.append(svg_y < SERVICE_LINE_SVG)

    # Detect contiguous "in front" runs >= MIN_FRAMES_IN_FRONT — each run is one approach.
    approaches: list[int] = []
    i = 0
    while i < len(in_front):
        if in_front[i]:
            j = i
            while j < len(in_front) and in_front[j]:
                j += 1
            if j - i >= MIN_FRAMES_IN_FRONT:
                approaches.append(i)
            i = j
        else:
            i += 1

    # Classify outcome.
    sorted_bounces = sorted(bounces)
    fps_f = float(fps) if fps else 30.0
    rally_end_window_frames = int(round(3.0 * fps_f))

    events: list[dict] = []
    wins = 0
    for start_frame in approaches:
        # Last bounce at or after the approach within the rally-end window
        bounces_after = [b for b in sorted_bounces if start_frame <= b <= start_frame + rally_end_window_frames]
        if not bounces_after:
            # No bounce within 3s => rally died on the approach. Treat as "won" only if
            # there was already an OUT-of-bounds bounce just before (opponent error
            # forced the approach). Otherwise call it lost to be conservative.
            preceding = [b for b in sorted_bounces if start_frame - rally_end_window_frames <= b < start_frame]
            preceding_out = any(b not in in_bounds_set for b in preceding)
            outcome = "won" if preceding_out else "lost"
        else:
            last_bounce = bounces_after[-1]
            outcome = "won" if last_bounce not in in_bounds_set else "lost"
        if outcome == "won":
            wins += 1
        events.append({
            "frame": int(start_frame),
            "time_s": round(start_frame / fps_f, 2),
            "outcome": outcome,
        })

    total = len(approaches)
    win_pct = round(wins / total * 100, 1) if total > 0 else 0.0
    return {
        "approaches": total,
        "wins": wins,
        "win_pct": win_pct,
        "events": events[:50],  # cap to keep payload reasonable
    }


def build_error_summary(
    bounces,
    ball_track,
    homography_matrices,
    court_ref,
    in_bounds_set,
    fps,
    swing_events=None,
    near_player_only: bool = True,
):
    """Direction-only error breakdown attributed to the near (P1) player.

    Tennis convention: an error belongs to the player who failed to make the
    shot. With a fixed near-side camera and only P1's strokes classified, a
    bounce that lands on the FAR half (svg_y < 39) is P1's miss; one on the
    NEAR half is P2's miss and irrelevant to the player we're coaching.

    Net-line bounces (37 ≤ svg_y ≤ 41) are side-ambiguous because the bounce
    is right at the divide. For these we fall back to swing-event pairing —
    the swing whose peak_frame immediately precedes the bounce wins the
    attribution. Net bounces with no nearby swing are dropped (better to
    under-count than mis-attribute).

    Each event carries ``player: 1`` so the frontend can later split out
    opponent errors if/when we model them.

    Classify each kept bounce by direction:
      - long: past either baseline (svg_y < 0 or svg_y > 78)
      - wide: past either sideline (svg_x < 1 or svg_x > 26)
      - net:  y near the net line (37..41) — proxy since we don't track
              ball height directly
    """
    left_x   = court_ref.left_court_line[0][0]
    right_x  = court_ref.right_court_line[0][0]
    top_y    = court_ref.baseline_top[0][1]
    bottom_y = court_ref.baseline_bottom[0][1]
    span_x = max(1.0, right_x - left_x)
    span_y = max(1.0, bottom_y - top_y)
    fps_f = float(fps) if fps else 30.0

    # Strict-sequential bounce → swing pairing for net-line attribution.
    # Each swing claims the next bounce that occurs after its peak_frame
    # and before the following swing's peak_frame. Mirrors the first pass
    # of build_shots' bounce_to_swing builder, scoped to net errors only.
    bounce_to_swing: dict[int, dict] = {}
    if swing_events:
        sorted_swings = sorted(swing_events, key=lambda e: int(e.get("peak_frame", 0)))
        sorted_bounces_for_pair = sorted(bounces)
        cursor = 0
        for i, swing in enumerate(sorted_swings):
            pf = int(swing.get("peak_frame", -1))
            if pf < 0:
                continue
            next_pf = (
                int(sorted_swings[i + 1].get("peak_frame", -1))
                if i + 1 < len(sorted_swings) else None
            )
            while cursor < len(sorted_bounces_for_pair) and sorted_bounces_for_pair[cursor] <= pf:
                cursor += 1
            if cursor >= len(sorted_bounces_for_pair):
                break
            candidate = sorted_bounces_for_pair[cursor]
            if next_pf is not None and candidate >= next_pf:
                continue
            bounce_to_swing[candidate] = swing

    long_n = 0
    wide_n = 0
    net_n = 0
    events: list[dict] = []
    skipped_far_player = 0  # opponent misses we intentionally dropped

    for bframe in sorted(bounces):
        if bframe in in_bounds_set:
            continue
        if bframe >= len(ball_track) or bframe >= len(homography_matrices):
            continue
        bp = ball_track[bframe]
        H = homography_matrices[bframe]
        if bp is None or H is None or bp[0] is None or bp[1] is None:
            continue
        pt = np.array([[[float(bp[0]), float(bp[1])]]], dtype=np.float32)
        try:
            mapped = cv2.perspectiveTransform(pt, H)
        except cv2.error:
            continue
        cx, cy = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
        svg_x = (cx - left_x) / span_x * 27.0
        svg_y = (cy - top_y) / span_y * 78.0

        # ---- Plausibility gate ----
        # The bounce detector occasionally flags transient ball-track noise as
        # bounces — when projected through the homography these land tens of
        # court units past the baselines (e.g. y=-65 = ~65 ft behind the far
        # baseline, physically in the stands). Those aren't real misses and
        # they were inflating the error count to 6 on clips with 0 actual
        # P1 misses. Allow ~6 units past each baseline / ~3 past each
        # sideline; anything farther is treated as detector noise.
        if svg_y < -6.0 or svg_y > 84.0 or svg_x < -3.0 or svg_x > 30.0:
            continue

        # ---- Player attribution ----
        is_net_zone = 37.0 <= svg_y <= 41.0
        if is_net_zone:
            paired = bounce_to_swing.get(int(bframe))
            paired_tid = int(paired.get("track_id", 0)) if paired else 0
            if near_player_only and paired_tid <= 0:
                # Either no swing pairing, or the pairing is the far player.
                # Drop rather than guess.
                skipped_far_player += 1
                continue
        else:
            # svg_y < 39 → ball landed on far half → P1 hit it (and missed)
            # svg_y > 41 → ball landed on near half → P2 hit it (skip)
            on_far_half = svg_y < 39.0
            if near_player_only and not on_far_half:
                skipped_far_player += 1
                continue

        # ---- Direction label ----
        if is_net_zone:
            label = "net"
            net_n += 1
        elif svg_y < 0 or svg_y > 78:
            label = "long"
            long_n += 1
        elif svg_x < 1 or svg_x > 26:
            label = "wide"
            wide_n += 1
        else:
            # Inside the boundary box but flagged OOB by count_in_out_bounces — rare
            # rounding case; lump with "wide" for simplicity.
            label = "wide"
            wide_n += 1
        events.append({
            "frame": int(bframe),
            "time_s": round(bframe / fps_f, 2),
            "direction": label,
            "court_x": round(svg_x, 2),
            "court_y": round(svg_y, 2),
            "player": 1,
        })

    # ---- Missed returns ----
    # An opponent's ball lands IN-BOUNDS on the near (P1) side and P1 doesn't
    # make a swing within a reasonable response window. That's a failure to
    # return — distinct from the OOB errors above (where P1 hit but missed).
    missed_return_n = 0
    if swing_events:
        p1_swings = sorted(
            int(s.get("peak_frame", -1))
            for s in swing_events
            if int(s.get("track_id", 0)) > 0 and int(s.get("peak_frame", -1)) >= 0
        )
        # ±0.5s back / 1.5s forward — covers a P1 mishit (swing slightly
        # before the bounce) AND a normal P1 return (swing 0.5-1.5s after
        # opponent's bounce). Outside this window means P1 made no contact
        # near this ball.
        look_back = int(0.5 * fps_f)
        look_forward = int(1.5 * fps_f)

        for bframe in sorted(bounces):
            if bframe not in in_bounds_set:
                continue  # OOB bounces handled above
            if bframe >= len(ball_track) or bframe >= len(homography_matrices):
                continue
            bp = ball_track[bframe]
            H = homography_matrices[bframe]
            if bp is None or H is None or bp[0] is None or bp[1] is None:
                continue
            pt = np.array([[[float(bp[0]), float(bp[1])]]], dtype=np.float32)
            try:
                mapped = cv2.perspectiveTransform(pt, H)
            except cv2.error:
                continue
            cx, cy = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
            svg_x = (cx - left_x) / span_x * 27.0
            svg_y = (cy - top_y) / span_y * 78.0

            if svg_y < -6.0 or svg_y > 84.0 or svg_x < -3.0 or svg_x > 30.0:
                continue
            if svg_y <= 41.0:
                continue  # far half or right at the net — not P1 territory

            has_p1_contact = any(
                bframe - look_back <= pf <= bframe + look_forward
                for pf in p1_swings
            )
            if has_p1_contact:
                continue  # P1 swung at it (or near it) — not a missed return

            missed_return_n += 1
            events.append({
                "frame": int(bframe),
                "time_s": round(bframe / fps_f, 2),
                "direction": "missed return",
                "court_x": round(svg_x, 2),
                "court_y": round(svg_y, 2),
                "player": 1,
            })

    total = long_n + wide_n + net_n + missed_return_n
    if skipped_far_player or missed_return_n:
        print(
            f"[Errors] near-player only: kept {total} "
            f"(missed_return={missed_return_n}, oob={long_n + wide_n + net_n}); "
            f"skipped {skipped_far_player} opponent/unattributed bounces"
        )
    return {
        "total": total,
        "long": long_n,
        "wide": wide_n,
        "net_err": net_n,
        "missed_return": missed_return_n,
        "events": events[:100],
    }


def count_in_out_bounces(bounces, ball_track, homography_matrices, court_ref):
    """
    Classify each detected bounce as in-bounds or out-of-bounds using the
    court-space coordinates obtained via homography projection.
    """
    left_x   = court_ref.left_court_line[0][0]    # 286
    right_x  = court_ref.right_court_line[0][0]   # 1379
    top_y    = court_ref.baseline_top[0][1]        # 561
    bottom_y = court_ref.baseline_bottom[0][1]     # 2935

    in_bounds = 0
    out_bounds = 0
    in_bounds_set: set[int] = set()
    n_frames = len(homography_matrices)

    for frame_idx in bounces:
        if frame_idx >= len(ball_track) or frame_idx >= n_frames:
            continue
        pos = ball_track[frame_idx]
        H = homography_matrices[frame_idx]
        if pos is None or H is None:
            continue
        bx, by = pos
        if bx is None or by is None:
            continue
        pt = np.array([[[float(bx), float(by)]]], dtype=np.float32)
        try:
            mapped = cv2.perspectiveTransform(pt, H)
        except cv2.error:
            continue
        cx, cy = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

        if left_x <= cx <= right_x and top_y <= cy <= bottom_y:
            in_bounds += 1
            in_bounds_set.add(frame_idx)
        else:
            out_bounds += 1

    return in_bounds, out_bounds, in_bounds_set


def compute_spatial_stats(
    bounces: set,
    ball_track: list,
    homography_matrices: list,
    player_detections: list,
    court_ref,
) -> dict:
    """
    Project bounce and player positions into court reference space and return zone
    statistics used to enrich the AI scouting report.

    Court reference coordinate system (from CourtReference):
      x: 286 (left sideline) → 1379 (right sideline), center=832
      y: 561 (far baseline) → 2935 (near baseline), net=1748
      Singles inner lines: x ∈ [423, 1242]
      Service lines: y=1110 (far), y=2386 (near)
    """
    left_x      = court_ref.left_court_line[0][0]      # 286
    right_x     = court_ref.right_court_line[0][0]     # 1379
    top_y       = court_ref.baseline_top[0][1]          # 561
    bottom_y    = court_ref.baseline_bottom[0][1]       # 2935
    net_y       = court_ref.net[0][1]                   # 1748
    center_x    = court_ref.middle_line[0][0]           # 832
    inner_left  = court_ref.left_inner_line[0][0]      # 423
    inner_right = court_ref.right_inner_line[0][0]     # 1242
    svc_bot_y   = court_ref.bottom_inner_line[0][1]    # 2386  (near service line)


    def project(bx, by, H):
        pt = np.array([[[float(bx), float(by)]]], dtype=np.float32)
        try:
            mapped = cv2.perspectiveTransform(pt, H)
            return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
        except cv2.error:
            return None, None

    def pct(n, total):
        return round(n / total * 100) if total > 0 else 0

    # --- Bounce spatial distribution (in-bounds only) ---
    b_near = b_far = b_left = b_right = b_deep_near = b_alley = valid = 0

    for frame_idx in bounces:
        if frame_idx >= len(ball_track) or frame_idx >= len(homography_matrices):
            continue
        pos = ball_track[frame_idx]
        H = homography_matrices[frame_idx]
        if pos is None or H is None:
            continue
        bx, by = pos
        if bx is None or by is None:
            continue
        cx, cy = project(bx, by, H)
        if cx is None:
            continue
        if not (left_x <= cx <= right_x and top_y <= cy <= bottom_y):
            continue  # out-of-bounds, skip for zone analysis
        valid += 1
        if cy > net_y:
            b_near += 1
            if cy > svc_bot_y:
                b_deep_near += 1
        else:
            b_far += 1
        if cx < center_x:
            b_left += 1
        else:
            b_right += 1
        if cx < inner_left or cx > inner_right:
            b_alley += 1

    bounce_stats = {
        "near_pct":      pct(b_near, valid),
        "far_pct":       pct(b_far, valid),
        "left_pct":      pct(b_left, valid),
        "right_pct":     pct(b_right, valid),
        "deep_near_pct": pct(b_deep_near, b_near) if b_near > 0 else 0,
        "alley_pct":     pct(b_alley, valid),
        "total":         valid,
    }

    # --- Player positioning (near half, sampled every 5 frames) ---
    p_near_left = p_near_right = p_deep = p_forward = p_samples = 0

    for frame_idx in range(0, len(player_detections), 5):
        pd = player_detections[frame_idx]
        H = homography_matrices[frame_idx] if frame_idx < len(homography_matrices) else None
        if not pd or H is None:
            continue
        for bbox in pd.values():
            x1, y1, x2, y2 = bbox
            cx, cy = project((x1 + x2) / 2.0, float(y2), H)
            if cx is None:
                continue
            # Loose bounds check to filter projection artifacts
            if not (left_x - 300 <= cx <= right_x + 300 and top_y - 300 <= cy <= bottom_y + 300):
                continue
            p_samples += 1
            if cy > net_y:
                if cx < center_x:
                    p_near_left += 1
                else:
                    p_near_right += 1
                if cy > svc_bot_y:
                    p_deep += 1
                else:
                    p_forward += 1
    near_samples = p_near_left + p_near_right
    player_stats = {
        "near_pct":         pct(near_samples, p_samples),
        "near_left_pct":    pct(p_near_left, near_samples),
        "near_right_pct":   pct(p_near_right, near_samples),
        "near_deep_pct":    pct(p_deep, near_samples),
        "near_forward_pct": pct(p_forward, near_samples),
        "samples":          p_samples,
    }

    return {"bounces": bounce_stats, "player": player_stats}


def generate_scouting_report(
    stats: dict,
    fps: float,
    num_frames: int,
    position_summary: "dict | None" = None,
    net_approach_summary: "dict | None" = None,
    error_summary: "dict | None" = None,
    rally_summary: "dict | None" = None,
    shots: "list | None" = None,
) -> "str | None":
    """Call GPT-4o-mini to generate a grounded scouting report.

    Grounded in the real coach-insight calculators (position / net / error /
    rally summaries + per-shot records) instead of coarse bounce-zone blobs,
    so the model cites actual numbers rather than inventing them. Every stat
    is passed explicitly; the model is instructed to omit any sentence whose
    stat is N/A.
    """
    try:
        import openai

        duration_sec = round(num_frames / fps) if fps else 0
        duration_str = f"{duration_sec // 60}m {duration_sec % 60}s"

        handedness = stats.get("handedness", "right")

        fh = stats.get("forehand_count", 0)
        bh = stats.get("backhand_count", 0)
        srv = stats.get("serve_count", 0)
        total_strokes = fh + bh + srv

        # Per-stroke error rate + far-court placement, derived from P1 shots[].
        per_oob = {"forehand": 0, "backhand": 0, "serve": 0, "unknown": 0}
        per_total = {"forehand": 0, "backhand": 0, "serve": 0, "unknown": 0}
        far_left = far_right = 0
        if shots:
            for s in shots:
                if s.get("player") != 1:
                    continue
                key = s.get("stroke", "unknown")
                per_total[key] = per_total.get(key, 0) + 1
                if not s.get("in", True):
                    per_oob[key] = per_oob.get(key, 0) + 1
                if s.get("court_y", 99) < 39 and s.get("in"):
                    if s.get("court_x", 13.5) < 13.5:
                        far_left += 1
                    else:
                        far_right += 1

        # Accuracy is the analyzed player's OWN shots only (P1), not the
        # match's combined in/out bounce count.
        p1_total = sum(per_total.values())
        p1_oob = sum(per_oob.values())
        p1_in = p1_total - p1_oob
        p1_acc = f"{round(p1_in / p1_total * 100)}%" if p1_total > 0 else "N/A"

        def err_rate(key: str) -> str:
            t = per_total.get(key, 0)
            return f"{round(per_oob.get(key, 0) / t * 100)}%" if t else "N/A"

        rs = rally_summary or {}
        rs_total = rs.get("total", 0) or 0
        total_rallies = rs_total if rs_total > 0 else stats.get("rally_count", 0)
        avg_length = rs.get("avg_length", "N/A") if rs_total > 0 else "N/A"
        p1_win_rate = rs.get("p1_win_rate", "N/A") if rs_total > 0 else "N/A"
        win_rate_str = f"{p1_win_rate}%" if p1_win_rate != "N/A" else "N/A"
        end_reasons = rs.get("end_reasons", {}) if rs_total > 0 else {}

        ps = position_summary or {}
        na = net_approach_summary or {}
        es = error_summary or {}

        prompt = (
            "You are a Division I tennis performance analyst reviewing a single "
            "match clip.\n"
            "The player analyzed is the one closer to the camera (near side, "
            "P1).\n"
            f"Player handedness: {handedness}.\n\n"
            "Write a concise, data-driven match report under 280 words.\n"
            "Use direct second-person language ('You...').\n"
            "Do NOT mention AI, models, cameras, or homography.\n"
            "Use ONLY the numbers provided — do not invent statistics.\n"
            "If a stat is N/A, omit the sentence that would cite it.\n\n"
            "Structure your response EXACTLY with these six sections:\n"
            "1) Match Snapshot\n"
            "2) Positioning Tendencies\n"
            "3) Error Patterns\n"
            "4) Strengths\n"
            "5) Areas to Improve\n"
            "6) One-Line Coaching Adjustment\n\n"
            "=== MATCH DATA ===\n"
            f"Duration: {duration_str}\n"
            f"Total rallies: {total_rallies} | Avg rally length: {avg_length} "
            f"shots | P1 rally win rate: {win_rate_str}\n"
            f"Rally end reasons: {end_reasons}\n\n"
            f"Strokes (near player): Forehand {fh}, Backhand {bh}, Serve {srv} "
            f"(total {total_strokes})\n"
            f"Your shot accuracy (P1 shots only): {p1_acc} "
            f"({p1_in} of {p1_total} of your shots landed in)\n"
            f"Error breakdown — total P1 errors: {es.get('total', 0)} "
            f"(long: {es.get('long', 0)}, wide: {es.get('wide', 0)}, "
            f"net: {es.get('net_err', 0)}, "
            f"missed returns: {es.get('missed_return', 0)})\n"
            f"Error rate by stroke: forehand {err_rate('forehand')}, "
            f"backhand {err_rate('backhand')}, serve {err_rate('serve')}\n"
            f"Far-court placement (P1 in-bounds shots): left {far_left}, "
            f"right {far_right}\n\n"
            "=== POSITIONING (near player, % of frames) ===\n"
            f"Inside baseline: {ps.get('inside_pct', 'N/A')}% | "
            f"On baseline: {ps.get('on_pct', 'N/A')}% | "
            f"1-10 ft behind: {ps.get('behind_5_10_pct', 'N/A')}% | "
            f"10+ ft behind: {ps.get('behind_10_plus_pct', 'N/A')}%\n\n"
            "=== NET GAME ===\n"
            f"Net approaches: {na.get('approaches', 0)} | "
            f"Win rate on approaches: {na.get('win_pct', 0)}%\n"
        )

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=550,
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[SCOUTING] Report generation failed: {e}")
        return None


_OPPONENT_COURT_MAP = {
    "2": "uc_davis_court2",
    "4": "uc_davis_court4",
    "6": "uc_davis_court6",
}


def _resolve_camera_id(video_path: str) -> str | None:
    """
    Infer camera_id from the video filename.
    Matches filenames containing 'Court' followed by a court number (e.g. StMarys_Court2.mp4).
    Returns None if no match — pipeline falls back to per-frame court detection.
    """
    import re
    stem = Path(video_path).stem
    match = re.search(r'[Cc]ourt(\d+)', stem)
    if match:
        return _OPPONENT_COURT_MAP.get(match.group(1))
    return None


def run_pipeline(video_path: str, match_id: str, local_mode: bool = False, config: PipelineConfig = None):
    """
    Main pipeline entry point (Modal-compatible).
    """
    import sys
    # Belt-and-suspenders in case the Modal Image env var doesn't take.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print(f"[Pipeline] Starting — match_id={match_id[:8]}", flush=True)
    try:
        if config is None:
            config = PipelineConfig()

        # Hold onto the original source path before ensure_720p potentially
        # reassigns video_path to a resized intermediate. The streamable mp4
        # step muxes audio from this so annotated playback isn't silent.
        original_video_path = video_path

        # Auto-resolve camera_id from filename unless explicitly set by caller
        if config.camera_id == PipelineConfig.camera_id:
            resolved = _resolve_camera_id(video_path)
            config.camera_id = resolved
            if resolved:
                print(f"[Pipeline] Camera ID resolved from filename: {resolved}", flush=True)
            else:
                print(f"[Pipeline] No Court# match in filename — using per-frame court detection", flush=True)

        device = config.device

        supabase = None
        # Near-player handedness. Defaults to 'right' if no player_id is
        # bound to this match, the player has no handedness set, or we're in
        # local_mode without DB access. Used inside the swing-classification
        # loop to mirror lefty pose sequences before TCN inference.
        near_player_handedness: str = "right"
        if not local_mode:
            supabase = get_supabase()
            # Heartbeat write — happens BEFORE any model loading so the
            # frontend's processing pane reaches a non-zero progress value the
            # moment Modal enters run_pipeline. Was previously delayed until
            # `update_progress(0.01, "Loading models")` AFTER ~10s of model
            # imports / weight loads, so users saw STARTING for 30-60s.
            # Heartbeat lands AFTER /api/trigger-process and app.py's stage
            # writes already bumped progress to ~0.012. Use 0.018 here so the
            # bar moves forward (not backward) and the user sees handoff into
            # the actual pipeline.
            try:
                supabase.table("matches").update({
                    "status": "processing",
                    "progress": 0.018,
                    "processing_stage": "Starting pipeline",
                }).eq("id", match_id).execute()
                print(f"[Heartbeat] Wrote status=processing progress=0.018 to {match_id[:8]}", flush=True)
            except Exception as e:
                # If processing_stage column missing, retry without it.
                try:
                    supabase.table("matches").update({
                        "status": "processing",
                        "progress": 0.018,
                    }).eq("id", match_id).execute()
                    print(f"[Heartbeat] Wrote status=processing progress=0.018 (no stage col)", flush=True)
                except Exception as e2:
                    print(f"[Heartbeat] FAILED: {e2}", flush=True)

            # Resolve near-player handedness from matches.player_id ->
            # players.handedness. Failure paths (no player_id, missing
            # column, no row) all fall back to 'right' so the pipeline
            # keeps its prior behavior when the migration hasn't been
            # applied or the upload had no player binding.
            try:
                match_resp = (
                    supabase.table("matches")
                    .select("player_id")
                    .eq("id", match_id)
                    .single()
                    .execute()
                )
                player_id = (match_resp.data or {}).get("player_id") if match_resp else None
                if player_id:
                    player_resp = (
                        supabase.table("players")
                        .select("handedness")
                        .eq("id", player_id)
                        .single()
                        .execute()
                    )
                    h = (player_resp.data or {}).get("handedness") if player_resp else None
                    if h in ("left", "right"):
                        near_player_handedness = h
                print(
                    f"[Handedness] near player = {near_player_handedness}"
                    + (f" (player_id={str(player_id)[:8]})" if player_id else " (no player binding)"),
                    flush=True,
                )
            except Exception as e:
                print(f"[Handedness] lookup failed, defaulting to right: {e}", flush=True)

        def update_progress(value: float, stage: "str | None" = None):
            pct = round(float(value) * 100, 1)
            payload: dict = {"progress": round(float(value), 3)}
            if stage is not None:
                payload["processing_stage"] = stage
            label = f" — {stage}" if stage else ""
            if supabase:
                try:
                    resp = supabase.table("matches").update(payload).eq("id", match_id).execute()
                    if hasattr(resp, "data") and resp.data is not None and len(resp.data) == 0:
                        print(f"[Progress] WARNING {pct}%{label} updated 0 rows (match_id={match_id} not found?)", flush=True)
                    else:
                        print(f"[Progress] {pct}%{label}", flush=True)
                except Exception as e:
                    if "processing_stage" in payload:
                        payload.pop("processing_stage")
                        try:
                            supabase.table("matches").update(payload).eq("id", match_id).execute()
                            print(f"[Progress] {pct}%{label} (stage column missing — wrote progress only)", flush=True)
                        except Exception as e2:
                            print(f"[Progress] supabase update failed: {e2}", flush=True)
                    else:
                        print(f"[Progress] supabase update failed: {e}", flush=True)
            else:
                print(f"Progress: {pct}%{label}", flush=True)

        # 0.022 stays monotonic above the 0.018 heartbeat and above app.py's
        # earlier "Loading models" write (0.012). Stage label stays the same
        # so the UI doesn't flicker between identical labels.
        update_progress(0.022, "Loading models")

        # ---------- Initialize models ----------
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

        ball_detector = BallDetector(
            path_model=os.path.join(WEIGHTS_DIR, config.ball_model_weights),
            device=device,
        )

        # Lazy-load: only needed when calibration is absent
        court_detector = None

        player_model_path = os.path.join(WEIGHTS_DIR, config.player_model)
        if not os.path.exists(player_model_path):
            # Not in weights dir — pass name directly so ultralytics auto-downloads
            player_model_path = config.player_model
        player_tracker = PlayerTracker(model_path=player_model_path, device=device, imgsz=config.player_imgsz, conf=config.player_conf)

        bounce_detector = BounceDetector(
            os.path.join(WEIGHTS_DIR, "bounce_detection_weights.cbm"),
            threshold=config.bounce_threshold,
            min_gap_frames=config.bounce_min_gap_frames,
        )

        # Legacy stroke model (kept for fallback)
        legacy_stroke_path = os.path.join(WEIGHTS_DIR, "stroke_classifier_weights.pth")
        stroke_model = None
        if os.path.exists(legacy_stroke_path):
            try:
                stroke_model = ActionRecognition(model_saved_state=legacy_stroke_path)
            except Exception as e:
                print(f"Legacy stroke model load failed: {e}")

        # Pose-based TCN stroke classifier
        tcn_weights_path = None
        if config.stroke_classifier_weights_tcn:
            tcn_weights_path = os.path.join(WEIGHTS_DIR, config.stroke_classifier_weights_tcn)
        pose_stroke_classifier = PoseStrokeClassifier(
            weights_path=tcn_weights_path,
            device=device,
        )
        swing_detector = SwingDetector(
            velocity_threshold=config.swing_velocity_threshold,
            ball_proximity=config.swing_ball_proximity,
        )

        stroke_mode = 'tcn' if tcn_weights_path else 'rule-based'
        print(f"[Pipeline] Models: ball=tracknet, player={config.player_model.replace('.pt','')}@{config.player_imgsz}, bounce=catboost, stroke={stroke_mode}")

        homography_estimator = HomographyEstimator()
        court_ref = CourtReference()

        # ---------- Ensure target resolution ----------
        if config.enforce_720p:
            path_intermediate_video = str(Path(video_path).with_name("intermediate_720p.mp4"))
            video_path = ensure_720p(
                video_path,
                path_intermediate_video,
                target_width=config.target_width,
                target_height=config.target_height
            )
        update_progress(0.05, "Calibrating the court")

        # ---------- Court calibration (skip per-frame detection if available) ----------
        calibrated_H_ref = None
        calibrated_H_frame = None
        calibrated_keypoints = None
        if config.calibration_path and config.camera_id:
            cal_H_ref, cal_H_frame, cal_kps = load_calibration(config.calibration_path, config.camera_id)
            if cal_H_ref is not None:
                calibrated_H_ref = cal_H_ref
                calibrated_H_frame = cal_H_frame
                calibrated_keypoints = cal_kps
                print(f"[Pipeline] Calibration: {config.camera_id} loaded")

        # If no calibration was loaded, detect the court once on the first N frames.
        # This replaces per-frame detection in Pass 2 — the camera is fixed.
        if calibrated_H_ref is None:
            print(f"[Pipeline] No calibration — running court detection on first {config.court_detection_startup_frames} frames")
            court_detector = CourtLineDetector(
                model_path=os.path.join(WEIGHTS_DIR, "keypoints_model.pth"),
                device=device,
            )
            cap_startup = cv2.VideoCapture(video_path)
            startup_frames = []
            for _ in range(config.court_detection_startup_frames):
                ret_s, frm_s = cap_startup.read()
                if ret_s:
                    startup_frames.append(frm_s)
            cap_startup.release()
            calibrated_H_ref, calibrated_keypoints = _detect_court_once(
                startup_frames, court_detector, homography_estimator
            )
            if calibrated_H_ref is not None:
                print(f"[Pipeline] Court detected from startup — reusing for all frames")
            else:
                print("[Pipeline] WARNING: Court detection failed on startup frames")

        # ---------- Pass 1: Ball + Player tracking + pose extraction ----------
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_source = cap.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ball_track = []
        player_detections = []
        pose_keypoints_per_frame = []  # list of {track_id: np.ndarray(17,3)}

        for i in tqdm(range(total_frames), desc="Pass 1: Ball + Player tracking"):
            ret, frame = cap.read()
            if not ret:
                break
            if i % config.ball_detection_interval == 0:
                ball_pos = ball_detector.infer_single(frame)
            else:
                ball_pos = None  # filled by _interpolate_ball_track after loop
            ball_track.append(ball_pos)

            if i % config.player_detection_interval == 0:
                if calibrated_H_frame is not None:
                    player_dict, kps_dict = player_tracker.detect_frame_with_far_roi(frame, calibrated_H_frame)
                else:
                    player_dict, kps_dict = player_tracker.detect_frame(frame)
            else:
                player_dict, kps_dict = {}, {}
            player_detections.append(player_dict)
            pose_keypoints_per_frame.append(kps_dict)

            if i % max(1, total_frames // config.progress_update_frequency) == 0:
                update_progress(0.05 + 0.4 * (i / total_frames), "Following the ball and players")

        # Fill gaps between detection frames using linear bbox interpolation
        player_detections = _interpolate_player_detections(player_detections)
        ball_track_raw = list(ball_track)          # raw detections — used for bounce detection

        # InpaintNet trajectory rectification — fills missed detections using learned physics.
        # Runs on raw detections before linear interpolation.
        # Bounce detector keeps ball_track_raw (pre-rectification) to avoid false inflections.
        if config.enable_inpaint_net:
            inpaint_weights_path = os.path.join(WEIGHTS_DIR, config.inpaint_net_weights)
            if os.path.exists(inpaint_weights_path):
                rectifier = TrajectoryRectifier(
                    weights_path=inpaint_weights_path,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    device=device,
                )
                ball_track = rectifier.rectify(ball_track)
                print(f"[InpaintNet] Rectified trajectory: "
                      f"{sum(1 for p in ball_track_raw if p is not None)} -> "
                      f"{sum(1 for p in ball_track if p is not None)} detected frames")
            else:
                print(f"[InpaintNet] Weights not found at {inpaint_weights_path}, skipping")

        ball_track = _interpolate_ball_track(ball_track)  # smoothed — used for drawing
        # Pose keypoints: forward-fill, then backward-fill to cover frames before first detection
        _last_kps: dict = {}
        for i, kd in enumerate(pose_keypoints_per_frame):
            if kd:
                _last_kps = kd
            elif _last_kps:
                pose_keypoints_per_frame[i] = dict(_last_kps)
        _last_kps = {}
        for i in range(len(pose_keypoints_per_frame) - 1, -1, -1):
            kd = pose_keypoints_per_frame[i]
            if kd:
                _last_kps = kd
            elif _last_kps:
                pose_keypoints_per_frame[i] = dict(_last_kps)
        # Note: bboxes use linear interpolation (smooth player movement);
        # pose keypoints use fill (carry-forward/back) since interpolating
        # 17-keypoint arrays would not produce meaningful intermediate poses.
        # Don't release cap — Pass 2 reuses it by seeking to frame 0

        # Filter players using homography: pick 1 player per side of the net
        H_ref_for_filter = calibrated_H_ref
        if H_ref_for_filter is None:
            print("[Pipeline] WARNING: No homography available — player filtering skipped. Load a calibration file.")

        # Keep raw detections for visualization — include all detections (ROI synthetic IDs too for debugging)
        player_detections_raw = [dict(d) for d in player_detections]

        # DEBUG: log all raw track IDs before filtering
        all_raw_tids = {tid for frame in player_detections for tid in frame}
        print(f"[Player DEBUG] Raw YOLO IDs across all frames: {sorted(all_raw_tids)} ({len(all_raw_tids)} unique)")

        if H_ref_for_filter is not None and len(player_detections) > 0:
            player_detections, pose_keypoints_per_frame = player_tracker.choose_and_filter_players(
                H_ref_for_filter,
                player_detections,
                pose_keypoints_per_frame,
                court_ref=court_ref,
                x_margin=config.far_player_court_x_margin,
                far_player_max_height=config.far_player_max_height,
                far_max_jump_px=config.far_player_max_jump_px,
                far_hold_frames=config.far_player_hold_frames,
            )

        # ---------- Bounce + shot detection ----------
        cap_meta = cv2.VideoCapture(video_path)
        fps = cap_meta.get(cv2.CAP_PROP_FPS)
        cap_meta.release()

        bounces_all = set()
        if bounce_detector.model is not None:
            # Feed the RAW (pre-interpolation) ball track to the bounce detector.
            # smooth_predictions inside the bounce detector handles gap-filling itself.
            # Pre-interpolating before this call double-processes gaps and creates synthetic
            # trajectory points that mimic bounce direction reversals.
            x_ball_raw = [float(p[0]) if p and p[0] is not None else None for p in ball_track_raw]
            y_ball_raw = [float(p[1]) if p and p[1] is not None else None for p in ball_track_raw]

            non_none_raw = sum(1 for x in x_ball_raw if x is not None)

            # Sub-sample to ~30fps if video is higher frame rate
            TARGET_FPS = 30.0
            if fps > TARGET_FPS * 1.1:
                step = fps / TARGET_FPS
                sample_indices = [int(round(i * step)) for i in range(int(len(ball_track) / step))]
                sample_indices = [min(i, len(ball_track) - 1) for i in sample_indices]
                x_sub = [x_ball_raw[i] for i in sample_indices]
                y_sub = [y_ball_raw[i] for i in sample_indices]
                bounces_sub = bounce_detector.predict(x_sub, y_sub, smooth=True)
                bounces_all = {sample_indices[b] for b in bounces_sub if b < len(sample_indices)}
            else:
                bounces_all = bounce_detector.predict(x_ball_raw, y_ball_raw, smooth=True)
            print(f"[Bounce] {len(bounces_all)} bounces | ball tracked: {non_none_raw}/{len(x_ball_raw)} frames")

        shot_frames = detect_shot_frames(ball_track)
        shot_frames_set = set(shot_frames)
        rally_count = calculate_rally_count(shot_frames, fps=fps)

        # ---------- Pose-based swing detection + TCN stroke classification ----------
        # New 4-class stroke counts (replaces legacy 3-class counts when TCN runs)
        pose_stroke_counts = {"Forehand": 0, "Backhand": 0, "Serve/Overhead": 0}
        frame_stroke_labels: dict[int, dict[int, str]] = {}
        swing_events = []
        if config.enable_stroke_recognition:
            swing_events = swing_detector.detect(
                pose_keypoints_per_frame=pose_keypoints_per_frame,
                ball_track=ball_track,
                player_detections=player_detections,
            )
            # frame_stroke_labels: {frame_idx: {track_id: label}}
            # Populated for the 30 frames after each swing peak so Pass 2 can draw it.
            frame_stroke_labels: dict[int, dict[int, str]] = {}
            for event in swing_events:
                # Only the near player (positive track_id) is bound to a
                # roster player_id and therefore has a known handedness.
                # Far player (synthetic negative IDs) keeps the default
                # 'right' until we model opponent handedness too.
                event_handedness = (
                    near_player_handedness if int(event["track_id"]) > 0 else "right"
                )
                seq = extract_pose_sequence(
                    pose_keypoints_per_frame,
                    track_id=event["track_id"],
                    window_start=event["window_start"],
                    window_end=event["window_end"],
                    handedness=event_handedness,
                )
                _probs, label = pose_stroke_classifier.predict(seq)
                event["label"] = label
                pose_stroke_counts[label] = pose_stroke_counts.get(label, 0) + 1
                # Show label from peak frame for 30 frames
                display_end = min(total_frames - 1, event["peak_frame"] + 30)
                for f in range(event["peak_frame"], display_end + 1):
                    frame_stroke_labels.setdefault(f, {})[event["track_id"]] = label
            print(f"[Stroke] {len(swing_events)} swings → FH={pose_stroke_counts.get('Forehand',0)} BH={pose_stroke_counts.get('Backhand',0)} Srv={pose_stroke_counts.get('Serve/Overhead',0)}")

        update_progress(0.5, "Detecting bounce points and stroke types")

        # ---------- Pass 2: Drawing ----------
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Use MJPEG for the intermediate output — fast to write since
        # make_streamable_mp4 re-encodes to H.264 as its final step anyway.
        local_output_path = f"/tmp/{match_id}_processed.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(local_output_path, fourcc, fps, (width, height))

        frame_idx = 0
        last_kps = None
        last_H_ref = None
        homography_matrices = [None] * total_frames

        # Legacy stroke recognition state (for when pose-based is not used)
        stroke_counts = {"Forehand": 0, "Backhand": 0, "Service/Smash": 0}

        # Pre-fill homography and keypoints from calibration if available
        if calibrated_H_ref is not None:
            homography_matrices = [calibrated_H_ref] * total_frames
            last_H_ref = calibrated_H_ref
            last_kps = calibrated_keypoints  # use saved keypoints to draw court lines

        # Pre-compute static minimap background once — court reference never changes.
        # Lighter mid steel-blue so saturated bounce dots really pop. The
        # previous deeper tone made colored dots disappear after the ~3.4×
        # downscale. BGR(180, 145, 110) ≈ RGB(110, 145, 180).
        _minimap_raw = court_ref.build_court_reference()
        _minimap_raw = cv2.dilate(_minimap_raw, np.ones((10, 10), np.uint8))
        minimap_bg = np.full((*_minimap_raw.shape, 3), (180, 145, 110), dtype=np.uint8)
        minimap_bg[_minimap_raw.astype(bool)] = (250, 250, 250)

        for i in tqdm(range(total_frames), desc="Pass 2: Drawing"):
            ret, frame = cap.read()
            if not ret:
                break

            # Court was detected once at startup — use fixed result for all frames.
            kps = last_kps

            output = frame.copy()
            output = draw_ball_trace(output, ball_track, frame_idx,
                                     trace_length=config.trace_length_main,
                                     base_color=config.ball_trace_color)
            output = draw_court_keypoints_and_lines(output, kps)

            if frame_idx < len(player_detections):
                # Draw only the 2 filtered players (near + far) — no coaches, nets, spectators
                output = draw_player_bboxes(output, player_detections[frame_idx])
                output = draw_stroke_labels(output, player_detections[frame_idx],
                                            frame_stroke_labels, frame_idx)

            minimap = minimap_bg.copy()

            if calibrated_H_ref is not None:
                H_ref = calibrated_H_ref
            elif kps is not None:
                H_ref, _ = homography_estimator.estimate(kps)
                if H_ref is not None:
                    last_H_ref = H_ref
                else:
                    H_ref = last_H_ref
            else:
                H_ref = last_H_ref

            homography_matrices[frame_idx] = H_ref

            minimap = draw_minimap_ball_and_bounces(
                minimap=minimap, homography_inv=H_ref, ball_track=ball_track,
                frame_idx=frame_idx, bounces=bounces_all,
                trace_length=config.trace_length_minimap,
                trace_color=config.ball_trace_color,
                trace_min_alpha=config.ball_trace_min_alpha,
                frame_stroke_labels=frame_stroke_labels,
            )
            if frame_idx < len(player_detections):
                minimap = draw_minimap_players(
                    minimap=minimap, homography_inv=H_ref,
                    player_dict=player_detections[frame_idx],
                    color=config.player_bbox_color,
                )

            minimap = cv2.resize(minimap, (config.minimap_width, config.minimap_height))
            # Thin border
            cv2.rectangle(minimap, (1, 1), (config.minimap_width - 2, config.minimap_height - 2),
                          (180, 180, 180), 1, cv2.LINE_AA)
            # Alpha-blend into frame (80% minimap, 20% video)
            roi = output[0:config.minimap_height, 0:config.minimap_width]
            cv2.addWeighted(minimap, 0.82, roi, 0.18, 0, roi)
            output[0:config.minimap_height, 0:config.minimap_width] = roi

            out.write(output)

            # Legacy stroke recognition (runs only when pose-based swing detection
            # produced no events AND the legacy model is loaded).
            if config.enable_stroke_recognition and stroke_model is not None and len(swing_events) == 0:
                pd_at_frame = player_detections[frame_idx] if frame_idx < len(player_detections) else {}
                if pd_at_frame:
                    player_box = list(pd_at_frame.values())[0]
                    try:
                        probs, stroke_label = stroke_model.predict_stroke(frame, player_box)
                        if frame_idx in shot_frames_set and probs is not None:
                            stroke_counts[stroke_label] = stroke_counts.get(stroke_label, 0) + 1
                    except Exception as e:
                        print(f"Stroke prediction error at frame {frame_idx}: {e}")

            frame_idx += 1
            if i % max(1, total_frames // config.progress_update_frequency) == 0:
                update_progress(0.50 + 0.45 * (i / total_frames), "Rendering your annotated recording")

        update_progress(0.95, "Generating heatmaps and scouting report")
        cap.release()
        out.release()

        # ---------- In-bounds / out-of-bounds bounce classification ----------
        in_bounds_bounces, out_bounds_bounces, in_bounds_set = count_in_out_bounces(
            bounces_all, ball_track, homography_matrices, court_ref
        )

        # ---------- Per-shot courtmap data (court coords + stroke type) ----------
        shots_data = build_shots(
            bounces=bounces_all,
            ball_track=ball_track,
            homography_matrices=homography_matrices,
            swing_events=swing_events,
            player_detections=player_detections,
            court_ref=court_ref,
            in_bounds_set=in_bounds_set,
            fps=fps,
        )
        print(f"[Shots] Built {len(shots_data)} courtmap records")

        # ---------- 12x8 occupancy grid for the Coverage viz ----------
        coverage_grid = build_coverage_grid(
            player_detections=player_detections,
            homography_matrices=homography_matrices,
            court_ref=court_ref,
        )
        cov_nonzero = sum(1 for row in coverage_grid for v in row if v > 0)
        print(f"[Coverage] Built 12x8 grid, {cov_nonzero}/96 cells populated")

        # ---------- Coach insights: position / net approach / errors ----------
        position_summary = build_position_summary(
            player_detections=player_detections,
            homography_matrices=homography_matrices,
            court_ref=court_ref,
        )
        print(
            f"[Position] inside={position_summary['inside_pct']}% on={position_summary['on_pct']}% "
            f"5-10ft={position_summary['behind_5_10_pct']}% 10ft+={position_summary['behind_10_plus_pct']}% "
            f"(n={position_summary['n_frames']})"
        )

        net_approach_summary = build_net_approach_summary(
            player_detections=player_detections,
            homography_matrices=homography_matrices,
            court_ref=court_ref,
            bounces=bounces_all,
            in_bounds_set=in_bounds_set,
            fps=fps,
        )
        print(
            f"[NetApproach] approaches={net_approach_summary['approaches']} "
            f"wins={net_approach_summary['wins']} ({net_approach_summary['win_pct']}%)"
        )

        error_summary = build_error_summary(
            bounces=bounces_all,
            ball_track=ball_track,
            homography_matrices=homography_matrices,
            court_ref=court_ref,
            in_bounds_set=in_bounds_set,
            fps=fps,
            swing_events=swing_events,
            near_player_only=True,
        )
        print(
            f"[Errors] total={error_summary['total']} long={error_summary['long']} "
            f"wide={error_summary['wide']} net={error_summary['net_err']}"
        )

        # ---------- Rally state machine ----------
        rallies = build_rallies(
            bounces=bounces_all,
            ball_track=ball_track,
            homography_matrices=homography_matrices,
            swing_events=swing_events,
            in_bounds_set=in_bounds_set,
            court_ref=court_ref,
            fps=fps,
        )
        rally_summary = build_rally_summary(rallies)
        decisive = rally_summary["p1_wins"] + rally_summary["p2_wins"]
        print(
            f"[Rallies] {rally_summary['total']} rallies | "
            f"avg_len={rally_summary['avg_length']} | "
            f"P1 wins {rally_summary['p1_wins']}/{decisive}"
        )

        # Merge stroke counts: prefer pose-based counts when swing events were detected
        if len(swing_events) > 0:
            final_forehand = pose_stroke_counts.get("Forehand", 0)
            final_backhand = pose_stroke_counts.get("Backhand", 0)
            final_serve    = pose_stroke_counts.get("Serve/Overhead", 0)
        else:
            # Fall back to legacy LSTM counts
            final_forehand = stroke_counts.get("Forehand", 0)
            final_backhand = stroke_counts.get("Backhand", 0)
            final_serve    = stroke_counts.get("Service/Smash", 0)

        # ---------- AI Scouting Report ----------
        scouting_report = generate_scouting_report(
            stats={
                "shot_count": len(shot_frames),
                "rally_count": rally_count,
                "bounce_count": len(bounces_all),
                "in_bounds_bounces": in_bounds_bounces,
                "out_bounds_bounces": out_bounds_bounces,
                "forehand_count": final_forehand,
                "backhand_count": final_backhand,
                "serve_count": final_serve,
                "handedness": near_player_handedness,
            },
            fps=fps,
            num_frames=total_frames,
            position_summary=position_summary,
            net_approach_summary=net_approach_summary,
            error_summary=error_summary,
            rally_summary=rally_summary,
            shots=shots_data,
        )

        # ---------- Heatmap generation ----------
        bounce_heatmap_path = None
        player_heatmap_path = None
        player_shot_map_path = None
        local_bounce_path = None
        local_player_path = None
        local_shot_map_path = None

        if config.generate_heatmaps:
            heatmap_dir = f"/tmp/{match_id}_heatmaps"
            local_bounce_path = os.path.join(heatmap_dir, "bounce_heatmap.png")
            local_player_path = os.path.join(heatmap_dir, "player_heatmap.png")
            local_shot_map_path = os.path.join(heatmap_dir, "player_shot_map.png")

            generate_minimap_heatmaps(
                homography_matrices=homography_matrices,
                ball_track=ball_track,
                bounces=bounces_all,
                player_detections=player_detections,
                output_bounce_heatmap=local_bounce_path,
                output_player_heatmap=local_player_path,
                ball_shot_frames=shot_frames,
                frame_stroke_labels=frame_stroke_labels,
                in_bounds_bounces=in_bounds_set,
            )

            generate_player_shot_dot_map(
                homography_matrices=homography_matrices,
                player_detections=player_detections,
                shot_frames=shot_frames,
                frame_stroke_labels=frame_stroke_labels,
                output_path=local_shot_map_path,
            )

            saved = [n for n, p in [("bounce", local_bounce_path), ("player", local_player_path), ("shot_map", local_shot_map_path)] if p and os.path.exists(p)]
            print(f"[Heatmap] Saved: {', '.join(saved)}")

        # ---------- Make streamable + upload ----------
        streamable_output = make_streamable_mp4(
            local_output_path,
            source_audio_path=original_video_path,
        )
        results_path = None

        if not local_mode:
            upload_results = upload_results_parallel(
                local_video_path=streamable_output,
                match_id=match_id,
                local_bounce_path=local_bounce_path,
                local_player_path=local_player_path,
                local_shot_map_path=local_shot_map_path,
            )
            results_path = upload_results.get("results_path")
            bounce_heatmap_path = upload_results.get("bounce_heatmap_path")
            player_heatmap_path = upload_results.get("player_heatmap_path")
            player_shot_map_path = upload_results.get("player_shot_map_path")

            update_data = {
                "status": "done",
                "progress": 1.0,
                "results_path": results_path,
                "fps": fps,
                "num_frames": total_frames,
                # Tennis stats
                "bounce_count": len(bounces_all),
                "shot_count": len(shot_frames),
                "rally_count": rally_count,
                "in_bounds_bounces": in_bounds_bounces,
                "out_bounds_bounces": out_bounds_bounces,
                "forehand_count": final_forehand,
                "backhand_count": final_backhand,
                "serve_count": final_serve,
                "scouting_report": scouting_report,
            }
            # shots column requires the 20260513_add_shots migration — skip gracefully
            try:
                supabase.table("matches").update({"shots": shots_data}).eq("id", match_id).execute()
            except Exception:
                pass
            # coverage_grid requires the 20260513_add_coverage_grid migration — skip gracefully
            try:
                supabase.table("matches").update({"coverage_grid": coverage_grid}).eq("id", match_id).execute()
            except Exception:
                pass
            # Coach-insight summaries require the 20260513_add_coach_insights migration.
            # Each is wrapped independently so a partial migration still persists what it can.
            for col, payload in (
                ("position_summary", position_summary),
                ("net_approach_summary", net_approach_summary),
                ("error_summary", error_summary),
                ("rallies", rallies),
                ("rally_summary", rally_summary),
            ):
                try:
                    supabase.table("matches").update({col: payload}).eq("id", match_id).execute()
                except Exception:
                    pass
            if bounce_heatmap_path:
                update_data["bounce_heatmap_path"] = bounce_heatmap_path
            if player_heatmap_path:
                update_data["player_heatmap_path"] = player_heatmap_path
            # player_shot_map_path requires the column to exist in Supabase;
            # skip gracefully if not yet migrated
            if player_shot_map_path:
                try:
                    supabase.table("matches").update({"player_shot_map_path": player_shot_map_path}).eq("id", match_id).execute()
                except Exception:
                    pass  # column not yet migrated

            supabase.table("matches").update(update_data).eq("id", match_id).execute()
            print(f"[Pipeline] Done — bounces={len(bounces_all)} shots={len(shot_frames)} rallies={rally_count} | in={in_bounds_bounces} out={out_bounds_bounces} | FH={final_forehand} BH={final_backhand} Srv={final_serve}")

            # ---------- Storage cleanup: drop the raw upload ----------
            # Once the processed video and metadata are persisted, the raw clip
            # in `raw-videos` is dead weight — the live app reads from `results`.
            # Keeps storage cost roughly halved (raw ≈ processed in size).
            # Skipped on failed runs so the user can retry without re-uploading.
            # Set DELETE_RAW_AFTER_PROCESS=0 in the Modal env to keep raws.
            if os.environ.get("DELETE_RAW_AFTER_PROCESS", "1") != "0":
                # `input_path` on the matches row is the raw-videos key. Look it
                # up rather than threading it through every callsite, so the
                # cleanup stays correct even if the key was rewritten.
                try:
                    raw_key_row = (
                        supabase.table("matches")
                        .select("input_path")
                        .eq("id", match_id)
                        .single()
                        .execute()
                    )
                    raw_key = (raw_key_row.data or {}).get("input_path")
                    if raw_key and results_path:
                        supabase.storage.from_("raw-videos").remove([raw_key])
                        print(f"[Storage] Removed raw upload {raw_key}")
                except Exception as e:
                    # Non-fatal — log and move on; the recording is already
                    # marked done and the processed video is live.
                    print(f"[Storage] Could not remove raw upload: {e}")
        else:
            print(f"\n✅ Local processing complete!")
            print(f"   Output: {streamable_output}")
            print(f"   FPS: {fps}, Frames: {total_frames}")
            print(f"   Bounces: {len(bounces_all)} ({in_bounds_bounces} in / {out_bounds_bounces} out)")
            print(f"   Shots: {len(shot_frames)}, Rallies: {rally_count}")
            print(f"   Strokes — FH: {final_forehand}, BH: {final_backhand}, Serve: {final_serve}")
            if len(swing_events) > 0:
                print(f"   (Pose-based: {len(swing_events)} swing events detected)")

        events_export = {
            "fps": float(fps),
            "total_frames": int(total_frames),
            "strokes": [
                {
                    "frame": int(e["peak_frame"]),
                    "player": 1 if int(e["track_id"]) > 0 else 2,
                    "track_id": int(e["track_id"]),
                    "label": e.get("label", "Unknown"),
                }
                for e in swing_events
            ],
            "bounces": [
                {"frame": int(b), "in_bounds": b in in_bounds_set}
                for b in sorted(bounces_all)
            ],
            "shots": shots_data,
            "coverage_grid": coverage_grid,
            "rallies": rallies,
            "rally_summary": rally_summary,
        }

        return {
            "status": "done",
            "results_path": results_path,
            "output_file": streamable_output,
            "events": events_export,
            "meta": {
                "fps": fps,
                "num_frames": total_frames,
                "bounce_count": len(bounces_all),
                "shot_count": len(shot_frames),
                "rally_count": rally_count,
            },
        }
    except Exception as e:
        if supabase:
            import logging
            logging.exception("Pipeline failed for match %s", match_id)
            supabase.table("matches").update({
                "status": "failed",
                "error": "Processing failed. Please try again.",
                "progress": 0
            }).eq("id", match_id).execute()
        raise


if __name__ == "__main__":
    import argparse
    import uuid
    import shutil

    parser = argparse.ArgumentParser(description="Run tennis analysis pipeline locally")
    parser.add_argument("--video", type=str, required=True, help="Path to input .mp4 video")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to output video")
    args = parser.parse_args()

    test_match_id = str(uuid.uuid4())
    print(f"🎾 Running tennis analysis pipeline locally")
    print(f"   Input: {args.video}")
    print(f"   Output: {args.output}")
    print(f"   Test match_id: {test_match_id}\n")

    try:
        result = run_pipeline(
            video_path=args.video,
            match_id=test_match_id,
            local_mode=True,
        )
        if result.get("output_file") and os.path.exists(result["output_file"]):
            shutil.copy(result["output_file"], args.output)
            print(f"\n✅ Pipeline complete! Output saved to: {args.output}")
        else:
            print(f"\n⚠️  Warning: Output file not found")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
