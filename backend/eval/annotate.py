"""
Ground-truth annotation tool for short test clips.

Usage:
    python -m backend.eval.annotate path/to/clip.mp4

Hotkeys:
    space      play / pause
    j / l      step -1 / +1 frame
    , / .      step -10 / +10 frames
    1 / 2      set current player tag (P1 = near, P2 = far)
    f          stamp Forehand at current frame
    b          stamp Backhand
    s          stamp Serve
    v          stamp Volley
    i          stamp bounce IN-bounds at current frame
    o          stamp bounce OUT-of-bounds at current frame
    u          undo most recent event
    w          save (write JSON, stay open)
    q          save and quit
    ESC        quit without saving

Output: <video>.gt.json next to the clip. Re-running on a clip with an existing
.gt.json loads it for further editing.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2


STROKE_HOTKEYS = {"f": "FH", "b": "BH", "s": "Serve", "v": "Volley"}
HUD_HEIGHT = 90


def load_gt(gt_path: Path) -> dict:
    if gt_path.exists():
        return json.loads(gt_path.read_text())
    return {"strokes": [], "bounces": []}


def save_gt(gt_path: Path, video_path: Path, fps: float, total: int, strokes: list, bounces: list) -> None:
    payload = {
        "video": str(video_path),
        "fps": float(fps),
        "total_frames": int(total),
        "strokes": sorted(strokes, key=lambda e: e["frame"]),
        "bounces": sorted(bounces, key=lambda e: e["frame"]),
    }
    gt_path.write_text(json.dumps(payload, indent=2))


def draw_hud(frame, frame_idx: int, total: int, player: int, last_event: str, n_strokes: int, n_bounces: int):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - HUD_HEIGHT), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    info = f"Frame {frame_idx:>4}/{total - 1}   Player P{player}   Strokes {n_strokes}   Bounces {n_bounces}"
    cv2.putText(frame, info, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if last_event:
        cv2.putText(frame, last_event, (10, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)
    keys = "space play  j/l +-1  ,/. +-10  1/2 player  f/b/s/v stroke  i/o bounce  u undo  w save  q quit"
    cv2.putText(frame, keys, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)


def annotate(video_path: Path) -> None:
    if not video_path.exists():
        print(f"ERROR: video not found at {video_path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: failed to open {video_path}", file=sys.stderr)
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    gt_path = video_path.with_suffix(video_path.suffix + ".gt.json")
    existing = load_gt(gt_path)
    strokes = list(existing.get("strokes", []))
    bounces = list(existing.get("bounces", []))
    history: list[tuple[str, int]] = [
        ("stroke", i) for i in range(len(strokes))
    ] + [("bounce", i) for i in range(len(bounces))]
    if strokes or bounces:
        print(f"Loaded {len(strokes)} strokes, {len(bounces)} bounces from {gt_path}")

    frame_idx = 0
    current_player = 1
    playing = False
    last_event = ""
    window = "annotate"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    cached_frame_idx = -1
    cached_frame = None

    while True:
        if frame_idx != cached_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame_idx = max(0, min(frame_idx, total - 1))
                continue
            cached_frame_idx = frame_idx
            cached_frame = frame

        display = cached_frame.copy()
        draw_hud(display, frame_idx, total, current_player, last_event, len(strokes), len(bounces))
        cv2.imshow(window, display)

        wait_ms = max(1, int(1000 / fps)) if playing else 0
        key = cv2.waitKey(wait_ms)

        if playing:
            if frame_idx >= total - 1:
                playing = False
            else:
                frame_idx += 1

        if key == -1:
            continue

        if key == 27:  # ESC
            cv2.destroyAllWindows()
            cap.release()
            print("Quit without saving.")
            return

        ch = chr(key & 0xFF) if 0 <= (key & 0xFF) < 128 else ""

        if ch == "q":
            save_gt(gt_path, video_path, fps, total, strokes, bounces)
            print(f"Saved {len(strokes)} strokes, {len(bounces)} bounces -> {gt_path}")
            break
        if ch == "w":
            save_gt(gt_path, video_path, fps, total, strokes, bounces)
            last_event = f"saved -> {gt_path.name}"
            print(last_event)
        elif ch == " ":
            playing = not playing
        elif ch == "j":
            frame_idx = max(0, frame_idx - 1)
        elif ch == "l":
            frame_idx = min(total - 1, frame_idx + 1)
        elif ch == ",":
            frame_idx = max(0, frame_idx - 10)
        elif ch == ".":
            frame_idx = min(total - 1, frame_idx + 10)
        elif ch == "1":
            current_player = 1
            last_event = "player -> P1"
        elif ch == "2":
            current_player = 2
            last_event = "player -> P2"
        elif ch in STROKE_HOTKEYS:
            label = STROKE_HOTKEYS[ch]
            ev = {"frame": frame_idx, "player": current_player, "type": label}
            strokes.append(ev)
            history.append(("stroke", len(strokes) - 1))
            last_event = f"+ stroke {label} P{current_player} @ frame {frame_idx}"
            print(last_event)
        elif ch in ("i", "o"):
            in_bounds = ch == "i"
            ev = {"frame": frame_idx, "in_bounds": in_bounds}
            bounces.append(ev)
            history.append(("bounce", len(bounces) - 1))
            last_event = f"+ bounce {'IN' if in_bounds else 'OUT'} @ frame {frame_idx}"
            print(last_event)
        elif ch == "u":
            if not history:
                last_event = "nothing to undo"
                continue
            kind, _ = history.pop()
            if kind == "stroke" and strokes:
                removed = strokes.pop()
                last_event = f"- stroke {removed['type']} P{removed['player']} @ frame {removed['frame']}"
            elif kind == "bounce" and bounces:
                removed = bounces.pop()
                last_event = f"- bounce {'IN' if removed['in_bounds'] else 'OUT'} @ frame {removed['frame']}"
            else:
                last_event = "undo: stale history entry skipped"
            print(last_event)

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Annotate ground truth events on a test clip.")
    ap.add_argument("video", help="Path to .mp4")
    args = ap.parse_args()
    annotate(Path(args.video).resolve())


if __name__ == "__main__":
    main()
