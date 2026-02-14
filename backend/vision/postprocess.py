## Postprocess
from sympy.geometry import Line, Point2D
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance

def line_intersection(line1, line2):
    """
    Find 2 lines intersection point
    """
    l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
    l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))

    intersection = l1.intersection(l2)
    point = None
    if len(intersection) > 0:
        if isinstance(intersection[0], Point2D):
            point = intersection[0].coordinates
    return point


def refine_kps(img, x_ct, y_ct, crop_size=40):
    refined_x_ct, refined_y_ct = x_ct, y_ct

    img_height, img_width = img.shape[:2]
    x_min = max(x_ct - crop_size, 0)
    x_max = min(img_height, x_ct + crop_size)
    y_min = max(y_ct - crop_size, 0)
    y_max = min(img_width, y_ct + crop_size)

    img_crop = img[x_min:x_max, y_min:y_max]
    lines = detect_lines(img_crop)
    # print('lines = ', lines)

    if len(lines) > 1:
        lines = merge_lines(lines)
        if len(lines) == 2:
            inters = line_intersection(lines[0], lines[1])
            if inters:
                new_x_ct = int(inters[1])
                new_y_ct = int(inters[0])
                if (
                    new_x_ct > 0
                    and new_x_ct < img_crop.shape[0]
                    and new_y_ct > 0
                    and new_y_ct < img_crop.shape[1]
                ):
                    refined_x_ct = x_min + new_x_ct
                    refined_y_ct = y_min + new_y_ct
    return refined_y_ct, refined_x_ct


def is_scene_cut(
    prev_frame, curr_frame, frame_idx=None, threshold=0.03, pixel_diff_thresh=12
):
    """
    Hybrid scene cut detector using HSV histograms + pixel difference fallback.

    Args:
        prev_frame: Previous BGR frame.
        curr_frame: Current BGR frame.
        frame_idx: Frame index for logging (optional).
        threshold: Threshold for histogram Bhattacharyya distance.
        pixel_diff_thresh: Optional pixel diff fallback threshold.

    Returns:
        True if scene cut is detected.
    """
    if prev_frame is None or curr_frame is None:
        return False

    # HSV Histogram comparison
    prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)

    hist_prev = cv2.calcHist([prev_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_curr = cv2.calcHist([curr_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])

    cv2.normalize(hist_prev, hist_prev)
    cv2.normalize(hist_curr, hist_curr)

    hist_diff = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_BHATTACHARYYA)

    # Pixel-wise mean abs diff fallback
    pixel_diff = np.mean(cv2.absdiff(prev_frame, curr_frame))

    return hist_diff > threshold or pixel_diff > pixel_diff_thresh


def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)[1]
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=30)
    lines = np.squeeze(lines)
    if len(lines.shape) > 0:
        if len(lines) == 4 and not isinstance(lines[0], np.ndarray):
            lines = [lines]
    else:
        lines = []
    return lines


def merge_lines(lines):
    lines = sorted(lines, key=lambda item: item[0])
    mask = [True] * len(lines)
    new_lines = []

    for i, line in enumerate(lines):
        if mask[i]:
            for j, s_line in enumerate(lines[i + 1 :]):
                if mask[i + j + 1]:
                    x1, y1, x2, y2 = line
                    x3, y3, x4, y4 = s_line
                    dist1 = distance.euclidean((x1, y1), (x3, y3))
                    dist2 = distance.euclidean((x2, y2), (x4, y4))
                    if dist1 < 20 and dist2 < 20:
                        line = np.array(
                            [
                                int((x1 + x3) / 2),
                                int((y1 + y3) / 2),
                                int((x2 + x4) / 2),
                                int((y2 + y4) / 2),
                            ]
                        )
                        mask[i + j + 1] = False
            new_lines.append(line)
    return new_lines


def detect_shot_frames(ball_track, min_change_frames=25):
    """
    Detect frames where a player hits the ball by finding y-direction reversals
    in the ball trajectory.

    Args:
        ball_track: list of (x, y) | None — ball center positions per frame
        min_change_frames: minimum sustained direction change to count as a hit

    Returns:
        list of frame indices where shots occurred
    """
    # Extract y-coordinates, using NaN for missing detections
    y_values = []
    for pos in ball_track:
        if pos is not None and pos[0] is not None:
            y_values.append(float(pos[1]))
        else:
            y_values.append(np.nan)

    df = pd.DataFrame({'mid_y': y_values})
    df['mid_y_rolling_mean'] = df['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
    df['delta_y'] = df['mid_y_rolling_mean'].diff()

    shot_frames = []
    threshold = int(min_change_frames * 1.2)

    for i in range(1, len(df) - threshold):
        delta_curr = df['delta_y'].iloc[i]
        delta_next = df['delta_y'].iloc[i + 1]

        if pd.isna(delta_curr) or pd.isna(delta_next):
            continue

        negative_change = delta_curr > 0 and delta_next < 0
        positive_change = delta_curr < 0 and delta_next > 0

        if negative_change or positive_change:
            change_count = 0
            for j in range(i + 1, i + threshold + 1):
                delta_j = df['delta_y'].iloc[j]
                if pd.isna(delta_j):
                    continue

                if negative_change and delta_curr > 0 and delta_j < 0:
                    change_count += 1
                elif positive_change and delta_curr < 0 and delta_j > 0:
                    change_count += 1

            if change_count > min_change_frames - 1:
                shot_frames.append(i)

    return shot_frames

