## Postprocess
import numpy as np
import pandas as pd


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

