"""
Video Processing Module.

This module handles the main video processing pipeline combining
court detection, ball tracking, and visualization.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Main video processor that combines all detection modules.
    """
    
    def __init__(
        self,
        court_processor,
        ball_detector,
        stroke_classifier=None,
        bounce_detector=None,
    ):
        """
        Initialize the video processor.
        
        Args:
            court_processor: CourtDetectionProcessor instance
            ball_detector: BallDetector instance
            stroke_classifier: StrokeClassifier instance (optional)
            bounce_detector: BounceDetector instance (optional)
        """
        self.court_processor = court_processor
        self.ball_detector = ball_detector
        self.stroke_classifier = stroke_classifier
        self.bounce_detector = bounce_detector
    
    def process(
        self,
        input_path: str,
        output_path: str,
        draw_court: bool = True,
        draw_ball: bool = True,
        draw_trace: bool = True,
        trace_length: int = 7,
    ) -> Dict:
        """
        Process a video file and generate output.
        
        Args:
            input_path: Path to input video
            output_path: Path to save output video
            draw_court: Whether to draw court keypoints and lines
            draw_ball: Whether to draw ball position
            draw_trace: Whether to draw ball trajectory
            trace_length: Number of previous positions to show in trace
            
        Returns:
            Dictionary with processing results
        """
        # Read video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {width}x{height} @ {fps} fps, {total_frames} frames")
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        logger.info(f"Loaded {len(frames)} frames")
        
        # Process frames
        processed_frames = []
        ball_track = []
        homography_matrices = []
        stroke_predictions = []
        
        # First pass: detect ball in all frames
        logger.info("Pass 1/2: Ball detection")
        for i, frame in enumerate(frames):
            # Court detection (optional)
            court_result = None
            if self.court_processor is not None:
                court_result = self.court_processor.process_frame(frame)
                homography_matrices.append(court_result["homography_matrix"])
            else:
                homography_matrices.append(None)
            
            # Ball detection
            x, y = self.ball_detector.detect_ball(frame)
            ball_track.append((x, y))
            
            if (i + 1) % 100 == 0:
                logger.info(f"Ball detection: {i + 1}/{len(frames)} frames")
        
        # Detect bounces (requires full ball track)
        logger.info("Detecting bounces...")
        bounces = []
        if self.bounce_detector is not None:
            bounces = self.bounce_detector.detect_bounces(ball_track)
            logger.info(f"Detected {len(bounces)} bounces")
        
        # Second pass: annotate frames
        logger.info("Pass 2/2: Annotation and stroke classification")
        for i, frame in enumerate(frames):
            output_frame = frame.copy()
            x, y = ball_track[i]
            
            # Draw court (if available)
            if draw_court and self.court_processor is not None:
                court_result = self.court_processor.process_frame(frame)
                if court_result is not None and court_result["keypoints"] is not None:
                    output_frame = self.court_processor.visualize_keypoints(
                        output_frame,
                        court_result["keypoints"],
                        draw_lines=True,
                    )
            
            # Stroke classification (optional)
            stroke_type, stroke_conf = ("Unknown", 0.0)
            if self.stroke_classifier is not None:
                stroke_type, stroke_conf = self.stroke_classifier.classify_stroke(frame)
                stroke_predictions.append((stroke_type, stroke_conf))
                
                # Draw stroke label (if confident)
                if stroke_conf > 0.6:
                    cv2.putText(
                        output_frame,
                        f"{stroke_type} ({stroke_conf:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 255),  # Magenta
                        2
                    )
            
            # Draw ball and trace
            if draw_ball and x is not None and y is not None:
                # Draw ball trace
                if draw_trace:
                    trace_start = max(0, i - trace_length)
                    for j in range(trace_start, i):
                        bx, by = ball_track[j]
                        if bx is not None and by is not None:
                            radius = max(2, trace_length - (i - j))
                            color = (255, 255, 0)  # Yellow
                            cv2.circle(output_frame, (int(bx), int(by)), radius, color, -1)
                
                # Draw current ball position
                cv2.circle(output_frame, (int(x), int(y)), 5, (0, 255, 255), -1)
            
            # Draw bounce indicator
            if i in bounces and self.bounce_detector is not None:
                output_frame = self.bounce_detector.annotate_bounces(
                    output_frame,
                    (x, y) if x is not None else (0, 0),
                    is_bounce=True
                )
            
            processed_frames.append(output_frame)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Annotation: {i + 1}/{len(frames)} frames")
        
        # Write output video
        self._write_video(processed_frames, output_path, fps, width, height)
        
        logger.info(f"Processing complete. Output saved to: {output_path}")
        
        # Calculate statistics
        total_bounces = len(bounces)
        ball_detected_frames = sum(1 for x, y in ball_track if x is not None)
        
        # Stroke statistics
        stroke_stats = {}
        if stroke_predictions:
            for stroke_type, conf in stroke_predictions:
                if conf > 0.6:  # Only count confident predictions
                    stroke_stats[stroke_type] = stroke_stats.get(stroke_type, 0) + 1
        
        return {
            "output_path": output_path,
            "total_frames": len(frames),
            "ball_track": ball_track,
            "homography_matrices": homography_matrices,
            "bounces": bounces,
            "total_bounces": total_bounces,
            "ball_detected_frames": ball_detected_frames,
            "ball_detection_rate": ball_detected_frames / len(frames) if frames else 0,
            "stroke_statistics": stroke_stats,
            "fps": fps,
            "duration_seconds": len(frames) / fps if fps > 0 else 0,
        }
    
    def _write_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: float,
        width: int,
        height: int,
    ):
        """Write processed frames to video file."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        logger.info(f"Video written to {output_path}")
