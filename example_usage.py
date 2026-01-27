"""
Example usage of CourtCheck modules.

This script demonstrates how to use the refactored modules
for court detection and ball tracking.
"""

import cv2
import numpy as np
from pathlib import Path

# Example 1: Court Detection Only
def example_court_detection():
    """Example of using court detection on a single image."""
    from court_detection_module import CourtDetectionProcessor
    
    # Initialize processor
    processor = CourtDetectionProcessor(
        model_path="./models/weights/court_detection_model.pth",
        device="cuda",  # or "cpu"
    )
    
    # Load image
    img = cv2.imread("test_frame.jpg")
    
    # Detect keypoints
    result = processor.process_frame(img)
    
    # Visualize
    output = processor.visualize_keypoints(img, result["keypoints"])
    cv2.imwrite("output_court.jpg", output)
    
    print(f"Detected {len(result['keypoints'])} keypoints")
    print(f"Homography matrix computed: {result['homography_matrix'] is not None}")


# Example 2: Ball Detection Only
def example_ball_detection():
    """Example of using ball detection on a video."""
    from ball_detection import BallDetector
    
    # Initialize detector
    detector = BallDetector(
        model_path="./models/weights/tracknet_weights.pt",
        device="cuda",
    )
    
    # Load video
    cap = cv2.VideoCapture("test_video.mp4")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # Detect ball
    ball_track = detector.process_video(frames)
    
    print(f"Detected ball in {sum(1 for x, y in ball_track if x is not None)} frames")


# Example 3: Full Video Processing
def example_full_processing():
    """Example of full video processing pipeline."""
    from court_detection_module import CourtDetectionProcessor
    from ball_detection import BallDetector
    from video_processor import VideoProcessor
    
    # Initialize components
    court_processor = CourtDetectionProcessor(
        model_path="./models/weights/court_detection_model.pth",
        device="cuda",
    )
    
    ball_detector = BallDetector(
        model_path="./models/weights/tracknet_weights.pt",
        device="cuda",
    )
    
    # Create processor
    processor = VideoProcessor(
        court_processor=court_processor,
        ball_detector=ball_detector,
    )
    
    # Process video
    result = processor.process(
        input_path="input_video.mp4",
        output_path="output_video.mp4",
        draw_court=True,
        draw_ball=True,
        draw_trace=True,
        trace_length=7,
    )
    
    print(f"Processing complete!")
    print(f"Total frames: {result['total_frames']}")
    print(f"Output saved to: {result['output_path']}")


# Example 4: Using with Modal
def example_modal_usage():
    """Example of using Modal deployment."""
    import modal
    
    # Connect to deployed app
    app = modal.App.lookup("courtcheck")
    
    # Process video
    result = app.process_video.remote(
        video_path="/videos/input_video.mp4",
        output_path="/videos/output_video.mp4",
        model_weights_path="/models/weights",
    )
    
    print(f"Processing complete: {result}")


if __name__ == "__main__":
    print("CourtCheck Usage Examples")
    print("=" * 50)
    print("\nAvailable examples:")
    print("1. example_court_detection() - Court detection on single image")
    print("2. example_ball_detection() - Ball detection on video")
    print("3. example_full_processing() - Full video processing")
    print("4. example_modal_usage() - Modal deployment usage")
    print("\nUncomment the example you want to run below:")
    
    # Uncomment to run:
    # example_court_detection()
    # example_ball_detection()
    # example_full_processing()
    # example_modal_usage()
