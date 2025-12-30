"""Utility functions for video I/O and logging setup."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_video_frames(
    video_path: Union[str, Path],
    max_frames: int = -1
) -> list[np.ndarray]:
    """Load frames from video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum frames to load (-1 for all frames)
        
    Returns:
        List of video frames as numpy arrays
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened or has no frames
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    logger.info(f"Loading video from: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    frames: list[np.ndarray] = []
    count = 0
    
    try:
        while cap.isOpened():
            if max_frames > 0 and count >= max_frames:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame)
            count += 1
    finally:
        cap.release()
    
    if not frames:
        raise ValueError(f"No frames loaded from video: {video_path}")
    
    logger.info(f"Loaded {len(frames)} frames from video")
    return frames


def apply_frame_transformations(
    frames: list[np.ndarray],
    rotation: Optional[str] = None,
    flip: bool = False
) -> list[np.ndarray]:
    """Apply rotation and/or flip transformations to frames.
    
    Args:
        frames: List of input frames
        rotation: Rotation direction ("CW" or "CCW") or None
        flip: Whether to horizontally flip frames
        
    Returns:
        List of transformed frames
    """
    transformed = frames.copy()
    
    # Apply rotation
    if rotation == "CW":
        logger.info("Rotating frames: 90° Clockwise")
        transformed = [
            cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE) for f in transformed
        ]
    elif rotation == "CCW":
        logger.info("Rotating frames: 90° Counter-Clockwise")
        transformed = [
            cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE) for f in transformed
        ]
    
    # Apply flip
    if flip:
        logger.info("Flipping frames horizontally")
        transformed = [cv2.flip(f, 1) for f in transformed]
    
    return transformed


def ensure_output_directory(output_path: Union[str, Path]) -> Path:
    """Ensure output directory exists, create if needed.
    
    Args:
        output_path: Path to output file
        
    Returns:
        Path object for the output directory
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.parent

