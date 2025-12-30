"""Optical flow tracking using Lucas-Kanade with RANSAC filtering."""

import logging
from typing import Optional

import cv2
import numpy as np

from .config import MosaicConfig
from .geometry import manual_ransac

logger = logging.getLogger(__name__)


class RobustLKTracker:
    """Robust Lucas-Kanade tracker with RANSAC-based outlier rejection.
    
    Tracks features between consecutive frames using optical flow, then uses
    RANSAC to estimate the dominant rigid motion while filtering outliers.
    """
    
    def __init__(self, config: MosaicConfig) -> None:
        """Initialize tracker with configuration.
        
        Args:
            config: MosaicConfig instance with tracking parameters
        """
        self.enable_x = config.enable_x
        self.enable_y = config.enable_y
        self.enable_rotation = config.enable_rotation
        self.y_range = config.tracking_y_range
        self.ransac_threshold = config.ransac_threshold
        
        # Feature detection parameters (Shi-Tomasi)
        self.feature_params = {
            "maxCorners": config.feature_params.max_corners,
            "qualityLevel": config.feature_params.quality_level,
            "minDistance": config.feature_params.min_distance,
            "blockSize": config.feature_params.block_size
        }
        
        # Lucas-Kanade parameters
        self.lk_params = {
            "winSize": config.lk_params.win_size,
            "maxLevel": config.lk_params.max_level,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        }
        
        logger.debug(
            f"Tracker initialized: x={self.enable_x}, y={self.enable_y}, "
            f"rot={self.enable_rotation}, y_range={self.y_range}"
        )
    
    def track_frame_pair(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> np.ndarray:
        """Track motion between two consecutive frames.
        
        Args:
            img1: First frame (BGR or grayscale)
            img2: Second frame (BGR or grayscale)
            
        Returns:
            Motion vector [dx, dy, theta_degrees]. Returns zeros if tracking fails.
        """
        # Convert to grayscale if needed
        gray1 = self._to_grayscale(img1)
        gray2 = self._to_grayscale(img2)
        
        h, w = gray1.shape
        
        # Create mask for tracking region (y_range)
        mask = np.zeros_like(gray1)
        y_start = max(0, int(h * self.y_range[0]))
        y_end = min(h, int(h * self.y_range[1]))
        
        if y_start >= y_end:
            logger.warning(f"Invalid y_range, using full frame")
            y_start, y_end = 0, h
        
        mask[y_start:y_end, :] = 255
        
        # Detect features
        p0 = cv2.goodFeaturesToTrack(gray1, mask=mask, **self.feature_params)
        if p0 is None or len(p0) < 5:
            logger.warning("Insufficient features detected")
            return np.zeros(3)
        
        # Track features using optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, p0, None, **self.lk_params
        )
        if p1 is None:
            logger.warning("Optical flow tracking failed")
            return np.zeros(3)
        
        # Filter successful tracks
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        if len(good_new) < 5:
            logger.warning("Insufficient good tracks")
            return np.zeros(3)
        
        # Estimate motion using RANSAC
        final_p = manual_ransac(
            good_old,
            good_new,
            self.enable_x,
            self.enable_y,
            self.enable_rotation,
            threshold=self.ransac_threshold
        )
        
        # Return negative (inverse) motion
        return -final_p
    
    def track_video_sequence(
        self,
        video_frames: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Track motion across entire video sequence.
        
        Args:
            video_frames: List of video frames
            
        Returns:
            List of motion vectors, one per frame transition
        """
        if len(video_frames) < 2:
            logger.error("Need at least 2 frames for tracking")
            return []
        
        motions: list[np.ndarray] = []
        logger.info(f"Tracking {len(video_frames)} frames...")
        
        for i in range(len(video_frames) - 1):
            motion = self.track_frame_pair(video_frames[i], video_frames[i + 1])
            motions.append(motion)
            
            if (i + 1) % 50 == 0:
                logger.debug(f"Tracked {i + 1}/{len(video_frames) - 1} frame pairs")
        
        logger.info(f"Tracking complete: {len(motions)} motion vectors computed")
        return motions
    
    @staticmethod
    def _to_grayscale(img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed.
        
        Args:
            img: Input image (BGR or grayscale)
            
        Returns:
            Grayscale image
        """
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

