"""Main pipeline orchestration class for video mosaicing."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import MosaicConfig
from .stitching import calculate_strip_boundaries_dynamic, stitch_from_strips
from .tracking import RobustLKTracker
from .utils import (
    apply_frame_transformations,
    ensure_output_directory,
    load_video_frames,
)

logger = logging.getLogger(__name__)


class VideoMosaicPipeline:
    """Main pipeline for generating video mosaics from input videos.
    
    Orchestrates the complete pipeline: frame loading, tracking, stitching,
    and video generation with proper error handling and logging.
    """
    
    def __init__(self, config: MosaicConfig) -> None:
        """Initialize pipeline with configuration.
        
        Args:
            config: MosaicConfig instance
        """
        config.validate()
        self.config = config
        self.tracker = RobustLKTracker(config)
        logger.info("VideoMosaicPipeline initialized")
    
    def run(self) -> Path:
        """Execute the complete video mosaicing pipeline.
        
        Returns:
            Path to generated video file
            
        Raises:
            FileNotFoundError: If input video doesn't exist
            ValueError: If insufficient frames or processing fails
        """
        logger.info("Starting video mosaic pipeline")
        
        # 1. Load frames
        raw_frames = load_video_frames(
            self.config.video_path,
            max_frames=self.config.frames_limit
        )
        frames = raw_frames[::self.config.stride]
        
        if len(frames) < 2:
            raise ValueError(
                f"Insufficient frames after stride ({len(frames)} < 2)"
            )
        
        logger.info(
            f"Processing {len(frames)} frames (stride={self.config.stride})"
        )
        
        # 2. Apply transformations
        frames = apply_frame_transformations(
            frames,
            rotation=self.config.input_rotation,
            flip=self.config.flip_result
        )
        
        # 3. Track motion
        motions = self.tracker.track_video_sequence(frames)
        
        if not motions:
            raise ValueError("Motion tracking failed - no motion vectors computed")
        
        # 4. Generate output video
        output_path = self._generate_output_path()
        ensure_output_directory(output_path)
        
        self._generate_video_mosaic(
            frames,
            motions,
            output_path
        )
        
        logger.info(f"Pipeline complete. Output: {output_path}")
        return output_path
    
    def _generate_output_path(self) -> Path:
        """Generate output file path from video name.
        
        Returns:
            Path to output video file
        """
        video_name = Path(self.config.video_path).stem
        output_dir = Path("outputs") / video_name
        output_path = output_dir / f"{video_name}_mosaic.mp4"
        return output_path
    
    def _generate_video_mosaic(
        self,
        frames: list[np.ndarray],
        motions: list[np.ndarray],
        output_path: Path
    ) -> None:
        """Generate video mosaic with centered panoramas.
        
        Creates a video where each frame contains one panorama, but all panoramas
        are centered in the frame so the central image changes rather than panning.
        
        Args:
            frames: List of processed video frames
            motions: List of motion vectors from tracking
            output_path: Path to save output video
        """
        logger.info("Generating video mosaic with centered panoramas")
        
        n_frames = self.config.video_mosaic_frames
        fps = self.config.video_mosaic_fps
        
        # Slit ratios: 1.0 (right) -> 0.0 (left)
        slit_ratios = np.linspace(1.0, 0.0, n_frames)
        
        frame_w = frames[0].shape[1]
        is_color = len(frames[0].shape) == 3
        
        # Generate all panoramas first to determine dimensions
        panoramas = []
        panorama_widths = []
        panorama_heights = []
        
        for idx, ratio in enumerate(slit_ratios):
            if idx % 10 == 0:
                logger.debug(
                    f"Generating panorama {idx}/{n_frames} (Ratio: {ratio:.2f})"
                )
            
            # Calculate strips for this slit ratio
            strips = calculate_strip_boundaries_dynamic(
                motions, frame_w, slit_ratio=ratio
            )
            
            # Stitch panorama
            pano = stitch_from_strips(frames, strips, motions)
            panoramas.append(pano)
            panorama_widths.append(pano.shape[1])
            panorama_heights.append(pano.shape[0])
        
        # Find minimum dimensions for video (crop during video writing, not before)
        min_width = min(panorama_widths)
        min_height = min(panorama_heights)
        
        # Final video dimensions (after rotation if needed)
        if self.config.rotate_result_back and self.config.input_rotation:
            final_h, final_w = min_width, min_height
        else:
            final_h, final_w = min_height, min_width
        
        # Initialize video writer with minimum dimensions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(
            str(output_path), fourcc, fps, (final_w, final_h)
        )
        
        if not video.isOpened():
            raise RuntimeError(f"Failed to initialize video writer: {output_path}")
        
        try:
            # Generate each frame - one panorama per frame, centered
            for idx, ratio in enumerate(slit_ratios):
                if idx % 10 == 0:
                    logger.debug(
                        f"Rendering frame {idx}/{n_frames} (Ratio: {ratio:.2f})"
                    )
                
                # Get the panorama for this frame
                pano = panoramas[idx]
                pano_h, pano_w = pano.shape[:2]
                
                # Crop from center to match minimum dimensions (like ex4.py)
                start_y = (pano_h - min_height) // 2
                start_x = (pano_w - min_width) // 2
                cropped = pano[start_y:start_y + min_height, start_x:start_x + min_width]
                
                # Create video frame from cropped panorama
                video_frame = cropped.copy()
                
                # Rotate entire frame if needed
                if self.config.rotate_result_back and self.config.input_rotation:
                    if self.config.input_rotation == "CW":
                        video_frame = cv2.rotate(
                            video_frame, cv2.ROTATE_90_COUNTERCLOCKWISE
                        )
                    elif self.config.input_rotation == "CCW":
                        video_frame = cv2.rotate(
                            video_frame, cv2.ROTATE_90_CLOCKWISE
                        )
                
                # Ensure dimensions match after rotation
                if video_frame.shape[0] != final_h or video_frame.shape[1] != final_w:
                    video_frame = cv2.resize(video_frame, (final_w, final_h))
                
                # Write frame
                video.write(video_frame)
            
            logger.info(f"Video saved to: {output_path}")
        
        finally:
            video.release()

