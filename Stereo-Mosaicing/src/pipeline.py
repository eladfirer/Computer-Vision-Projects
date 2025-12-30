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
        """Generate video mosaic with animated slit scanning.
        
        Creates a video where the panorama is gradually revealed by scanning
        the slit position from right to left across all frames.
        
        Args:
            frames: List of processed video frames
            motions: List of motion vectors from tracking
            output_path: Path to save output video
        """
        logger.info("Generating video mosaic with animated slit scanning")
        
        n_frames = self.config.video_mosaic_frames
        fps = self.config.video_mosaic_fps
        
        # Slit ratios: 1.0 (right) -> 0.0 (left)
        slit_ratios = np.linspace(1.0, 0.0, n_frames)
        
        # Calculate base panorama dimensions (unrotated)
        temp_strips = calculate_strip_boundaries_dynamic(
            motions, frames[0].shape[1], slit_ratio=0.5
        )
        temp_pano = stitch_from_strips(frames, temp_strips, motions)
        
        unrotated_h, unrotated_w = temp_pano.shape[:2]
        frame_w = frames[0].shape[1]
        
        # Canvas dimensions (before rotation)
        canvas_w = unrotated_w + frame_w
        canvas_h = unrotated_h
        
        # Final video dimensions (after rotation if needed)
        if self.config.rotate_result_back and self.config.input_rotation:
            # After rotation, width and height swap
            final_h, final_w = canvas_w, canvas_h
        else:
            final_h, final_w = canvas_h, canvas_w
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(
            str(output_path), fourcc, fps, (final_w, final_h)
        )
        
        if not video.isOpened():
            raise RuntimeError(f"Failed to initialize video writer: {output_path}")
        
        try:
            # Generate each frame
            for idx, ratio in enumerate(slit_ratios):
                if idx % 10 == 0:
                    logger.debug(
                        f"Rendering frame {idx}/{n_frames} (Ratio: {ratio:.2f})"
                    )
                
                # Calculate strips for this slit ratio
                strips = calculate_strip_boundaries_dynamic(
                    motions, frame_w, slit_ratio=ratio
                )
                
                # Stitch panorama
                pano_frame = stitch_from_strips(frames, strips, motions)
                
                # Resize to match base dimensions
                if pano_frame.shape[0] != unrotated_h or pano_frame.shape[1] != unrotated_w:
                    pano_frame = cv2.resize(pano_frame, (unrotated_w, unrotated_h))
                
                # Create video frame canvas
                video_frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                
                # Calculate panorama position (animated from right to left)
                start_x = int(ratio * (canvas_w - unrotated_w))
                end_x = start_x + unrotated_w
                start_x = max(0, start_x)
                end_x = min(canvas_w, end_x)
                
                # Place panorama on canvas
                video_frame[:, start_x:end_x] = pano_frame
                
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

