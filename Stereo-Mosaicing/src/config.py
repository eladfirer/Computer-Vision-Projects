"""Configuration management using dataclasses for type safety and validation."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class FeatureParams:
    """Parameters for feature detection (Shi-Tomasi corner detector)."""
    
    max_corners: int = 200
    quality_level: float = 0.01
    min_distance: int = 30
    block_size: int = 9
    
    @classmethod
    def from_dict(cls, data: dict) -> "FeatureParams":
        """Create FeatureParams from dictionary."""
        return cls(
            max_corners=data.get("maxCorners", 200),
            quality_level=data.get("qualityLevel", 0.01),
            min_distance=data.get("minDistance", 30),
            block_size=data.get("blockSize", 9)
        )


@dataclass
class LKParams:
    """Parameters for Lucas-Kanade optical flow."""
    
    win_size: tuple[int, int] = (21, 21)
    max_level: int = 3
    
    @classmethod
    def from_dict(cls, data: dict) -> "LKParams":
        """Create LKParams from dictionary."""
        win_size = data.get("winSize", [21, 21])
        return cls(
            win_size=(win_size[0], win_size[1]),
            max_level=data.get("maxLevel", 3)
        )


@dataclass
class MosaicConfig:
    """Main configuration dataclass for video mosaicing pipeline.
    
    Validates all inputs and provides type-safe access to configuration parameters.
    """
    
    # Video input
    video_path: str
    frames_limit: int = 200
    stride: int = 1
    
    # Transformations
    input_rotation: Optional[Literal["CW", "CCW"]] = None
    rotate_result_back: bool = False
    flip_result: bool = False
    
    # Tracking parameters
    tracking_y_range: tuple[float, float] = (0.0, 1.0)
    feature_params: FeatureParams = field(default_factory=FeatureParams)
    lk_params: LKParams = field(default_factory=LKParams)
    ransac_threshold: float = 2.0
    
    # Motion constraints
    enable_rotation: bool = False
    enable_x: bool = True
    enable_y: bool = True
    
    # Video mosaic output
    video_mosaic_frames: int = 60
    video_mosaic_fps: int = 30
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "MosaicConfig":
        """Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            MosaicConfig instance with loaded parameters
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is malformed
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
        
        # Validate required fields
        if "video_path" not in data:
            raise ValueError("Missing required field: video_path")
        
        # Extract nested configs
        feature_params = FeatureParams.from_dict(
            data.get("feature_params", {})
        )
        lk_params = LKParams.from_dict(
            data.get("lk_params", {})
        )
        
        # Extract tracking_y_range
        y_range = data.get("tracking_y_range", [0.0, 1.0])
        if not isinstance(y_range, list) or len(y_range) != 2:
            raise ValueError("tracking_y_range must be a list of 2 floats")
        
        config = cls(
            video_path=data["video_path"],
            frames_limit=data.get("frames_limit", 200),
            stride=data.get("stride", 1),
            input_rotation=data.get("input_rotation"),
            rotate_result_back=data.get("rotate_result_back", False),
            flip_result=data.get("flip_result", False),
            tracking_y_range=(float(y_range[0]), float(y_range[1])),
            feature_params=feature_params,
            lk_params=lk_params,
            ransac_threshold=data.get("ransac_threshold", 2.0),
            enable_rotation=data.get("enable_rotation", False),
            enable_x=data.get("enable_x", True),
            enable_y=data.get("enable_y", True),
            video_mosaic_frames=data.get("video_mosaic_frames", 60),
            video_mosaic_fps=data.get("video_mosaic_fps", 30)
        )
        
        logger.info(f"Loaded config from {json_path}")
        return config
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.frames_limit < -1:
            raise ValueError("frames_limit must be >= -1 (-1 means no limit)")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if not (0.0 <= self.tracking_y_range[0] < self.tracking_y_range[1] <= 1.0):
            raise ValueError("tracking_y_range must be [0.0, 1.0] with start < end")
        if self.ransac_threshold <= 0:
            raise ValueError("ransac_threshold must be > 0")
        if self.video_mosaic_frames < 1:
            raise ValueError("video_mosaic_frames must be >= 1")
        if self.video_mosaic_fps <= 0:
            raise ValueError("video_mosaic_fps must be > 0")

