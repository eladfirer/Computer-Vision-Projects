# Video Mosaic - Production-Grade Video Mosaicing Pipeline

A modular, production-ready Python package for generating video mosaics using optical flow tracking and dynamic strip stitching.

## Package Structure

```
Stereo-Mosaicing/
├── src/
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration dataclasses with validation
│   ├── geometry.py          # SVD-based rigid transforms and RANSAC
│   ├── tracking.py         # Lucas-Kanade optical flow tracker
│   ├── stitching.py        # Dynamic strip calculation and panorama stitching
│   ├── utils.py            # Video I/O and logging utilities
│   └── pipeline.py         # Main orchestration class
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Features

- **Type Safety**: Full type hints throughout the codebase
- **Modular Design**: Clean separation of concerns with dedicated modules
- **Error Handling**: Comprehensive validation and error handling
- **Logging**: Structured logging with configurable levels
- **Configuration**: Type-safe dataclass-based configuration with validation
- **Documentation**: Google-style docstrings for all public APIs

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
python main.py --config configTrees.json
python main.py --config configTrees.json --verbose
```

### Programmatic Usage

```python
from src.config import MosaicConfig
from src.pipeline import VideoMosaicPipeline

# Load configuration
config = MosaicConfig.from_json("configTrees.json")

# Create and run pipeline
pipeline = VideoMosaicPipeline(config)
output_path = pipeline.run()
```

## Configuration

The configuration file is a JSON file with the following structure:

```json
{
    "video_path": "inputs/video.mp4",
    "frames_limit": 200,
    "stride": 1,
    "input_rotation": "CW",
    "rotate_result_back": true,
    "flip_result": false,
    "tracking_y_range": [0.0, 1.0],
    "feature_params": {
        "maxCorners": 200,
        "qualityLevel": 0.01,
        "minDistance": 30,
        "blockSize": 9
    },
    "lk_params": {
        "winSize": [21, 21],
        "maxLevel": 3
    },
    "ransac_threshold": 2.0,
    "enable_rotation": false,
    "enable_x": true,
    "enable_y": true,
    "video_mosaic_frames": 200,
    "video_mosaic_fps": 50
}
```

## Module Documentation

### `src/config.py`
- `MosaicConfig`: Main configuration dataclass with validation
- `FeatureParams`: Feature detection parameters
- `LKParams`: Lucas-Kanade optical flow parameters

### `src/geometry.py`
- `compute_rigid_transform_manual()`: SVD-based rigid transform computation
- `manual_ransac()`: RANSAC-based robust transform estimation

### `src/tracking.py`
- `RobustLKTracker`: Lucas-Kanade tracker with RANSAC filtering

### `src/stitching.py`
- `calculate_strip_boundaries_dynamic()`: Dynamic strip boundary calculation
- `stitch_from_strips()`: Panorama stitching from strips

### `src/utils.py`
- `load_video_frames()`: Video frame loading with error handling
- `apply_frame_transformations()`: Frame rotation and flipping
- `setup_logging()`: Logging configuration

### `src/pipeline.py`
- `VideoMosaicPipeline`: Main pipeline orchestration class

## Requirements

- Python 3.9+
- numpy >= 1.24.0
- opencv-python >= 4.8.0
- matplotlib >= 3.7.0

## License

This code is part of a Computer Vision engineering portfolio project.

