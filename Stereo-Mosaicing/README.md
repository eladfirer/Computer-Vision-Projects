
#### INPUT
![Slit Scan Demo](gifs/ReadmeGif1.gif)
#### OUTPUT
![Slit Scan Demo](gifs/ReadmeGIf.gif)
# Video Mosaic 

A professional computer vision system that generates panoramic video mosaics from video sequences. This project implements a robust pipeline combining sparse optical flow, rigid motion estimation, and dynamic strip stitching to create seamless panoramas and "slit-scan" video effects.
## 🧠 Algorithmic Core

This pipeline relies on a sequence of classical computer vision algorithms to estimate camera motion and reconstruct the scene, such as Shi-Tomasi, Lucas-Kanade, SVD, and RANSAC.

### 1. Motion Estimation Pipeline

The system does not rely on simple frame differencing. Instead, it uses a robust feature-tracking approach:

**Feature Detection (Shi-Tomasi):**
We utilize the Shi-Tomasi Corner Detector (`cv2.goodFeaturesToTrack`) to identify high-contrast "corners" in the image that are invariant to translation. This provides a stable set of points to track.

**Sparse Optical Flow (Lucas-Kanade):**
We track these specific keypoints into the next frame using Pyramidal Lucas-Kanade Optical Flow (`cv2.calcOpticalFlowPyrLK`). The pyramidal approach allows the tracker to handle larger displacements by processing the image at multiple scales.

### 2. Geometry & Robust Estimation

Raw optical flow data is noisy due to moving objects in the scene or sensor noise. To resolve the global camera motion:

**Rigid Transform via SVD:**
We model the camera motion as a Rigid Transformation (Rotation $\theta$ + Translation $t_x, t_y$). To solve this, we compute the optimal transformation between point cloud $A$ (previous frame) and $B$ (current frame).

1. Center the datasets.
2. Compute the covariance matrix $H = A_{centered}^T B_{centered}$.
3. Perform Singular Value Decomposition (SVD) on $H$ to derive the optimal Rotation Matrix $R$.
4. Calculate translation $t$ based on the rotated centroids.

**Outlier Rejection (RANSAC):**
To ignore moving objects (e.g., a car moving across a static background), we implement a manual RANSAC (Random Sample Consensus) loop:

1. Randomly sample a minimal set of point pairs.
2. Compute the rigid transform model for this sample.
3. Project all points using this model and calculate the projection error (Euclidean distance).
4. Count "inliers" (points with error < `ransac_threshold`).
5. Repeat and keep the model with the highest inlier count.

### 3. Dynamic Stitching & Manifold Projection

Once the global motion vectors are known, we construct the mosaic:

**Dynamic Strip Calculation:**
Instead of projecting entire frames (which causes perspective distortion), we extract specific vertical "strips" from each frame. The width of these strips is dynamic:
- Fast camera motion $\rightarrow$ Wider strips (to prevent gaps).
- Slow camera motion $\rightarrow$ Narrower strips (to minimize redundancy).

The strip boundaries are calculated based on the differential motion $dx$ between frames.

**Vertical Drift Compensation:**
Handheld video often contains unstable vertical jitter. The stitching engine accumulates the vertical motion ($dy$) over time (`cum_dy`) and dynamically adjusts the vertical plotting position on the canvas. The canvas height is automatically calculated to accommodate the full vertical range of the camera path.

**Slit-Scan Video Generation:**
The pipeline generates an animated video output. It iterates the "slit position" (the column index from which strips are sampled) from the right side of the frame to the left (Ratio $1.0 \rightarrow 0.0$). This creates a parallax-like video effect where the panorama appears to be scanned across time.

## 🏗️ Package Structure

The project is structured as a modular Python package to ensure type safety, ease of testing, and maintainability.

```
Stereo-Mosaicing/
├── src/
│   ├── config.py           # Data Validation Layer
│   │                       # Uses @dataclass to strictly type-check JSON inputs
│   │                       # Validates logic (e.g., frames_limit >= -1)
│   │
│   ├── geometry.py         # The Mathematical Core
│   │                       # Implements compute_rigid_transform_manual (SVD)
│   │                       # Implements manual_ransac loop
│   │
│   ├── tracking.py         # Computer Vision Layer
│   │                       # RobustLKTracker class
│   │                       # Wraps OpenCV functions and injects RANSAC logic
│   │
│   ├── stitching.py        # Image Processing Layer
│   │                       # calculate_strip_boundaries_dynamic
│   │                       # stitch_from_strips (Canvas creation & pixel copying)
│   │
│   ├── pipeline.py         # Orchestration Layer
│   │                       # VideoMosaicPipeline class
│   │                       # Manages the flow: Load -> Track -> Generate Video
│   │
│   └── utils.py            # Infrastructure
│                           # Logging setup, Video I/O, Matrix transformations
│
├── main.py                 # CLI Entry Point
├── inputs/                 # Directory for source videos
├── outputs/                # Directory for generated results
└── configTrees.json        # Example configuration
```

## 🚀 Installation & Usage

### Prerequisites

- Python 3.9+
- NumPy, OpenCV, Matplotlib

### Installation

```bash
pip install -r requirements.txt
```

### Running the Pipeline

You can run a specific configuration or process all configuration files in the directory.

**Run a single experiment:**
```bash
python main.py --config configTrees.json
```

**Run with verbose logging (Debug mode):**
```bash
python main.py --config configTrees.json --verbose
```

**Batch process all configs:**
```bash
python main.py --all-configs
```

## ⚙️ Configuration

The system is fully data-driven via JSON configuration files.

```json
{
    "video_path": "inputs/video.mp4",
    "frames_limit": 300,             // Process only first N frames (-1 for all)
    "stride": 1,                     // Skip every Nth frame for speed
    
    // Transformations
    "input_rotation": "CW",          // Rotate input: "CW", "CCW", or null
    "rotate_result_back": true,      // Rotate final video back to original orientation
    "flip_result": false,            // Horizontally flip frames
    
    // Tracking Region
    "tracking_y_range": [0.0, 1.0],  // Vertical range [start, end] for feature tracking (0.0-1.0)
    
    // Feature Detection (Shi-Tomasi)
    "feature_params": {
        "maxCorners": 200,           // Maximum number of corners to detect
        "qualityLevel": 0.01,        // Minimum quality threshold (0-1)
        "minDistance": 30,           // Minimum distance between corners (pixels)
        "blockSize": 9               // Size of neighborhood for corner detection
    },
    
    // Lucas-Kanade Optical Flow
    "lk_params": {
        "winSize": [21, 21],         // Search window size [width, height]
        "maxLevel": 3                // Maximum pyramid level (0 = no pyramid)
    },
    
    // Motion Estimation
    "ransac_threshold": 2.0,         // Inlier threshold in pixels for RANSAC
    "enable_rotation": false,        // Solve for rotation (theta)?
    "enable_x": true,                // Solve for horizontal translation (dx)?
    "enable_y": true,                // Solve for vertical translation (dy)?
    
    // Output Settings
    "video_mosaic_frames": 200,      // Number of frames in output animation
    "video_mosaic_fps": 50           // Frames per second for output video
}
```

## License

This project is part of Image Proccessing Course at The Hebrew University.
