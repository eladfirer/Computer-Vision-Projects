# Computer Vision Projects

A collection of production-grade computer vision projects demonstrating knowledge in image processing, video analysis, and classical CV algorithms. Each project is implemented with clean architecture, comprehensive documentation, and modular design.

---

## Projects Overview

| Project | Description | Tech Stack |
|---------|-------------|------------|
| [**Stereo Mosaicing**](Stereo-Mosaicing/) | Panoramic video mosaics using optical flow & RANSAC | Python, OpenCV, NumPy |
| [**Image Blending**](Image-Blending/) | Laplacian pyramid blending & hybrid images | Python, OpenCV, NumPy, Matplotlib |
| [**Image Editing Tool**](Image-Editing-Tool/) | JSON-configurable image processing pipeline | Java, AWT/Swing |

---

## Stereo Mosaicing

A robust pipeline for generating panoramic video mosaics from video sequences using sparse optical flow, rigid motion estimation, and dynamic strip stitching.

**Key Algorithms:**
- Shi-Tomasi Corner Detection
- Pyramidal Lucas-Kanade Optical Flow
- SVD-based Rigid Transform Estimation
- RANSAC Outlier Rejection
- Dynamic Strip Stitching with Vertical Drift Compensation

```bash
cd Stereo-Mosaicing
pip install -r requirements.txt
python main.py --config configTrees.json
```

[View Full Documentation →](Stereo-Mosaicing/README.md)

---

## Image Blending

A multi-scale image processing toolkit implementing Laplacian Pyramid Blending for seamless compositing and Hybrid Images for frequency-based optical illusions.

**Key Algorithms:**
- Gaussian & Laplacian Pyramid Construction
- Separable 1D Convolution
- Multi-Scale Frequency Blending
- FFT Spectrum Analysis

```bash
cd Image-Blending
pip install -r requirements.txt
python main.py imageblending inputs/source.jpg inputs/target.jpg
```

[View Full Documentation →](Image-Blending/README.md)

---

## Image Editing Tool

A Java-based image processing application that applies a sequence of filters and adjustments based on JSON configuration files.

**Supported Operations:**
- Brightness & Contrast Adjustment
- Saturation Control
- Box Blur & Sharpening
- Sobel Edge Detection

```bash
cd Image-Editing-Tool/AdvancedImageEditingSystem
./edit-image --config config.json
```

[View Full Documentation →](Image-Editing-Tool/README.md)

---

## Repository Structure

```
Computer-Vision-Projects/
├── Stereo-Mosaicing/          # Video mosaic generation
│   ├── src/                   # Modular Python package
│   ├── inputs/                # Source videos
│   ├── outputs/               # Generated results
│   └── config*.json           # Configuration files
│
├── Image-Blending/            # Pyramid blending toolkit
│   ├── src/                   # Core algorithms & pipelines
│   ├── inputs/                # Source images
│   ├── outputs/               # Blended results
│   └── main.py                # CLI entry point
│
├── Image-Editing-Tool/        # Java image processor
│   └── AdvancedImageEditingSystem/
│       ├── src/               # Java source files
│       └── edit-image.jar     # Runnable JAR
│
└── README.md                  # This file
```

---

## Technical Highlights

### Software Engineering
- **Modular Architecture** — Clean separation of concerns (core algorithms, pipelines, utilities)
- **Type Safety** — Full type hints (Python) and strict validation
- **Configuration-Driven** — JSON-based configuration with dataclass validation
- **Comprehensive Logging** — Structured logging replacing print statements

### Computer Vision
- **Motion Estimation** — Optical flow with robust outlier rejection
- **Multi-Scale Processing** — Gaussian/Laplacian pyramids for frequency manipulation
- **Interactive Tools** — Custom OpenCV UI for mask drawing and alignment
- **Frequency Analysis** — FFT visualization and spectrum comparison

---

## Author

**Elad Firer**

Some of the projects were developed as part of the Image Processing Course at The Hebrew University of Jerusalem.

---

## License

Educational use. See individual project READMEs for details.
