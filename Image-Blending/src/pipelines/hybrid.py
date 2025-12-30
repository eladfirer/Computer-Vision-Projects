"""Hybrid image creation pipeline.

Hybrid images combine low-frequency content from one image with high-frequency
content from another, creating images that appear different when viewed from
different distances (far = low-freq image, close = high-freq image).
"""

import cv2
import numpy as np
from typing import Tuple

from src.core.filters import get_gaussian_kernel, low_pass_filter_pyramid, high_pass_filter_pyramid
from src.utils.visualization import save_hybrid_visualization


def create_hybrid_image(
    im1_path: str,
    im2_path: str,
    depth: int = 4,
    low_weight: float = 1.3,
    high_weight: float = 0.7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates a hybrid image from two input images.
    
    A hybrid image combines:
    - Low-frequency component from im1 (visible from far)
    - High-frequency component from im2 (visible from close)
    
    The formula: $H = w_{low} \\cdot L(im1) + w_{high} \\cdot H(im2)$
    
    where:
    - $L(im1)$ = low-pass filtered im1 (blurred, smooth)
    - $H(im2)$ = high-pass filtered im2 (edges, details)
    
    Args:
        im1_path: Path to low-frequency image (seen from far).
        im2_path: Path to high-frequency image (seen from close).
        depth: Number of pyramid reduction levels (blur level). Recommended: 4-5.
        low_weight: Weight for low-frequency component (1.0-1.5).
        high_weight: Weight for high-frequency component (0.5-0.7).
    
    Returns:
        Tuple of (hybrid_image, low_freq_component, high_freq_component).
    
    Raises:
        FileNotFoundError: If images cannot be loaded.
    """
    print(f"Creating Hybrid Image...")
    print(f"Low Freq (Far): {im1_path} [Weight: {low_weight}]")
    print(f"High Freq (Close): {im2_path} [Weight: {high_weight}]")
    
    img_far = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)
    img_close = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE)
    
    if img_far is None:
        raise FileNotFoundError(f"Cannot load {im1_path}")
    if img_close is None:
        raise FileNotFoundError(f"Cannot load {im2_path}")
    
    # Resize to same dimensions
    img_far = cv2.resize(img_far, (1024, 1024))
    img_close = cv2.resize(img_close, (1024, 1024))
    
    kernel = get_gaussian_kernel()
    
    # Extract frequency components
    low_freq_component = low_pass_filter_pyramid(img_far, depth, kernel).astype(np.float32)
    high_freq_component = high_pass_filter_pyramid(img_close, depth, kernel).astype(np.float32)
    
    # Combine with weights
    hybrid_float = (low_freq_component * low_weight) + (high_freq_component * high_weight)
    hybrid_image = np.clip(hybrid_float, 0, 255).astype(np.uint8)
    
    return hybrid_image, low_freq_component, high_freq_component


def hybrid_images(im1_path: str, im2_path: str) -> None:
    """Complete hybrid image creation pipeline.
    
    Creates a hybrid image and saves:
    - Hybrid result
    - Low-frequency component
    - High-frequency component
    - Multi-scale visualization
    
    Args:
        im1_path: Path to low-frequency image (far view).
        im2_path: Path to high-frequency image (close view).
    """
    res, low, high = create_hybrid_image(im1_path, im2_path, depth=4)
    
    # Save components
    cv2.imwrite("hybrid_low_freq_component.jpg", np.clip(low, 0, 255).astype(np.uint8))
    cv2.imwrite("hybrid_high_freq_component.jpg", np.clip(high + 128, 0, 255).astype(np.uint8))
    
    # Save result
    cv2.imwrite("hybrid_result.jpg", res)
    
    # Save multi-scale visualization
    save_hybrid_visualization(res, "hybrid_pyramid_view.jpg")
    
    print("Hybrid process done.")
    cv2.imshow("Hybrid Result", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

