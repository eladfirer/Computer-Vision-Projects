"""Convolution, Gaussian kernels, and frequency filtering operations.

This module implements separable convolution and custom Gaussian kernels
for efficient image filtering operations used in pyramid construction.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def get_gaussian_kernel() -> np.ndarray:
    """Returns the standard 1D Gaussian kernel for pyramid operations.
    
    The kernel [1, 4, 6, 4, 1] / 16 is a binomial approximation of a Gaussian.
    This kernel is separable, meaning we can apply it first horizontally (1x5)
    and then vertically (5x1) for efficient 2D convolution.
    
    Returns:
        A 1x5 numpy array representing the normalized Gaussian kernel.
    """
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32).reshape(1, 5)
    return kernel / 16.0


def create_custom_gaussian_kernel(k_size: int, sigma: float) -> np.ndarray:
    """Generates a 1D Gaussian kernel with specified size and standard deviation.
    
    The Gaussian kernel is computed using the formula:
    $G(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{x^2}{2\\sigma^2}}$
    
    After computing the unnormalized values, the kernel is normalized to sum to 1.
    
    Args:
        k_size: The size of the kernel (must be odd, e.g., 35).
        sigma: The standard deviation (spread) of the Gaussian blur.
    
    Returns:
        A 1D numpy array of shape (1, k_size) representing the normalized kernel.
    """
    center = k_size // 2
    x = np.arange(-center, center + 1, dtype=np.float32)
    
    # Compute Gaussian: exp(-x^2 / (2 * sigma^2))
    kernel_1d = np.exp(-(x**2) / (2 * (sigma**2)))
    
    # Normalize so sum = 1
    kernel_1d /= kernel_1d.sum()
    
    return kernel_1d.reshape(1, k_size)


def convolve(image: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    """Performs separable 2D convolution using a 1D kernel.
    
    Separable convolution reduces computational complexity from O(n²k²) to O(n²k)
    where n is image size and k is kernel size. The process:
    1. Convolve horizontally with the 1D kernel (1x5 or 1xk)
    2. Convolve vertically with the transposed kernel (5x1 or kx1)
    
    Uses BORDER_REFLECT_101 to handle edge pixels by mirroring.
    
    Args:
        image: Input image (grayscale or color) as numpy array.
        kernel_1d: 1D kernel of shape (1, k) for separable convolution.
    
    Returns:
        Convolved image with same shape as input.
    """
    # Horizontal pass: apply kernel_1d
    temp_image = cv2.filter2D(image, -1, kernel_1d, borderType=cv2.BORDER_REFLECT_101)
    # Vertical pass: apply transposed kernel
    result_image = cv2.filter2D(temp_image, -1, kernel_1d.T, borderType=cv2.BORDER_REFLECT_101)
    
    return result_image


def reduce_image(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Reduces image size by half using blur and decimation.
    
    This implements the "reduce" operation for Gaussian pyramids:
    1. Blur the image using convolution (anti-aliasing)
    2. Decimate by keeping every 2nd pixel (::2, ::2)
    
    The blur step is critical to prevent aliasing when downsampling.
    
    Args:
        image: Input image (grayscale or color).
        kernel: 1D Gaussian kernel for blurring.
    
    Returns:
        Reduced image with dimensions (h/2, w/2).
    """
    blurred = convolve(image, kernel)
    reduced = blurred[::2, ::2]
    return reduced


def expand_image(
    image: np.ndarray, 
    kernel: np.ndarray, 
    output_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Expands image size by 2x using zero-insertion and blur.
    
    This implements the "expand" operation for Gaussian pyramids:
    1. Create a 2x larger grid with zeros
    2. Place original pixels at even coordinates (0, 2, 4, ...)
    3. Blur with the kernel and multiply by 4 to preserve brightness
    
    The multiplication by 4 compensates for the energy loss from zero-insertion.
    
    Args:
        image: Input image (grayscale or color).
        kernel: 1D Gaussian kernel for blurring.
        output_shape: Optional (height, width) tuple to force exact size match.
                     Useful when original image had odd dimensions.
    
    Returns:
        Expanded image with dimensions (h*2, w*2) or matching output_shape.
    """
    h, w = image.shape[:2]
    
    # Create 2x larger canvas
    new_h = h * 2
    new_w = w * 2
    
    # Initialize with zeros
    if len(image.shape) == 3:
        expanded = np.zeros((new_h, new_w, 3), dtype=np.float32)
    else:
        expanded = np.zeros((new_h, new_w), dtype=np.float32)
    
    # Place original pixels at even coordinates
    expanded[::2, ::2] = image
    
    # Blur and multiply by 4 to preserve energy
    expanded = convolve(expanded, kernel) * 4.0
    
    # Crop to exact shape if specified (handles odd dimensions)
    if output_shape is not None:
        target_h, target_w = output_shape
        expanded = expanded[:target_h, :target_w]
    
    return expanded


def low_pass_filter_pyramid(
    image: np.ndarray, 
    depth: int, 
    kernel: np.ndarray
) -> np.ndarray:
    """Applies a low-pass filter using pyramid reduction/expansion.
    
    The filter removes high frequencies by:
    1. Reducing the image 'depth' times (blur + downsample)
    2. Expanding it back 'depth' times (upsample + blur)
    
    This effectively removes fine details while preserving overall structure.
    
    Args:
        image: Input image to filter.
        depth: Number of reduction/expansion cycles (blur level).
        kernel: 1D Gaussian kernel for pyramid operations.
    
    Returns:
        Low-pass filtered image with same dimensions as input.
    """
    temp = image.astype(np.float32)
    original_shapes = []
    
    # Reduce depth times
    for _ in range(depth):
        original_shapes.append(temp.shape[:2])
        temp = reduce_image(temp, kernel)
    
    # Expand back depth times
    for i in range(depth):
        target_shape = original_shapes.pop()
        temp = expand_image(temp, kernel, output_shape=target_shape)
    
    return temp


def high_pass_filter_pyramid(
    image: np.ndarray, 
    depth: int, 
    kernel: np.ndarray
) -> np.ndarray:
    """Applies a high-pass filter using pyramid techniques.
    
    High-pass filtering extracts fine details by subtracting the low-pass
    component from the original: $H = I - L(I)$
    
    Args:
        image: Input image to filter.
        depth: Number of reduction/expansion cycles for low-pass.
        kernel: 1D Gaussian kernel for pyramid operations.
    
    Returns:
        High-pass filtered image (fine details only).
    """
    low_passed = low_pass_filter_pyramid(image, depth, kernel)
    high_passed = image.astype(np.float32) - low_passed
    return high_passed

