"""Gaussian and Laplacian pyramid construction and reconstruction.

This module implements the core pyramid operations for multi-scale image
representation, which is fundamental to seamless image blending.
"""

import numpy as np
from typing import List

from src.core.filters import reduce_image, expand_image


def build_gaussian_pyramid(
    image: np.ndarray, 
    levels: int, 
    kernel: np.ndarray
) -> List[np.ndarray]:
    """Builds a Gaussian pyramid of specified depth.
    
    A Gaussian pyramid is a sequence of images where each level is a blurred
    and downsampled version of the previous level. Level 0 is the original image,
    and each subsequent level has half the resolution.
    
    The pyramid is constructed by iteratively applying:
    $G_{i+1} = Reduce(G_i)$
    
    Args:
        image: Input image (grayscale or color).
        levels: Number of reduction levels (depth of pyramid).
        kernel: 1D Gaussian kernel for blurring during reduction.
    
    Returns:
        List of images where index 0 is the original and index N is the smallest.
    """
    pyramid = [image.astype(np.float32)]
    temp_img = image.astype(np.float32)
    
    for _ in range(levels):
        temp_img = reduce_image(temp_img, kernel)
        pyramid.append(temp_img)
    
    return pyramid


def build_laplacian_pyramid(
    gaussian_pyramid: List[np.ndarray], 
    kernel: np.ndarray
) -> List[np.ndarray]:
    """Builds a Laplacian pyramid from a Gaussian pyramid.
    
    A Laplacian pyramid stores the difference between consecutive Gaussian levels,
    capturing detail at different scales. The formula is:
    $L_i = G_i - Expand(G_{i+1})$
    
    The top level (smallest) is just the top of the Gaussian pyramid since
    there's no higher level to subtract from.
    
    The Laplacian pyramid is lossless: the original image can be perfectly
    reconstructed by expanding and adding all levels.
    
    Args:
        gaussian_pyramid: List of Gaussian pyramid levels.
        kernel: 1D Gaussian kernel for expansion operations.
    
    Returns:
        List of Laplacian pyramid levels (detail images).
    """
    pyramid = []
    num_levels = len(gaussian_pyramid)
    
    # For each level except the last, compute difference
    for i in range(num_levels - 1):
        current_gaussian = gaussian_pyramid[i]
        next_gaussian = gaussian_pyramid[i + 1]
        
        # Expand the next level to match current level's size
        expanded_next = expand_image(
            next_gaussian, 
            kernel, 
            output_shape=current_gaussian.shape[:2]
        )
        
        # Laplacian = current - expanded_next
        laplacian = current_gaussian - expanded_next
        pyramid.append(laplacian)
    
    # Top level is just the top Gaussian (no higher level to subtract)
    pyramid.append(gaussian_pyramid[-1])
    
    return pyramid


def reconstruct_from_pyramid(
    laplacian_pyramid: List[np.ndarray], 
    kernel: np.ndarray
) -> np.ndarray:
    """Reconstructs the original image from a Laplacian pyramid.
    
    Reconstruction works by starting from the top (smallest) level and
    iteratively expanding and adding the next level:
    $G_i = Expand(G_{i+1}) + L_i$
    
    This process perfectly reconstructs the original image at level 0.
    
    Args:
        laplacian_pyramid: List of Laplacian pyramid levels.
        kernel: 1D Gaussian kernel for expansion operations.
    
    Returns:
        Reconstructed image (uint8, clipped to [0, 255]).
    """
    # Start with the top level (smallest, coarsest detail)
    current_image = laplacian_pyramid[-1]
    
    # Reconstruct from top to bottom
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        laplacian_level = laplacian_pyramid[i]
        
        # Expand current image to match laplacian level size
        expanded_image = expand_image(
            current_image, 
            kernel, 
            output_shape=laplacian_level.shape[:2]
        )
        
        # Add the detail back
        current_image = expanded_image + laplacian_level
    
    # Clip and convert to uint8
    final_image = np.clip(current_image, 0, 255).astype(np.uint8)
    
    return final_image

