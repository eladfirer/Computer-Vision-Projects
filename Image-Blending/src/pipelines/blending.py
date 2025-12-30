"""Image blending pipeline using Laplacian pyramid blending.

This module orchestrates the complete image blending workflow:
1. Interactive mask drawing and alignment
2. Image warping and alignment
3. Naive blending (baseline)
4. Laplacian pyramid blending (advanced)
5. FFT analysis and visualization
"""

import cv2
import numpy as np
from typing import Optional

from src.core.filters import get_gaussian_kernel, create_custom_gaussian_kernel, convolve
from src.core.pyramids import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    reconstruct_from_pyramid
)
from src.utils.ui import draw_polygon_mask, place_mask_on_source, align_source_to_target
from src.utils.visualization import (
    create_and_save_pyramid_viz,
    create_and_save_fft_pyramid_viz,
    save_fourier_magnitude,
    save_fft_comparison_with_colorbar
)


def pyramid_blend(
    img_source: np.ndarray,
    img_target: np.ndarray,
    mask: np.ndarray,
    levels: int = 6,
    debug: bool = False
) -> np.ndarray:
    """Performs Laplacian Pyramid Blending.
    
    The blending process:
    1. Build Gaussian pyramids for source, target, and mask
    2. Build Laplacian pyramids for source and target
    3. Blend each level: $B_i = L_{source,i} \\cdot M_i + L_{target,i} \\cdot (1 - M_i)$
    4. Reconstruct the final image from the blended pyramid
    
    This produces seamless blending by mixing frequencies at appropriate scales.
    
    Args:
        img_source: Source image to blend in.
        img_target: Target image to blend onto.
        mask: Binary mask (255 = use source, 0 = use target).
        levels: Number of pyramid levels for blending.
        debug: If True, generates pyramid visualizations and FFT analysis.
    
    Returns:
        Blended image (uint8).
    """
    print(f"--- Running Laplacian Pyramid Blending ({levels} levels) ---")
    
    kernel = get_gaussian_kernel()
    mask_float = mask.astype(np.float32) / 255.0
    mask_pyramid = build_gaussian_pyramid(mask_float, levels, kernel)
    
    reconstructed_channels = []
    
    # Process each color channel separately
    for channel_idx in range(3):
        s_chan = img_source[:, :, channel_idx]
        t_chan = img_target[:, :, channel_idx]
        
        # Build pyramids
        s_gauss = build_gaussian_pyramid(s_chan, levels, kernel)
        s_lap = build_laplacian_pyramid(s_gauss, kernel)
        
        t_gauss = build_gaussian_pyramid(t_chan, levels, kernel)
        t_lap = build_laplacian_pyramid(t_gauss, kernel)
        
        # Blend each level
        blended_pyramid = []
        for k in range(len(s_lap)):
            mask_level = mask_pyramid[k]
            blended_level = s_lap[k] * mask_level + t_lap[k] * (1.0 - mask_level)
            blended_pyramid.append(blended_level)
        
        # Reconstruct channel
        reconstructed_channel = reconstruct_from_pyramid(blended_pyramid, kernel)
        reconstructed_channels.append(reconstructed_channel)
    
    result = cv2.merge(reconstructed_channels)
    
    # Save visualizations only in debug mode
    if debug:
        create_and_save_pyramid_viz(img_source, "pyramids-source.jpg", levels=levels)
        create_and_save_pyramid_viz(img_target, "pyramids-target.jpg", levels=levels)
        create_and_save_pyramid_viz(mask, "pyramids-mask.jpg", levels=levels)
        create_and_save_pyramid_viz(result, "pyramids-result.jpg", levels=levels)
        
        create_and_save_fft_pyramid_viz(img_source, "fft-pyramids-source.jpg", levels=levels)
        create_and_save_fft_pyramid_viz(img_target, "fft-pyramids-target.jpg", levels=levels)
        create_and_save_fft_pyramid_viz(mask, "fft-pyramids-mask.jpg", levels=levels)
        create_and_save_fft_pyramid_viz(result, "fft-pyramids-result.jpg", levels=levels)
    
    return result


def image_blending(im1_path: str, im2_path: str, debug: bool = False) -> None:
    """Complete image blending pipeline with interactive mask drawing.
    
    Workflow:
    1. Load and determine target/source (larger = target)
    2. Draw polygon mask on target
    3. Align mask on source
    4. Warp source to match target region
    5. Perform naive blending (baseline)
    6. Perform pyramid blending (advanced)
    7. Generate FFT analysis and visualizations (if debug=True)
    
    Args:
        im1_path: Path to first image.
        im2_path: Path to second image.
        debug: If True, generates pyramid visualizations and FFT analysis.
    
    Raises:
        FileNotFoundError: If images cannot be loaded.
    """
    print(f"Starting Image Blending with:\n1. {im1_path}\n2. {im2_path}")
    
    img_a = cv2.imread(im1_path)
    img_b = cv2.imread(im2_path)
    
    if img_a is None:
        raise FileNotFoundError(f"Could not load: {im1_path}")
    if img_b is None:
        raise FileNotFoundError(f"Could not load: {im2_path}")
    
    # Determine target (larger image) and source
    area_a = img_a.shape[0] * img_a.shape[1]
    area_b = img_b.shape[0] * img_b.shape[1]
    
    if area_a >= area_b:
        im_target, im_source = img_a, img_b
    else:
        im_target, im_source = img_b, img_a
    
    # Step 1: Draw mask on target
    print("Step 1: Draw mask on Target...")
    target_mask = draw_polygon_mask(im_target)
    
    # Step 2: Align mask on source
    print("Step 2: Align mask on Source...")
    source_mask_overlay, _ = place_mask_on_source(im_source, target_mask)
    
    # Step 3: Warp source to match target
    print("Step 3: Warping source to match target...")
    im_source_aligned = align_source_to_target(
        im_source, im_target.shape, target_mask, source_mask_overlay
    )
    
    # Save intermediate files
    print("Saving intermediate files...")
    cv2.imwrite("aligned_source.jpg", im_source_aligned)
    cv2.imwrite("aligned_target.jpg", im_target)
    cv2.imwrite("mask.png", target_mask)
    
    # Naive blending (baseline)
    mask_float = target_mask.astype(float) / 255.0
    if len(im_target.shape) == 3:
        mask_float_3c = np.dstack([mask_float] * 3)
    
    naive_blend = (
        im_source_aligned.astype(float) * mask_float_3c +
        im_target.astype(float) * (1 - mask_float_3c)
    )
    naive_blend = np.clip(naive_blend, 0, 255).astype(np.uint8)
    cv2.imwrite("naive_blend_result.jpg", naive_blend)
    
    # Step 4: Pyramid blending
    print("Step 4: Performing Laplacian Pyramid Blending...")
    pyramid_result = pyramid_blend(im_source_aligned, im_target, target_mask, levels=8, debug=debug)
    
    # FFT analysis only in debug mode
    if debug:
        save_fourier_magnitude(naive_blend, "fft_naive.png", "Naive Blend FFT")
        save_fourier_magnitude(pyramid_result, "fft_pyramid.png", "Pyramid Blend FFT")
        save_fft_comparison_with_colorbar(naive_blend, pyramid_result, "fft_comparison.png")
    
    cv2.imwrite("pyramid_blend_result.jpg", pyramid_result)
    print("Done! Result saved as 'pyramid_blend_result.jpg'")
    
    # Display results
    cv2.imshow("Naive Blend", naive_blend)
    cv2.imshow("Pyramid Blend", pyramid_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def blend_with_provided_mask(im1_path: str, im2_path: str, mask_path: str, debug: bool = False) -> None:
    """Image blending pipeline using a pre-provided mask.
    
    Skips the interactive mask drawing and alignment steps.
    Useful when you already have a prepared mask.
    
    Args:
        im1_path: Path to first image.
        im2_path: Path to second image.
        mask_path: Path to binary mask image.
        debug: If True, generates pyramid visualizations and FFT analysis.
    
    Raises:
        FileNotFoundError: If images or mask cannot be loaded.
    """
    print(f"Starting Blending with provided mask:\n1. {im1_path}\n2. {im2_path}\n3. Mask: {mask_path}")
    
    img_b = cv2.imread(im1_path)
    img_a = cv2.imread(im2_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img_a is None:
        raise FileNotFoundError(f"Could not load: {im1_path}")
    if img_b is None:
        raise FileNotFoundError(f"Could not load: {im2_path}")
    if mask is None:
        raise FileNotFoundError(f"Could not load: {mask_path}")
    
    # Ensure all images and mask have the same dimensions
    mask_h, mask_w = mask.shape[:2]
    img_b_h, img_b_w = img_b.shape[:2]
    img_a_h, img_a_w = img_a.shape[:2]
    
    # Determine target dimensions (use mask dimensions as reference, or largest image)
    if mask_h == img_b_h and mask_w == img_b_w:
        # Mask matches img_b, resize img_a to match
        target_h, target_w = img_b_h, img_b_w
        img_a = cv2.resize(img_a, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        print(f"Resized img_a to match mask/img_b: {target_h}x{target_w}")
    elif mask_h == img_a_h and mask_w == img_a_w:
        # Mask matches img_a, resize img_b to match
        target_h, target_w = img_a_h, img_a_w
        img_b = cv2.resize(img_b, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        print(f"Resized img_b to match mask/img_a: {target_h}x{target_w}")
    else:
        # Mask doesn't match either, resize all to mask dimensions
        target_h, target_w = mask_h, mask_w
        img_a = cv2.resize(img_a, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_b = cv2.resize(img_b, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        print(f"Resized both images to match mask: {target_h}x{target_w}")
    
    # Ensure mask matches final dimensions
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # FFT of input images (only in debug mode)
    if debug:
        save_fourier_magnitude(img_a, "fft_img_a.png", "Image A FFT")
        save_fourier_magnitude(img_b, "fft_img_b.png", "Image B FFT")
    
    # Binarize mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    big_kernel = create_custom_gaussian_kernel(35, sigma=6.0)
    mask_blurred = convolve(mask, big_kernel)
    
    # Naive blending
    print("Calculating Naive Blend...")
    mask_float = mask.astype(float) / 255.0
    mask_float_3c = np.dstack([mask_float] * 3)
    
    naive_blend = (
        img_a.astype(float) * mask_float_3c +
        img_b.astype(float) * (1 - mask_float_3c)
    )
    naive_blend = np.clip(naive_blend, 0, 255).astype(np.uint8)
    cv2.imwrite("naive_blend_result.jpg", naive_blend)
    
    # Pyramid blending
    print("Step 4: Performing Laplacian Pyramid Blending...")
    pyramid_result = pyramid_blend(img_a, img_b, mask, levels=8, debug=debug)
    
    cv2.imwrite("pyramid_blend_result.jpg", pyramid_result)
    
    # FFT analysis only in debug mode
    if debug:
        save_fourier_magnitude(naive_blend, "fft_naive.png", "Naive Blend FFT")
        save_fourier_magnitude(pyramid_result, "fft_pyramid.png", "Pyramid Blend FFT")
        save_fft_comparison_with_colorbar(img_b, pyramid_result, "fft_comparison.png")
    print("Done! Result saved as 'pyramid_blend_result.jpg'")
    
    # Display results
    cv2.imshow("Naive Blend", naive_blend)
    cv2.imshow("Pyramid Blend", pyramid_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

