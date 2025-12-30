"""Visualization functions for pyramids and FFT analysis.

This module provides tools for visualizing:
- Gaussian and Laplacian pyramid levels
- Fourier Transform magnitude spectra
- FFT comparisons between images
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from src.core.filters import get_gaussian_kernel
from src.core.pyramids import build_gaussian_pyramid, build_laplacian_pyramid


def create_and_save_pyramid_viz(
    image: np.ndarray, 
    filename: str, 
    levels: int = 4
) -> None:
    """Creates and saves a visualization of Gaussian and Laplacian pyramids.
    
    The visualization shows:
    - Top row: Gaussian pyramid levels (G0, G1, ..., GN)
    - Bottom row: Laplacian pyramid levels (L0, L1, ..., LN)
    
    Args:
        image: Input image to build pyramids from.
        filename: Output filename for the visualization.
        levels: Number of pyramid levels to display.
    """
    kernel = get_gaussian_kernel()
    g_pyr = build_gaussian_pyramid(image, levels, kernel)
    l_pyr = build_laplacian_pyramid(g_pyr, kernel)
    
    fig, axes = plt.subplots(2, len(g_pyr), figsize=(16, 6))
    fig.suptitle(f'Pyramid Analysis: {filename}', fontsize=16)
    
    for i in range(len(g_pyr)):
        # Gaussian (Top Row)
        ax_g = axes[0, i]
        g_img_disp = np.clip(g_pyr[i], 0, 255).astype(np.uint8)
        ax_g.imshow(cv2.cvtColor(g_img_disp, cv2.COLOR_BGR2RGB))
        ax_g.set_title(f'G{i}')
        ax_g.axis('off')
        
        # Laplacian (Bottom Row)
        ax_l = axes[1, i]
        if i < len(l_pyr):
            l_data = l_pyr[i]
            l_disp = cv2.normalize(l_data, None, 0, 255, cv2.NORM_MINMAX)
            l_disp = np.clip(l_disp, 0, 255).astype(np.uint8)
            ax_l.imshow(cv2.cvtColor(l_disp, cv2.COLOR_BGR2RGB))
            ax_l.set_title(f'L{i}')
            ax_l.axis('off')
    
    plt.tight_layout()
    print(f"Saving visualization to: {filename}")
    plt.savefig(filename)
    plt.close(fig)


def create_and_save_fft_pyramid_viz(
    image: np.ndarray, 
    filename: str, 
    levels: int = 4
) -> None:
    """Creates and saves FFT spectrum visualization for pyramid levels.
    
    Computes the Fourier Transform magnitude spectrum for each pyramid level
    to visualize frequency content at different scales.
    
    Args:
        image: Input image to analyze.
        filename: Output filename for the visualization.
        levels: Number of pyramid levels to analyze.
    """
    kernel = get_gaussian_kernel()
    g_pyr = build_gaussian_pyramid(image, levels, kernel)
    l_pyr = build_laplacian_pyramid(g_pyr, kernel)
    
    fig, axes = plt.subplots(2, len(g_pyr), figsize=(16, 6))
    fig.suptitle(f'FFT Spectrum Analysis: {filename}', fontsize=16)
    
    def get_fft_mag(img: np.ndarray) -> np.ndarray:
        """Compute log-magnitude FFT spectrum."""
        if len(img.shape) == 3:
            if img.dtype == np.uint8:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = np.mean(img, axis=2)
        else:
            gray = img
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        return 20 * np.log(np.abs(fshift) + 1)
    
    for i in range(len(g_pyr)):
        # Gaussian FFT (Top Row)
        ax_g = axes[0, i]
        g_fft = get_fft_mag(g_pyr[i])
        ax_g.imshow(g_fft, cmap='gray')
        ax_g.set_title(f'G{i} FFT')
        ax_g.axis('off')
        
        # Laplacian FFT (Bottom Row)
        ax_l = axes[1, i]
        if i < len(l_pyr):
            l_fft = get_fft_mag(l_pyr[i])
            ax_l.imshow(l_fft, cmap='gray')
            ax_l.set_title(f'L{i} FFT')
            ax_l.axis('off')
    
    plt.tight_layout()
    print(f"Saving FFT visualization to: {filename}")
    plt.savefig(filename)
    plt.close(fig)


def save_fourier_magnitude(
    image: np.ndarray, 
    filename: str, 
    title: str = "Magnitude Spectrum"
) -> None:
    """Calculates and saves the Fourier Transform magnitude spectrum.
    
    The FFT magnitude spectrum shows the frequency content of an image.
    Low frequencies (center) represent smooth regions, while high frequencies
    (edges) represent fine details.
    
    Args:
        image: Input image (grayscale or color).
        filename: Output filename for the visualization.
        title: Title for the plot.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Compute FFT and shift zero frequency to center
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # Compute log-magnitude for visualization
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(title)
    plt.axis('off')
    
    print(f"Saving Fourier visualization to: {filename}")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_fft_comparison_with_colorbar(
    naive_img: np.ndarray, 
    pyramid_img: np.ndarray, 
    filename: str
) -> None:
    """Creates and saves FFT comparison with difference map.
    
    Compares the FFT spectra of two images and visualizes:
    1. FFT of first image
    2. FFT of second image
    3. Difference map with colorbar
    
    Also saves separate files:
    - `*_diff_with_bar.*`: Difference map with colorbar
    - `*_diff_clean.*`: Clean difference map only
    
    Args:
        naive_img: First image for comparison.
        pyramid_img: Second image for comparison.
        filename: Base filename for outputs.
    """
    # Convert to grayscale if needed
    if len(naive_img.shape) == 3:
        gray_naive = cv2.cvtColor(naive_img, cv2.COLOR_BGR2GRAY)
        gray_pyr = cv2.cvtColor(pyramid_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_naive, gray_pyr = naive_img, pyramid_img
    
    def calc_log_fft(img: np.ndarray) -> np.ndarray:
        """Compute log-magnitude FFT."""
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        return 20 * np.log(np.abs(fshift) + 1)
    
    mag_naive = calc_log_fft(gray_naive)
    mag_pyr = calc_log_fft(gray_pyr)
    
    # Compute difference
    diff = np.abs(mag_naive - mag_pyr)
    vmax_val = np.percentile(diff, 99)
    if vmax_val < 1e-5:
        vmax_val = diff.max()
    
    # Full comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].imshow(mag_naive, cmap='gray')
    axes[0].set_title('Pyramid Blend FFT level 3')
    axes[0].axis('off')
    
    axes[1].imshow(mag_pyr, cmap='gray')
    axes[1].set_title('Pyramid Blend FFT level 8')
    axes[1].axis('off')
    
    im = axes[2].imshow(diff, cmap='jet', vmin=0, vmax=vmax_val)
    axes[2].set_title(f'Difference Map')
    axes[2].axis('off')
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Magnitude Difference')
    
    print(f"Saving Full Report to: {filename}")
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    # Difference with colorbar
    plt.figure(figsize=(8, 6))
    plt.imshow(diff, cmap='jet', vmin=0, vmax=vmax_val)
    plt.title('Difference Map')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04, label='Magnitude Difference')
    
    bar_filename = filename.replace('.png', '_diff_with_bar.png').replace('.jpg', '_diff_with_bar.jpg')
    print(f"Saving Diff with Bar to: {bar_filename}")
    plt.savefig(bar_filename, bbox_inches='tight', dpi=150)
    plt.close()
    
    # Clean difference (no axes, no colorbar)
    clean_filename = filename.replace('.png', '_diff_clean.png').replace('.jpg', '_diff_clean.jpg')
    print(f"Saving Clean Diff to: {clean_filename}")
    plt.imsave(clean_filename, diff, cmap='jet', vmin=0, vmax=vmax_val)


def save_hybrid_visualization(hybrid_img: np.ndarray, filename: str) -> None:
    """Saves a visualization of hybrid image at progressive downsampling levels.
    
    Creates a horizontal strip showing the hybrid image at multiple scales,
    demonstrating how it appears when viewed from different distances.
    
    Args:
        hybrid_img: Hybrid image to visualize (grayscale or color).
        filename: Output filename.
    """
    print(f"Saving Hybrid visualization to {filename}...")
    
    levels = 5
    padding = 20
    h, w = hybrid_img.shape[:2]
    
    # Calculate total canvas width
    total_width = 0
    temp_w = w
    for _ in range(levels):
        total_width += temp_w + padding
        temp_w = int(temp_w / 2)
    
    # Convert to BGR if grayscale
    if len(hybrid_img.shape) == 2:
        hybrid_img = cv2.cvtColor(hybrid_img, cv2.COLOR_GRAY2BGR)
    
    canvas = np.ones((h, total_width, 3), dtype=np.uint8) * 255
    
    current_x = 0
    current_img = hybrid_img.copy()
    
    for i in range(levels):
        h_curr, w_curr = current_img.shape[:2]
        
        # Place image on canvas (bottom-aligned)
        canvas[h - h_curr:h, current_x:current_x + w_curr] = current_img
        
        current_x += w_curr + padding
        
        # Downsample for next level
        current_img = cv2.resize(current_img, (0, 0), fx=0.5, fy=0.5)
    
    cv2.imwrite(filename, canvas)

