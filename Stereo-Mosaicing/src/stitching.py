"""Panorama stitching from dynamic strips with vertical motion compensation."""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def calculate_strip_boundaries_dynamic(
    motions: list[np.ndarray],
    frame_w: int,
    slit_ratio: float = 0.5
) -> list[Tuple[int, int]]:
    """Calculate dynamic strip boundaries for each frame.
    
    Computes the horizontal strip (x_start, x_end) to extract from each frame
    based on motion history. The strip position adapts to motion magnitude
    to minimize overlap and gaps.
    
    Args:
        motions: List of motion vectors [dx, dy, theta] for each frame transition
        frame_w: Frame width in pixels
        slit_ratio: Slit position ratio (0.0=left, 0.5=center, 1.0=right)
        
    Returns:
        List of (x_start, x_end) tuples, one per frame
    """
    strips: list[Tuple[int, int]] = []
    num_frames = len(motions) + 1
    
    # Safe limits to avoid edge artifacts
    min_safe_limit = 0.2
    max_safe_limit = 0.8
    effective_pos = min_safe_limit + (slit_ratio * (max_safe_limit - min_safe_limit))
    base_center = frame_w * effective_pos
    
    for i in range(num_frames):
        # Get motion magnitudes for adjacent frames
        if i == 0:
            dx_prev = 0.0
        else:
            dx_prev = abs(motions[i - 1][0])
        
        if i == num_frames - 1:
            dx_curr = 0.0
        else:
            dx_curr = abs(motions[i][0])
        
        # Calculate strip boundaries based on motion
        x_start = int(base_center - (dx_prev / 2.0))
        x_end = int(base_center + (dx_curr / 2.0))
        
        # Clamp to frame boundaries
        x_start = max(0, x_start)
        x_end = min(frame_w, x_end)
        
        # Safety check: ensure valid strip
        if x_start >= x_end:
            safe_center = int(base_center)
            safe_center = max(0, min(safe_center, frame_w - 1))
            x_start, x_end = safe_center, safe_center + 1
        
        strips.append((x_start, x_end))
    
    return strips


def stitch_from_strips(
    frames: list[np.ndarray],
    strips: list[Tuple[int, int]],
    motions: list[np.ndarray]
) -> np.ndarray:
    """Stitch panorama from frame strips with vertical motion compensation.
    
    Extracts strips from each frame and stitches them horizontally, accounting
    for vertical motion (dy) between frames. The canvas height is dynamically
    sized to accommodate all vertical offsets.
    
    Args:
        frames: List of video frames
        strips: List of (x_start, x_end) tuples for each frame
        motions: List of motion vectors [dx, dy, theta] for each transition
        
    Returns:
        Stitched panorama image
    """
    if not frames:
        raise ValueError("Empty frame list")
    
    h, w = frames[0].shape[:2]
    is_color = len(frames[0].shape) == 3
    
    # Compute cumulative vertical offsets
    cum_dy = [0.0]
    curr_y = 0.0
    for m in motions:
        curr_y += m[1]  # dy component
        cum_dy.append(curr_y)
    
    # Calculate canvas dimensions
    strips_width = sum([(s[1] - s[0]) for s in strips])
    total_width = strips_width
    
    min_dy = min(cum_dy)
    max_dy = max(cum_dy)
    total_height = int(h + (max_dy - min_dy)) + 50  # Extra padding
    
    # Create canvas
    if is_color:
        canvas = np.zeros(
            (total_height, total_width, frames[0].shape[2]),
            dtype=np.uint8
        )
    else:
        canvas = np.zeros((total_height, total_width), dtype=np.uint8)
    
    # Stitch strips
    curr_x = 0
    y_global_offset = -min_dy + 25  # Center vertically
    all_y_starts = []
    
    for i, frame in enumerate(frames):
        x_start, x_end = strips[i]
        strip_w = x_end - x_start
        strip = frame[:, x_start:x_end]
        
        # Calculate vertical position
        offset_y = cum_dy[i] + y_global_offset
        y_start = int(offset_y)
        all_y_starts.append(y_start)
        
        h_cut = min(h, total_height - y_start)
        
        # Place strip on canvas
        if strip_w > 0 and h_cut > 0:
            canvas[y_start:y_start + h_cut, curr_x:curr_x + strip_w] = strip[:h_cut, :]
        
        curr_x += strip_w
    
    # Crop top and bottom based on actual y positions
    crop_top = int(np.max(all_y_starts))
    crop_bottom = int(np.min([y + h for y in all_y_starts]))
    
    if crop_bottom > crop_top + 10:
        canvas = canvas[crop_top:crop_bottom, :]
    
    logger.debug(
        f"Stitched panorama: {canvas.shape[0]}x{canvas.shape[1]} from {len(frames)} frames"
    )
    return canvas

