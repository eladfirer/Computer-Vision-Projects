"""OpenCV-based UI functions for interactive mask drawing and alignment.

This module provides interactive tools for:
- Drawing polygon masks on images
- Aligning and scaling masks between images
- Warping source images to match target regions
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional


def draw_polygon_mask(image: np.ndarray) -> np.ndarray:
    """Interactive polygon mask drawing tool.
    
    Opens an OpenCV window that allows the user to:
    - Left-click to add vertices
    - Click-and-drag existing vertices to reshape
    - Press 'z' to undo last vertex
    - Press 'r' to reset
    - Press 'Enter' to confirm
    - Press 'ESC' to cancel
    
    The area inside the polygon becomes the mask (white = 255, black = 0).
    
    Args:
        image: Input image to draw mask on.
    
    Returns:
        Binary mask (uint8) with same height/width as input image.
        White (255) inside polygon, black (0) outside.
    
    Raises:
        RuntimeError: If user cancels the operation (ESC key).
    """
    window_name = (
        "Step 1: Polygon mask (LMB add/move, Z undo, R reset, Enter confirm, ESC cancel)"
    )
    polygon_pts = []
    selected_idx: Optional[int] = None
    hover_pos: Optional[Tuple[int, int]] = None
    
    def find_vertex(x: int, y: int, radius: int = 15) -> Optional[int]:
        """Find vertex index near (x, y) within radius."""
        for idx, (px, py) in enumerate(polygon_pts):
            if (px - x) ** 2 + (py - y) ** 2 <= radius**2:
                return idx
        return None
    
    def mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse events for polygon drawing."""
        nonlocal selected_idx, hover_pos
        hover_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = find_vertex(x, y)
            if idx is not None:
                selected_idx = idx
            else:
                polygon_pts.append([x, y])
        elif event == cv2.EVENT_MOUSEMOVE and selected_idx is not None:
            polygon_pts[selected_idx] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            selected_idx = None
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        display = image.copy()
        overlay = display.copy()
        
        # Draw polygon if we have at least 3 points
        if len(polygon_pts) >= 3:
            pts = np.array(polygon_pts, dtype=np.int32)
            cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
            display = cv2.addWeighted(overlay, 0.3, display, 0.7, 0)
        
        # Draw vertices
        if polygon_pts:
            for idx, (px, py) in enumerate(polygon_pts):
                cv2.circle(display, (px, py), 6, (0, 255, 255), -1)
        
        # Draw preview line to hover position
        if hover_pos and polygon_pts:
            cv2.line(display, polygon_pts[-1], hover_pos, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(
            display,
            "Add/move vertices to define mask. Enter confirms.",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, display)
        key = cv2.waitKey(16) & 0xFF
        
        if key == 27:  # ESC
            cv2.destroyWindow(window_name)
            raise RuntimeError("Mask drawing cancelled by user.")
        elif key == ord("r"):
            polygon_pts.clear()
        elif key == ord("z"):
            if polygon_pts:
                polygon_pts.pop()
        elif key == 13:  # Enter
            if len(polygon_pts) < 3:
                print("Polygon needs at least 3 points.")
                continue
            break
    
    cv2.destroyWindow(window_name)
    
    # Create binary mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array(polygon_pts, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def place_mask_on_source(
    image: np.ndarray, 
    reference_mask: np.ndarray
) -> Tuple[np.ndarray, Dict[str, any]]:
    """Interactive mask alignment and scaling tool.
    
    Allows the user to align a reference mask onto a source image by:
    - Arrow keys or WASD to move the mask
    - Pressing 'B' or '+' to scale up
    - Pressing 'S' or '-' to scale down
    - Pressing 'R' to reset position/scale
    - Pressing 'Enter' to confirm
    - Pressing 'ESC' to cancel
    
    Args:
        image: Target image to place mask on.
        reference_mask: Reference mask to align (from draw_polygon_mask).
    
    Returns:
        Tuple of:
        - Binary mask (uint8) aligned on source image
        - Dictionary with placement info (scale, top_left, mask_shape)
    
    Raises:
        RuntimeError: If user cancels the operation (ESC key).
    """
    window_name = (
        "Step 2: Align mask (Arrows=move, B/+ bigger, S/- smaller, Enter confirm, R reset)"
    )
    instructions_name = "Mask Controls"
    
    base_mask = reference_mask.astype(np.uint8)
    scale = 1.0
    min_scale = 0.05
    max_scale = 5.0
    center = [image.shape[1] // 2, image.shape[0] // 2]
    move_step = 5  # Pixels to move per key press
    overlay_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    scaled_mask_shape = reference_mask.shape
    scaled_top_left = (0, 0)
    
    # Create instruction window
    instructions_img = np.zeros((210, 520, 3), dtype=np.uint8)
    lines = [
        "Mask Alignment Controls:",
        "  Arrow keys     -> Move mask",
        "  B / +          -> Enlarge mask",
        "  S / -          -> Shrink mask",
        "  R              -> Reset position/scale",
        "  Enter          -> Confirm placement",
        "  ESC            -> Cancel",
    ]
    y = 35
    for text in lines:
        cv2.putText(
            instructions_img,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28
    cv2.namedWindow(instructions_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(instructions_name, instructions_img)
    
    def render_overlay() -> np.ndarray:
        """Render the scaled and positioned mask overlay."""
        nonlocal overlay_mask, scaled_mask_shape, scaled_top_left
        scaled_mask = cv2.resize(
            base_mask,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )
        _, scaled_mask = cv2.threshold(scaled_mask, 127, 255, cv2.THRESH_BINARY)
        h, w = scaled_mask.shape
        scaled_mask_shape = (h, w)
        tl_x = int(center[0] - w / 2)
        tl_y = int(center[1] - h / 2)
        scaled_top_left = (tl_x, tl_y)
        mask_canvas = np.zeros_like(overlay_mask)
        
        # Clip to image bounds
        x1 = max(tl_x, 0)
        y1 = max(tl_y, 0)
        x2 = min(tl_x + w, image.shape[1])
        y2 = min(tl_y + h, image.shape[0])
        if x1 < x2 and y1 < y2:
            src_x1 = x1 - tl_x
            src_y1 = y1 - tl_y
            mask_canvas[y1:y2, x1:x2] = scaled_mask[
                src_y1 : src_y1 + (y2 - y1), src_x1 : src_x1 + (x2 - x1)
            ]
        overlay_mask = mask_canvas
        return overlay_mask
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    overlay_mask = render_overlay()
    needs_redraw = True
    
    while True:
        if needs_redraw:
            display = image.copy()
            tinted = np.zeros_like(display)
            tinted[:, :] = (0, 0, 255)
            alpha_mask = (overlay_mask > 0).astype(np.float32)[:, :, None]
            display = (display * (1 - 0.4 * alpha_mask) + tinted * 0.4 * alpha_mask).astype(
                np.uint8
            )
            cv2.putText(
                display,
                "Use Arrow keys to move mask. B/+ to enlarge, S/- to shrink. Enter confirm, R reset, ESC cancel.",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, display)
            needs_redraw = False
        
        # Get full key code to detect arrow keys
        # Use waitKeyEx on some systems for better arrow key detection
        try:
            key_code = cv2.waitKeyEx(30)  # waitKeyEx better handles special keys
        except AttributeError:
            key_code = cv2.waitKey(30)  # Fallback to waitKey if waitKeyEx not available
        key = key_code & 0xFF if key_code != -1 else -1
        
        if key == 27:  # ESC
            cv2.destroyWindow(window_name)
            cv2.destroyWindow(instructions_name)
            raise RuntimeError("Mask alignment cancelled by user.")
        elif key == ord("r"):
            scale = 1.0
            center = [image.shape[1] // 2, image.shape[0] // 2]
            overlay_mask = render_overlay()
            needs_redraw = True
        elif key in (ord("+"), ord("="), ord("b"), ord("B")):
            scale *= 1.08
            scale = float(np.clip(scale, min_scale, max_scale))
            overlay_mask = render_overlay()
            needs_redraw = True
        elif key in (ord("-"), ord("_"), ord("s"), ord("S")):
            scale /= 1.08
            scale = float(np.clip(scale, min_scale, max_scale))
            overlay_mask = render_overlay()
            needs_redraw = True
        elif key == 13:  # Enter
            if np.count_nonzero(overlay_mask) == 0:
                print("Mask overlay is empty. Adjust before confirming.")
                continue
            break
        else:
            # Arrow key movement only
            moved = False
            # Arrow keys detection - OpenCV waitKey returns different values on different systems
            # On macOS/Linux, arrow keys often return: Up=82, Down=84, Left=81, Right=83
            # But sometimes they return values > 255, so we check both
            if key_code != -1:
                # Check for arrow keys
                if key_code == 82:  # Up arrow
                    center[1] = max(0, center[1] - move_step)
                    moved = True
                elif key_code == 84:  # Down arrow
                    center[1] = min(image.shape[0] - 1, center[1] + move_step)
                    moved = True
                elif key_code == 81:  # Left arrow
                    center[0] = max(0, center[0] - move_step)
                    moved = True
                elif key_code == 83:  # Right arrow
                    center[0] = min(image.shape[1] - 1, center[0] + move_step)
                    moved = True
                # Alternative: check if it's a special key (key_code > 255)
                elif key_code > 255:
                    # Extract the low byte which might contain arrow key info
                    low_byte = key_code & 0xFF
                    if low_byte == 0:  # Up
                        center[1] = max(0, center[1] - move_step)
                        moved = True
                    elif low_byte == 1:  # Down
                        center[1] = min(image.shape[0] - 1, center[1] + move_step)
                        moved = True
                    elif low_byte == 2:  # Left
                        center[0] = max(0, center[0] - move_step)
                        moved = True
                    elif low_byte == 3:  # Right
                        center[0] = min(image.shape[1] - 1, center[0] + move_step)
                        moved = True
            
            if moved:
                overlay_mask = render_overlay()
                needs_redraw = True
    
    cv2.destroyWindow(window_name)
    cv2.destroyWindow(instructions_name)
    overlay_mask = np.clip(overlay_mask, 0, 255).astype(np.uint8)
    placement_info = {
        "scale": scale,
        "top_left": scaled_top_left,
        "mask_shape": scaled_mask_shape,
    }
    return overlay_mask, placement_info


def align_source_to_target(
    source_img: np.ndarray,
    target_img_shape: Tuple[int, int],
    target_mask: np.ndarray,
    source_mask: np.ndarray
) -> np.ndarray:
    """Robust alignment using bounding box-based affine transformation.
    
    This function warps the source image so that the source mask region
    fits exactly into the target mask region. It uses an affine transformation
    computed from the bounding boxes of both masks.
    
    The transformation is computed using three control points:
    - Top-left corner
    - Top-right corner  
    - Bottom-left corner
    
    This allows for translation, scaling, and shearing to align the regions.
    
    Args:
        source_img: Source image to warp.
        target_img_shape: (height, width) of target image.
        target_mask: Binary mask on target image (where to place source).
        source_mask: Binary mask on source image (what region to extract).
    
    Returns:
        Warped source image aligned to target mask region.
    
    Raises:
        ValueError: If either mask is empty.
    """
    # Find bounding box of target mask (destination)
    ty, tx = np.where(target_mask > 0)
    if len(tx) == 0:
        raise ValueError("Target mask is empty!")
    
    tgt_x, tgt_y = np.min(tx), np.min(ty)
    tgt_w = np.max(tx) - tgt_x
    tgt_h = np.max(ty) - tgt_y
    
    # Find bounding box of source mask (source)
    sy, sx = np.where(source_mask > 0)
    if len(sx) == 0:
        raise ValueError("Source mask is empty!")
    
    src_x, src_y = np.min(sx), np.min(sy)
    src_w = np.max(sx) - src_x
    src_h = np.max(sy) - src_y
    
    # Define three control points for affine transformation
    # (top-left, top-right, bottom-left)
    src_tri = np.float32([
        [src_x, src_y],
        [src_x + src_w, src_y],
        [src_x, src_y + src_h]
    ])
    
    dst_tri = np.float32([
        [tgt_x, tgt_y],
        [tgt_x + tgt_w, tgt_y],
        [tgt_x, tgt_y + tgt_h]
    ])
    
    # Compute affine transformation matrix
    M = cv2.getAffineTransform(src_tri, dst_tri)
    
    # Apply warp
    aligned_source = cv2.warpAffine(
        source_img,
        M,
        (target_img_shape[1], target_img_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    return aligned_source

