import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt



# region: mask drawing functions
def draw_polygon_mask(image):
    """
    Open a window that lets the user draw a polygon mask.
    - Left click to add a vertex.
    - Click-and-drag an existing vertex to move it (reshape).
    - Press 'z' to undo last vertex, 'r' to reset, 'Enter' to confirm, 'ESC' to cancel.
    The area inside the polygon becomes the mask.
    """
    window_name = (
        "Step 1: Polygon mask (LMB add/move, Z undo, R reset, Enter confirm, ESC cancel)"
    )
    polygon_pts = []
    selected_idx = None
    hover_pos = None

    def find_vertex(x, y, radius=15):
        for idx, (px, py) in enumerate(polygon_pts):
            if (px - x) ** 2 + (py - y) ** 2 <= radius**2:
                return idx
        return None

    def mouse_callback(event, x, y, flags, param):
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
        if len(polygon_pts) >= 3:
            pts = np.array(polygon_pts, dtype=np.int32)
            cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
            display = cv2.addWeighted(overlay, 0.3, display, 0.7, 0)
        if polygon_pts:
            for idx, (px, py) in enumerate(polygon_pts):
                cv2.circle(display, (px, py), 6, (0, 255, 255), -1)
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
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array(polygon_pts, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def place_mask_on_source(image, reference_mask):
    """
    Lets the user align/scale the drawn mask onto the second image.
    Drag with the mouse to move the mask.
    Press 'B' (bigger) or '+' to scale up, 'S' (smaller) or '-' to scale down.
    Returns a binary mask the same size as the source image.
    """
    window_name = (
        "Step 2: Align mask (Drag=move, B/+ bigger, S/- smaller, Enter confirm, R reset)"
    )
    instructions_name = "Mask Controls"

    base_mask = reference_mask.astype(np.uint8)
    scale = 1.0
    min_scale = 0.05
    max_scale = 5.0
    center = [image.shape[1] // 2, image.shape[0] // 2]
    dragging = False
    drag_offset = (0, 0)
    overlay_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    scaled_mask_shape = reference_mask.shape
    scaled_top_left = (0, 0)

    # Instruction window
    instructions_img = np.zeros((210, 520, 3), dtype=np.uint8)
    lines = [
        "Mask Alignment Controls:",
        "  Drag mouse      -> Move mask",
        "  B / +           -> Enlarge mask",
        "  S / -           -> Shrink mask",
        "  R               -> Reset position/scale",
        "  Enter           -> Confirm placement",
        "  ESC             -> Cancel",
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

    def render_overlay():
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

    def mouse_callback(event, x, y, flags, param):
        nonlocal dragging, drag_offset, scale, center, overlay_mask
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            drag_offset = (x - center[0], y - center[1])
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            center[0] = x - drag_offset[0]
            center[1] = y - drag_offset[1]
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Mouse wheel no longer changes scale; ignore to avoid accidental resizing
            return

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    overlay_mask = render_overlay()
    while True:
        display = image.copy()
        tinted = np.zeros_like(display)
        tinted[:, :] = (0, 0, 255)
        alpha_mask = (overlay_mask > 0).astype(np.float32)[:, :, None]
        display = (display * (1 - 0.4 * alpha_mask) + tinted * 0.4 * alpha_mask).astype(
            np.uint8
        )
        cv2.putText(
            display,
            "Drag mask. Use B/+ to enlarge, S/- to shrink. Enter confirm, R reset, ESC cancel.",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, display)
        key = cv2.waitKey(16) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow(window_name)
            cv2.destroyWindow(instructions_name)
            raise RuntimeError("Mask alignment cancelled by user.")
        elif key == ord("r"):
            scale = 1.0
            center = [image.shape[1] // 2, image.shape[0] // 2]
            overlay_mask = render_overlay()
        elif key in (ord("+"), ord("="), ord("b"), ord("B")):
            scale *= 1.08
            scale = float(np.clip(scale, min_scale, max_scale))
            overlay_mask = render_overlay()
            continue
        elif key in (ord("-"), ord("_"), ord("s"), ord("S")):
            scale /= 1.08
            scale = float(np.clip(scale, min_scale, max_scale))
            overlay_mask = render_overlay()
            continue
        elif key == 13:  # Enter
            if np.count_nonzero(overlay_mask) == 0:
                print("Mask overlay is empty. Adjust before confirming.")
                continue
            break
        overlay_mask = render_overlay()

    cv2.destroyWindow(window_name)
    cv2.destroyWindow(instructions_name)
    overlay_mask = np.clip(overlay_mask, 0, 255).astype(np.uint8)
    placement_info = {
        "scale": scale,
        "top_left": scaled_top_left,
        "mask_shape": scaled_mask_shape,
    }
    return overlay_mask, placement_info




def align_source_to_target(source_img, target_img_shape, target_mask, source_mask):
    """
    Robust alignment based on pixel bounding boxes.
    It finds the white pixels in both masks and warps the source image
    so that the source mask area fits exactly into the target mask area.
    """
    # 1. מציאת הריבוע החוסם של המסכה בתמונת היעד (לאן רוצים להגיע)
    ty, tx = np.where(target_mask > 0)
    if len(tx) == 0: raise ValueError("Target mask is empty!")
    
    tgt_x, tgt_y = np.min(tx), np.min(ty)
    tgt_w = np.max(tx) - tgt_x
    tgt_h = np.max(ty) - tgt_y
    
    # 2. מציאת הריבוע החוסם של המסכה בתמונת המקור (מאיפה לוקחים)
    sy, sx = np.where(source_mask > 0)
    if len(sx) == 0: raise ValueError("Source mask is empty!")
    
    src_x, src_y = np.min(sx), np.min(sy)
    src_w = np.max(sx) - src_x
    src_h = np.max(sy) - src_y

    # 3. הגדרת נקודות עיגון (משולש: שמאל-למעלה, ימין-למעלה, שמאל-למטה)
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

    # 4. חישוב הטרנספורמציה
    # המחשב מחשב לבד כמה למתוח וכמה להזיז
    M = cv2.getAffineTransform(src_tri, dst_tri)

    # 5. ביצוע ה-Warp
    aligned_source = cv2.warpAffine(
        source_img, 
        M, 
        (target_img_shape[1], target_img_shape[0]), 
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0)
    )

    return aligned_source

# endregion

# region convolution, gaussian, reduce and expand image functions
def get_gaussian_kernel():
    """
    Returns the 1D Gaussian kernel: [1, 4, 6, 4, 1] / 16
    We will use this single vector for both X and Y passes.
    """
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32).reshape(1, 5)
    return kernel / 16.0


def create_custom_gaussian_kernel(k_size, sigma):
    """
    Generates a 1D Gaussian kernel manually using Numpy.
    k_size: The size of the kernel (must be odd, e.g., 35).
    sigma: The standard deviation (spread) of the blur.
    """
    
    center = k_size // 2
    x = np.arange(-center, center + 1, dtype=np.float32)
    
    
    kernel_1d = np.exp(-(x**2) / (2 * (sigma**2)))
    
    
    kernel_1d /= kernel_1d.sum()
    

    return kernel_1d.reshape(1, k_size)

def convolve(image, kernel_1d):
    """
    Performs separable convolution:
    1. Convolve horizontally with the 1x5 kernel.
    2. Convolve vertically with the 5x1 kernel (transpose of the 1x5).
    """
    
    temp_image = cv2.filter2D(image, -1, kernel_1d, borderType=cv2.BORDER_REFLECT_101)
    result_image = cv2.filter2D(temp_image, -1, kernel_1d.T, borderType=cv2.BORDER_REFLECT_101)
    
    return result_image


def reduce_image(image, kernel):
    """
    Reduces the image size by half:
    1. Blur (convolve)
    2. Decimate (keep every 2nd pixel)
    """
    
    blurred = convolve(image, kernel)
   
    reduced = blurred[::2, ::2]
    
    return reduced

def expand_image(image, kernel, output_shape=None):
    """
    Expands the image size by x2:
    1. Create separate grid with zeros
    2. Place original pixels in even coordinates
    3. Blur and multiply by 4 to keep the brightness 
    
    output_shape: (height, width) - Optional, to force exact size match 
                  (useful if original image had odd dimensions).
    """
    h, w = image.shape[:2]
    
    
    new_h = h * 2
    new_w = w * 2
    
    
    if len(image.shape) == 3:
        expanded = np.zeros((new_h, new_w, 3), dtype=np.float32)
    else:
        expanded = np.zeros((new_h, new_w), dtype=np.float32)
        
    expanded[::2, ::2] = image
    
    expanded = convolve(expanded, kernel) * 4.0
    
    if output_shape is not None:
        target_h, target_w = output_shape
        expanded = expanded[:target_h, :target_w]
        
    return expanded

# endregion

# region gaussian and laplacian pyramid building functions

def build_gaussian_pyramid(image, levels, kernel):
    """
    Builds a Gaussian pyramid of 'levels' depth.
    Returns a list of images (level 0 is original, level N is smallest).
    """
    pyramid = [image.astype(np.float32)]
    temp_img = image.astype(np.float32)

    for _ in range(levels):
        temp_img = reduce_image(temp_img, kernel)
        pyramid.append(temp_img)

    return pyramid

def build_laplacian_pyramid(gaussian_pyramid, kernel):
    """
    Builds a Laplacian pyramid from a given Gaussian pyramid.
    Formula: L_i = G_i - Expand(G_{i+1})
    The last level is just the top of the Gaussian pyramid.
    """
    pyramid = []
    num_levels = len(gaussian_pyramid)

    for i in range(num_levels - 1):
        current_gaussian = gaussian_pyramid[i]
        next_gaussian = gaussian_pyramid[i+1]
        
        expanded_next = expand_image(next_gaussian, kernel, output_shape=current_gaussian.shape[:2])
        
        laplacian = current_gaussian - expanded_next
        pyramid.append(laplacian)

    pyramid.append(gaussian_pyramid[-1])

    return pyramid
# endregion

# region reconstructing from pyramid functions
def reconstruct_from_pyramid(laplacian_pyramid, kernel):
    """
    Collapses the Laplacian pyramid back into a single image.
    Starts from the top (smallest) level and expands downwards.
    """
    current_image = laplacian_pyramid[-1]
    
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        laplacian_level = laplacian_pyramid[i]
        
        expanded_image = expand_image(current_image, kernel, output_shape=laplacian_level.shape[:2])
        
        current_image = expanded_image + laplacian_level
        
    final_image = np.clip(current_image, 0, 255).astype(np.uint8)
    
    return final_image
# endregion

# region pyramid visualization functions   

def create_and_save_pyramid_viz(image, filename, levels=4):
    """
    Creates the pyramids, draws them and saves them to a file.
    """
    kernel = get_gaussian_kernel()
    g_pyr = build_gaussian_pyramid(image, levels, kernel)
    l_pyr = build_laplacian_pyramid(g_pyr, kernel)

    fig, axes = plt.subplots(2, len(g_pyr), figsize=(16, 6))
    fig.suptitle(f'Pyramid Analysis: {filename}', fontsize=16)

    for i in range(len(g_pyr)):
        # --- Gaussian (Top Row) ---
        ax_g = axes[0, i]
        g_img_disp = np.clip(g_pyr[i], 0, 255).astype(np.uint8)
        ax_g.imshow(cv2.cvtColor(g_img_disp, cv2.COLOR_BGR2RGB))
        ax_g.set_title(f'G{i}')
        ax_g.axis('off')

        # --- Laplacian (Bottom Row) ---
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

# endregion

# region fft pyramid visualization functions
def create_and_save_fft_pyramid_viz(image, filename, levels=4):
    """
    Builds the pyramids, calculates FFT for each level, and displays the spectrum.
    """
   
    kernel = get_gaussian_kernel()
    g_pyr = build_gaussian_pyramid(image, levels, kernel)
    l_pyr = build_laplacian_pyramid(g_pyr, kernel)


    fig, axes = plt.subplots(2, len(g_pyr), figsize=(16, 6))
    fig.suptitle(f'FFT Spectrum Analysis: {filename}', fontsize=16)

    #
    def get_fft_mag(img):
       
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
        # --- Gaussian FFT (Top Row) ---
        ax_g = axes[0, i]
        g_fft = get_fft_mag(g_pyr[i])
        ax_g.imshow(g_fft, cmap='gray')
        ax_g.set_title(f'G{i} FFT')
        ax_g.axis('off')

        # --- Laplacian FFT (Bottom Row) ---
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

# endregion

# region pyramid blending functions
def pyramid_blend(img_source, img_target, mask, levels=6):
    """
    Performs Laplacian Pyramid Blending.
    
    save_viz_name: (Optional) String. If provided, saves a visualization 
                   of the result's pyramids to this filename.
    """
    print(f"--- Running Laplacian Pyramid Blending ({levels} levels) ---")
    
    kernel = get_gaussian_kernel()
    mask_float = mask.astype(np.float32) / 255.0
    mask_pyramid = build_gaussian_pyramid(mask_float, levels, kernel)

    reconstructed_channels = []

   
    for channel_idx in range(3):
        s_chan = img_source[:, :, channel_idx]
        t_chan = img_target[:, :, channel_idx]

        s_gauss = build_gaussian_pyramid(s_chan, levels, kernel)
        s_lap = build_laplacian_pyramid(s_gauss, kernel)
        
        t_gauss = build_gaussian_pyramid(t_chan, levels, kernel)
        t_lap = build_laplacian_pyramid(t_gauss, kernel)

        blended_pyramid = []
        for k in range(len(s_lap)):
            mask_level = mask_pyramid[k]
            blended_level = s_lap[k] * mask_level + t_lap[k] * (1.0 - mask_level)
            blended_pyramid.append(blended_level)

        reconstructed_channel = reconstruct_from_pyramid(blended_pyramid, kernel)
        reconstructed_channels.append(reconstructed_channel)

    result = cv2.merge(reconstructed_channels)

    create_and_save_pyramid_viz(img_source, "pyramids-source.jpg", levels=levels)
    create_and_save_pyramid_viz(img_target, "pyramids-target.jpg", levels=levels)
    create_and_save_pyramid_viz(mask, "pyramids-mask.jpg", levels=levels)
    create_and_save_pyramid_viz(result, "pyramids-result.jpg", levels=levels)

    create_and_save_fft_pyramid_viz(img_source, "fft-pyramids-source.jpg", levels=levels)
    create_and_save_fft_pyramid_viz(img_target, "fft-pyramids-target.jpg", levels=levels)
    create_and_save_fft_pyramid_viz(mask, "fft-pyramids-mask.jpg", levels=levels)
    create_and_save_fft_pyramid_viz(result, "fft-pyramids-result.jpg", levels=levels)

    return result
# endregion

# region fourier magnitude functions
def save_fourier_magnitude(image, filename, title="Magnitude Spectrum"):
    """
    Calculates the Fourier Transform (FFT) of the image,
    Saves the magnitude spectrum (Magnitude) as an image file.
    """
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

   
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(title)
    plt.axis('off') 
    
    print(f"Saving Fourier visualization to: {filename}")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_fft_comparison_with_colorbar(naive_img, pyramid_img, filename):
    """
    Calculates FFT, finds the difference and performs color stretching.
    Saves 3 versions:
    1. filename -> full report.
    2. ..._diff_with_bar -> difference with colorbar.
    3. ..._diff_clean -> clean difference (only the pixels).
    """
   
    if len(naive_img.shape) == 3:
        gray_naive = cv2.cvtColor(naive_img, cv2.COLOR_BGR2GRAY)
        gray_pyr = cv2.cvtColor(pyramid_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_naive, gray_pyr = naive_img, pyramid_img

    def calc_log_fft(img):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        return 20 * np.log(np.abs(fshift) + 1)

    mag_naive = calc_log_fft(gray_naive)
    mag_pyr = calc_log_fft(gray_pyr)

    diff = np.abs(mag_naive - mag_pyr)
    vmax_val = np.percentile(diff, 99)
    if vmax_val < 1e-5: vmax_val = diff.max()

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

    plt.figure(figsize=(8, 6))
    plt.imshow(diff, cmap='jet', vmin=0, vmax=vmax_val)
    plt.title('Difference Map')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04, label='Magnitude Difference')
    
    bar_filename = filename.replace('.png', '_diff_with_bar.png').replace('.jpg', '_diff_with_bar.jpg')
    print(f"Saving Diff with Bar to: {bar_filename}")
    plt.savefig(bar_filename, bbox_inches='tight', dpi=150)
    plt.close()

  
    clean_filename = filename.replace('.png', '_diff_clean.png').replace('.jpg', '_diff_clean.jpg')
    print(f"Saving Clean Diff to: {clean_filename}")
    

    plt.imsave(clean_filename, diff, cmap='jet', vmin=0, vmax=vmax_val)


#endregion

# region image blending 
def image_blending(im1_path, im2_path):
    print(f"Starting Image Blending with:\n1. {im1_path}\n2. {im2_path}")
    img_a = cv2.imread(im1_path)
    img_b = cv2.imread(im2_path)
    
    if img_a is None: raise FileNotFoundError(f"Could not load: {im1_path}")
    if img_b is None: raise FileNotFoundError(f"Could not load: {im2_path}")


    area_a = img_a.shape[0] * img_a.shape[1]
    area_b = img_b.shape[0] * img_b.shape[1]
    
    if area_a >= area_b:
        im_target, im_source = img_a, img_b
    else:
        im_target, im_source = img_b, img_a

   
    print("Step 1: Draw mask on Target...")
    target_mask = draw_polygon_mask(im_target)
    
  
    print("Step 2: Align mask on Source...")
    source_mask_overlay, _ = place_mask_on_source(im_source, target_mask)
    
    
    print("Step 3: Warping source to match target...")
    im_source_aligned = align_source_to_target(im_source, im_target.shape, target_mask, source_mask_overlay)
    
  
    print("Saving intermediate files...")
    cv2.imwrite("aligned_source.jpg", im_source_aligned)
    cv2.imwrite("aligned_target.jpg", im_target)
    cv2.imwrite("mask.png", target_mask)

   
    mask_float = target_mask.astype(float) / 255.0
    if len(im_target.shape) == 3:
        mask_float_3c = np.dstack([mask_float]*3)
    
    naive_blend = (im_source_aligned.astype(float) * mask_float_3c + 
                   im_target.astype(float) * (1 - mask_float_3c))
    naive_blend = np.clip(naive_blend, 0, 255).astype(np.uint8)
    cv2.imwrite("naive_blend_result.jpg", naive_blend)

    print("Step 4: Performing Laplacian Pyramid Blending...")
    
    pyramid_result = pyramid_blend(im_source_aligned, im_target, target_mask, levels=8)
    save_fourier_magnitude(naive_blend, "fft_naive.png", "Naive Blend FFT")
    save_fourier_magnitude(pyramid_result, "fft_pyramid.png", "Pyramid Blend FFT")
    save_fft_comparison_with_colorbar(naive_blend, pyramid_result, "fft_comparison.png")
    cv2.imwrite("pyramid_blend_result.jpg", pyramid_result)
    print("Done! Result saved as 'pyramid_blend_result.jpg'")

   
    cv2.imshow("Naive Blend", naive_blend)
    cv2.imshow("Pyramid Blend", pyramid_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def blend_with_provided_mask(im1_path, im2_path, mask_path):
    """
    Performs blending when we already have two images and a prepared mask.
    Skips the alignment and manual cutting steps.
    """
    print(f"Starting Blending with provided mask:\n1. {im1_path}\n2. {im2_path}\n3. Mask: {mask_path}")
    
   
    img_b = cv2.imread(im1_path)
    img_a = cv2.imread(im2_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 

    if img_a is None: raise FileNotFoundError(f"Could not load: {im1_path}")
    if img_b is None: raise FileNotFoundError(f"Could not load: {im2_path}")
    if mask is None: raise FileNotFoundError(f"Could not load: {mask_path}")

    save_fourier_magnitude(img_a, "fft_img_a.png", "Image A FFT")
    save_fourier_magnitude(img_b, "fft_img_b.png", "Image B FFT")
    

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    big_kernel = create_custom_gaussian_kernel(35, sigma=6.0)

    mask_blurred = convolve(mask, big_kernel)
    
    print("Calculating Naive Blend...")
    mask_float = mask.astype(float) / 255.0
    mask_float_3c = np.dstack([mask_float]*3) 
    
    naive_blend = (img_a.astype(float) * mask_float_3c + 
                   img_b.astype(float) * (1 - mask_float_3c))
    naive_blend = np.clip(naive_blend, 0, 255).astype(np.uint8)
    cv2.imwrite("naive_blend_result.jpg", naive_blend)

    print("Step 4: Performing Laplacian Pyramid Blending...")
    
    pyramid_result = pyramid_blend(
        img_a,
        img_b,
        mask, 
        levels=8,
    )
    
    cv2.imwrite("pyramid_blend_result.jpg", pyramid_result)
    save_fourier_magnitude(naive_blend, "fft_naive.png", "Naive Blend FFT")
    save_fourier_magnitude(pyramid_result, "fft_pyramid.png", "Pyramid Blend FFT")
    save_fft_comparison_with_colorbar(img_b,pyramid_result, "fft_comparison.png")
    print("Done! Result saved as 'pyramid_blend_result.jpg'")

    cv2.imshow("Naive Blend", naive_blend)
    cv2.imshow("Pyramid Blend", pyramid_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# endregion

# region low and high pass filter functions

def low_pass_filter_pyramid(image, depth, kernel):
    """
    Applies a Low-Pass Filter using Pyramid techniques:
    Reduces the image 'depth' times, then Expands it back 'depth' times.
    This effectively removes high frequencies.
    """
    temp = image.astype(np.float32)
    original_shapes = [] 
    
    for _ in range(depth):
        original_shapes.append(temp.shape[:2])
        temp = reduce_image(temp, kernel)
        
    for i in range(depth):
        target_shape = original_shapes.pop() 
        temp = expand_image(temp, kernel, output_shape=target_shape)
        
    return temp

def high_pass_filter_pyramid(image, depth, kernel):
    """
    Applies a High-Pass Filter using Pyramid techniques:
    Original Image - Low_Pass(Image).
    """
    low_passed = low_pass_filter_pyramid(image, depth, kernel)
    
    high_passed = image.astype(np.float32) - low_passed
    
    return high_passed

# endregion

# region hybrid image functions

def create_hybrid_image(im1_path, im2_path, depth=4, low_weight=1.3, high_weight=0.7):
    """
    Creates a Hybrid Image with optimization parameters.
    
    im1_path: Low Freq Image (the image seen from far - low frequency)
    im2_path: High Freq Image (the image seen from close - high frequency)
    depth: how many levels to reduce in the pyramid (the blur level). Recommended 4 or 5.
    low_weight: strengthen the low frequency image (to make it look blurry from far). Try 1.0 to 1.5.
    high_weight: weaken the high frequency image (to not hide the low frequency). Try 0.5 to 0.7.
    """
    print(f"Creating Hybrid Image...")
    print(f"Low Freq (Far): {im1_path} [Weight: {low_weight}]")
    print(f"High Freq (Close): {im2_path} [Weight: {high_weight}]")

    img_far = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)
    img_close = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE)
    
    if img_far is None: raise FileNotFoundError(f"Cannot load {im1_path}")
    if img_close is None: raise FileNotFoundError(f"Cannot load {im2_path}")
    
    img_far = cv2.resize(img_far, (1024, 1024))
    img_close = cv2.resize(img_close, (1024, 1024))

    kernel = get_gaussian_kernel()

    low_freq_component = low_pass_filter_pyramid(img_far, depth, kernel).astype(np.float32)
    
    high_freq_component = high_pass_filter_pyramid(img_close, depth, kernel).astype(np.float32)

    hybrid_float = (low_freq_component * low_weight) + (high_freq_component * high_weight)
    
    hybrid_image = np.clip(hybrid_float, 0, 255).astype(np.uint8)

    return hybrid_image, low_freq_component, high_freq_component

# endregion

# region save hybrid visualization function

def save_hybrid_visualization(hybrid_img, filename):
    """
    Saves a visualization of the hybrid image at progressive downsampling levels.
    Fixed to calculate exact canvas width to prevent crashes.
    """
    print(f"Saving Hybrid visualization to {filename}...")
    
    levels = 5
    padding = 20 
    h, w = hybrid_img.shape[:2]
    
    total_width = 0
    temp_w = w
    for _ in range(levels):
        total_width += temp_w + padding
        temp_w = int(temp_w / 2) 

    if len(hybrid_img.shape) == 2:
        hybrid_img = cv2.cvtColor(hybrid_img, cv2.COLOR_GRAY2BGR)
        
    canvas = np.ones((h, total_width, 3), dtype=np.uint8) * 255 
    
    current_x = 0
    current_img = hybrid_img.copy()
    
    for i in range(levels):
        h_curr, w_curr = current_img.shape[:2]
        
      
        canvas[h-h_curr:h, current_x:current_x+w_curr] = current_img
        
        current_x += w_curr + padding
        
       
        current_img = cv2.resize(current_img, (0,0), fx=0.5, fy=0.5)
        
    cv2.imwrite(filename, canvas)

# endregion

# region hybrid image main function

def hybrid_images(im1_path, im2_path):
    
    res, low, high = create_hybrid_image(im1_path, im2_path, depth=4)
    
    cv2.imwrite("hybrid_low_freq_component.jpg", np.clip(low, 0, 255).astype(np.uint8))
    cv2.imwrite("hybrid_high_freq_component.jpg", np.clip(high + 128, 0, 255).astype(np.uint8))
    
   
    cv2.imwrite("hybrid_result.jpg", res)
    
   
    save_hybrid_visualization(res, "hybrid_pyramid_view.jpg")
    
    print("Hybrid process done.")
    cv2.imshow("Hybrid Result", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# endregion

# region main function

def main():
    
    if len(sys.argv) < 4:
        print("Usage:")
        print("1. Manual Align: python ex3.py <img1> <img2> imageblending")
        print("2. Hybrid:       python ex3.py <img1> <img2> hybridimages")
        print("3. With Mask:    python ex3.py <img1> <img2> <mask_path> imageblending")
        print("4. Compare FFT:  python ex3.py <img1> <img2> compare_fft")
        return
    

    if len(sys.argv) == 5 and sys.argv[4] == 'imageblending':
        im1_path = sys.argv[1]
        im2_path = sys.argv[2]
        mask_path = sys.argv[3]
        blend_with_provided_mask(im1_path, im2_path, mask_path)
        return

    
    image_path1 = sys.argv[1]
    image_path2 = sys.argv[2]
    mode = sys.argv[3]

    if mode == 'imageblending':
        image_blending(image_path1, image_path2)
    
    elif mode == 'hybridimages':
        hybrid_images(image_path1, image_path2)

    elif mode == "compare_fft":
        img_a = cv2.imread(image_path1)
        img_b = cv2.imread(image_path2)
        save_fft_comparison_with_colorbar(img_a, img_b, "fft_comparison.png")
        
    else:
        print(f"Error: Unknown mode '{mode}'.")

if __name__ == "__main__":
    main()

# endregion