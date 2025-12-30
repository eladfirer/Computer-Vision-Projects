"""Geometric transformations: SVD-based rigid transform and RANSAC."""

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


def compute_rigid_transform_manual(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    enable_x: bool,
    enable_y: bool,
    enable_rot: bool
) -> np.ndarray:
    """Compute rigid transform using SVD decomposition.
    
    Computes the optimal rigid transformation (translation + rotation) between
    two point sets using Singular Value Decomposition (SVD). The method centers
    the point sets, computes the covariance matrix H = src_centered^T @ dst_centered,
    performs SVD on H to extract the rotation matrix, and computes translation.
    
    Args:
        src_pts: Source points (N, 2) array
        dst_pts: Destination points (N, 2) array
        enable_x: Enable translation in X direction
        enable_y: Enable translation in Y direction
        enable_rot: Enable rotation component
        
    Returns:
        Array [dx, dy, theta_degrees] representing the transform
    """
    src_mean = np.mean(src_pts, axis=0)
    dst_mean = np.mean(dst_pts, axis=0)
    
    if not enable_rot:
        theta = 0.0
        dx = dst_mean[0] - src_mean[0]
        dy = dst_mean[1] - src_mean[1]
    else:
        # Center the point sets
        src_centered = src_pts - src_mean
        dst_centered = dst_pts - dst_mean
        
        # Compute covariance matrix
        H = src_centered.T @ dst_centered
        
        # SVD decomposition
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T
        
        # Extract rotation angle
        theta = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(theta), np.sin(theta)
        
        # Compute translation accounting for rotation
        rx = src_mean[0] * c - src_mean[1] * s
        ry = src_mean[0] * s + src_mean[1] * c
        dx = dst_mean[0] - rx
        dy = dst_mean[1] - ry
    
    # Apply motion constraints
    if not enable_x:
        dx = 0
    if not enable_y:
        dy = 0
    
    return np.array([dx, dy, np.degrees(theta)])


def manual_ransac(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    enable_x: bool,
    enable_y: bool,
    enable_rot: bool,
    threshold: float,
    max_iters: int = 200
) -> np.ndarray:
    """RANSAC-based robust estimation of rigid transform.
    
    Uses Random Sample Consensus to find the best rigid transformation model
    that fits the majority of point correspondences, filtering out outliers.
    
    Args:
        src_pts: Source points (N, 2) array
        dst_pts: Destination points (N, 2) array
        enable_x: Enable translation in X direction
        enable_y: Enable translation in Y direction
        enable_rot: Enable rotation component
        threshold: Inlier threshold in pixels
        max_iters: Maximum RANSAC iterations
        
    Returns:
        Array [dx, dy, theta_degrees] representing the best transform
    """
    num_points = len(src_pts)
    best_inliers_count = -1
    src = src_pts.reshape(-1, 2)
    dst = dst_pts.reshape(-1, 2)
    
    # Minimum sample size: 2 for rotation, 1 for translation only
    sample_size = 2 if enable_rot else 1
    
    if num_points < sample_size:
        logger.warning(f"Insufficient points ({num_points}) for RANSAC")
        return np.zeros(3)
    
    best_inliers_mask = None
    
    for iteration in range(max_iters):
        # Random sample
        idxs = np.random.choice(num_points, sample_size, replace=False)
        s_sample, d_sample = src[idxs], dst[idxs]
        
        # Fit model to sample
        curr_model = compute_rigid_transform_manual(
            s_sample, d_sample, enable_x, enable_y, enable_rot
        )
        dx, dy, deg = curr_model
        theta_rad = np.radians(deg)
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        
        # Apply transform to all points
        proj_x = src[:, 0] * c - src[:, 1] * s + dx
        proj_y = src[:, 0] * s + src[:, 1] * c + dy
        
        # Compute errors
        diff_x = proj_x - dst[:, 0]
        diff_y = proj_y - dst[:, 1]
        errors = np.sqrt(diff_x**2 + diff_y**2)
        
        # Count inliers
        current_inliers = errors < threshold
        count = np.sum(current_inliers)
        
        if count > best_inliers_count:
            best_inliers_count = count
            best_inliers_mask = current_inliers
    
    # Refit model using all inliers
    if best_inliers_count > sample_size and best_inliers_mask is not None:
        final_src = src[best_inliers_mask]
        final_dst = dst[best_inliers_mask]
        best_model = compute_rigid_transform_manual(
            final_src, final_dst, enable_x, enable_y, enable_rot
        )
        logger.debug(f"RANSAC found {best_inliers_count}/{num_points} inliers")
        return best_model
    
    logger.warning("RANSAC failed to find sufficient inliers")
    return np.zeros(3)

