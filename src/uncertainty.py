"""Pixel-level uncertainty map computation for VSR hybrid fusion.

This module provides functions to compute uncertainty maps that guide the fusion
between conservative BasicVSR++ reconstruction and generative/texture-enhancement branches.

Uncertainty U(x,y) ∈ [0,1] where:
- Low U: pixel is confident, trust BasicVSR++ reconstruction
- High U: pixel is uncertain, may benefit from texture enhancement or hallucination
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np


def compute_temporal_residual(
    frames: Sequence[np.ndarray],
    gaussian_blur: bool = True,
    blur_ksize: int = 5,
) -> list[np.ndarray]:
    """Compute temporal residual (frame difference) as motion/instability cue.
    
    For each frame, compute the absolute difference with the previous frame.
    This serves as a proxy for temporal instability / motion strength.
    
    Args:
        frames: Sequence of grayscale or color frames. Color frames will be
                converted to grayscale internally.
        gaussian_blur: Whether to apply Gaussian blur to smooth the residual.
        blur_ksize: Kernel size for Gaussian blur (must be odd).
    
    Returns:
        List of residual maps, one per frame, normalized to [0, 1].
        The first frame's residual is set to 0 (no prior frame).
        
    Note:
        Current implementation uses simple frame difference. Future versions
        can replace this with optical-flow warping residual for better motion
        alignment, e.g.:
            I_t_warped = warp(I_{t-1}, optical_flow)
            residual = |I_t - I_t_warped|
    """
    if not frames:
        return []
    
    residuals: list[np.ndarray] = []
    prev_gray = None
    
    for frame in frames:
        # Convert to grayscale if needed
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.astype(np.uint8) if frame.dtype != np.uint8 else frame
        
        if prev_gray is None:
            # First frame has no temporal residual
            residual = np.zeros(gray.shape, dtype=np.float32)
        else:
            # Compute absolute difference
            diff = cv2.absdiff(gray.astype(np.float32), prev_gray.astype(np.float32))
            residual = diff / 255.0  # Normalize to [0, 1]
        
        if gaussian_blur and blur_ksize > 1:
            residual = cv2.GaussianBlur(residual, (blur_ksize, blur_ksize), 0)
        
        residuals.append(np.clip(residual, 0, 1).astype(np.float32))
        prev_gray = gray
    
    return residuals


def compute_texture_complexity(frame: np.ndarray) -> np.ndarray:
    """Compute pixel-level texture complexity as uncertainty cue.
    
    High texture complexity suggests the pixel contains fine details that
    might benefit from enhancement or hallucination.
    
    Texture complexity is estimated using Laplacian response (edge/detail detection).
    
    Args:
        frame: Single frame (H, W, 3) in BGR or grayscale (H, W).
    
    Returns:
        Texture complexity map (H, W), normalized to [0, 1], dtype float32.
        High values indicate high-complexity regions (edges, fine details).
    """
    # Convert to grayscale if needed
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    gray = gray.astype(np.float32)
    
    # Compute Laplacian (edge/detail response)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    laplacian_abs = np.abs(laplacian)
    
    # Normalize to [0, 1]
    max_val = np.max(laplacian_abs)
    if max_val > 0:
        texture_complexity = laplacian_abs / max_val
    else:
        texture_complexity = np.zeros_like(laplacian_abs)
    
    # Optional: apply spatial smoothing to avoid isolated spikes
    texture_complexity = cv2.GaussianBlur(texture_complexity, (5, 5), 1.0)
    
    return np.clip(texture_complexity, 0, 1).astype(np.float32)


def compute_structure_confidence(frame: np.ndarray) -> np.ndarray:
    """Compute pixel-level structure confidence map.
    
    High confidence in structured regions (edges, text, faces) to protect them
    from excessive hallucination or enhancement by the generative branch.
    
    Uses edge strength (Sobel gradient magnitude) as a proxy for structure.
    
    Args:
        frame: Single frame (H, W, 3) in BGR or grayscale (H, W).
    
    Returns:
        Structure confidence map (H, W), normalized to [0, 1], dtype float32.
        High values indicate strong structure/edges.
    """
    # Convert to grayscale if needed
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    gray = gray.astype(np.float32)
    
    # Compute Sobel gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Gradient magnitude
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize to [0, 1]
    max_val = np.max(gradient_mag)
    if max_val > 0:
        structure_confidence = gradient_mag / max_val
    else:
        structure_confidence = np.zeros_like(gradient_mag)
    
    return np.clip(structure_confidence, 0, 1).astype(np.float32)


def compute_uncertainty_map(
    frame: np.ndarray,
    temporal_residual: Optional[np.ndarray] = None,
    texture_complexity: Optional[np.ndarray] = None,
    structure_confidence: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 0.8,
    gamma: float = 0.6,
    sigmoid_temperature: float = 1.0,
) -> np.ndarray:
    """Compute pixel-level uncertainty map by combining multiple cues.
    
    Uncertainty formula:
        U = sigmoid( (alpha * R + beta * T - gamma * S) / sigmoid_temperature )
    
    where:
        R = temporal_residual: motion/instability cue
        T = texture_complexity: fine-detail cue
        S = structure_confidence: edge/structure cue
    
    Interpretation:
        - Pixels with high R, T but low S → high uncertainty
        - Pixels with high S (strong structure) → low uncertainty (protected)
        - Pixels with low R, T → low uncertainty (stable, smooth)
    
    Args:
        frame: Single frame for computing auto cues if not provided.
        temporal_residual: Pre-computed temporal residual map. If None, will compute.
        texture_complexity: Pre-computed texture complexity map. If None, will compute.
        structure_confidence: Pre-computed structure confidence map. If None, will compute.
        alpha: Weight for temporal residual term.
        beta: Weight for texture complexity term.
        gamma: Weight for structure confidence term (higher gamma → lower uncertainty for high-S regions).
        sigmoid_temperature: Temperature for sigmoid (lower = sharper transition).
    
    Returns:
        Uncertainty map (H, W), normalized to [0, 1], dtype float32.
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Compute missing cues
    if temporal_residual is None:
        temporal_residual = np.zeros((frame_height, frame_width), dtype=np.float32)
    
    if texture_complexity is None:
        texture_complexity = compute_texture_complexity(frame)
    
    if structure_confidence is None:
        structure_confidence = compute_structure_confidence(frame)
    
    # Ensure all maps have the same shape
    if temporal_residual.shape != (frame_height, frame_width):
        temporal_residual = cv2.resize(
            temporal_residual, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR
        )
    if texture_complexity.shape != (frame_height, frame_width):
        texture_complexity = cv2.resize(
            texture_complexity, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR
        )
    if structure_confidence.shape != (frame_height, frame_width):
        structure_confidence = cv2.resize(
            structure_confidence, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR
        )
    
    # Compute logit
    logit = (
        alpha * temporal_residual +
        beta * texture_complexity -
        gamma * structure_confidence
    )
    logit = logit / max(sigmoid_temperature, 1e-6)
    
    # Apply sigmoid
    uncertainty = 1.0 / (1.0 + np.exp(-logit))
    
    return np.clip(uncertainty, 0, 1).astype(np.float32)


def smooth_uncertainty_maps(
    uncertainty_maps: Sequence[np.ndarray],
    spatial_blur_ksize: int = 9,
    temporal_smooth_lambda: float = 0.7,
) -> list[np.ndarray]:
    """Apply spatial and temporal smoothing to uncertainty maps.
    
    Spatial smoothing: Gaussian blur to avoid isolated spikes.
    Temporal smoothing: Exponential moving average to reduce frame-to-frame flickering.
    
    Args:
        uncertainty_maps: Sequence of uncertainty maps (H, W), dtype float32, values in [0, 1].
        spatial_blur_ksize: Kernel size for spatial Gaussian blur (must be odd, >= 3).
        temporal_smooth_lambda: Weight for exponential moving average.
                                 U_smooth_t = lambda * U_t + (1 - lambda) * U_smooth_{t-1}
                                 Higher lambda = more weight on current frame.
    
    Returns:
        List of smoothed uncertainty maps, same shape and dtype as input.
    """
    if not uncertainty_maps:
        return []
    
    # Ensure kernel size is odd and >= 3
    if spatial_blur_ksize < 3 or spatial_blur_ksize % 2 == 0:
        spatial_blur_ksize = max(3, spatial_blur_ksize | 1)  # Make odd
    
    # Spatial smoothing
    spatially_smoothed: list[np.ndarray] = []
    for u_map in uncertainty_maps:
        smoothed = cv2.GaussianBlur(u_map, (spatial_blur_ksize, spatial_blur_ksize), 1.5)
        spatially_smoothed.append(smoothed)
    
    # Temporal smoothing using exponential moving average
    temporally_smoothed: list[np.ndarray] = []
    u_smooth_prev = None
    
    for u_smooth_t in spatially_smoothed:
        if u_smooth_prev is None:
            u_smooth_final = u_smooth_t.copy()
        else:
            u_smooth_final = (
                temporal_smooth_lambda * u_smooth_t +
                (1.0 - temporal_smooth_lambda) * u_smooth_prev
            )
        temporally_smoothed.append(np.clip(u_smooth_final, 0, 1).astype(np.float32))
        u_smooth_prev = u_smooth_final
    
    return temporally_smoothed


def save_uncertainty_visualizations(
    uncertainty_maps: Sequence[np.ndarray],
    output_dir,
    threshold: float = 0.5,
) -> None:
    """Save uncertainty heatmaps and visualizations.
    
    Outputs:
        - heatmap_{i:08d}.png: RGB heatmap (red=high uncertainty, blue=low uncertainty)
        - binary_mask_{i:08d}.png: Binary mask (white where U > threshold)
    
    Args:
        uncertainty_maps: Sequence of uncertainty maps [0, 1].
        output_dir: Output directory path.
        threshold: Threshold for binary mask visualization.
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for frame_idx, u_map in enumerate(uncertainty_maps):
        # Heatmap: apply colormap (red=high, blue=low)
        u_map_uint8 = (u_map * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(u_map_uint8, cv2.COLORMAP_JET)
        heatmap_path = output_dir / f"heatmap_{frame_idx:08d}.png"
        cv2.imwrite(str(heatmap_path), heatmap_bgr)
        
        # Binary mask
        binary_mask = ((u_map > threshold).astype(np.uint8) * 255)
        binary_path = output_dir / f"binary_mask_{frame_idx:08d}.png"
        cv2.imwrite(str(binary_path), binary_mask)


def compute_uncertainty_statistics(uncertainty_maps: Sequence[np.ndarray]) -> dict:
    """Compute and return statistics about uncertainty maps.
    
    Args:
        uncertainty_maps: Sequence of uncertainty maps [0, 1].
    
    Returns:
        Dictionary with keys:
            - avg_uncertainty: Average pixel uncertainty across all frames.
            - high_uncertainty_ratio: Fraction of pixels with U > 0.5.
            - frame_count: Number of frames.
    """
    if not uncertainty_maps:
        return {
            "avg_uncertainty": 0.0,
            "high_uncertainty_ratio": 0.0,
            "frame_count": 0,
        }
    
    all_values = np.concatenate([u.flatten() for u in uncertainty_maps])
    avg_uncertainty = float(np.mean(all_values))
    high_uncertainty_ratio = float(np.mean(all_values > 0.5))
    
    return {
        "avg_uncertainty": avg_uncertainty,
        "high_uncertainty_ratio": high_uncertainty_ratio,
        "frame_count": len(uncertainty_maps),
    }
