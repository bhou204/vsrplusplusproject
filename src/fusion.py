"""Fusion utilities for combining ROI-heavy results with light background."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .roi_motion import BoundingBox


def _resize_to_bbox(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    target_width = max(1, bbox.width)
    target_height = max(1, bbox.height)
    if frame.shape[1] == target_width and frame.shape[0] == target_height:
        return frame
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)


def paste_roi_into_frame(
    base_frame: np.ndarray,
    roi_frame: np.ndarray,
    bbox: BoundingBox,
    feather_blend: bool = False,
    feather_sigma: float = 8.0,
) -> np.ndarray:
    """Paste an ROI frame back into a full-resolution base frame."""

    if not bbox.is_valid():
        return base_frame.copy()

    result = base_frame.copy()
    roi_resized = _resize_to_bbox(roi_frame, bbox)
    x1, y1, x2, y2 = bbox.as_tuple()
    result[y1:y2, x1:x2] = roi_resized

    if not feather_blend:
        return result

    blend_mask = np.zeros(base_frame.shape[:2], dtype=np.float32)
    blend_mask[y1:y2, x1:x2] = 1.0
    blend_mask = cv2.GaussianBlur(blend_mask, (0, 0), feather_sigma)
    blend_mask = np.clip(blend_mask, 0.0, 1.0)[..., None]
    blended = (
        result.astype(np.float32) * blend_mask +
        base_frame.astype(np.float32) * (1.0 - blend_mask)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def paste_roi_sequence(
    base_frames: Sequence[np.ndarray],
    roi_frames: Sequence[np.ndarray],
    bboxes: Sequence[BoundingBox],
    feather_blend: bool = False,
    feather_sigma: float = 8.0,
) -> list[np.ndarray]:
    """Paste ROI frames into a sequence of light background frames."""

    if len(base_frames) != len(roi_frames):
        raise ValueError("Base frames and ROI frames must have the same length")

    if len(bboxes) == 1 and len(base_frames) > 1:
        bboxes = list(bboxes) * len(base_frames)

    if len(bboxes) != len(base_frames):
        raise ValueError("The number of bboxes must match the frame count")

    fused_frames: list[np.ndarray] = []
    for base_frame, roi_frame, bbox in zip(base_frames, roi_frames, bboxes):
        fused_frames.append(
            paste_roi_into_frame(
                base_frame=base_frame,
                roi_frame=roi_frame,
                bbox=bbox,
                feather_blend=feather_blend,
                feather_sigma=feather_sigma,
            ))
    return fused_frames


def pixel_uncertainty_fusion(
    basicvsr_frame: np.ndarray,
    texture_frame: np.ndarray,
    uncertainty_map: np.ndarray,
    clamp_output: bool = True,
) -> np.ndarray:
    """Fuse two frames using pixel-level uncertainty weighting.
    
    Final pixel = (1 - U) * basicvsr_frame + U * texture_frame
    
    where U is uncertainty map in [0, 1]:
    - U ≈ 0: trust BasicVSR++ (conservative reconstruction)
    - U ≈ 1: trust texture/generative branch (enhanced/hallucinated)
    
    Args:
        basicvsr_frame: Conservative reconstruction frame (H, W, 3) uint8.
        texture_frame: Texture-enhanced frame (H, W, 3) uint8, same shape as basicvsr_frame.
        uncertainty_map: Uncertainty map (H, W) float32, values in [0, 1].
        clamp_output: Whether to clamp output to [0, 255].
    
    Returns:
        Fused frame (H, W, 3) uint8.
    
    Raises:
        ValueError: If frame shapes don't match or uncertainty_map shape is incorrect.
    """
    if basicvsr_frame.shape != texture_frame.shape:
        raise ValueError(
            f"Frame shape mismatch: {basicvsr_frame.shape} vs {texture_frame.shape}"
        )
    
    frame_height, frame_width = basicvsr_frame.shape[:2]
    
    if uncertainty_map.shape != (frame_height, frame_width):
        # Try to resize uncertainty map
        uncertainty_map = cv2.resize(
            uncertainty_map, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR
        )
    
    # Ensure uncertainty is in [0, 1]
    uncertainty_map = np.clip(uncertainty_map, 0.0, 1.0).astype(np.float32)
    
    # Expand to 3 channels
    uncertainty_3ch = uncertainty_map[..., None].astype(np.float32)
    
    # Fuse: final = (1 - U) * basicvsr + U * texture
    basicvsr_f = basicvsr_frame.astype(np.float32)
    texture_f = texture_frame.astype(np.float32)
    
    fused = (1.0 - uncertainty_3ch) * basicvsr_f + uncertainty_3ch * texture_f
    
    if clamp_output:
        fused = np.clip(fused, 0, 255)
    
    return fused.astype(np.uint8)


def fuse_sequence_with_uncertainty(
    basicvsr_frames: Sequence[np.ndarray],
    texture_frames: Sequence[np.ndarray],
    uncertainty_maps: Sequence[np.ndarray],
    clamp_output: bool = True,
) -> list[np.ndarray]:
    """Fuse entire frame sequences using pixel-level uncertainty weighting.
    
    Args:
        basicvsr_frames: Sequence of conservative reconstruction frames.
        texture_frames: Sequence of texture-enhanced frames.
        uncertainty_maps: Sequence of uncertainty maps, one per frame.
        clamp_output: Whether to clamp output to [0, 255].
    
    Returns:
        List of fused frames.
    
    Raises:
        ValueError: If frame counts or shapes don't match.
    """
    if len(basicvsr_frames) != len(texture_frames):
        raise ValueError(
            f"Frame count mismatch: {len(basicvsr_frames)} vs {len(texture_frames)}"
        )
    
    if len(basicvsr_frames) != len(uncertainty_maps):
        raise ValueError(
            f"Frame count mismatch: {len(basicvsr_frames)} vs {len(uncertainty_maps)}"
        )
    
    fused_frames: list[np.ndarray] = []
    for basicvsr_frame, texture_frame, uncertainty_map in zip(
        basicvsr_frames, texture_frames, uncertainty_maps
    ):
        fused_frame = pixel_uncertainty_fusion(
            basicvsr_frame=basicvsr_frame,
            texture_frame=texture_frame,
            uncertainty_map=uncertainty_map,
            clamp_output=clamp_output,
        )
        fused_frames.append(fused_frame)
    
    return fused_frames
