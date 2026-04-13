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
