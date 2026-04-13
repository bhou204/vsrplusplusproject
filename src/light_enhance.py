"""Lightweight enhancement utilities for the full frame background."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def bicubic_upscale_frames(
    frames: Sequence[np.ndarray],
    scale_factor: int,
) -> list[np.ndarray]:
    """Upscale a frame sequence with bicubic interpolation."""

    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")

    upscaled_frames: list[np.ndarray] = []
    for frame in frames:
        height, width = frame.shape[:2]
        new_size = (width * scale_factor, height * scale_factor)
        upscaled = cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)
        upscaled_frames.append(upscaled)
    return upscaled_frames
