"""Motion-based ROI detection for lightweight video region selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass
class MotionRoiResult:
    """Motion ROI result for a full frame sequence."""

    per_frame_boxes: list[Optional[BoundingBox]]
    per_frame_masks: list[np.ndarray]
    global_box: BoundingBox
    motion_scores: list[float]
    average_roi_area_ratio: float


def _to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _clip_box(box: BoundingBox, width: int, height: int) -> BoundingBox:
    x1 = max(0, min(box.x1, width - 1))
    y1 = max(0, min(box.y1, height - 1))
    x2 = max(x1 + 1, min(box.x2, width))
    y2 = max(y1 + 1, min(box.y2, height))
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def expand_bbox(
    box: BoundingBox,
    expand_ratio: float,
    width: int,
    height: int,
) -> BoundingBox:
    """Expand a bbox by ratio while keeping it inside image bounds."""

    if not box.is_valid():
        return BoundingBox(0, 0, width, height)

    pad_x = int(round(box.width * expand_ratio))
    pad_y = int(round(box.height * expand_ratio))
    return _clip_box(
        BoundingBox(
            x1=box.x1 - pad_x,
            y1=box.y1 - pad_y,
            x2=box.x2 + pad_x,
            y2=box.y2 + pad_y,
        ), width, height)


def _mask_to_bbox(mask: np.ndarray, min_roi_area: int) -> Optional[BoundingBox]:
    """Convert a binary mask into a bbox around the largest connected blob."""

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return None

    best_label = -1
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label

    if best_label < 0 or best_area < min_roi_area:
        return None

    left = int(stats[best_label, cv2.CC_STAT_LEFT])
    top = int(stats[best_label, cv2.CC_STAT_TOP])
    width = int(stats[best_label, cv2.CC_STAT_WIDTH])
    height = int(stats[best_label, cv2.CC_STAT_HEIGHT])
    return BoundingBox(left, top, left + width, top + height)


def _smooth_boxes(
    boxes: Sequence[Optional[BoundingBox]],
    window_size: int,
) -> list[Optional[BoundingBox]]:
    """Smooth per-frame boxes with a simple moving average."""

    if window_size <= 1:
        return list(boxes)

    half_window = window_size // 2
    smoothed: list[Optional[BoundingBox]] = []
    last_valid: Optional[BoundingBox] = None

    for index in range(len(boxes)):
        start = max(0, index - half_window)
        end = min(len(boxes), index + half_window + 1)
        window_boxes = [box for box in boxes[start:end] if box is not None]
        if window_boxes:
            x1 = int(round(sum(box.x1 for box in window_boxes) / len(window_boxes)))
            y1 = int(round(sum(box.y1 for box in window_boxes) / len(window_boxes)))
            x2 = int(round(sum(box.x2 for box in window_boxes) / len(window_boxes)))
            y2 = int(round(sum(box.y2 for box in window_boxes) / len(window_boxes)))
            current = BoundingBox(x1, y1, x2, y2)
            last_valid = current
        else:
            current = last_valid
        smoothed.append(current)
    return smoothed


def _union_boxes(
    boxes: Sequence[Optional[BoundingBox]],
    width: int,
    height: int,
) -> BoundingBox:
    """Compute a union bbox from a box sequence."""

    valid_boxes = [box for box in boxes if box is not None and box.is_valid()]
    if not valid_boxes:
        return BoundingBox(0, 0, width, height)

    x1 = min(box.x1 for box in valid_boxes)
    y1 = min(box.y1 for box in valid_boxes)
    x2 = max(box.x2 for box in valid_boxes)
    y2 = max(box.y2 for box in valid_boxes)
    return _clip_box(BoundingBox(x1, y1, x2, y2), width, height)


def detect_motion_rois(
    frames: Sequence[np.ndarray],
    motion_threshold: float,
    min_roi_area: int,
    roi_expand_ratio: float,
    roi_smooth_window: int = 1,
    use_mask: bool = False,
) -> MotionRoiResult:
    """Detect motion-driven ROIs from a frame sequence.

    The prototype focuses on a stable bbox-based ROI. Binary masks are also
    returned for later mask-based extensions, but the current end-to-end flow
    uses bboxes for cropping and fusion.
    """

    if not frames:
        raise ValueError("Cannot detect ROI from an empty frame sequence")

    height, width = frames[0].shape[:2]
    raw_boxes: list[Optional[BoundingBox]] = []
    masks: list[np.ndarray] = []
    motion_scores: list[float] = []

    previous_gray = _to_gray(frames[0])
    zeros = np.zeros((height, width), dtype=np.uint8)
    raw_boxes.append(None)
    masks.append(zeros)
    motion_scores.append(0.0)

    blur_kernel = (5, 5)
    morph_kernel = np.ones((5, 5), dtype=np.uint8)

    for frame in frames[1:]:
        current_gray = _to_gray(frame)
        diff = cv2.absdiff(previous_gray, current_gray)
        diff = cv2.GaussianBlur(diff, blur_kernel, 0)
        motion_scores.append(float(diff.mean()))

        _, binary = cv2.threshold(diff, motion_threshold, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel)

        box = _mask_to_bbox(binary, min_roi_area=min_roi_area)
        if box is not None:
            box = expand_bbox(box, roi_expand_ratio, width=width, height=height)

        raw_boxes.append(box)
        masks.append(binary if use_mask else binary)
        previous_gray = current_gray

    smoothed_boxes = _smooth_boxes(raw_boxes, roi_smooth_window)
    global_box = _union_boxes(smoothed_boxes, width=width, height=height)

    area_ratios = [
        box.area / float(width * height)
        for box in smoothed_boxes
        if box is not None and box.is_valid()
    ]
    average_roi_area_ratio = float(np.mean(area_ratios)) if area_ratios else 0.0

    return MotionRoiResult(
        per_frame_boxes=smoothed_boxes,
        per_frame_masks=masks,
        global_box=global_box,
        motion_scores=motion_scores,
        average_roi_area_ratio=average_roi_area_ratio,
    )
