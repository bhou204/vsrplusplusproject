"""Helpers for exporting frame folders as videos."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .io_utils import ensure_dir, list_frame_paths, read_frames_from_dir, write_video


def export_frame_dir_to_video(
    frame_dir: str | Path,
    output_video_path: str | Path,
    fps: float = 25.0,
) -> Path:
    """Convert a directory of image frames into a video file."""

    frames = read_frames_from_dir(frame_dir)
    return write_video(frames, output_video_path, fps=fps)


def export_results_videos(
    input_frames_dir: str | Path,
    full_heavy_frames_dir: str | Path,
    roi_heavy_frames_dir: str | Path,
    results_video_dir: str | Path,
    fps: float = 25.0,
) -> dict[str, Path]:
    """Export source, full-heavy and ROI-heavy frame folders to videos."""

    video_dir = ensure_dir(results_video_dir)
    outputs: dict[str, Path] = {}
    outputs["original"] = export_frame_dir_to_video(
        input_frames_dir, video_dir / "original.mp4", fps=fps)
    outputs["full_heavy"] = export_frame_dir_to_video(
        full_heavy_frames_dir, video_dir / "full_heavy.mp4", fps=fps)
    outputs["roi_heavy"] = export_frame_dir_to_video(
        roi_heavy_frames_dir, video_dir / "roi_heavy.mp4", fps=fps)
    return outputs


def export_side_by_side_video(
    left_frames_dir: str | Path,
    right_frames_dir: str | Path,
    output_video_path: str | Path,
    fps: float = 25.0,
    left_label: Optional[str] = None,
    right_label: Optional[str] = None,
) -> Path:
    """Create a side-by-side comparison video from two frame directories.

    This is optional and useful when you want a single video to compare
    original/full-heavy/roi-heavy differences visually.
    """

    import cv2
    import numpy as np

    left_frames = read_frames_from_dir(left_frames_dir)
    right_frames = read_frames_from_dir(right_frames_dir)
    if len(left_frames) != len(right_frames):
        raise ValueError(
            f"Frame count mismatch: {len(left_frames)} vs {len(right_frames)}"
        )

    paired_frames: list[np.ndarray] = []
    for left_frame, right_frame in zip(left_frames, right_frames):
        left_height, left_width = left_frame.shape[:2]
        right_height, right_width = right_frame.shape[:2]
        target_height = max(left_height, right_height)

        if left_height != target_height:
            left_frame = cv2.resize(
                left_frame,
                (left_width, target_height),
                interpolation=cv2.INTER_CUBIC,
            )
        if right_height != target_height:
            right_frame = cv2.resize(
                right_frame,
                (right_width, target_height),
                interpolation=cv2.INTER_CUBIC,
            )

        separator = np.full((target_height, 8, 3), 255, dtype=np.uint8)
        paired_frames.append(np.concatenate([left_frame, separator, right_frame], axis=1))

    return write_video(paired_frames, output_video_path, fps=fps)
