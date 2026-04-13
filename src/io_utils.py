"""I/O helpers for reading frames, saving frames, and writing videos."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_video_path(path: str | Path) -> bool:
    """Return True if the path points to a video file."""

    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_image_path(path: str | Path) -> bool:
    """Return True if the path points to an image file."""

    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def list_video_paths(video_dir: str | Path) -> list[Path]:
    """List video files from a directory in lexicographic order."""

    directory = Path(video_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Video directory not found: {directory}")

    video_paths = [
        path for path in sorted(directory.iterdir())
        if path.is_file() and is_video_path(path)
    ]
    if not video_paths:
        raise FileNotFoundError(f"No video files found in: {directory}")
    return video_paths


def list_frame_paths(frame_dir: str | Path) -> list[Path]:
    """List image paths from a frame directory in lexicographic order."""

    directory = Path(frame_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {directory}")

    frame_paths = [
        path for path in sorted(directory.iterdir())
        if path.is_file() and is_image_path(path)
    ]
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in: {directory}")
    return frame_paths


def read_image(path: str | Path) -> np.ndarray:
    """Read a single image in BGR format."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def read_frames_from_dir(frame_dir: str | Path) -> list[np.ndarray]:
    """Read all frames from a directory of images."""

    return [read_image(path) for path in list_frame_paths(frame_dir)]


def read_frames_from_video(video_path: str | Path) -> list[np.ndarray]:
    """Decode a video file into a list of BGR frames."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    frames: list[np.ndarray] = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame)

    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return frames


def read_frames(source_path: str | Path) -> list[np.ndarray]:
    """Read frames from either a video file or a frame directory."""

    source = Path(source_path)
    if source.is_dir():
        return read_frames_from_dir(source)
    if source.is_file() and is_video_path(source):
        return read_frames_from_video(source)
    if source.is_file() and is_image_path(source):
        return [read_image(source)]
    raise FileNotFoundError(f"Unsupported input source: {source}")


def save_frames(
    frames: Sequence[np.ndarray],
    output_dir: str | Path,
    filename_tmpl: str = "{:08d}.png",
    start_index: int = 0,
) -> list[Path]:
    """Save a sequence of frames to a directory."""

    directory = ensure_dir(output_dir)
    saved_paths: list[Path] = []
    for offset, frame in enumerate(frames):
        frame_index = start_index + offset
        save_path = directory / filename_tmpl.format(frame_index)
        if not cv2.imwrite(str(save_path), frame):
            raise IOError(f"Failed to write frame: {save_path}")
        saved_paths.append(save_path)
    return saved_paths


def write_video(
    frames: Sequence[np.ndarray],
    output_path: str | Path,
    fps: float = 25.0,
) -> Path:
    """Write frames to a video file using mp4v."""

    if not frames:
        raise ValueError("Cannot write an empty video sequence")

    path = Path(output_path)
    ensure_dir(path.parent)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Failed to open video writer: {path}")

    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        writer.write(frame)

    writer.release()
    return path


def crop_frame(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a frame by an inclusive-exclusive bbox."""

    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2].copy()


def crop_frames(
    frames: Sequence[np.ndarray],
    bbox: tuple[int, int, int, int],
) -> list[np.ndarray]:
    """Crop all frames using the same bbox."""

    return [crop_frame(frame, bbox) for frame in frames]
