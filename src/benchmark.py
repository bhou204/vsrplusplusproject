"""Benchmark utilities for timing and rough GPU memory tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Optional

import cv2
import numpy as np


@dataclass
class BenchmarkResult:
    """Compact benchmark summary."""

    mode: str
    total_seconds: float
    frame_count: int
    avg_frame_seconds: float
    fps: float
    peak_gpu_memory_mib: Optional[float]
    average_roi_area_ratio: Optional[float]
    output_dir: Path


@dataclass
class FrameComparisonResult:
    """Average image-quality metrics for two aligned frame sequences."""

    frame_count: int
    psnr: float
    ssim: float


def _compute_psnr(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    frame_a = frame_a.astype(np.float64)
    frame_b = frame_b.astype(np.float64)
    mse = float(np.mean((frame_a - frame_b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10((255.0 * 255.0) / mse)


def _compute_ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    frame_a = frame_a.astype(np.float64)
    frame_b = frame_b.astype(np.float64)
    if frame_a.ndim == 2:
        frame_a = frame_a[..., None]
        frame_b = frame_b[..., None]

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    channel_scores: list[float] = []

    for channel_index in range(frame_a.shape[2]):
        channel_a = frame_a[..., channel_index]
        channel_b = frame_b[..., channel_index]
        mu_a = cv2.GaussianBlur(channel_a, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(channel_b, (11, 11), 1.5)
        mu_a_sq = mu_a * mu_a
        mu_b_sq = mu_b * mu_b
        mu_ab = mu_a * mu_b

        sigma_a_sq = cv2.GaussianBlur(channel_a * channel_a, (11, 11), 1.5) - mu_a_sq
        sigma_b_sq = cv2.GaussianBlur(channel_b * channel_b, (11, 11), 1.5) - mu_b_sq
        sigma_ab = cv2.GaussianBlur(channel_a * channel_b, (11, 11), 1.5) - mu_ab

        numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
        denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
        ssim_map = numerator / denominator
        channel_scores.append(float(np.mean(ssim_map)))

    return float(np.mean(channel_scores))


def compare_frame_sequences(
    reference_frames: list[np.ndarray],
    compared_frames: list[np.ndarray],
) -> FrameComparisonResult:
    """Compare two frame sequences with average PSNR and SSIM.

    The frames must already be spatially aligned and have the same length.
    """

    if len(reference_frames) != len(compared_frames):
        raise ValueError(
            f"Frame count mismatch: {len(reference_frames)} vs {len(compared_frames)}"
        )
    if not reference_frames:
        raise ValueError("Cannot compare empty frame sequences")

    psnr_scores: list[float] = []
    ssim_scores: list[float] = []
    for reference_frame, compared_frame in zip(reference_frames, compared_frames):
        if reference_frame.shape != compared_frame.shape:
            raise ValueError(
                f"Frame shape mismatch: {reference_frame.shape} vs {compared_frame.shape}"
            )
        psnr_scores.append(_compute_psnr(reference_frame, compared_frame))
        ssim_scores.append(_compute_ssim(reference_frame, compared_frame))

    return FrameComparisonResult(
        frame_count=len(reference_frames),
        psnr=float(np.mean(psnr_scores)),
        ssim=float(np.mean(ssim_scores)),
    )


class GpuMemorySampler:
    """Best-effort peak GPU memory sampler using nvidia-smi."""

    def __init__(self, gpu_index: int = 0, interval_sec: float = 0.5) -> None:
        self.gpu_index = gpu_index
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak_memory_mib: float = 0.0
        self._enabled = shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if not self._enabled:
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                        "-i",
                        str(self.gpu_index),
                    ],
                    text=True,
                ).strip()
                value = float(output.splitlines()[0])
                self._peak_memory_mib = max(self._peak_memory_mib, value)
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def stop(self) -> Optional[float]:
        if not self._enabled:
            return None

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec * 4)
        return self._peak_memory_mib if self._peak_memory_mib > 0 else None


def build_result(
    mode: str,
    elapsed_seconds: float,
    frame_count: int,
    output_dir: str | Path,
    peak_gpu_memory_mib: Optional[float] = None,
    average_roi_area_ratio: Optional[float] = None,
) -> BenchmarkResult:
    """Build a benchmark result from primitive measurements."""

    avg_frame_seconds = elapsed_seconds / frame_count if frame_count > 0 else 0.0
    fps = frame_count / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return BenchmarkResult(
        mode=mode,
        total_seconds=elapsed_seconds,
        frame_count=frame_count,
        avg_frame_seconds=avg_frame_seconds,
        fps=fps,
        peak_gpu_memory_mib=peak_gpu_memory_mib,
        average_roi_area_ratio=average_roi_area_ratio,
        output_dir=Path(output_dir),
    )


def format_result(result: BenchmarkResult) -> str:
    """Format a benchmark result as human-readable text."""

    peak_memory = (
        f"{result.peak_gpu_memory_mib:.2f} MiB"
        if result.peak_gpu_memory_mib is not None else "N/A"
    )
    roi_ratio = (
        f"{result.average_roi_area_ratio:.4f}"
        if result.average_roi_area_ratio is not None else "N/A"
    )
    return (
        f"mode={result.mode}, total={result.total_seconds:.3f}s, "
        f"avg_frame={result.avg_frame_seconds:.3f}s, fps={result.fps:.2f}, "
        f"peak_gpu_memory={peak_memory}, avg_roi_area_ratio={roi_ratio}, "
        f"output_dir={result.output_dir}"
    )
