"""Benchmark utilities for timing and rough GPU memory tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import json
import shutil
import subprocess
import threading
import time
from typing import Any, Optional

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
class UncertaintyBenchmarkResult:
    """Extended benchmark for uncertainty-aware VSR.
    
    Tracks time breakdown across different pipeline stages.
    """
    mode: str
    total_seconds: float
    frame_count: int
    avg_frame_seconds: float
    fps: float
    peak_gpu_memory_mib: Optional[float]
    output_dir: Path
    
    # Time breakdown (seconds)
    bicubic_time: Optional[float] = None
    basicvsr_time: Optional[float] = None
    texture_branch_time: Optional[float] = None
    uncertainty_time: Optional[float] = None
    fusion_time: Optional[float] = None
    
    # Uncertainty statistics
    avg_uncertainty: Optional[float] = None
    high_uncertainty_ratio: Optional[float] = None
    
    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameComparisonResult:
    """Average image-quality metrics for two aligned frame sequences."""

    frame_count: int
    psnr: float
    ssim: float
    lpips: Optional[float] = None


def _compute_psnr(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    frame_a = frame_a.astype(np.float64)
    frame_b = frame_b.astype(np.float64)
    mse = float(np.mean((frame_a - frame_b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10((255.0 * 255.0) / mse)


def _compute_lpips(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute LPIPS distance between two frames."""
    try:
        import torch
        import lpips
    except ImportError:
        return float("nan")
    
    # LPIPS expects RGB, but frames are BGR, so convert
    frame_a_rgb = cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB)
    frame_b_rgb = cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB)
    
    # Normalize to [-1, 1] as expected by LPIPS
    frame_a_tensor = torch.from_numpy(frame_a_rgb).permute(2, 0, 1).float() / 127.5 - 1.0
    frame_b_tensor = torch.from_numpy(frame_b_rgb).permute(2, 0, 1).float() / 127.5 - 1.0
    
    # Add batch dimension
    frame_a_tensor = frame_a_tensor.unsqueeze(0)
    frame_b_tensor = frame_b_tensor.unsqueeze(0)
    
    # Use AlexNet backbone for LPIPS
    loss_fn = lpips.LPIPS(net='alex')
    with torch.no_grad():
        lpips_value = loss_fn(frame_a_tensor, frame_b_tensor).item()
    
    return lpips_value


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
    """Compare two frame sequences with average PSNR, SSIM, and LPIPS.

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
    lpips_scores: list[float] = []
    for reference_frame, compared_frame in zip(reference_frames, compared_frames):
        if reference_frame.shape != compared_frame.shape:
            raise ValueError(
                f"Frame shape mismatch: {reference_frame.shape} vs {compared_frame.shape}"
            )
        psnr_scores.append(_compute_psnr(reference_frame, compared_frame))
        ssim_scores.append(_compute_ssim(reference_frame, compared_frame))
        lpips_scores.append(_compute_lpips(reference_frame, compared_frame))

    return FrameComparisonResult(
        frame_count=len(reference_frames),
        psnr=float(np.mean(psnr_scores)),
        ssim=float(np.mean(ssim_scores)),
        lpips=float(np.nanmean(lpips_scores)) if lpips_scores else None,
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


def format_uncertainty_result(result: UncertaintyBenchmarkResult) -> str:
    """Format an uncertainty benchmark result as human-readable text."""
    peak_memory = (
        f"{result.peak_gpu_memory_mib:.2f} MiB"
        if result.peak_gpu_memory_mib is not None else "N/A"
    )
    
    lines = [
        f"mode={result.mode}",
        f"total_time={result.total_seconds:.3f}s",
        f"avg_frame_time={result.avg_frame_seconds:.3f}s",
        f"fps={result.fps:.2f}",
        f"peak_gpu_memory={peak_memory}",
    ]
    
    if result.bicubic_time is not None:
        lines.append(f"bicubic_time={result.bicubic_time:.3f}s")
    if result.basicvsr_time is not None:
        lines.append(f"basicvsr_time={result.basicvsr_time:.3f}s")
    if result.texture_branch_time is not None:
        lines.append(f"texture_branch_time={result.texture_branch_time:.3f}s")
    if result.uncertainty_time is not None:
        lines.append(f"uncertainty_time={result.uncertainty_time:.3f}s")
    if result.fusion_time is not None:
        lines.append(f"fusion_time={result.fusion_time:.3f}s")
    
    if result.avg_uncertainty is not None:
        lines.append(f"avg_uncertainty={result.avg_uncertainty:.4f}")
    if result.high_uncertainty_ratio is not None:
        lines.append(f"high_uncertainty_ratio={result.high_uncertainty_ratio:.4f}")
    
    lines.append(f"output_dir={result.output_dir}")
    
    return ", ".join(lines)


def save_benchmark_results(
    results: list[BenchmarkResult | UncertaintyBenchmarkResult],
    output_dir: Path | str,
    json_output: bool = True,
    csv_output: bool = True,
) -> None:
    """Save benchmark results to JSON and/or CSV files.
    
    Args:
        results: List of benchmark results (can mix regular and uncertainty results).
        output_dir: Directory to save results.
        json_output: Whether to save as JSON.
        csv_output: Whether to save as CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict format
    results_dicts = []
    for result in results:
        if isinstance(result, UncertaintyBenchmarkResult):
            result_dict = asdict(result)
            result_dict["output_dir"] = str(result_dict["output_dir"])
            # Flatten metadata if any
            if result_dict["metadata"]:
                result_dict.update(result_dict.pop("metadata"))
        else:
            result_dict = asdict(result)
            result_dict["output_dir"] = str(result_dict["output_dir"])
        results_dicts.append(result_dict)
    
    # Save JSON
    if json_output:
        json_path = output_dir / "benchmark.json"
        with open(json_path, "w") as f:
            json.dump(results_dicts, f, indent=2)
    
    # Save CSV
    if csv_output and results_dicts:
        csv_path = output_dir / "benchmark.csv"
        keys = results_dicts[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results_dicts)
