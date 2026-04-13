"""Command-line entrypoint for the ROI-aware VSR prototype."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

import cv2

from src.benchmark import (
    GpuMemorySampler,
    build_result,
    compare_frame_sequences,
    format_result,
)
from src.fusion import paste_roi_sequence
from src.heavy_bvsr import BasicVsrRunner
from src.io_utils import (
    crop_frames,
    ensure_dir,
    list_video_paths,
    read_frames,
    read_frames_from_video,
    save_frames,
)
from src.light_enhance import bicubic_upscale_frames
from src.roi_motion import BoundingBox, detect_motion_rois
from src.video_export import export_results_videos

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_config(config_path: Path) -> dict[str, Any]:
    suffix = config_path.suffix.lower()
    content = config_path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configs")
        data = yaml.safe_load(content)
    elif suffix == ".json":
        data = json.loads(content)
    else:
        raise ValueError(f"Unsupported config format: {config_path}")

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def _resolve_path(value: Optional[str], base_dir: Path = PROJECT_ROOT) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _get_bool_arg(value: Optional[bool], default: Any) -> bool:
    if value is None:
        return bool(default)
    return bool(value)


def _ensure_bbox(box: BoundingBox | None, frame_width: int, frame_height: int) -> BoundingBox:
    if box is None or not box.is_valid():
        return BoundingBox(0, 0, frame_width, frame_height)
    return box


def _source_stem(source_path: Path) -> str:
    return source_path.stem if source_path.is_file() else source_path.name


def _normalize_source_name(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(".mp4"):
        lowered = lowered[:-4]
    return lowered


def _find_gt_video_for_source(source_video: Path) -> Optional[Path]:
    candidates = [
        source_video.with_name(f"{source_video.stem}gt{source_video.suffix}"),
        source_video.with_name(f"{source_video.stem}_gt{source_video.suffix}"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _discover_sources(config: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    source_video_names = args.source_video_names or config.get("source_video_names") or ["grass"]
    allowed_names = {_normalize_source_name(name) for name in source_video_names}

    if args.input_video_dir is not None:
        video_dir = _resolve_path(args.input_video_dir)
        if video_dir is None:
            raise ValueError("input_video_dir cannot be resolved")
        return [path for path in list_video_paths(video_dir) if _source_stem(path).lower() in allowed_names]

    if args.input_video_path is not None:
        video_path = _resolve_path(args.input_video_path)
        if video_path is None:
            raise ValueError("input_video_path cannot be resolved")
        return [video_path]

    if args.input_frames_dir is not None:
        frames_path = _resolve_path(args.input_frames_dir)
        if frames_path is None:
            raise ValueError("input_frames_dir cannot be resolved")
        return [frames_path]

    input_video_dir = config.get("input_video_dir")
    if input_video_dir:
        video_dir = _resolve_path(input_video_dir)
        if video_dir is None:
            raise ValueError("input_video_dir cannot be resolved")
        return [path for path in list_video_paths(video_dir) if _source_stem(path).lower() in allowed_names]

    input_video_path = config.get("input_video_path")
    if input_video_path:
        video_path = _resolve_path(input_video_path)
        if video_path is None:
            raise ValueError("input_video_path cannot be resolved")
        return [video_path]

    input_frames_dir = config.get("input_frames_dir")
    if input_frames_dir:
        frames_path = _resolve_path(input_frames_dir)
        if frames_path is None:
            raise ValueError("input_frames_dir cannot be resolved")
        return [frames_path]

    raise ValueError("No input source configured")


def _load_input_frames(source_path: Path, temp_dir: Path) -> tuple[list[Any], Path]:
    if source_path.is_dir():
        frames = read_frames(source_path)
        return frames, source_path

    frames = read_frames(source_path)
    extracted_dir = temp_dir / _source_stem(source_path) / "input_frames"
    ensure_dir(extracted_dir)
    save_frames(frames, extracted_dir)
    return frames, extracted_dir


def _crop_with_bbox(frames: list[Any], bbox: BoundingBox) -> list[Any]:
    return crop_frames(frames, bbox.as_tuple())


def _prepare_runner(config: dict[str, Any], args: argparse.Namespace) -> BasicVsrRunner:
    basicvsr_root = _resolve_path(args.basicvsr_root or config["basicvsr_root"])
    basicvsr_config = _resolve_path(args.basicvsr_config or config["basicvsr_config"])
    basicvsr_checkpoint = _resolve_path(args.basicvsr_checkpoint or config["basicvsr_checkpoint"])
    if basicvsr_root is None or basicvsr_config is None or basicvsr_checkpoint is None:
        raise ValueError("BasicVSR++ paths cannot be resolved")

    python_executable = args.python_executable or sys.executable
    return BasicVsrRunner(
        basicvsr_root=basicvsr_root,
        basicvsr_config=basicvsr_config,
        basicvsr_checkpoint=basicvsr_checkpoint,
        device=int(args.device if args.device is not None else config.get("device", 0)),
        python_executable=python_executable,
    )


def _run_full_heavy(
    config: dict[str, Any],
    args: argparse.Namespace,
    frames: list[Any],
    input_frames_dir: Path,
    output_root: Path,
    temp_dir: Path,
) -> Any:
    runner = _prepare_runner(config, args)
    mode_output_dir = ensure_dir(output_root / "full_heavy")
    sampler = GpuMemorySampler(gpu_index=int(args.device if args.device is not None else config.get("device", 0)))
    sampler.start()

    import time

    start_time = time.perf_counter()
    runner.run(
        input_dir=input_frames_dir,
        output_dir=mode_output_dir,
        window_size=int(args.window_size if args.window_size is not None else config.get("window_size", 0)),
        start_idx=int(config.get("start_idx", 0)),
        filename_tmpl=config.get("filename_tmpl", "{:08d}.png"),
        max_seq_len=args.max_seq_len if args.max_seq_len is not None else config.get("max_seq_len"),
    )

    elapsed_seconds = time.perf_counter() - start_time
    peak_memory = sampler.stop()

    result = build_result(
        mode="full_heavy",
        elapsed_seconds=elapsed_seconds,
        frame_count=len(frames),
        output_dir=mode_output_dir,
        peak_gpu_memory_mib=peak_memory,
        average_roi_area_ratio=None,
    )
    print(format_result(result))
    return result


def _run_roi_heavy(
    config: dict[str, Any],
    args: argparse.Namespace,
    frames: list[Any],
    input_frames_dir: Path,
    output_root: Path,
    temp_dir: Path,
) -> Any:
    runner = _prepare_runner(config, args)
    mode_output_dir = ensure_dir(output_root / "roi_heavy")
    mode_temp_dir = ensure_dir(temp_dir / "roi_heavy")

    upscale_factor = int(args.upscale_factor if args.upscale_factor is not None else config.get("upscale_factor", 4))
    motion_threshold = float(args.motion_threshold if args.motion_threshold is not None else config.get("motion_threshold", 20.0))
    min_roi_area = int(args.min_roi_area if args.min_roi_area is not None else config.get("min_roi_area", 400))
    roi_expand_ratio = float(args.roi_expand_ratio if args.roi_expand_ratio is not None else config.get("roi_expand_ratio", 0.15))
    roi_smooth_window = int(args.roi_smooth_window if args.roi_smooth_window is not None else config.get("roi_smooth_window", 5))
    feather_blend = bool(args.feather_blend if args.feather_blend is not None else config.get("feather_blend", False))
    use_mask = bool(args.use_mask if args.use_mask is not None else config.get("use_mask", False))

    roi_result = detect_motion_rois(
        frames=frames,
        motion_threshold=motion_threshold,
        min_roi_area=min_roi_area,
        roi_expand_ratio=roi_expand_ratio,
        roi_smooth_window=roi_smooth_window,
        use_mask=use_mask,
    )

    frame_height, frame_width = frames[0].shape[:2]
    roi_box = _ensure_bbox(roi_result.global_box, frame_width, frame_height)

    roi_input_dir = ensure_dir(mode_temp_dir / "roi_input")
    roi_output_dir = ensure_dir(mode_temp_dir / "roi_output")
    light_output_dir = ensure_dir(mode_temp_dir / "light_output")

    cropped_frames = _crop_with_bbox(frames, roi_box)
    save_frames(cropped_frames, roi_input_dir)

    sampler = GpuMemorySampler(gpu_index=int(args.device if args.device is not None else config.get("device", 0)))
    sampler.start()
    import time

    start_time = time.perf_counter()
    runner.run(
        input_dir=roi_input_dir,
        output_dir=roi_output_dir,
        window_size=int(args.window_size if args.window_size is not None else config.get("window_size", 0)),
        start_idx=int(config.get("start_idx", 0)),
        filename_tmpl=config.get("filename_tmpl", "{:08d}.png"),
        max_seq_len=args.max_seq_len if args.max_seq_len is not None else config.get("max_seq_len"),
    )
    elapsed_seconds = time.perf_counter() - start_time
    peak_memory = sampler.stop()

    roi_enhanced_frames = read_frames(roi_output_dir)
    light_frames = bicubic_upscale_frames(frames, upscale_factor)

    if len(roi_enhanced_frames) != len(light_frames):
        raise ValueError(
            f"ROI output frame count mismatch: {len(roi_enhanced_frames)} vs {len(light_frames)}"
        )

    hr_box = BoundingBox(
        x1=roi_box.x1 * upscale_factor,
        y1=roi_box.y1 * upscale_factor,
        x2=roi_box.x2 * upscale_factor,
        y2=roi_box.y2 * upscale_factor,
    )
    fused_frames = paste_roi_sequence(
        base_frames=light_frames,
        roi_frames=roi_enhanced_frames,
        bboxes=[hr_box],
        feather_blend=feather_blend,
    )

    save_frames(fused_frames, mode_output_dir)
    if bool(args.save_intermediate if args.save_intermediate is not None else config.get("save_intermediate", False)):
        save_frames(light_frames, light_output_dir)

    result = build_result(
        mode="roi_heavy",
        elapsed_seconds=elapsed_seconds,
        frame_count=len(frames),
        output_dir=mode_output_dir,
        peak_gpu_memory_mib=peak_memory,
        average_roi_area_ratio=roi_result.average_roi_area_ratio,
    )
    print(format_result(result))
    return result


def _write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resize_frame_to_match(frame: Any, target_shape: tuple[int, int, int]) -> Any:
    target_height, target_width = target_shape[:2]
    if frame.shape[:2] == (target_height, target_width):
        return frame
    interpolation = cv2.INTER_CUBIC if frame.shape[0] < target_height or frame.shape[1] < target_width else cv2.INTER_AREA
    return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)


def _compare_result_video_to_gt(result_video_path: Path, gt_video_path: Path) -> dict[str, Any]:
    result_frames = read_frames_from_video(result_video_path)
    gt_frames = read_frames_from_video(gt_video_path)
    frame_count = min(len(result_frames), len(gt_frames))
    if frame_count == 0:
        raise ValueError(f"Empty result/gt videos: {result_video_path}, {gt_video_path}")
    result_frames = result_frames[:frame_count]
    gt_frames = gt_frames[:frame_count]
    aligned_gt_frames = [
        _resize_frame_to_match(gt_frame, result_frame.shape)
        for gt_frame, result_frame in zip(gt_frames, result_frames)
    ]
    comparison = compare_frame_sequences(aligned_gt_frames, result_frames)
    return {
        "psnr": comparison.psnr,
        "ssim": comparison.ssim,
        "frame_count": comparison.frame_count,
    }


def _run_video_pipeline(
    source_path: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
    temp_root: Path,
) -> dict[str, Any]:
    source_name = _source_stem(source_path)
    source_temp_dir = ensure_dir(temp_root / source_name)
    source_output_dir = ensure_dir(output_root / source_name)
    source_video_dir = ensure_dir(source_output_dir / "video")
    source_summary_dir = ensure_dir(source_output_dir / "summary")

    frames, input_frames_dir = _load_input_frames(source_path, source_temp_dir)
    original_frames_dir = source_output_dir / "original_frames"
    if bool(config.get("save_intermediate", False)):
        save_frames(frames, original_frames_dir)

    full_result = _run_full_heavy(
        config=config,
        args=args,
        frames=frames,
        input_frames_dir=input_frames_dir,
        output_root=source_output_dir,
        temp_dir=source_temp_dir,
    )
    roi_result = _run_roi_heavy(
        config=config,
        args=args,
        frames=frames,
        input_frames_dir=input_frames_dir,
        output_root=source_output_dir,
        temp_dir=source_temp_dir,
    )

    export_videos = _get_bool_arg(args.save_videos, config.get("save_videos", True))
    exported_videos: dict[str, str] = {}
    if export_videos:
        exported = export_results_videos(
            input_frames_dir=original_frames_dir if original_frames_dir.is_dir() else input_frames_dir,
            full_heavy_frames_dir=source_output_dir / "full_heavy",
            roi_heavy_frames_dir=source_output_dir / "roi_heavy",
            results_video_dir=source_video_dir,
            fps=float(args.fps if args.fps is not None else config.get("fps", 25.0)),
        )
        exported_videos = {name: str(path) for name, path in exported.items()}

    comparison_summary: dict[str, Any]
    try:
        full_video_path = Path(exported_videos["full_heavy"]) if "full_heavy" in exported_videos else source_video_dir / "full_heavy.mp4"
        roi_video_path = Path(exported_videos["roi_heavy"]) if "roi_heavy" in exported_videos else source_video_dir / "roi_heavy.mp4"
        gt_video_path = _find_gt_video_for_source(source_path)
        if gt_video_path is None:
            raise FileNotFoundError(f"GT video not found for source: {source_path}")

        full_vs_gt = _compare_result_video_to_gt(full_video_path, gt_video_path)
        roi_vs_gt = _compare_result_video_to_gt(roi_video_path, gt_video_path)
        comparison_summary = {
            "gt_video": str(gt_video_path),
            "full_heavy_vs_gt": full_vs_gt,
            "roi_heavy_vs_gt": roi_vs_gt,
            "note": "PSNR/SSIM are computed from exported videos under output/<name>/video against the paired GT video.",
        }
    except Exception as exc:
        comparison_summary = {
            "error": str(exc),
            "note": "GT comparison failed, but exported videos were still kept.",
        }

    video_summary = {
        "source": str(source_path),
        "source_name": source_name,
        "full_heavy": {
            "total_seconds": full_result.total_seconds,
            "avg_frame_seconds": full_result.avg_frame_seconds,
            "fps": full_result.fps,
            "peak_gpu_memory_mib": full_result.peak_gpu_memory_mib,
        },
        "roi_heavy": {
            "total_seconds": roi_result.total_seconds,
            "avg_frame_seconds": roi_result.avg_frame_seconds,
            "fps": roi_result.fps,
            "peak_gpu_memory_mib": roi_result.peak_gpu_memory_mib,
            "average_roi_area_ratio": roi_result.average_roi_area_ratio,
        },
        "comparison": comparison_summary,
        "exported_videos": exported_videos,
    }

    _write_json(source_summary_dir / "metrics.json", video_summary)
    if "error" in comparison_summary:
        print(
            f"[{source_name}] comparison skipped: {comparison_summary['error']}; "
            f"full={full_result.fps:.2f} FPS, roi={roi_result.fps:.2f} FPS"
        )
    else:
        full_vs_gt = comparison_summary["full_heavy_vs_gt"]
        roi_vs_gt = comparison_summary["roi_heavy_vs_gt"]
        print(
            f"[{source_name}] full_vs_gt PSNR={full_vs_gt['psnr']:.4f}, SSIM={full_vs_gt['ssim']:.4f}; "
            f"roi_vs_gt PSNR={roi_vs_gt['psnr']:.4f}, SSIM={roi_vs_gt['ssim']:.4f}; "
            f"full={full_result.fps:.2f} FPS, roi={roi_result.fps:.2f} FPS"
        )
    return video_summary


def _print_comparison(full_result: Any, roi_result: Any) -> None:
    print("\n=== comparison ===")
    print(format_result(full_result))
    print(format_result(roi_result))
    if full_result.total_seconds > 0 and roi_result.total_seconds > 0:
        time_speedup = full_result.total_seconds / roi_result.total_seconds
        fps_speedup = roi_result.fps / full_result.fps if full_result.fps > 0 else 0.0
        print(f"speedup(full/roi)={time_speedup:.3f}x")
        print(f"fps_ratio(roi/full)={fps_speedup:.3f}x")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROI-aware VSR prototype")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the config file")
    parser.add_argument("--mode", default="compare", choices=["full_heavy", "roi_heavy", "compare"], help="Experiment mode")
    parser.add_argument("--input-video-path", default=None, help="Override input video path")
    parser.add_argument("--input-video-dir", default=None, help="Override input video directory for batch processing")
    parser.add_argument("--source-video-names", nargs="*", default=None, help="Source video stems to process, e.g. grass road")
    parser.add_argument("--input-frames-dir", default=None, help="Override input frame directory")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--temp-dir", default=None, help="Override temporary directory")
    parser.add_argument("--basicvsr-root", default=None, help="Override BasicVSR++ root path")
    parser.add_argument("--basicvsr-config", default=None, help="Override BasicVSR++ config path")
    parser.add_argument("--basicvsr-checkpoint", default=None, help="Override BasicVSR++ checkpoint path")
    parser.add_argument("--upscale-factor", type=int, default=None, help="Override upscale factor")
    parser.add_argument("--motion-threshold", type=float, default=None, help="Override motion threshold")
    parser.add_argument("--min-roi-area", type=int, default=None, help="Override minimum ROI area")
    parser.add_argument("--roi-expand-ratio", type=float, default=None, help="Override ROI expansion ratio")
    parser.add_argument("--roi-smooth-window", type=int, default=None, help="Override ROI smoothing window")
    parser.add_argument("--use-mask", action="store_true", default=None, help="Enable mask output for ROI detection")
    parser.add_argument("--use-bbox", action="store_true", default=None, help="Keep bbox-based fusion enabled")
    parser.add_argument("--save-intermediate", action="store_true", default=None, help="Save intermediate frames")
    parser.add_argument("--save-videos", action="store_true", default=None, help="Export output frames to mp4 automatically")
    parser.add_argument("--feather-blend", action="store_true", default=None, help="Enable feather blending at ROI borders")
    parser.add_argument("--window-size", type=int, default=None, help="BasicVSR++ window size")
    parser.add_argument("--max-seq-len", type=int, default=None, help="BasicVSR++ max sequence length")
    parser.add_argument("--device", type=int, default=None, help="CUDA device index")
    parser.add_argument("--fps", type=float, default=None, help="FPS used for exported videos")
    parser.add_argument("--python-executable", default=None, help="Python executable used to call BasicVSR++")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config_path = _resolve_path(args.config)
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = _load_config(config_path)
    output_root = _resolve_path(args.output_dir or config.get("output_dir", "results/output"))
    temp_dir = _resolve_path(args.temp_dir or config.get("temp_dir", "results/tmp"))
    summary_dir = _resolve_path(config.get("summary_dir", "results/summary"))
    video_root = _resolve_path(config.get("video_output_dir", "results/video"))
    if output_root is None or temp_dir is None:
        raise ValueError("Output or temp directory cannot be resolved")
    ensure_dir(output_root)
    ensure_dir(temp_dir)
    if summary_dir is not None:
        ensure_dir(summary_dir)
    if video_root is not None:
        ensure_dir(video_root)

    sources = _discover_sources(config, args)
    batch_summary: list[dict[str, Any]] = []

    if len(sources) > 1:
        print(f"发现 {len(sources)} 个输入视频，将按顺序批处理。")

    for source_path in sources:
        summary = _run_video_pipeline(
            source_path=source_path,
            config=config,
            args=args,
            output_root=output_root,
            temp_root=temp_dir,
        )
        batch_summary.append(summary)

    if summary_dir is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _write_json(summary_dir / f"run_summary_{timestamp}.json", batch_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
