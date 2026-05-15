"""Command-line entrypoint for the ROI-aware VSR prototype."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    UncertaintyBenchmarkResult,
    build_result,
    compare_frame_sequences,
    format_result,
    format_uncertainty_result,
    save_benchmark_results,
)
from src.fusion import (
    fuse_sequence_with_uncertainty,
    paste_roi_sequence,
    pixel_uncertainty_fusion,
)
from src.heavy_bvsr import BasicVsrRunner
from src.io_utils import (
    crop_frames,
    ensure_dir,
    is_image_path,
    list_video_paths,
    list_frame_paths,
    read_frames,
    read_frames_from_video,
    save_frames,
)
from src.light_enhance import bicubic_upscale_frames
from src.roi_motion import BoundingBox, detect_motion_rois
from src.texture_branch import enhance_texture_sequence
from src.uncertainty import (
    compute_structure_confidence,
    compute_temporal_residual,
    compute_texture_complexity,
    compute_uncertainty_map,
    compute_uncertainty_statistics,
    save_uncertainty_visualizations,
    smooth_uncertainty_maps,
)
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


def _find_gt_frames_dir(source_path: Path, gt_base_dir: Path) -> Optional[Path]:
    """Find GT frames directory for a source."""
    source_name = _source_stem(source_path)
    gt_candidates = [
        gt_base_dir / f"{source_name}gt",
        gt_base_dir / f"{source_name}_gt",
    ]
    for candidate in gt_candidates:
        if candidate.is_dir() and any(is_image_path(path) for path in candidate.iterdir() if path.is_file()):
            return candidate
    return None
    candidates = [
        source_video.with_name(f"{source_video.stem}gt{source_video.suffix}"),
        source_video.with_name(f"{source_video.stem}_gt{source_video.suffix}"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _discover_sources(config: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    explicit_source_names = args.source_video_names is not None
    if explicit_source_names:
        source_video_names = args.source_video_names
    elif args.input_video_dir is not None or args.input_frames_dir is not None:
        source_video_names = []
    else:
        source_video_names = config.get("source_video_names") or ["grass"]

    allowed_names = {_normalize_source_name(name) for name in source_video_names} if source_video_names else set()

    if args.input_video_dir is not None:
        video_dir = _resolve_path(args.input_video_dir)
        if video_dir is None:
            raise ValueError("input_video_dir cannot be resolved")
        return [
            path for path in list_video_paths(video_dir)
            if not allowed_names or _source_stem(path).lower() in allowed_names
        ]

    if args.input_video_path is not None:
        video_path = _resolve_path(args.input_video_path)
        if video_path is None:
            raise ValueError("input_video_path cannot be resolved")
        return [video_path]

    if args.input_frames_dir is not None:
        frames_path = _resolve_path(args.input_frames_dir)
        if frames_path is None:
            raise ValueError("input_frames_dir cannot be resolved")
        return _discover_frame_sources(frames_path, allowed_names)

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
        return _discover_frame_sources(frames_path, allowed_names)

    raise ValueError("No input source configured")


def _discover_frame_sources(frames_path: Path, allowed_names: set[str]) -> list[Path]:
    if frames_path.is_file():
        return [frames_path]

    if any(is_image_path(path) for path in sorted(frames_path.iterdir()) if path.is_file()):
        return [frames_path]

    subdirs: list[Path] = [path for path in sorted(frames_path.iterdir()) if path.is_dir()]
    frame_dirs = [subdir for subdir in subdirs if any(is_image_path(path) for path in sorted(subdir.iterdir()) if path.is_file())]
    if frame_dirs:
        if allowed_names:
            frame_dirs = [subdir for subdir in frame_dirs if _normalize_source_name(subdir.name) in allowed_names]
        if not frame_dirs:
            raise FileNotFoundError(
                f"No matching frame subdirectories found in {frames_path} for source names: {sorted(allowed_names)}"
            )
        return frame_dirs

    raise FileNotFoundError(f"No image frames found in: {frames_path}")


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


def _run_uncertainty_hybrid(
    config: dict[str, Any],
    args: argparse.Namespace,
    frames: list[Any],
    input_frames_dir: Path,
    output_root: Path,
    temp_dir: Path,
) -> UncertaintyBenchmarkResult:
    """Run uncertainty-aware hybrid VSR pipeline.
    
    Pipeline:
    1. Generate bicubic upsampled frames
    2. Run BasicVSR++ for conservative reconstruction
    3. Generate texture-enhanced frames
    4. Compute pixel-level uncertainty maps
    5. Smooth uncertainty maps
    6. Fuse frames using uncertainty weighting
    7. Save outputs and visualizations
    """
    runner = _prepare_runner(config, args)
    mode_output_dir = ensure_dir(output_root / "uncertainty_hybrid")
    mode_temp_dir = ensure_dir(temp_dir / "uncertainty_hybrid")
    
    # Get uncertainty config
    unc_config = config.get("uncertainty", {})
    alpha = float(unc_config.get("alpha", 1.0))
    beta = float(unc_config.get("beta", 0.8))
    gamma = float(unc_config.get("gamma", 0.6))
    sigmoid_temp = float(unc_config.get("sigmoid_temperature", 1.0))
    spatial_ksize = int(unc_config.get("spatial_smooth_ksize", 9))
    temporal_lambda = float(unc_config.get("temporal_smooth_lambda", 0.7))
    unc_threshold = float(unc_config.get("threshold", 0.5))
    save_heatmap = bool(unc_config.get("save_heatmap", True))
    save_overlay = bool(unc_config.get("save_overlay", True))
    
    # Get texture branch config
    tex_config = config.get("texture_branch", {})
    tex_method = str(tex_config.get("method", "unsharp"))
    sharpen_amount = float(tex_config.get("sharpen_amount", 1.0))
    blur_ksize = int(tex_config.get("blur_ksize", 5))
    realesrgan_config = tex_config.get("realesrgan", {})
    
    # Get fusion config
    fusion_config = config.get("fusion", {})
    clamp_output = bool(fusion_config.get("clamp_output", True))
    
    upscale_factor = int(args.upscale_factor if args.upscale_factor is not None else config.get("upscale_factor", 4))
    
    # Time tracking
    time_stats = {
        "bicubic": 0.0,
        "basicvsr": 0.0,
        "texture_branch": 0.0,
        "uncertainty": 0.0,
        "fusion": 0.0,
    }
    
    sampler = GpuMemorySampler(
        gpu_index=int(args.device if args.device is not None else config.get("device", 0))
    )
    sampler.start()
    start_time = time.perf_counter()
    
    # Step 1: Bicubic upsampling
    print("  [uncertainty_hybrid] Step 1/6: Bicubic upsampling...")
    bicubic_start = time.perf_counter()
    bicubic_frames = bicubic_upscale_frames(frames, upscale_factor)
    time_stats["bicubic"] = time.perf_counter() - bicubic_start
    
    # Step 2: BasicVSR++ reconstruction
    print("  [uncertainty_hybrid] Step 2/6: BasicVSR++ reconstruction...")
    basicvsr_input_dir = ensure_dir(mode_temp_dir / "basicvsr_input")
    basicvsr_output_dir = ensure_dir(mode_temp_dir / "basicvsr_output")
    save_frames(frames, basicvsr_input_dir)
    
    basicvsr_start = time.perf_counter()
    runner.run(
        input_dir=basicvsr_input_dir,
        output_dir=basicvsr_output_dir,
        window_size=int(args.window_size if args.window_size is not None else config.get("window_size", 0)),
        start_idx=int(config.get("start_idx", 0)),
        filename_tmpl=config.get("filename_tmpl", "{:08d}.png"),
        max_seq_len=args.max_seq_len if args.max_seq_len is not None else config.get("max_seq_len"),
    )
    basicvsr_frames = read_frames(basicvsr_output_dir)
    time_stats["basicvsr"] = time.perf_counter() - basicvsr_start
    
    if len(basicvsr_frames) != len(bicubic_frames):
        raise ValueError(
            f"BasicVSR++ output frame count mismatch: {len(basicvsr_frames)} vs {len(bicubic_frames)}"
        )
    
    # Step 3: Texture enhancement
    print("  [uncertainty_hybrid] Step 3/6: Texture branch enhancement...")
    texture_start = time.perf_counter()
    texture_source_frames = frames if tex_method.lower().strip() == "realesrgan" else basicvsr_frames
    texture_frames = enhance_texture_sequence(
        texture_source_frames,
        method=tex_method,
        sharpen_amount=sharpen_amount,
        blur_ksize=blur_ksize,
        realesrgan_config=realesrgan_config,
        gpu_id=int(args.device if args.device is not None else config.get("device", 0)),
    )
    time_stats["texture_branch"] = time.perf_counter() - texture_start
    
    # Step 4 & 5: Compute uncertainty maps and smooth
    print("  [uncertainty_hybrid] Step 4/5: Computing and smoothing uncertainty maps...")
    unc_start = time.perf_counter()
    
    # Compute temporal residual from input frames
    temporal_residuals = compute_temporal_residual(frames)
    
    # Compute uncertainty for each frame
    raw_uncertainties = []
    for frame_idx, frame in enumerate(frames):
        temporal_res = temporal_residuals[frame_idx] if frame_idx < len(temporal_residuals) else None
        uncertainty_map = compute_uncertainty_map(
            frame=frame,
            temporal_residual=temporal_res,
            texture_complexity=None,  # Will be computed automatically
            structure_confidence=None,  # Will be computed automatically
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            sigmoid_temperature=sigmoid_temp,
        )
        raw_uncertainties.append(uncertainty_map)
    
    # Smooth uncertainty maps spatially and temporally
    smoothed_uncertainties = smooth_uncertainty_maps(
        raw_uncertainties,
        spatial_blur_ksize=spatial_ksize,
        temporal_smooth_lambda=temporal_lambda,
    )
    time_stats["uncertainty"] = time.perf_counter() - unc_start
    
    # Step 6: Fusion
    print("  [uncertainty_hybrid] Step 6/6: Pixel-level uncertainty fusion...")
    fusion_start = time.perf_counter()
    
    # Fuse BasicVSR++ and texture frames using uncertainty
    # Need to resize uncertainty maps to match output frame size
    uncertainty_hr = []
    for u_map in smoothed_uncertainties:
        h, w = basicvsr_frames[0].shape[:2]
        if u_map.shape != (h, w):
            u_map_resized = cv2.resize(u_map, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            u_map_resized = u_map
        uncertainty_hr.append(u_map_resized)
    
    fused_frames = fuse_sequence_with_uncertainty(
        basicvsr_frames=basicvsr_frames,
        texture_frames=texture_frames,
        uncertainty_maps=uncertainty_hr,
        clamp_output=clamp_output,
    )
    time_stats["fusion"] = time.perf_counter() - fusion_start
    
    # Save fused output frames
    save_frames(fused_frames, mode_output_dir)
    
    # Save uncertainty visualizations
    if save_heatmap or save_overlay:
        unc_vis_dir = ensure_dir(mode_temp_dir / "uncertainty_vis")
        if save_heatmap:
            print("  [uncertainty_hybrid] Saving uncertainty heatmaps...")
            save_uncertainty_visualizations(
                uncertainty_hr,
                output_dir=unc_vis_dir / "heatmaps",
                threshold=unc_threshold,
            )
        if save_overlay:
            print("  [uncertainty_hybrid] Saving uncertainty overlays...")
            save_uncertainty_visualizations(
                uncertainty_hr,
                output_dir=unc_vis_dir / "overlays",
                threshold=unc_threshold,
            )
    
    elapsed_seconds = time.perf_counter() - start_time
    peak_memory = sampler.stop()
    
    # Compute uncertainty statistics
    unc_stats = compute_uncertainty_statistics(uncertainty_hr)
    
    # Build result
    result = UncertaintyBenchmarkResult(
        mode="uncertainty_hybrid",
        total_seconds=elapsed_seconds,
        frame_count=len(frames),
        avg_frame_seconds=elapsed_seconds / len(frames) if frames else 0.0,
        fps=len(frames) / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        peak_gpu_memory_mib=peak_memory,
        output_dir=mode_output_dir,
        bicubic_time=time_stats["bicubic"],
        basicvsr_time=time_stats["basicvsr"],
        texture_branch_time=time_stats["texture_branch"],
        uncertainty_time=time_stats["uncertainty"],
        fusion_time=time_stats["fusion"],
        avg_uncertainty=unc_stats.get("avg_uncertainty", None),
        high_uncertainty_ratio=unc_stats.get("high_uncertainty_ratio", None),
        metadata={
            "unc_alpha": alpha,
            "unc_beta": beta,
            "unc_gamma": gamma,
            "tex_method": tex_method,
            "upscale_factor": upscale_factor,
        },
    )
    
    print(format_uncertainty_result(result))
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


def _compare_frames_to_gt(gt_frames: list[Any], result_frames: list[Any]) -> dict[str, Any]:
    """Compare result frames to GT frames."""
    comparison = compare_frame_sequences(gt_frames, result_frames)
    return {
        "psnr": comparison.psnr,
        "ssim": comparison.ssim,
        "lpips": comparison.lpips,
        "frame_count": comparison.frame_count,
    }
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

    mode = str(args.mode) if args.mode is not None else "compare"
    
    full_result = None
    roi_result = None
    uncertainty_result = None

    # Run requested modes
    if mode in ["full_heavy", "compare"]:
        full_result = _run_full_heavy(
            config=config,
            args=args,
            frames=frames,
            input_frames_dir=input_frames_dir,
            output_root=source_output_dir,
            temp_dir=source_temp_dir,
        )
    
    if mode in ["roi_heavy", "compare"]:
        roi_result = _run_roi_heavy(
            config=config,
            args=args,
            frames=frames,
            input_frames_dir=input_frames_dir,
            output_root=source_output_dir,
            temp_dir=source_temp_dir,
        )
    
    if mode in ["uncertainty_hybrid", "compare"]:
        print(f"[{source_name}] Running uncertainty_hybrid mode...")
        uncertainty_result = _run_uncertainty_hybrid(
            config=config,
            args=args,
            frames=frames,
            input_frames_dir=input_frames_dir,
            output_root=source_output_dir,
            temp_dir=source_temp_dir,
        )

    export_videos = _get_bool_arg(args.save_videos, config.get("save_videos", True))
    exported_videos: dict[str, str] = {}
    if export_videos and mode in ["full_heavy", "roi_heavy", "compare"]:
        # Only export if we ran full_heavy or roi_heavy
        full_heavy_dir = source_output_dir / "full_heavy" if full_result else None
        roi_heavy_dir = source_output_dir / "roi_heavy" if roi_result else None
        
        if full_heavy_dir and full_heavy_dir.is_dir() and roi_heavy_dir and roi_heavy_dir.is_dir():
            exported = export_results_videos(
                input_frames_dir=original_frames_dir if original_frames_dir.is_dir() else input_frames_dir,
                full_heavy_frames_dir=full_heavy_dir,
                roi_heavy_frames_dir=roi_heavy_dir,
                results_video_dir=source_video_dir,
                fps=float(args.fps if args.fps is not None else config.get("fps", 25.0)),
            )
            exported_videos = {name: str(path) for name, path in exported.items()}

    comparison_summary: dict[str, Any] = {}
    frame_comparison_summary: dict[str, Any] = {}
    
    # Frame-level comparison if GT frames available
    gt_frames_dir = None
    if args.gt_frames_dir is not None:
        gt_base_dir = _resolve_path(args.gt_frames_dir)
        if gt_base_dir:
            gt_frames_dir = _find_gt_frames_dir(source_path, gt_base_dir)
    
    if gt_frames_dir:
        gt_frames = read_frames(gt_frames_dir)
        frame_comparison_summary["gt_frames_dir"] = str(gt_frames_dir)
        
        if full_result and (source_output_dir / "full_heavy").is_dir():
            full_heavy_frames = read_frames(source_output_dir / "full_heavy")
            frame_comparison_summary["full_heavy_vs_gt"] = _compare_frames_to_gt(gt_frames, full_heavy_frames)
        
        if uncertainty_result and (source_output_dir / "uncertainty_hybrid").is_dir():
            uncertainty_frames = read_frames(source_output_dir / "uncertainty_hybrid")
            frame_comparison_summary["uncertainty_hybrid_vs_gt"] = _compare_frames_to_gt(gt_frames, uncertainty_frames)
    
    if mode == "compare" and full_result and roi_result:
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
        "mode": mode,
        "exported_videos": exported_videos,
    }
    
    if full_result:
        video_summary["full_heavy"] = {
            "total_seconds": full_result.total_seconds,
            "avg_frame_seconds": full_result.avg_frame_seconds,
            "fps": full_result.fps,
            "peak_gpu_memory_mib": full_result.peak_gpu_memory_mib,
        }
    
    if roi_result:
        video_summary["roi_heavy"] = {
            "total_seconds": roi_result.total_seconds,
            "avg_frame_seconds": roi_result.avg_frame_seconds,
            "fps": roi_result.fps,
            "peak_gpu_memory_mib": roi_result.peak_gpu_memory_mib,
            "average_roi_area_ratio": roi_result.average_roi_area_ratio,
        }
    
    if uncertainty_result:
        video_summary["uncertainty_hybrid"] = {
            "total_seconds": uncertainty_result.total_seconds,
            "avg_frame_seconds": uncertainty_result.avg_frame_seconds,
            "fps": uncertainty_result.fps,
            "peak_gpu_memory_mib": uncertainty_result.peak_gpu_memory_mib,
            "avg_uncertainty": uncertainty_result.avg_uncertainty,
            "high_uncertainty_ratio": uncertainty_result.high_uncertainty_ratio,
            "time_breakdown": {
                "bicubic": uncertainty_result.bicubic_time,
                "basicvsr": uncertainty_result.basicvsr_time,
                "texture_branch": uncertainty_result.texture_branch_time,
                "uncertainty": uncertainty_result.uncertainty_time,
                "fusion": uncertainty_result.fusion_time,
            },
        }
    
    if comparison_summary:
        video_summary["comparison"] = comparison_summary
    
    if frame_comparison_summary:
        video_summary["frame_comparison"] = frame_comparison_summary

    _write_json(source_summary_dir / "metrics.json", video_summary)
    
    # Print summary
    if mode == "compare" and full_result and roi_result:
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
    parser = argparse.ArgumentParser(description="ROI-aware VSR prototype with uncertainty-aware hybrid support")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the config file")
    parser.add_argument(
        "--mode",
        default="compare",
        choices=["full_heavy", "roi_heavy", "uncertainty_hybrid", "compare"],
        help="Experiment mode",
    )
    parser.add_argument("--input-video-path", default=None, help="Override input video path")
    parser.add_argument("--input-video-dir", default=None, help="Override input video directory for batch processing")
    parser.add_argument("--source-video-names", nargs="*", default=None, help="Source video stems to process, e.g. grass road")
    parser.add_argument("--input-frames-dir", default=None, help="Override input frame directory")
    parser.add_argument("--gt-frames-dir", default=None, help="Directory containing GT frame folders for comparison")
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
