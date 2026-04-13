"""Compare already-generated outputs against paired GT videos without rerunning inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import compare_frame_sequences
from src.io_utils import ensure_dir, list_video_paths, read_frames_from_video


def _resolve(path_text: str, base_dir: Path = ROOT) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _is_gt_video(path: Path) -> bool:
    return path.stem.lower().endswith("gt")


def _find_gt_video(source_path: Path, all_video_paths: list[Path]) -> Path:
    candidates = [
        source_path.with_name(f"{source_path.stem}gt{source_path.suffix}"),
        source_path.with_name(f"{source_path.stem}_gt{source_path.suffix}"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    source_stem = source_path.stem.lower()
    for candidate in all_video_paths:
        if candidate == source_path:
            continue
        candidate_stem = candidate.stem.lower()
        if candidate_stem == f"{source_stem}gt" or candidate_stem == f"{source_stem}_gt":
            return candidate

    raise FileNotFoundError(f"Paired GT video not found for {source_path}")


def _load_existing_result_video(results_root: Path, source_name: str, mode: str) -> Path:
    candidates = [
        results_root / source_name / "video" / f"{mode}.mp4",
        results_root / source_name / mode / f"{mode}.mp4",
        results_root / source_name / f"{mode}.mp4",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Result video not found for source={source_name}, mode={mode}. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _resize_frame_to_match(frame: Any, target_shape: tuple[int, int, int]) -> Any:
    target_height, target_width = target_shape[:2]
    if frame.shape[:2] == (target_height, target_width):
        return frame

    interpolation = cv2.INTER_CUBIC if frame.shape[0] < target_height or frame.shape[1] < target_width else cv2.INTER_AREA
    return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)


def _compare_video_to_gt(
    result_video: Path,
    gt_video: Path,
    resize_gt_to_result: bool,
) -> dict[str, Any]:
    result_frames = read_frames_from_video(result_video)
    gt_frames = read_frames_from_video(gt_video)
    frame_count = min(len(result_frames), len(gt_frames))
    if frame_count == 0:
        raise ValueError(f"Empty video pair: {result_video}, {gt_video}")

    if len(result_frames) != len(gt_frames):
        result_frames = result_frames[:frame_count]
        gt_frames = gt_frames[:frame_count]

    if resize_gt_to_result:
        aligned_gt_frames = [
            _resize_frame_to_match(gt_frame, result_frame.shape)
            for gt_frame, result_frame in zip(gt_frames, result_frames)
        ]
    else:
        aligned_gt_frames = gt_frames

    comparison = compare_frame_sequences(aligned_gt_frames, result_frames)
    return {
        "result_video": str(result_video),
        "gt_video": str(gt_video),
        "frame_count": comparison.frame_count,
        "psnr": comparison.psnr,
        "ssim": comparison.ssim,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare existing outputs against GT videos")
    parser.add_argument("--data-dir", default="data", help="Directory containing source and GT videos")
    parser.add_argument("--results-root", default="results/output", help="Directory containing generated outputs")
    parser.add_argument("--output-json", default="results/summary/existing_video_metrics.json", help="Path to save JSON summary")
    parser.add_argument(
        "--resize-gt-to-result",
        action="store_true",
        help="Resize GT frames to result resolution before comparison (default: off, strict compare)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = _resolve(args.data_dir)
    results_root = _resolve(args.results_root)
    output_json = _resolve(args.output_json)

    all_video_paths = list_video_paths(data_dir)
    source_videos = [path for path in all_video_paths if not _is_gt_video(path)]

    summary: list[dict[str, Any]] = []
    for source_path in source_videos:
        gt_path = _find_gt_video(source_path, all_video_paths)
        source_name = source_path.stem
        full_video = _load_existing_result_video(results_root, source_name, "full_heavy")
        roi_video = _load_existing_result_video(results_root, source_name, "roi_heavy")

        full_metrics = _compare_video_to_gt(
            full_video,
            gt_path,
            resize_gt_to_result=args.resize_gt_to_result,
        )
        roi_metrics = _compare_video_to_gt(
            roi_video,
            gt_path,
            resize_gt_to_result=args.resize_gt_to_result,
        )

        item = {
            "source": str(source_path),
            "gt": str(gt_path),
            "full_heavy": full_metrics,
            "roi_heavy": roi_metrics,
        }
        summary.append(item)

        print(
            f"[{source_name}] full_vs_gt PSNR={full_metrics['psnr']:.4f}, SSIM={full_metrics['ssim']:.4f}; "
            f"roi_vs_gt PSNR={roi_metrics['psnr']:.4f}, SSIM={roi_metrics['ssim']:.4f}"
        )

    ensure_dir(output_json.parent)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary saved to: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
