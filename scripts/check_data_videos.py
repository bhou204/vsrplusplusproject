"""Inspect videos in data/ and verify whether source videos match paired GT videos."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from src.io_utils import list_video_paths


@dataclass
class VideoInfo:
    path: str
    stem: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


@dataclass
class PairCheckResult:
    source: VideoInfo
    gt: VideoInfo | None
    pair_found: bool
    source_is_gt_name: bool
    gt_name_match: bool
    width_ratio_gt_over_src: float | None
    height_ratio_gt_over_src: float | None
    fps_ratio_gt_over_src: float | None
    frame_count_difference: int | None
    looks_like_x2_scale: bool
    same_aspect_ratio: bool


def _resolve(path_text: str, base_dir: Path = ROOT) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _is_gt_video(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.endswith("gt") or stem.endswith("_gt")


def _find_gt_video(source_path: Path, all_video_paths: list[Path]) -> Path | None:
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
    return None


def _read_video_info(video_path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    duration_sec = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(
        path=str(video_path),
        stem=video_path.stem,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _same_aspect_ratio(source: VideoInfo, gt: VideoInfo, tolerance: float = 0.02) -> bool:
    if source.height == 0 or gt.height == 0:
        return False
    source_ratio = source.width / source.height
    gt_ratio = gt.width / gt.height
    return abs(source_ratio - gt_ratio) <= tolerance


def _looks_like_x2_scale(source: VideoInfo, gt: VideoInfo, tolerance: float = 0.08) -> bool:
    if source.width == 0 or source.height == 0:
        return False
    width_ratio = gt.width / source.width
    height_ratio = gt.height / source.height
    return abs(width_ratio - 2.0) <= tolerance and abs(height_ratio - 2.0) <= tolerance


def _check_pair(source_path: Path, all_video_paths: list[Path]) -> PairCheckResult:
    gt_path = _find_gt_video(source_path, all_video_paths)
    source_info = _read_video_info(source_path)
    gt_info = _read_video_info(gt_path) if gt_path is not None else None

    if gt_info is None:
        return PairCheckResult(
            source=source_info,
            gt=None,
            pair_found=False,
            source_is_gt_name=_is_gt_video(source_path),
            gt_name_match=False,
            width_ratio_gt_over_src=None,
            height_ratio_gt_over_src=None,
            fps_ratio_gt_over_src=None,
            frame_count_difference=None,
            looks_like_x2_scale=False,
            same_aspect_ratio=False,
        )

    width_ratio = _ratio(gt_info.width, source_info.width)
    height_ratio = _ratio(gt_info.height, source_info.height)
    frame_count_difference = gt_info.frame_count - source_info.frame_count

    return PairCheckResult(
        source=source_info,
        gt=gt_info,
        pair_found=True,
        source_is_gt_name=_is_gt_video(source_path),
        gt_name_match=gt_path.stem.lower() == f"{source_path.stem.lower()}gt" or gt_path.stem.lower() == f"{source_path.stem.lower()}_gt",
        width_ratio_gt_over_src=width_ratio,
        height_ratio_gt_over_src=height_ratio,
        fps_ratio_gt_over_src=(gt_info.fps / source_info.fps) if source_info.fps > 0 else None,
        frame_count_difference=frame_count_difference,
        looks_like_x2_scale=_looks_like_x2_scale(source_info, gt_info),
        same_aspect_ratio=_same_aspect_ratio(source_info, gt_info),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check source videos and paired GT videos in data/")
    parser.add_argument("--data-dir", default="data", help="Directory containing source and GT videos")
    parser.add_argument("--output-json", default="results/summary/video_check.json", help="Where to save the JSON summary")
    return parser.parse_args()


def _print_result(result: PairCheckResult) -> None:
    source = result.source
    if result.gt is None:
        print(f"[{source.stem}] GT not found")
        return

    gt = result.gt
    print(
        f"[{source.stem}] source={source.width}x{source.height} fps={source.fps:.3f} frames={source.frame_count} duration={source.duration_sec:.2f}s | "
        f"gt={gt.width}x{gt.height} fps={gt.fps:.3f} frames={gt.frame_count} duration={gt.duration_sec:.2f}s | "
        f"gt/source ratio={result.width_ratio_gt_over_src:.3f}x{result.height_ratio_gt_over_src:.3f}x | "
        f"same_aspect={result.same_aspect_ratio} x2_scale={result.looks_like_x2_scale}"
    )


def main() -> int:
    args = parse_args()
    data_dir = _resolve(args.data_dir)
    output_json = _resolve(args.output_json)

    all_video_paths = list_video_paths(data_dir)
    source_videos = [path for path in all_video_paths if not _is_gt_video(path)]

    results: list[dict[str, Any]] = []
    for source_path in source_videos:
        check_result = _check_pair(source_path, all_video_paths)
        _print_result(check_result)
        results.append(asdict(check_result))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary saved to: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
