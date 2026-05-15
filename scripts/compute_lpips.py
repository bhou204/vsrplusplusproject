#!/usr/bin/env python3
"""独立的LPIPS计算脚本，用于修复LPIPS计算问题。"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
import cv2
from typing import Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.io_utils import read_frames


def compute_lpips_single(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute LPIPS distance between two frames with proper error handling."""
    try:
        import torch
        import lpips
    except ImportError as e:
        print(f"Import error: {e}")
        return float("nan")

    try:
        # Convert BGR to RGB
        frame_a_rgb = cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB)
        frame_b_rgb = cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB)

        # Convert to float32 and normalize to [-1, 1]
        frame_a_tensor = torch.from_numpy(frame_a_rgb).float().permute(2, 0, 1) / 127.5 - 1.0
        frame_b_tensor = torch.from_numpy(frame_b_rgb).float().permute(2, 0, 1) / 127.5 - 1.0

        # Add batch dimension
        frame_a_tensor = frame_a_tensor.unsqueeze(0)
        frame_b_tensor = frame_b_tensor.unsqueeze(0)

        # Initialize LPIPS model (AlexNet backbone)
        loss_fn = lpips.LPIPS(net='alex', verbose=False)

        # Compute LPIPS
        with torch.no_grad():
            lpips_value = loss_fn(frame_a_tensor, frame_b_tensor).item()

        return lpips_value

    except Exception as e:
        print(f"LPIPS computation error: {e}")
        return float("nan")


def compute_lpips_batch(frames_a: list[np.ndarray], frames_b: list[np.ndarray]) -> list[float]:
    """Compute LPIPS for a batch of frame pairs."""
    if len(frames_a) != len(frames_b):
        raise ValueError(f"Frame count mismatch: {len(frames_a)} vs {len(frames_b)}")

    lpips_scores = []
    for i, (frame_a, frame_b) in enumerate(zip(frames_a, frames_b)):
        if frame_a.shape != frame_b.shape:
            print(f"Warning: Frame {i} shape mismatch: {frame_a.shape} vs {frame_b.shape}")
            lpips_scores.append(float("nan"))
            continue

        lpips_val = compute_lpips_single(frame_a, frame_b)
        lpips_scores.append(lpips_val)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(frames_a)} frames...")

    return lpips_scores


def compare_with_lpips(
    results_root: Path,
    gt_root: Path,
    source_names: list[str],
) -> dict[str, Any]:
    """Compare results with GT using LPIPS."""
    import cv2  # Import here to avoid import issues

    summary = {}

    for source_name in source_names:
        print(f"\nProcessing {source_name}...")
        source_summary = {}

        # Find GT frames
        gt_candidates = [
            gt_root / f"{source_name}gt",
            gt_root / f"{source_name}_gt",
        ]
        gt_frames_dir = None
        for candidate in gt_candidates:
            if candidate.is_dir():
                gt_frames_dir = candidate
                break

        if not gt_frames_dir:
            print(f"  GT not found for {source_name}, skipping")
            continue

        try:
            gt_frames = read_frames(gt_frames_dir)
            source_summary["gt_frames_dir"] = str(gt_frames_dir)
            source_summary["gt_frame_count"] = len(gt_frames)

            # Compare full_heavy
            full_heavy_dir = results_root / source_name / "full_heavy"
            if full_heavy_dir.is_dir():
                full_heavy_frames = read_frames(full_heavy_dir)
                if len(full_heavy_frames) == len(gt_frames):
                    print("  Computing LPIPS for full_heavy...")
                    lpips_scores = compute_lpips_batch(gt_frames, full_heavy_frames)
                    valid_scores = [s for s in lpips_scores if not np.isnan(s)]
                    avg_lpips = float(np.mean(valid_scores)) if valid_scores else float("nan")

                    source_summary["full_heavy_vs_gt"] = {
                        "lpips": avg_lpips,
                        "frame_count": len(lpips_scores),
                        "valid_frames": len(valid_scores),
                    }
                    print(f"  full_heavy LPIPS: {avg_lpips:.4f} (valid: {len(valid_scores)}/{len(lpips_scores)})")
                else:
                    print(f"  full_heavy frame count mismatch: {len(full_heavy_frames)} vs {len(gt_frames)}")

            # Compare uncertainty_hybrid
            uncertainty_dir = results_root / source_name / "uncertainty_hybrid"
            if uncertainty_dir.is_dir():
                uncertainty_frames = read_frames(uncertainty_dir)
                if len(uncertainty_frames) == len(gt_frames):
                    print("  Computing LPIPS for uncertainty_hybrid...")
                    lpips_scores = compute_lpips_batch(gt_frames, uncertainty_frames)
                    valid_scores = [s for s in lpips_scores if not np.isnan(s)]
                    avg_lpips = float(np.mean(valid_scores)) if valid_scores else float("nan")

                    source_summary["uncertainty_hybrid_vs_gt"] = {
                        "lpips": avg_lpips,
                        "frame_count": len(lpips_scores),
                        "valid_frames": len(valid_scores),
                    }
                    print(f"  uncertainty_hybrid LPIPS: {avg_lpips:.4f} (valid: {len(valid_scores)}/{len(lpips_scores)})")
                else:
                    print(f"  uncertainty_hybrid frame count mismatch: {len(uncertainty_frames)} vs {len(gt_frames)}")

        except Exception as e:
            print(f"  Error processing {source_name}: {e}")
            continue

        if source_summary:
            summary[source_name] = source_summary

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compute LPIPS between VSR results and GT frames")
    parser.add_argument("--results-root", default="results/output", help="Results root directory")
    parser.add_argument("--gt-root", default="data", help="GT frames root directory")
    parser.add_argument("--source-names", nargs="*", default=["000", "011", "015", "020"],
                       help="Source names to compare")
    parser.add_argument("--output-json", default="results/lpips_comparison.json",
                       help="Output JSON file")

    args = parser.parse_args()

    results_root = Path(args.results_root)
    gt_root = Path(args.gt_root)

    if not results_root.is_dir():
        raise ValueError(f"Results root not found: {results_root}")
    if not gt_root.is_dir():
        raise ValueError(f"GT root not found: {gt_root}")

    print("Starting LPIPS computation...")
    summary = compare_with_lpips(results_root, gt_root, args.source_names)

    # Save to JSON
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nLPIPS comparison saved to {output_path}")

    # Print summary
    print("\n=== LPIPS Results Summary ===")
    for source_name, data in summary.items():
        print(f"\n{source_name}:")
        if "full_heavy_vs_gt" in data:
            fh = data["full_heavy_vs_gt"]
            print(f"  full_heavy:     LPIPS={fh['lpips']:.4f}")
        if "uncertainty_hybrid_vs_gt" in data:
            uh = data["uncertainty_hybrid_vs_gt"]
            print(f"  uncertainty:    LPIPS={uh['lpips']:.4f}")

        # Compare the two methods
        if "full_heavy_vs_gt" in data and "uncertainty_hybrid_vs_gt" in data:
            fh_lpips = data["full_heavy_vs_gt"]["lpips"]
            uh_lpips = data["uncertainty_hybrid_vs_gt"]["lpips"]

            if not np.isnan(fh_lpips) and not np.isnan(uh_lpips):
                lpips_diff = uh_lpips - fh_lpips
                print(f"  uncertainty vs full_heavy: LPIPS {lpips_diff:+.4f}")


if __name__ == "__main__":
    main()