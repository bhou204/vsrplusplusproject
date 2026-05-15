#!/usr/bin/env python3
"""Compare full_heavy and uncertainty_hybrid results with GT frames using PSNR, SSIM, and LPIPS."""

import sys
from pathlib import Path
import argparse
import json
from typing import Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.benchmark import compare_frame_sequences
from src.io_utils import read_frames


def compare_results_with_gt(
    results_root: Path,
    gt_root: Path,
    source_names: list[str],
) -> dict[str, Any]:
    """Compare results with GT for given source names."""
    
    summary = {}
    
    for source_name in source_names:
        print(f"Processing {source_name}...")
        
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
            
        gt_frames = read_frames(gt_frames_dir)
        source_summary["gt_frames_dir"] = str(gt_frames_dir)
        source_summary["gt_frame_count"] = len(gt_frames)
        
        # Compare full_heavy
        full_heavy_dir = results_root / source_name / "full_heavy"
        if full_heavy_dir.is_dir():
            full_heavy_frames = read_frames(full_heavy_dir)
            if len(full_heavy_frames) == len(gt_frames):
                comparison = compare_frame_sequences(gt_frames, full_heavy_frames)
                source_summary["full_heavy_vs_gt"] = {
                    "psnr": comparison.psnr,
                    "ssim": comparison.ssim,
                    "lpips": None,  # Skip LPIPS for now
                    "frame_count": comparison.frame_count,
                }
                print(f"  full_heavy: PSNR={comparison.psnr:.4f}, SSIM={comparison.ssim:.4f}")
            else:
                print(f"  full_heavy frame count mismatch: {len(full_heavy_frames)} vs {len(gt_frames)}")
        
        # Compare uncertainty_hybrid
        uncertainty_dir = results_root / source_name / "uncertainty_hybrid"
        if uncertainty_dir.is_dir():
            uncertainty_frames = read_frames(uncertainty_dir)
            if len(uncertainty_frames) == len(gt_frames):
                comparison = compare_frame_sequences(gt_frames, uncertainty_frames)
                source_summary["uncertainty_hybrid_vs_gt"] = {
                    "psnr": comparison.psnr,
                    "ssim": comparison.ssim,
                    "lpips": None,  # Skip LPIPS for now
                    "frame_count": comparison.frame_count,
                }
                print(f"  uncertainty_hybrid: PSNR={comparison.psnr:.4f}, SSIM={comparison.ssim:.4f}")
            else:
                print(f"  uncertainty_hybrid frame count mismatch: {len(uncertainty_frames)} vs {len(gt_frames)}")
        
        if source_summary:
            summary[source_name] = source_summary
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare VSR results with GT frames")
    parser.add_argument("--results-root", default="results/output", help="Results root directory")
    parser.add_argument("--gt-root", default="data", help="GT frames root directory")
    parser.add_argument("--source-names", nargs="*", default=["000", "011", "015", "020"], 
                       help="Source names to compare")
    parser.add_argument("--output-json", default="results/comparison_with_gt.json", 
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    results_root = Path(args.results_root)
    gt_root = Path(args.gt_root)
    
    if not results_root.is_dir():
        raise ValueError(f"Results root not found: {results_root}")
    if not gt_root.is_dir():
        raise ValueError(f"GT root not found: {gt_root}")
    
    summary = compare_results_with_gt(results_root, gt_root, args.source_names)
    
    # Save to JSON
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nComparison saved to {output_path}")
    
    # Print overall summary
    print("\n=== Overall Summary ===")
    for source_name, data in summary.items():
        print(f"\n{source_name}:")
        if "full_heavy_vs_gt" in data:
            fh = data["full_heavy_vs_gt"]
            print(f"  full_heavy:     PSNR={fh['psnr']:.4f}, SSIM={fh['ssim']:.4f}")
        if "uncertainty_hybrid_vs_gt" in data:
            uh = data["uncertainty_hybrid_vs_gt"]
            print(f"  uncertainty:    PSNR={uh['psnr']:.4f}, SSIM={uh['ssim']:.4f}")
        
        # Compare the two methods
        if "full_heavy_vs_gt" in data and "uncertainty_hybrid_vs_gt" in data:
            fh_psnr = data["full_heavy_vs_gt"]["psnr"]
            uh_psnr = data["uncertainty_hybrid_vs_gt"]["psnr"]
            fh_lpips = data["full_heavy_vs_gt"]["lpips"]
            uh_lpips = data["uncertainty_hybrid_vs_gt"]["lpips"]
            
            psnr_diff = uh_psnr - fh_psnr
            lpips_diff = uh_lpips - fh_lpips
            
            print(f"  uncertainty vs full_heavy: PSNR {'+' if psnr_diff >= 0 else ''}{psnr_diff:.4f}")


if __name__ == "__main__":
    main()