#!/usr/bin/env python3
"""
Script to compare performance of full_heavy, roi_heavy, and uncertainty_hybrid modes.

This script:
1. Runs all three modes on the same video
2. Generates comparison charts
3. Outputs detailed benchmark report
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_metrics(summary_dir: Path) -> Optional[Dict[str, Any]]:
    """Load metrics.json from summary directory."""
    metrics_file = summary_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    
    with open(metrics_file, "r") as f:
        return json.load(f)


def print_comparison_report(metrics: Dict[str, Any]) -> None:
    """Print a formatted comparison report."""
    
    print("\n" + "=" * 80)
    print("  PERFORMANCE COMPARISON REPORT")
    print("=" * 80 + "\n")
    
    print(f"Source: {metrics.get('source', 'N/A')}")
    print(f"Video: {metrics.get('source_name', 'N/A')}\n")
    
    # Time comparison
    print("-" * 80)
    print("⏱️  TIME PERFORMANCE")
    print("-" * 80)
    
    print(f"{'Mode':<20} {'Total(s)':<12} {'Per Frame(s)':<15} {'FPS':<10}")
    print("-" * 80)
    
    modes_data = {}
    for mode in ["full_heavy", "roi_heavy", "uncertainty_hybrid"]:
        if mode in metrics:
            m = metrics[mode]
            total_sec = m.get("total_seconds", 0)
            per_frame = m.get("avg_frame_seconds", 0)
            fps = m.get("fps", 0)
            
            modes_data[mode] = {
                "total": total_sec,
                "per_frame": per_frame,
                "fps": fps,
            }
            
            print(f"{mode:<20} {total_sec:<12.2f} {per_frame:<15.4f} {fps:<10.2f}")
    
    # Speedup comparison
    print("\n📊 SPEEDUP RATIOS (vs full_heavy)\n")
    if "full_heavy" in modes_data:
        full_time = modes_data["full_heavy"]["total"]
        
        for mode in ["roi_heavy", "uncertainty_hybrid"]:
            if mode in modes_data:
                mode_time = modes_data[mode]["total"]
                if full_time > 0:
                    speedup = full_time / mode_time
                    savings = (1 - mode_time / full_time) * 100
                    print(f"{mode:<25} {speedup:.2f}x faster, {savings:.1f}% time saved")
    
    # GPU memory comparison
    print("\n" + "-" * 80)
    print("💾 GPU MEMORY")
    print("-" * 80)
    
    print(f"{'Mode':<20} {'Peak Memory(MiB)':<20}")
    print("-" * 80)
    
    peak_mems = {}
    for mode in ["full_heavy", "roi_heavy", "uncertainty_hybrid"]:
        if mode in metrics:
            m = metrics[mode]
            peak_mem = m.get("peak_gpu_memory_mib", None)
            if peak_mem:
                peak_mems[mode] = peak_mem
                print(f"{mode:<20} {peak_mem:<20.1f}")
            else:
                print(f"{mode:<20} {'N/A':<20}")
    
    # Memory savings
    if "full_heavy" in peak_mems:
        print(f"\n📊 MEMORY REDUCTION (vs full_heavy)\n")
        full_mem = peak_mems["full_heavy"]
        
        for mode in ["roi_heavy", "uncertainty_hybrid"]:
            if mode in peak_mems:
                mode_mem = peak_mems[mode]
                reduction = (1 - mode_mem / full_mem) * 100
                saved_mb = full_mem - mode_mem
                print(f"{mode:<25} {reduction:.1f}% reduction, {saved_mb:.1f} MiB saved")
    
    # Uncertainty-specific metrics
    if "uncertainty_hybrid" in metrics:
        print("\n" + "-" * 80)
        print("🔍 UNCERTAINTY HYBRID SPECIFIC")
        print("-" * 80)
        
        uhybrid = metrics["uncertainty_hybrid"]
        print(f"Average Uncertainty: {uhybrid.get('avg_uncertainty', 'N/A'):.4f}")
        print(f"High Uncertainty Ratio (U>0.5): {uhybrid.get('high_uncertainty_ratio', 'N/A'):.4f}")
        
        if "time_breakdown" in uhybrid:
            print("\nTIME BREAKDOWN:\n")
            breakdown = uhybrid["time_breakdown"]
            total_uh = uhybrid.get("total_seconds", 1)
            
            for stage, time_val in breakdown.items():
                if time_val:
                    percentage = (time_val / total_uh) * 100
                    print(f"  {stage:<25} {time_val:>8.3f}s ({percentage:>5.1f}%)")
    
    # ROI-specific metrics
    if "roi_heavy" in metrics:
        roi = metrics["roi_heavy"]
        if "average_roi_area_ratio" in roi and roi["average_roi_area_ratio"] is not None:
            print("\n" + "-" * 80)
            print("📦 ROI HEAVY SPECIFIC")
            print("-" * 80)
            ratio = roi["average_roi_area_ratio"]
            print(f"Average ROI Area Ratio: {ratio:.4f} ({ratio*100:.2f}% of frame)")
    
    print("\n" + "=" * 80 + "\n")


def generate_charts(metrics: Dict[str, Any], output_dir: Path) -> None:
    """Generate comparison charts using matplotlib."""
    
    if not HAS_MATPLOTLIB:
        print("⚠️  matplotlib not installed, skipping chart generation")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Chart 1: FPS Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Performance Comparison: {metrics.get('source_name', 'Video')}", fontsize=16)
    
    modes = []
    fps_values = []
    time_values = []
    memory_values = []
    
    for mode in ["full_heavy", "roi_heavy", "uncertainty_hybrid"]:
        if mode in metrics:
            m = metrics[mode]
            modes.append(mode.replace("_", "\n"))
            fps_values.append(m.get("fps", 0))
            time_values.append(m.get("total_seconds", 0))
            mem = m.get("peak_gpu_memory_mib", 0)
            memory_values.append(mem if mem else 0)
    
    # FPS chart
    ax = axes[0, 0]
    ax.bar(modes, fps_values, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
    ax.set_ylabel("FPS")
    ax.set_title("Frames Per Second (higher is better)")
    ax.grid(axis="y", alpha=0.3)
    
    # Time chart
    ax = axes[0, 1]
    ax.bar(modes, time_values, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Total Processing Time (lower is better)")
    ax.grid(axis="y", alpha=0.3)
    
    # Memory chart
    ax = axes[1, 0]
    if any(memory_values):
        ax.bar(modes, memory_values, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
        ax.set_ylabel("GPU Memory (MiB)")
        ax.set_title("Peak GPU Memory Usage (lower is better)")
        ax.grid(axis="y", alpha=0.3)
    
    # Time breakdown (if uncertainty_hybrid exists)
    ax = axes[1, 1]
    if "uncertainty_hybrid" in metrics and "time_breakdown" in metrics["uncertainty_hybrid"]:
        breakdown = metrics["uncertainty_hybrid"]["time_breakdown"]
        stages = list(breakdown.keys())
        times = list(breakdown.values())
        
        ax.barh(stages, times, color="#45B7D1")
        ax.set_xlabel("Time (seconds)")
        ax.set_title("uncertainty_hybrid Time Breakdown")
        ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    chart_path = output_dir / "performance_comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved: {chart_path}")
    plt.close()


def main():
    """Main entry point."""
    
    from main import main as run_main
    
    # Run all three modes
    modes = ["full_heavy", "roi_heavy", "uncertainty_hybrid"]
    
    print("\n🚀 Running Performance Comparison Test\n")
    print("This will run three modes on the same video:")
    print("  1. full_heavy")
    print("  2. roi_heavy")
    print("  3. uncertainty_hybrid\n")
    
    input("Press Enter to start...")
    
    # Run each mode
    for mode in modes:
        print(f"\n{'='*80}")
        print(f"  Running mode: {mode}")
        print(f"{'='*80}\n")
        
        argv = [
            "--config", "configs/default.yaml",
            "--mode", mode,
        ]
        
        result = run_main(argv)
        if result != 0:
            print(f"❌ Mode {mode} failed")
            return 1
    
    # Load and display metrics
    print(f"\n{'='*80}")
    print("  COMPARISON RESULTS")
    print(f"{'='*80}\n")
    
    # Find the first video's summary directory
    output_dir = Path("results/output")
    if output_dir.exists():
        for video_dir in output_dir.iterdir():
            if video_dir.is_dir():
                summary_dir = video_dir / "summary"
                if summary_dir.exists():
                    metrics = load_metrics(summary_dir)
                    if metrics:
                        print_comparison_report(metrics)
                        generate_charts(metrics, summary_dir)
                    break
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
