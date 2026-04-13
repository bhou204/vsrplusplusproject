"""Convenience wrapper for running both experiments and printing a comparison."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import main as project_main


if __name__ == "__main__":
    raise SystemExit(project_main(["--mode", "compare", "--config", "configs/default.yaml"]))
