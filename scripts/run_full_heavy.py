"""Convenience wrapper for the full-frame BasicVSR++ experiment."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import main as project_main


if __name__ == "__main__":
    raise SystemExit(project_main(["--mode", "full_heavy", "--config", "configs/default.yaml"]))
