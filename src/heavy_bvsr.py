"""Subprocess wrapper around the official BasicVSR++ demo script."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional


@dataclass
class BasicVsrRunner:
    """Command-line wrapper for the upstream BasicVSR++ inference demo."""

    basicvsr_root: Path
    basicvsr_config: Path
    basicvsr_checkpoint: Path
    device: int = 0
    python_executable: str = sys.executable

    def __post_init__(self) -> None:
        self.basicvsr_root = Path(self.basicvsr_root)
        self.basicvsr_config = Path(self.basicvsr_config)
        self.basicvsr_checkpoint = Path(self.basicvsr_checkpoint)

    @property
    def demo_script(self) -> Path:
        return self.basicvsr_root / "demo" / "restoration_video_demo.py"

    def validate(self) -> None:
        """Validate the configured paths before running inference."""

        if not self.basicvsr_root.is_dir():
            raise FileNotFoundError(f"BasicVSR++ root not found: {self.basicvsr_root}")
        if not self.demo_script.is_file():
            raise FileNotFoundError(f"Demo script not found: {self.demo_script}")
        if not self.basicvsr_config.is_file():
            raise FileNotFoundError(f"BasicVSR++ config not found: {self.basicvsr_config}")
        if not self.basicvsr_checkpoint.is_file():
            raise FileNotFoundError(f"BasicVSR++ checkpoint not found: {self.basicvsr_checkpoint}")

    def build_command(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        window_size: int = 0,
        start_idx: int = 0,
        filename_tmpl: str = "{:08d}.png",
        max_seq_len: Optional[int] = None,
    ) -> list[str]:
        """Build the subprocess command for the official demo entrypoint."""

        self.validate()
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        input_arg = os.path.relpath(str(input_path), start=str(self.basicvsr_root))
        output_arg = os.path.relpath(str(output_path), start=str(self.basicvsr_root))
        command: list[str] = [
            str(self.python_executable),
            str(self.demo_script),
            str(self.basicvsr_config),
            str(self.basicvsr_checkpoint),
            input_arg,
            output_arg,
            "--window-size",
            str(window_size),
            "--start-idx",
            str(start_idx),
            "--filename-tmpl",
            filename_tmpl,
            "--device",
            str(self.device),
        ]
        if max_seq_len is not None:
            command.extend(["--max-seq-len", str(max_seq_len)])
        return command

    def run(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        window_size: int = 0,
        start_idx: int = 0,
        filename_tmpl: str = "{:08d}.png",
        max_seq_len: Optional[int] = None,
        extra_env: Optional[dict[str, str]] = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run BasicVSR++ inference and return the completed process."""

        command = self.build_command(
            input_dir=input_dir,
            output_dir=output_dir,
            window_size=window_size,
            start_idx=start_idx,
            filename_tmpl=filename_tmpl,
            max_seq_len=max_seq_len,
        )
        environment = None
        if extra_env:
            import os

            environment = os.environ.copy()
            environment.update(extra_env)

        return subprocess.run(
            command,
            cwd=str(self.basicvsr_root),
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
