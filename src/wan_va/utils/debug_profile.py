# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import torch

from .logging import logger


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class DebugTrainProfiler:
    """Lightweight wall-clock timing for the last N training steps."""

    _SECTIONS = (
        "train/data_next_batch",
        "train/convert_input_format",
        "train/prepare_input_dict",
        "train/forward",
        "train/compute_loss",
        "train/backward",
        "train/clip_grad_norm",
        "train/optimizer_step",
        "train/step_total",
    )

    def __init__(
        self,
        *,
        enabled: bool,
        total_steps: int = 20,
        profile_steps: int = 10,
        output_dir: str | Path,
        is_main_process: bool = True,
    ):
        self.enabled = enabled and is_main_process
        self.total_steps = total_steps
        self.profile_steps = profile_steps
        self.warmup_steps = total_steps - profile_steps
        self.output_dir = Path(output_dir)
        self._active = False
        self._step_started = False
        self._step_start = 0.0
        self._step_sections: dict[str, float] = {}
        self._profiled_steps: list[dict[str, float]] = []

    @property
    def profile_start_step(self) -> int:
        return self.warmup_steps

    @property
    def profile_end_step(self) -> int:
        return self.total_steps - 1

    def on_step_start(self, step: int) -> None:
        if not self.enabled or step < self.profile_start_step:
            self._active = False
            return

        if step == self.profile_start_step:
            logger.info(
                "Debug timing: warmup finished, measuring steps "
                f"{self.profile_start_step + 1}-{self.total_steps} "
                f"(0-indexed {self.profile_start_step}-{self.profile_end_step})"
            )

        self._active = True
        self._step_started = False
        self._step_sections = {}
        _sync_cuda()
        self._step_start = time.perf_counter()

    def on_step_end(self, step: int) -> None:
        if not self._active:
            return

        _sync_cuda()
        self._step_sections["train/step_total"] = time.perf_counter() - self._step_start
        self._profiled_steps.append(dict(self._step_sections))
        self._active = False

    def finish(self) -> None:
        if not self.enabled or not self._profiled_steps:
            return

        totals = defaultdict(float)
        for step_times in self._profiled_steps:
            for name, value in step_times.items():
                totals[name] += value

        num_steps = len(self._profiled_steps)
        averages = {name: totals[name] / num_steps for name in totals}
        step_total = averages.get("train/step_total", 0.0) or 1e-9

        report_lines = [
            "Debug timing report (rank-0 wall clock, CUDA synchronized)",
            f"profiled_steps={num_steps} "
            f"(step index {self.profile_start_step}-{self.profile_end_step})",
            "",
            f"{'Section':<32} {'avg(s)':>8} {'pct':>7}",
            "-" * 50,
        ]

        for name in self._SECTIONS:
            if name not in averages:
                continue
            avg_s = averages[name]
            pct = 100.0 * avg_s / step_total if name != "train/step_total" else 100.0
            report_lines.append(f"{name:<32} {avg_s:8.3f} {pct:6.1f}%")

        report_lines.extend(
            [
                "",
                f"Average step time: {step_total:.3f}s ({1000.0 * step_total:.1f} ms)",
            ]
        )
        report_text = "\n".join(report_lines)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "profile_report.txt"
        report_path.write_text(report_text + "\n", encoding="utf-8")

        logger.info(f"Debug timing report saved to {report_path}")
        print("\n" + report_text + "\n", flush=True)

    def record(self, name: str):
        if not self._active:
            return _NullRecordContext()
        return _TimingContext(self, name)


class _TimingContext:
    def __init__(self, profiler: DebugTrainProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.start = 0.0

    def __enter__(self):
        _sync_cuda()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        _sync_cuda()
        self.profiler._step_sections[self.name] = (
            self.profiler._step_sections.get(self.name, 0.0)
            + time.perf_counter() - self.start
        )
        return False


class _NullRecordContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
