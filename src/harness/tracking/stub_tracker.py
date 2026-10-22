from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.config import TrackingConfig
from harness.factories import register_tracker


@register_tracker("none")
class StubTracker:
    def __init__(self, cfg: TrackingConfig) -> None:
        self._run_id = "stub-run-000"

    def start_run(self, run_name: str) -> str:
        print(f"[tracker:stub] start_run: {run_name}")
        return self._run_id

    def log_param(self, key: str, value: Any) -> None:
        print(f"[tracker:stub] param: {key}={value}")

    def log_metric(self, key: str, value: float) -> None:
        print(f"[tracker:stub] metric: {key}={value:.4f}")

    def log_artifact(self, local_path: Path) -> None:
        print(f"[tracker:stub] artifact: {local_path}")

    def end_run(self) -> None:
        print(f"[tracker:stub] end_run: {self._run_id}")
