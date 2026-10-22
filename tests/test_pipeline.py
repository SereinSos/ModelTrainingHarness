from __future__ import annotations

from pathlib import Path

import pytest

from harness.config import load_config
from harness.pipeline import Pipeline


@pytest.fixture
def cfg() -> HarnessConfig:
    return load_config(Path("config/harness.yaml"))


def test_pipeline_run_full(cfg) -> None:
    pipeline = Pipeline(cfg)
    report_paths = pipeline.run_full()
    assert len(report_paths) == 2
    for p in report_paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_pipeline_run_baseline(cfg) -> None:
    pipeline = Pipeline(cfg)
    report_paths = pipeline.run_baseline()
    assert len(report_paths) == 2
    for p in report_paths:
        assert p.exists()


def test_pipeline_report_json_valid(cfg) -> None:
    import json
    pipeline = Pipeline(cfg)
    report_paths = pipeline.run_full()
    json_path = next(p for p in report_paths if p.suffix == ".json")
    data = json.loads(json_path.read_text())
    assert "run_id" in data
    assert "model_scores" in data
    assert "storage_scores" in data
    assert isinstance(data["model_scores"], list)
    assert len(data["model_scores"]) > 0
