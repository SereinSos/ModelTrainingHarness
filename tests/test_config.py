from __future__ import annotations

import pytest
from pathlib import Path

from harness.config import load_config, HarnessConfig


@pytest.fixture
def config_path() -> Path:
    return Path("config/harness.yaml")


def test_load_config_default(config_path: Path) -> None:
    cfg = load_config(config_path)
    assert isinstance(cfg, HarnessConfig)
    assert cfg.vector_store.type == "chromadb"
    assert cfg.hpc.backend == "local"
    assert cfg.embedding.type == "sentence_transformer"
    assert cfg.evaluation.provider == "mteb"
    assert cfg.tracking.backend == "none"
    assert "text" in cfg.report.formats
    assert "json" in cfg.report.formats


def test_load_config_path_coercion(config_path: Path) -> None:
    cfg = load_config(config_path)
    assert isinstance(cfg.container.image_path, Path)
    assert isinstance(cfg.hpc.remote_dir, Path)
    assert isinstance(cfg.report.output_dir, Path)


def test_load_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))


def test_load_config_collection_fields(config_path: Path) -> None:
    cfg = load_config(config_path)
    assert cfg.vector_store.collection_name == "harness_vectors"
    assert cfg.vector_store.collection_dimension == 384
    assert cfg.vector_store.collection_metric == "cosine"


def test_load_config_evaluation_tasks(config_path: Path) -> None:
    cfg = load_config(config_path)
    assert len(cfg.evaluation.tasks) == 1
    assert cfg.evaluation.tasks[0].name == "BIOSSES"
    assert "en" in cfg.evaluation.tasks[0].languages


def test_load_config_hpc_resources(config_path: Path) -> None:
    cfg = load_config(config_path)
    assert cfg.hpc.resources.nodes == 1
    assert cfg.hpc.resources.gpus == 0
    assert cfg.hpc.resources.memory == "16G"
