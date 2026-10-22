from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ContainerConfig:
    runtime: str = "apptainer"
    image_path: Path = Path("/opt/containers/skdb_latest.sif")
    expected_version: str = "1.4.0"


@dataclass
class VectorStoreConfig:
    type: str = "memory"
    connection: dict[str, Any] = field(default_factory=dict)
    collection_name: str = "harness_vectors"
    collection_dimension: int = 1024
    collection_metric: str = "cosine"


@dataclass
class EmbeddingConfig:
    type: str = "sentence_transformer"
    model_name_or_path: str = "BAAI/bge-small-zh-v1.5"
    batch_size: int = 32
    device: str = "cpu"


@dataclass
class EvaluationTaskConfig:
    name: str
    languages: list[str] = field(default_factory=lambda: ["en"])


@dataclass
class EvaluationConfig:
    provider: str = "mteb"
    tasks: list[EvaluationTaskConfig] = field(default_factory=list)
    dataset_cache_dir: Path = Path("./data/mteb_cache")
    output_dir: Path = Path("./results/eval")


@dataclass
class HpcResourcesConfig:
    nodes: int = 1
    ntasks_per_node: int = 4
    gpus: int = 2
    memory: str = "64G"
    time_limit: str = "24:00:00"
    partition: str = "gpu"


@dataclass
class HpcConfig:
    backend: str = "local"
    host: str = "localhost"
    user: str = ""
    ssh_key: Path = Path("")
    remote_dir: Path = Path("/tmp/hpc")
    resources: HpcResourcesConfig = field(default_factory=HpcResourcesConfig)
    container_bind: str = "/data:/mnt"


@dataclass
class TrackingConfig:
    backend: str = "none"
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "model-training-harness"


@dataclass
class ReportConfig:
    output_dir: Path = Path("./reports")
    formats: list[str] = field(default_factory=lambda: ["text", "json"])


@dataclass
class HarnessConfig:
    container: ContainerConfig = field(default_factory=ContainerConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    hpc: HpcConfig = field(default_factory=HpcConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    report: ReportConfig = field(default_factory=ReportConfig)


def _build_dataclass(cls: type, data: dict | None) -> Any:
    if data is None:
        return cls()
    fields = {f.name: f for f in cls.__dataclass_fields__.values()}
    filtered = {}
    for k, v in data.items():
        if k not in fields:
            continue
        f = fields[k]
        if isinstance(v, str) and f.type in (Path, "Path"):
            v = Path(v)
        filtered[k] = v
    return cls(**filtered)


def load_config(path: Path | str | None = None) -> HarnessConfig:
    if path is None:
        import os
        path = os.environ.get("HARNESS_CONFIG", "config/harness.yaml")
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}

    container = _build_dataclass(ContainerConfig, raw.get("container"))

    vs_raw = raw.get("vector_store", {}) or {}
    vs_connection = vs_raw.pop("connection", {}) or {}
    vs_collection = vs_raw.pop("collection", {}) or {}
    vs_kwargs = {**vs_raw}
    if vs_collection:
        vs_kwargs["collection_name"] = vs_collection.get("name", "harness_vectors")
        vs_kwargs["collection_dimension"] = vs_collection.get("dimension", 1024)
        vs_kwargs["collection_metric"] = vs_collection.get("metric", "cosine")
    vs_kwargs["connection"] = vs_connection
    vector_store = _build_dataclass(VectorStoreConfig, vs_kwargs)

    embedding = _build_dataclass(EmbeddingConfig, raw.get("embedding"))

    eval_raw = raw.get("evaluation", {}) or {}
    eval_tasks_raw = eval_raw.pop("tasks", []) or []
    eval_tasks = [
        _build_dataclass(EvaluationTaskConfig, t) if isinstance(t, dict) else EvaluationTaskConfig(name=t)
        for t in eval_tasks_raw
    ]
    eval_kwargs = {**eval_raw, "tasks": eval_tasks}
    evaluation = _build_dataclass(EvaluationConfig, eval_kwargs)

    hpc_raw = raw.get("hpc", {}) or {}
    hpc_resources = _build_dataclass(HpcResourcesConfig, hpc_raw.pop("resources", None))
    hpc_kwargs = {**hpc_raw, "resources": hpc_resources}
    hpc = _build_dataclass(HpcConfig, hpc_kwargs)

    tracking = _build_dataclass(TrackingConfig, raw.get("tracking"))
    report = _build_dataclass(ReportConfig, raw.get("report"))

    return HarnessConfig(
        container=container,
        vector_store=vector_store,
        embedding=embedding,
        evaluation=evaluation,
        hpc=hpc,
        tracking=tracking,
        report=report,
    )
