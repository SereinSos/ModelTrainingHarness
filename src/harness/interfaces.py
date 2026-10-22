from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class Dataset:
    texts: list[str]
    labels: list[str] | None = None
    queries: list[str] | None = None
    corpus: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    model_name: str
    scores: dict[str, float]
    elapsed_seconds: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelScore:
    task: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageScore:
    metric: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HpcResult:
    job_id: str
    status: str
    output_path: Path


@dataclass
class Report:
    model_scores: list[ModelScore]
    storage_scores: list[StorageScore]
    run_id: str
    timestamp: str


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    fatal: bool = False


@runtime_checkable
class VectorStore(Protocol):
    def connect(self) -> None: ...
    def create_collection(self, name: str, dimension: int, metric: str = "cosine") -> None: ...
    def store(self, vectors: list[list[float]], metadata: list[dict]) -> None: ...
    def search(
        self, query: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[dict]: ...
    def delete_collection(self, name: str) -> None: ...
    def count(self) -> int: ...
    def close(self) -> None: ...


@runtime_checkable
class HPCBackend(Protocol):
    def check_connectivity(self) -> bool: ...
    def pack(self, local_dir: Path, output: Path) -> Path: ...
    def upload(self, local_path: Path, remote_path: Path) -> None: ...
    def submit(self, remote_path: Path, resources: dict) -> str: ...
    def status(self, job_id: str) -> str: ...
    def fetch(self, job_id: str, local_dir: Path) -> Path: ...
    def cancel(self, job_id: str) -> None: ...


@runtime_checkable
class EmbeddingModel(Protocol):
    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]: ...
    @property
    def dimension(self) -> int: ...
    @property
    def model_name(self) -> str: ...


@runtime_checkable
class DatasetProvider(Protocol):
    def load(self, task_name: str, split: str = "test") -> Dataset: ...
    def list_available(self) -> list[str]: ...


@runtime_checkable
class ExperimentTracker(Protocol):
    def start_run(self, run_name: str) -> str: ...
    def log_param(self, key: str, value: Any) -> None: ...
    def log_metric(self, key: str, value: float) -> None: ...
    def log_artifact(self, local_path: Path) -> None: ...
    def end_run(self) -> None: ...
