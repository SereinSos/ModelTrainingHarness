from __future__ import annotations

from typing import Any

from harness.config import (
    EmbeddingConfig,
    EvaluationConfig,
    HpcConfig,
    TrackingConfig,
    VectorStoreConfig,
)
from harness.interfaces import (
    DatasetProvider,
    EmbeddingModel,
    ExperimentTracker,
    HPCBackend,
    VectorStore,
)

_VECTOR_STORE_REGISTRY: dict[str, type] = {}
_HPC_BACKEND_REGISTRY: dict[str, type] = {}
_EMBEDDING_REGISTRY: dict[str, type] = {}
_DATASET_REGISTRY: dict[str, type] = {}
_TRACKING_REGISTRY: dict[str, type] = {}


def register_vector_store(name: str):
    def decorator(cls: type) -> type:
        _VECTOR_STORE_REGISTRY[name] = cls
        return cls
    return decorator


def register_hpc_backend(name: str):
    def decorator(cls: type) -> type:
        _HPC_BACKEND_REGISTRY[name] = cls
        return cls
    return decorator


def register_embedding(name: str):
    def decorator(cls: type) -> type:
        _EMBEDDING_REGISTRY[name] = cls
        return cls
    return decorator


def register_dataset_provider(name: str):
    def decorator(cls: type) -> type:
        _DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def register_tracker(name: str):
    def decorator(cls: type) -> type:
        _TRACKING_REGISTRY[name] = cls
        return cls
    return decorator


def create_vector_store(cfg: VectorStoreConfig) -> VectorStore:
    _ensure_loaded()
    cls = _VECTOR_STORE_REGISTRY.get(cfg.type)
    if cls is None:
        raise ValueError(
            f"Unknown vector_store type: {cfg.type!r}. "
            f"Available: {list(_VECTOR_STORE_REGISTRY)}"
        )
    return cls(cfg)


def create_hpc_backend(cfg: HpcConfig) -> HPCBackend:
    _ensure_loaded()
    cls = _HPC_BACKEND_REGISTRY.get(cfg.backend)
    if cls is None:
        raise ValueError(
            f"Unknown hpc backend: {cfg.backend!r}. "
            f"Available: {list(_HPC_BACKEND_REGISTRY)}"
        )
    return cls(cfg)


def create_embedding_model(cfg: EmbeddingConfig) -> EmbeddingModel:
    _ensure_loaded()
    cls = _EMBEDDING_REGISTRY.get(cfg.type)
    if cls is None:
        raise ValueError(
            f"Unknown embedding type: {cfg.type!r}. "
            f"Available: {list(_EMBEDDING_REGISTRY)}"
        )
    return cls(cfg)


def create_dataset_provider(cfg: EvaluationConfig) -> DatasetProvider:
    _ensure_loaded()
    cls = _DATASET_REGISTRY.get(cfg.provider)
    if cls is None:
        raise ValueError(
            f"Unknown evaluation provider: {cfg.provider!r}. "
            f"Available: {list(_DATASET_REGISTRY)}"
        )
    return cls(cfg)


def create_experiment_tracker(cfg: TrackingConfig) -> ExperimentTracker:
    _ensure_loaded()
    cls = _TRACKING_REGISTRY.get(cfg.backend)
    if cls is None:
        raise ValueError(
            f"Unknown tracking backend: {cfg.backend!r}. "
            f"Available: {list(_TRACKING_REGISTRY)}"
        )
    return cls(cfg)


_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True

    import harness.vector.memory  # noqa: F401
    import harness.hpc.local  # noqa: F401
    import harness.embedding.stub  # noqa: F401
    import harness.eval.stub_provider  # noqa: F401
    import harness.tracking.stub_tracker  # noqa: F401

    try:
        import harness.embedding.st_model  # noqa: F401
    except ImportError:
        pass

    try:
        import harness.eval.mteb_runner  # noqa: F401
    except ImportError:
        pass

    try:
        import harness.vector.milvus  # noqa: F401
    except ImportError:
        pass

    try:
        import harness.vector.qdrant  # noqa: F401
    except ImportError:
        pass

    try:
        import harness.vector.chromadb_backend  # noqa: F401
    except ImportError:
        pass

    try:
        import harness.hpc.slurm  # noqa: F401
    except ImportError:
        pass

    try:
        import harness.tracking.mlflow_tracker  # noqa: F401
    except ImportError:
        pass


def list_registries() -> dict[str, list[str]]:
    _ensure_loaded()
    return {
        "vector_store": list(_VECTOR_STORE_REGISTRY),
        "hpc_backend": list(_HPC_BACKEND_REGISTRY),
        "embedding": list(_EMBEDDING_REGISTRY),
        "dataset_provider": list(_DATASET_REGISTRY),
        "tracker": list(_TRACKING_REGISTRY),
    }
