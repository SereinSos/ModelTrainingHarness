from __future__ import annotations

from harness.factories import (
    create_vector_store,
    create_hpc_backend,
    create_embedding_model,
    create_dataset_provider,
    create_experiment_tracker,
    list_registries,
)
from harness.config import (
    VectorStoreConfig,
    HpcConfig,
    EmbeddingConfig,
    EvaluationConfig,
    TrackingConfig,
)
from harness.interfaces import VectorStore, HPCBackend, EmbeddingModel, DatasetProvider, ExperimentTracker


def test_registries_populated() -> None:
    regs = list_registries()
    assert "memory" in regs["vector_store"]
    assert "local" in regs["hpc_backend"]
    assert "stub" in regs["embedding"]
    assert "stub" in regs["dataset_provider"]
    assert "none" in regs["tracker"]


def test_create_vector_store_memory() -> None:
    cfg = VectorStoreConfig(type="memory")
    store = create_vector_store(cfg)
    assert isinstance(store, VectorStore)
    store.connect()
    store.create_collection("test", 128)
    store.store([[1.0, 0.0]], [{"id": 1}])
    assert store.count() == 1
    results = store.search([1.0, 0.0], top_k=1)
    assert len(results) == 1
    store.close()


def test_create_hpc_backend_local() -> None:
    cfg = HpcConfig(backend="local")
    hpc = create_hpc_backend(cfg)
    assert isinstance(hpc, HPCBackend)
    assert hpc.check_connectivity() is True


def test_create_embedding_stub() -> None:
    cfg = EmbeddingConfig(type="stub")
    model = create_embedding_model(cfg)
    assert isinstance(model, EmbeddingModel)
    vecs = model.encode(["hello", "world"])
    assert len(vecs) == 2
    assert len(vecs[0]) == model.dimension


def test_create_dataset_provider_stub() -> None:
    cfg = EvaluationConfig(provider="stub")
    provider = create_dataset_provider(cfg)
    assert isinstance(provider, DatasetProvider)
    available = provider.list_available()
    assert len(available) > 0
    ds = provider.load(available[0])
    assert len(ds.texts) > 0


def test_create_tracker_stub() -> None:
    cfg = TrackingConfig(backend="none")
    tracker = create_experiment_tracker(cfg)
    assert isinstance(tracker, ExperimentTracker)
    run_id = tracker.start_run("test")
    assert isinstance(run_id, str)
    tracker.end_run()


def test_create_unknown_type_raises() -> None:
    cfg = VectorStoreConfig(type="nonexistent_db")
    try:
        create_vector_store(cfg)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent_db" in str(e)


def test_create_embedding_sentence_transformer() -> None:
    cfg = EmbeddingConfig(
        type="sentence_transformer",
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    model = create_embedding_model(cfg)
    assert isinstance(model, EmbeddingModel)
    assert model.dimension == 384
    assert model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    vecs = model.encode(["hello", "world"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 384
    norm = sum(x * x for x in vecs[0]) ** 0.5
    assert abs(norm - 1.0) < 0.01
