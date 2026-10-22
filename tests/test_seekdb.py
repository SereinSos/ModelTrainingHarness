from __future__ import annotations

import pytest

from harness.config import VectorStoreConfig


@pytest.fixture
def seekdb_config() -> VectorStoreConfig:
    return VectorStoreConfig(
        type="seekdb",
        connection={"db_path": "./test_seekdb.db", "database": "test"},
        collection_name="test_collection",
        collection_dimension=384,
        collection_metric="cosine",
    )


def test_seekdb_vector_store_init(seekdb_config: VectorStoreConfig) -> None:
    from harness.vector.seekdb import SeekDBVectorStore

    store = SeekDBVectorStore(seekdb_config)
    assert store._collection_name == "test_collection"
    assert store._dimension == 384
    assert store._metric == "cosine"
    assert store._client is None


def test_seekdb_fulltext_search_init(seekdb_config: VectorStoreConfig) -> None:
    from harness.vector.seekdb import SeekDBFullTextSearch

    fts = SeekDBFullTextSearch(seekdb_config)
    assert fts._table_name == "test_collection"
    assert fts._client is None


def test_seekdb_scalar_search_init(seekdb_config: VectorStoreConfig) -> None:
    from harness.vector.seekdb import SeekDBScalarSearch

    scalar = SeekDBScalarSearch(seekdb_config)
    assert scalar._client is None


def test_seekdb_gis_search_init(seekdb_config: VectorStoreConfig) -> None:
    from harness.vector.seekdb import SeekDBGISSearch

    gis = SeekDBGISSearch(seekdb_config)
    assert gis._client is None


def test_seekdb_hybrid_search_init(seekdb_config: VectorStoreConfig) -> None:
    from harness.vector.seekdb import SeekDBHybridSearch

    hybrid = SeekDBHybridSearch(seekdb_config)
    assert hybrid._client is None


def test_seekdb_registered() -> None:
    from harness.factories import list_registries

    regs = list_registries()
    assert "seekdb" in regs["vector_store"]


def test_all_seekdb_classes_importable() -> None:
    from harness.vector.seekdb import (
        SeekDBVectorStore,
        SeekDBFullTextSearch,
        SeekDBScalarSearch,
        SeekDBGISSearch,
        SeekDBHybridSearch,
    )

    assert SeekDBVectorStore is not None
    assert SeekDBFullTextSearch is not None
    assert SeekDBScalarSearch is not None
    assert SeekDBGISSearch is not None
    assert SeekDBHybridSearch is not None
