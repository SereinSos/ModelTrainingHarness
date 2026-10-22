from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

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


@pytest.fixture
def mock_pyseekdb_module():
    mock_module = MagicMock()
    mock_client_instance = MagicMock()
    mock_module.Client.return_value = mock_client_instance
    with patch.dict(sys.modules, {"pyseekdb": mock_module}):
        yield mock_module, mock_client_instance


@pytest.fixture
def mock_sql_modules():
    mock_pylibseekdb = MagicMock()
    mock_seekdb_instance = MagicMock()
    mock_seekdb_instance.connection_options.return_value = {
        "host": "127.0.0.1",
        "port": 2881,
        "user": "root",
        "password": "",
    }
    mock_pylibseekdb.open.return_value = mock_seekdb_instance

    mock_pymysql = MagicMock()
    mock_connection = MagicMock(spec=['cursor', 'close', 'commit'])
    mock_cursor = MagicMock()
    
    # Set up context manager properly
    cursor_context = MagicMock()
    cursor_context.__enter__ = MagicMock(return_value=mock_cursor)
    cursor_context.__exit__ = MagicMock(return_value=False)
    mock_connection.cursor.return_value = cursor_context
    
    mock_pymysql.connect.return_value = mock_connection
    mock_pymysql.cursors = MagicMock()

    with patch.dict(sys.modules, {
        "pylibseekdb": mock_pylibseekdb,
        "pymysql": mock_pymysql,
        "pymysql.cursors": mock_pymysql.cursors,
    }):
        yield mock_connection, mock_cursor


def test_seekdb_vector_store_init(seekdb_config: VectorStoreConfig) -> None:
    from harness.vector.seekdb import SeekDBVectorStore

    store = SeekDBVectorStore(seekdb_config)
    assert store._collection_name == "test_collection"
    assert store._dimension == 384
    assert store._metric == "cosine"
    assert store._client is None
    assert store._collection is None


def test_seekdb_vector_store_connect_pyseekdb(seekdb_config: VectorStoreConfig, mock_pyseekdb_module) -> None:
    from harness.vector.seekdb import SeekDBVectorStore

    mock_module, mock_client_instance = mock_pyseekdb_module
    seekdb_config.connection = {"path": "./test.db"}
    store = SeekDBVectorStore(seekdb_config)
    store.connect()

    assert store._client is mock_client_instance
    mock_module.Client.assert_called_once_with(path="./test.db")


def test_seekdb_vector_store_connect_pylibseekdb(seekdb_config: VectorStoreConfig, mock_sql_modules) -> None:
    from harness.vector.seekdb import SeekDBVectorStore

    connection, cursor = mock_sql_modules
    store = SeekDBVectorStore(seekdb_config)
    store.connect()

    assert store._client is connection


def test_seekdb_vector_store_create_collection_pyseekdb(seekdb_config: VectorStoreConfig, mock_pyseekdb_module) -> None:
    from harness.vector.seekdb import SeekDBVectorStore

    mock_module, mock_client_instance = mock_pyseekdb_module
    seekdb_config.connection = {"path": "./test.db"}
    store = SeekDBVectorStore(seekdb_config)
    store.connect()

    mock_collection = MagicMock()
    mock_client_instance.get_or_create_collection.return_value = mock_collection

    store.create_collection("new_collection", 512, "l2")

    assert store._collection_name == "new_collection"
    assert store._dimension == 512
    assert store._metric == "l2"
    mock_client_instance.get_or_create_collection.assert_called_once_with(name="new_collection")


def test_seekdb_vector_store_create_collection_sql(seekdb_config: VectorStoreConfig, mock_sql_modules) -> None:
    from harness.vector.seekdb import SeekDBVectorStore

    connection, cursor = mock_sql_modules
    store = SeekDBVectorStore(seekdb_config)
    store.connect()

    store.create_collection("test_table", 384, "cosine")

    assert store._collection_name == "test_table"
    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args[0][0]
    assert "CREATE TABLE IF NOT EXISTS test_table" in sql
    assert "VECTOR(384)" in sql
    assert "DISTANCE=cosine" in sql


def test_seekdb_vector_store_store_pyseekdb(seekdb_config: VectorStoreConfig, mock_pyseekdb_module) -> None:
    from harness.vector.seekdb import SeekDBVectorStore

    mock_module, mock_client_instance = mock_pyseekdb_module
    seekdb_config.connection = {"path": "./test.db"}
    store = SeekDBVectorStore(seekdb_config)
    store.connect()

    mock_collection = MagicMock()
    store._collection = mock_collection

    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    metadata = [{"key": "value1"}, {"key": "value2"}]

    store.store(vectors, metadata)

    mock_collection.upsert.assert_called_once()
    call_args = mock_collection.upsert.call_args
    assert call_args[1]["embeddings"] == vectors
    assert call_args[1]["metadatas"] == metadata


def test_seekdb_vector_store_store_sql(seekdb_config: VectorStoreConfig, mock_sql_modules) -> None:
    from harness.vector.seekdb import SeekDBVectorStore

    connection, cursor = mock_sql_modules
    store = SeekDBVectorStore(seekdb_config)
    store.connect()
    store._collection_name = "test_table"

    vectors = [[0.1, 0.2, 0.3]]
    metadata = [{"key": "value"}]

    store.store(vectors, metadata)

    assert cursor.execute.call_count == 1
    sql = cursor.execute.call_args[0][0]
    assert "INSERT INTO test_table" in sql


def test_seekdb_vector_store_search_pyseekdb(seekdb_config: VectorStoreConfig, mock_pyseekdb_module) -> None:
    from harness.vector.seekdb import SeekDBVectorStore

    mock_module, mock_client_instance = mock_pyseekdb_module
    seekdb_config.connection = {"path": "./test.db"}
    store = SeekDBVectorStore(seekdb_config)
    store.connect()

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["0", "1"]],
        "distances": [[0.1, 0.2]],
        "metadatas": [[{"key": "v1"}, {"key": "v2"}]],
    }
    store._collection = mock_collection

    results = store.search([0.1, 0.2, 0.3], top_k=2)

    assert len(results) == 2
    assert results[0]["score"] == 0.1
    assert results[0]["metadata"] == {"key": "v1"}
    mock_collection.query.assert_called_once()


def test_seekdb_vector_store_search_sql(seekdb_config: VectorStoreConfig, mock_sql_modules) -> None:
    from harness.vector.seekdb import SeekDBVectorStore

    connection, cursor = mock_sql_modules
    cursor.fetchall.return_value = [
        {"id": 1, "metadata": {"key": "v1"}, "dist": 0.1},
        {"id": 2, "metadata": {"key": "v2"}, "dist": 0.2},
    ]

    store = SeekDBVectorStore(seekdb_config)
    store.connect()
    store._collection_name = "test_table"
    store._metric = "cosine"

    results = store.search([0.1, 0.2, 0.3], top_k=2)

    assert len(results) == 2
    assert results[0]["score"] == 0.9
    assert results[1]["score"] == 0.8


def test_seekdb_vector_store_count(seekdb_config: VectorStoreConfig, mock_sql_modules) -> None:
    from harness.vector.seekdb import SeekDBVectorStore

    connection, cursor = mock_sql_modules
    cursor.fetchone.return_value = {"cnt": 42}

    store = SeekDBVectorStore(seekdb_config)
    store.connect()
    store._collection_name = "test_table"

    count = store.count()

    assert count == 42
    cursor.execute.assert_called_once()
    assert "SELECT COUNT(*)" in cursor.execute.call_args[0][0]


def test_seekdb_vector_store_delete_collection(seekdb_config: VectorStoreConfig, mock_sql_modules) -> None:
    from harness.vector.seekdb import SeekDBVectorStore

    connection, cursor = mock_sql_modules

    store = SeekDBVectorStore(seekdb_config)
    store.connect()

    store.delete_collection("old_table")

    cursor.execute.assert_called_once()
    assert "DROP TABLE IF EXISTS old_table" in cursor.execute.call_args[0][0]


def test_seekdb_fulltext_search_init(seekdb_config: VectorStoreConfig) -> None:
    from harness.vector.seekdb import SeekDBFullTextSearch

    fts = SeekDBFullTextSearch(seekdb_config)
    assert fts._table_name == "test_collection"
    assert fts._client is None


def test_seekdb_fulltext_search_create_index(seekdb_config: VectorStoreConfig, mock_sql_modules) -> None:
    from harness.vector.seekdb import SeekDBFullTextSearch

    connection, cursor = mock_sql_modules

    fts = SeekDBFullTextSearch(seekdb_config)
    fts.connect()
    fts.create_index("articles", "content", "ik")

    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args[0][0]
    assert "CREATE FULLTEXT INDEX" in sql
    assert "articles" in sql
    assert "content" in sql
    assert "ik" in sql


def test_seekdb_fulltext_search_search(seekdb_config: VectorStoreConfig, mock_sql_modules) -> None:
    from harness.vector.seekdb import SeekDBFullTextSearch

    connection, cursor = mock_sql_modules
    cursor.fetchall.return_value = [
        {"id": 1, "content": "test", "score": 0.9},
        {"id": 2, "content": "example", "score": 0.7},
    ]

    fts = SeekDBFullTextSearch(seekdb_config)
    fts.connect()

    results = fts.search("query", "articles", "content", limit=10)

    assert len(results) == 2
    assert results[0]["score"] == 0.9
    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args[0][0]
    assert "MATCH" in sql
    assert "AGAINST" in sql


def test_seekdb_scalar_search_init(seekdb_config: VectorStoreConfig) -> None:
    from harness.vector.seekdb import SeekDBScalarSearch

    scalar = SeekDBScalarSearch(seekdb_config)
    assert scalar._client is None


def test_seekdb_scalar_search_insert(seekdb_config: VectorStoreConfig, mock_sql_modules) -> None:
    from harness.vector.seekdb import SeekDBScalarSearch

    connection, cursor = mock_sql_modules
    cursor.execute.return_value = 1

    scalar = SeekDBScalarSearch(seekdb_config)
    scalar.connect()

    data = {"name": "test", "value": 42}
    result = scalar.insert("users", data)

    assert result == 1
    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args[0][0]
    assert "INSERT INTO users" in sql
    assert "name, value" in sql


def test_seekdb_scalar_search_insert_batch(seekdb_config: VectorStoreConfig, mock_sql_modules) -> None:
    from harness.vector.seekdb import SeekDBScalarSearch

    connection, cursor = mock_sql_modules
    cursor.executemany.return_value = 3

    scalar = SeekDBScalarSearch(seekdb_config)
    scalar.connect()

    records = [
        {"name": "user1", "value": 1},
        {"name": "user2", "value": 2},
        {"name": "user3", "value": 3},
    ]
    result = scalar.insert_batch("users", records)

    assert result == 3
    cursor.executemany.assert_called_once()
    sql = cursor.executemany.call_args[0][0]
    assert "INSERT INTO users" in sql


def test_seekdb_scalar_search_delete(seekdb_config: VectorStoreConfig, mock_sql_modules) -> None:
    from harness.vector.seekdb import SeekDBScalarSearch

    connection, cursor = mock_sql_modules
    cursor.execute.return_value = 1

    scalar = SeekDBScalarSearch(seekdb_config)
    scalar.connect()

    result = scalar.delete("users", "id = %s", (42,))

    assert result == 1
    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args[0][0]
    assert "DELETE FROM users" in sql
    assert "WHERE id = %s" in sql


def test_seekdb_scalar_search_update(seekdb_config: VectorStoreConfig, mock_sql_modules) -> None:
    from harness.vector.seekdb import SeekDBScalarSearch

    connection, cursor = mock_sql_modules
    cursor.execute.return_value = 1

    scalar = SeekDBScalarSearch(seekdb_config)
    scalar.connect()

    data = {"name": "updated", "value": 100}
    result = scalar.update("users", data, "id = %s", (42,))

    assert result == 1
    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args[0][0]
    assert "UPDATE users SET" in sql
    assert "name = %s, value = %s" in sql
    assert "WHERE id = %s" in sql


def test_seekdb_scalar_search_query(seekdb_config: VectorStoreConfig, mock_sql_modules) -> None:
    from harness.vector.seekdb import SeekDBScalarSearch

    connection, cursor = mock_sql_modules
    cursor.fetchall.return_value = [
        {"id": 1, "name": "user1"},
        {"id": 2, "name": "user2"},
    ]

    scalar = SeekDBScalarSearch(seekdb_config)
    scalar.connect()

    results = scalar.query("SELECT * FROM users WHERE value > %s", (50,))

    assert len(results) == 2
    assert results[0]["name"] == "user1"
    cursor.execute.assert_called_once()


def test_seekdb_scalar_search_execute(seekdb_config: VectorStoreConfig, mock_sql_modules) -> None:
    from harness.vector.seekdb import SeekDBScalarSearch

    connection, cursor = mock_sql_modules
    cursor.execute.return_value = 0

    scalar = SeekDBScalarSearch(seekdb_config)
    scalar.connect()

    result = scalar.execute("CREATE TABLE test (id INT PRIMARY KEY)")

    assert result == 0
    cursor.execute.assert_called_once()


def test_seekdb_gis_search_init(seekdb_config: VectorStoreConfig) -> None:
    from harness.vector.seekdb import SeekDBGISSearch

    gis = SeekDBGISSearch(seekdb_config)
    assert gis._client is None


def test_seekdb_gis_search_nearby(seekdb_config: VectorStoreConfig, mock_sql_modules) -> None:
    from harness.vector.seekdb import SeekDBGISSearch

    connection, cursor = mock_sql_modules
    cursor.fetchall.return_value = [
        {"id": 1, "name": "place1", "distance_km": 1.5},
        {"id": 2, "name": "place2", "distance_km": 2.3},
    ]

    gis = SeekDBGISSearch(seekdb_config)
    gis.connect()

    results = gis.search_nearby("locations", "geo_point", 116.4, 39.9, 5.0, limit=10)

    assert len(results) == 2
    assert results[0]["distance_km"] == 1.5
    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args[0][0]
    assert "ST_Distance_Sphere" in sql


def test_seekdb_gis_search_within_polygon(seekdb_config: VectorStoreConfig, mock_sql_modules) -> None:
    from harness.vector.seekdb import SeekDBGISSearch

    connection, cursor = mock_sql_modules
    cursor.fetchall.return_value = [
        {"id": 1, "name": "inside_place"},
    ]

    gis = SeekDBGISSearch(seekdb_config)
    gis.connect()

    polygon_wkt = "POLYGON((116.0 39.0, 117.0 39.0, 117.0 40.0, 116.0 40.0, 116.0 39.0))"
    results = gis.search_within_polygon("locations", "geo_point", polygon_wkt)

    assert len(results) == 1
    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args[0][0]
    assert "ST_Within" in sql
    assert "ST_GeomFromText" in sql


def test_seekdb_hybrid_search_init(seekdb_config: VectorStoreConfig) -> None:
    from harness.vector.seekdb import SeekDBHybridSearch

    hybrid = SeekDBHybridSearch(seekdb_config)
    assert hybrid._client is None


def test_seekdb_hybrid_search(seekdb_config: VectorStoreConfig, mock_sql_modules) -> None:
    from harness.vector.seekdb import SeekDBHybridSearch

    connection, cursor = mock_sql_modules
    cursor.fetchall.return_value = [
        {"id": 1, "title": "doc1", "vec_dist": 0.1, "text_score": 0.8},
        {"id": 2, "title": "doc2", "vec_dist": 0.2, "text_score": 0.9},
    ]

    hybrid = SeekDBHybridSearch(seekdb_config)
    hybrid.connect()

    results = hybrid.search(
        table="documents",
        vector_column="embedding",
        text_column="content",
        query_vector=[0.1, 0.2, 0.3],
        query_text="search query",
        scalar_filter="category = 'tech'",
        limit=10,
    )

    assert len(results) == 2
    assert "hybrid_score" in results[0]
    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args[0][0]
    assert "l2_distance" in sql
    assert "MATCH" in sql
    assert "WHERE category = 'tech'" in sql


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
