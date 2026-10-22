from __future__ import annotations

from typing import Any

from harness.config import VectorStoreConfig
from harness.factories import register_vector_store


@register_vector_store("seekdb")
class SeekDBVectorStore:
    """SeekDB 向量检索适配器（实现 VectorStore Protocol）"""

    def __init__(self, cfg: VectorStoreConfig) -> None:
        self._cfg = cfg
        self._client = None
        self._collection = None
        self._collection_name = cfg.collection_name
        self._dimension = cfg.collection_dimension
        self._metric = cfg.collection_metric

    def connect(self) -> None:
        conn = self._cfg.connection
        if conn.get("path"):
            import pyseekdb
            self._client = pyseekdb.Client(path=conn["path"])
        else:
            import pylibseekdb
            import pymysql
            self._seekdb = pylibseekdb.open(conn.get("db_path", "./seekdb.db"))
            options = self._seekdb.connection_options()
            self._client = pymysql.connect(
                database=conn.get("database", "test"),
                autocommit=True,
                cursorclass=pymysql.cursors.DictCursor,
                **options
            )

    def create_collection(self, name: str, dimension: int, metric: str = "cosine") -> None:
        self._collection_name = name
        self._dimension = dimension
        self._metric = metric

        if hasattr(self._client, 'get_or_create_collection'):
            self._collection = self._client.get_or_create_collection(name=name)
        else:
            distance_map = {"cosine": "cosine", "l2": "l2", "ip": "ip"}
            sql = f"""
            CREATE TABLE IF NOT EXISTS {name} (
                id INT PRIMARY KEY AUTO_INCREMENT,
                embedding VECTOR({dimension}),
                metadata JSON,
                VECTOR INDEX idx_vec (embedding) WITH (DISTANCE={distance_map.get(metric, 'l2')}, TYPE=hnsw, LIB=vsag)
            ) ORGANIZATION = HEAP;
            """
            with self._client.cursor() as cursor:
                cursor.execute(sql)

    def store(self, vectors: list[list[float]], metadata: list[dict]) -> None:
        if hasattr(self._collection, 'upsert'):
            ids = [str(i) for i in range(len(vectors))]
            self._collection.upsert(ids=ids, embeddings=vectors, metadatas=metadata)
        else:
            with self._client.cursor() as cursor:
                for vec, meta in zip(vectors, metadata):
                    sql = f"INSERT INTO {self._collection_name} (embedding, metadata) VALUES (%s, %s)"
                    cursor.execute(sql, (str(vec), str(meta)))

    def search(
        self, query: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[dict]:
        if hasattr(self._collection, 'query'):
            results = self._collection.query(query_embeddings=[query], n_results=top_k)
            return [
                {"score": results["distances"][0][i], "metadata": results["metadatas"][0][i]}
                for i in range(len(results["ids"][0]))
            ]
        else:
            distance_func = "cosine_distance" if self._metric == "cosine" else "l2_distance"
            sql = f"""
            SELECT id, metadata, {distance_func}(embedding, %s) AS dist
            FROM {self._collection_name}
            ORDER BY dist APPROXIMATE LIMIT %s
            """
            with self._client.cursor() as cursor:
                cursor.execute(sql, (str(query), top_k))
                return [{"score": 1 - row["dist"], "metadata": row["metadata"]} for row in cursor.fetchall()]

    def delete_collection(self, name: str) -> None:
        if hasattr(self._client, 'delete_collection'):
            self._client.delete_collection(name)
        else:
            with self._client.cursor() as cursor:
                cursor.execute(f"DROP TABLE IF EXISTS {name}")

    def count(self) -> int:
        if hasattr(self._collection, 'count'):
            return self._collection.count()
        else:
            with self._client.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) as cnt FROM {self._collection_name}")
                return cursor.fetchone()["cnt"]

    def close(self) -> None:
        if hasattr(self._client, 'close'):
            self._client.close()
        elif hasattr(self._client, '__exit__'):
            self._client.close()


class SeekDBFullTextSearch:
    """SeekDB 全文检索适配器"""

    def __init__(self, cfg: VectorStoreConfig) -> None:
        self._cfg = cfg
        self._client = None
        self._table_name = cfg.collection_name

    def connect(self) -> None:
        conn = self._cfg.connection
        import pylibseekdb
        import pymysql
        self._seekdb = pylibseekdb.open(conn.get("db_path", "./seekdb.db"))
        options = self._seekdb.connection_options()
        self._client = pymysql.connect(
            database=conn.get("database", "test"),
            autocommit=True,
            cursorclass=pymysql.cursors.DictCursor,
            **options
        )

    def create_index(self, table: str, column: str, parser: str = "ik") -> None:
        self._table_name = table
        sql = f"CREATE FULLTEXT INDEX idx_ft_{column} ON {table}({column}) WITH PARSER {parser}"
        with self._client.cursor() as cursor:
            try:
                cursor.execute(sql)
            except Exception:
                pass

    def search(self, query: str, table: str, column: str, limit: int = 10) -> list[dict]:
        sql = f"""
        SELECT *, MATCH({column}) AGAINST(%s) AS score
        FROM {table}
        WHERE MATCH({column}) AGAINST(%s)
        ORDER BY score DESC
        LIMIT %s
        """
        with self._client.cursor() as cursor:
            cursor.execute(sql, (query, query, limit))
            return cursor.fetchall()

    def close(self) -> None:
        if self._client:
            self._client.close()


class SeekDBScalarSearch:
    """SeekDB 普通数据检索适配器（SQL 查询）"""

    def __init__(self, cfg: VectorStoreConfig) -> None:
        self._cfg = cfg
        self._client = None

    def connect(self) -> None:
        conn = self._cfg.connection
        import pylibseekdb
        import pymysql
        self._seekdb = pylibseekdb.open(conn.get("db_path", "./seekdb.db"))
        options = self._seekdb.connection_options()
        self._client = pymysql.connect(
            database=conn.get("database", "test"),
            autocommit=True,
            cursorclass=pymysql.cursors.DictCursor,
            **options
        )

    def insert(self, table: str, data: dict) -> int:
        """插入单条记录"""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["%s"] * len(data))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        with self._client.cursor() as cursor:
            return cursor.execute(sql, tuple(data.values()))

    def insert_batch(self, table: str, records: list[dict]) -> int:
        """批量插入"""
        if not records:
            return 0
        columns = ", ".join(records[0].keys())
        placeholders = ", ".join(["%s"] * len(records[0]))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        with self._client.cursor() as cursor:
            return cursor.executemany(sql, [tuple(r.values()) for r in records])

    def delete(self, table: str, where: str, params: tuple | None = None) -> int:
        """删除记录"""
        sql = f"DELETE FROM {table} WHERE {where}"
        with self._client.cursor() as cursor:
            return cursor.execute(sql, params or ())

    def update(self, table: str, data: dict, where: str, params: tuple | None = None) -> int:
        """更新记录"""
        set_clause = ", ".join([f"{k} = %s" for k in data.keys()])
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        values = tuple(data.values()) + (params or ())
        with self._client.cursor() as cursor:
            return cursor.execute(sql, values)

    def query(self, sql: str, params: tuple | None = None) -> list[dict]:
        """查询记录"""
        with self._client.cursor() as cursor:
            cursor.execute(sql, params or ())
            return cursor.fetchall()

    def execute(self, sql: str, params: tuple | None = None) -> int:
        """执行任意 SQL"""
        with self._client.cursor() as cursor:
            return cursor.execute(sql, params or ())

    def close(self) -> None:
        if self._client:
            self._client.close()


class SeekDBGISSearch:
    """SeekDB GIS 检索适配器"""

    def __init__(self, cfg: VectorStoreConfig) -> None:
        self._cfg = cfg
        self._client = None

    def connect(self) -> None:
        conn = self._cfg.connection
        import pylibseekdb
        import pymysql
        self._seekdb = pylibseekdb.open(conn.get("db_path", "./seekdb.db"))
        options = self._seekdb.connection_options()
        self._client = pymysql.connect(
            database=conn.get("database", "test"),
            autocommit=True,
            cursorclass=pymysql.cursors.DictCursor,
            **options
        )

    def search_nearby(
        self, table: str, geo_column: str, lon: float, lat: float, radius_km: float, limit: int = 10
    ) -> list[dict]:
        sql = f"""
        SELECT *, ST_Distance_Sphere(
            POINT(%s, %s),
            {geo_column}
        ) / 1000 AS distance_km
        FROM {table}
        WHERE ST_Distance_Sphere(POINT(%s, %s), {geo_column}) <= %s * 1000
        ORDER BY distance_km
        LIMIT %s
        """
        with self._client.cursor() as cursor:
            cursor.execute(sql, (lon, lat, lon, lat, radius_km, limit))
            return cursor.fetchall()

    def search_within_polygon(self, table: str, geo_column: str, polygon_wkt: str) -> list[dict]:
        sql = f"""
        SELECT *
        FROM {table}
        WHERE ST_Within({geo_column}, ST_GeomFromText(%s))
        """
        with self._client.cursor() as cursor:
            cursor.execute(sql, (polygon_wkt,))
            return cursor.fetchall()

    def close(self) -> None:
        if self._client:
            self._client.close()


class SeekDBHybridSearch:
    """SeekDB 混合检索适配器（向量 + 全文 + 标量过滤）"""

    def __init__(self, cfg: VectorStoreConfig) -> None:
        self._cfg = cfg
        self._client = None

    def connect(self) -> None:
        conn = self._cfg.connection
        import pylibseekdb
        import pymysql
        self._seekdb = pylibseekdb.open(conn.get("db_path", "./seekdb.db"))
        options = self._seekdb.connection_options()
        self._client = pymysql.connect(
            database=conn.get("database", "test"),
            autocommit=True,
            cursorclass=pymysql.cursors.DictCursor,
            **options
        )

    def search(
        self,
        table: str,
        vector_column: str,
        text_column: str,
        query_vector: list[float],
        query_text: str,
        scalar_filter: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        where_clause = f"WHERE {scalar_filter}" if scalar_filter else ""
        sql = f"""
        SELECT id, title,
               l2_distance({vector_column}, %s) AS vec_dist,
               MATCH({text_column}) AGAINST(%s) AS text_score
        FROM {table}
        {where_clause}
        ORDER BY vec_dist APPROXIMATE
        LIMIT %s
        """
        with self._client.cursor() as cursor:
            cursor.execute(sql, (str(query_vector), query_text, limit))
            results = cursor.fetchall()
            for r in results:
                r["hybrid_score"] = (1 - r["vec_dist"]) * 0.5 + r["text_score"] * 0.5
            return sorted(results, key=lambda x: x["hybrid_score"], reverse=True)

    def close(self) -> None:
        if self._client:
            self._client.close()
