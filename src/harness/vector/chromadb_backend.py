from __future__ import annotations

from harness.config import VectorStoreConfig
from harness.factories import register_vector_store


@register_vector_store("chromadb")
class ChromaDBVectorStore:
    def __init__(self, cfg: VectorStoreConfig) -> None:
        self._cfg = cfg
        self._client = None
        self._collection = None

    def connect(self) -> None:
        import chromadb

        conn = self._cfg.connection
        db_path = conn.get("path", "./chroma_data")
        self._client = chromadb.PersistentClient(path=db_path)

    def create_collection(self, name: str, dimension: int, metric: str = "cosine") -> None:
        metric_map = {"cosine": "cosine", "l2": "l2", "ip": "ip"}
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": metric_map.get(metric, "cosine")},
        )

    def store(self, vectors: list[list[float]], metadata: list[dict]) -> None:
        if self._collection is None:
            raise RuntimeError("Collection not created. Call create_collection() first.")
        ids = [str(i) for i in range(self._collection.count(), self._collection.count() + len(vectors))]
        self._collection.add(ids=ids, embeddings=vectors, metadatas=metadata)

    def search(
        self, query: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[dict]:
        if self._collection is None:
            raise RuntimeError("Collection not created.")
        kwargs: dict = {"query_embeddings": [query], "n_results": top_k}
        if filters:
            kwargs["where"] = filters
        results = self._collection.query(**kwargs)
        items = []
        for i in range(len(results["ids"][0])):
            items.append({
                "score": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
                "id": results["ids"][0][i],
            })
        return items

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(name)

    def count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()

    def close(self) -> None:
        self._collection = None
        self._client = None
