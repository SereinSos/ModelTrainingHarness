from __future__ import annotations

import math

from harness.config import VectorStoreConfig
from harness.factories import register_vector_store


@register_vector_store("memory")
class InMemoryVectorStore:
    def __init__(self, cfg: VectorStoreConfig) -> None:
        self._collections: dict[str, list[tuple[list[float], dict]]] = {}
        self._dimension: dict[str, int] = {}
        self._metric: dict[str, str] = {}

    def connect(self) -> None:
        pass

    def create_collection(self, name: str, dimension: int, metric: str = "cosine") -> None:
        self._collections[name] = []
        self._dimension[name] = dimension
        self._metric[name] = metric

    def store(self, vectors: list[list[float]], metadata: list[dict]) -> None:
        for name, items in self._collections.items():
            items.extend(zip(vectors, metadata))

    def search(
        self, query: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[dict]:
        results = []
        for name, items in self._collections.items():
            metric = self._metric.get(name, "cosine")
            for vec, meta in items:
                score = self._similarity(query, vec, metric)
                results.append({"score": score, "metadata": meta})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete_collection(self, name: str) -> None:
        self._collections.pop(name, None)
        self._dimension.pop(name, None)
        self._metric.pop(name, None)

    def count(self) -> int:
        return sum(len(items) for items in self._collections.values())

    def close(self) -> None:
        self._collections.clear()

    @staticmethod
    def _similarity(a: list[float], b: list[float], metric: str) -> float:
        if metric == "cosine":
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb) if na and nb else 0.0
        if metric == "l2":
            return -math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        dot = sum(x * y for x, y in zip(a, b))
        return dot
