from __future__ import annotations

from harness.interfaces import StorageScore, VectorStore


def compute_storage_scores(
    store: VectorStore,
    queries: list[list[float]],
) -> list[StorageScore]:
    scores: list[StorageScore] = []
    total_score = 0.0

    for q in queries:
        results = store.search(q, top_k=5)
        if results:
            total_score += results[0]["score"]

    count = len(queries) if queries else 1
    avg_score = total_score / count

    scores.append(StorageScore(
        metric="avg_top1_similarity",
        score=avg_score,
        details={"num_queries": len(queries), "store_count": store.count()},
    ))

    scores.append(StorageScore(
        metric="total_vectors",
        score=float(store.count()),
        details={"unit": "vectors"},
    ))

    return scores
