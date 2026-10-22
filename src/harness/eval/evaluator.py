from __future__ import annotations

import time

from harness.config import EvaluationConfig
from harness.interfaces import DatasetProvider, EmbeddingModel, EvalResult


def run_evaluation(
    embedder: EmbeddingModel,
    provider: DatasetProvider,
    cfg: EvaluationConfig,
) -> EvalResult:
    start = time.time()
    scores: dict[str, float] = {}

    for task_cfg in cfg.tasks:
        dataset = provider.load(task_cfg.name)
        vectors = embedder.encode(dataset.texts)
        score = _compute_stub_score(task_cfg.name, vectors)
        scores[task_cfg.name] = score

    elapsed = time.time() - start
    return EvalResult(
        model_name=embedder.model_name,
        scores=scores,
        elapsed_seconds=elapsed,
    )


def _compute_stub_score(task_name: str, vectors: list[list[float]]) -> float:
    if not vectors:
        return 0.0
    dim = len(vectors[0])
    total = 0.0
    count = 0
    for i in range(len(vectors)):
        for j in range(i + 1, min(i + 5, len(vectors))):
            dot = sum(a * b for a, b in zip(vectors[i], vectors[j]))
            total += abs(dot)
            count += 1
    avg = total / count if count else 0.0
    return min(avg * 100, 1.0)
