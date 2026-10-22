from __future__ import annotations

from harness.interfaces import EvalResult, ModelScore


def compute_model_scores(eval_result: EvalResult) -> list[ModelScore]:
    return [
        ModelScore(task=task, score=score, details={"source": "mteb"})
        for task, score in eval_result.scores.items()
    ]
