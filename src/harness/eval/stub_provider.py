from __future__ import annotations

from harness.config import EvaluationConfig
from harness.factories import register_dataset_provider
from harness.interfaces import Dataset


_STUB_TASKS = {
    "STS22": Dataset(
        texts=["The cat sat on the mat", "A feline rested on a rug", "Stocks surged today"],
        labels=["1.0", "0.9", "0.1"],
        metadata={"task_type": "STS", "languages": ["en"]},
    ),
    "Banking77Classification": Dataset(
        texts=[
            "I need to dispute a charge",
            "What is my account balance",
            "Transfer money to savings",
        ],
        labels=["dispute_charge", "balance", "transfer"],
        metadata={"task_type": "Classification", "languages": ["en"]},
    ),
}


@register_dataset_provider("stub")
class StubDatasetProvider:
    def __init__(self, cfg: EvaluationConfig) -> None:
        self._cfg = cfg

    def load(self, task_name: str, split: str = "test") -> Dataset:
        if task_name in _STUB_TASKS:
            return _STUB_TASKS[task_name]
        return Dataset(
            texts=[f"sample text for {task_name}"],
            labels=["unknown"],
            metadata={"task_type": "Unknown", "task_name": task_name},
        )

    def list_available(self) -> list[str]:
        return list(_STUB_TASKS.keys())
