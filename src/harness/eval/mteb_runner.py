from __future__ import annotations

from harness.config import EvaluationConfig
from harness.factories import register_dataset_provider
from harness.interfaces import Dataset


@register_dataset_provider("mteb")
class MtebDatasetProvider:
    def __init__(self, cfg: EvaluationConfig) -> None:
        self._cfg = cfg

    def load(self, task_name: str, split: str = "test") -> Dataset:
        import mteb

        tasks = mteb.get_tasks(tasks=[task_name])
        if not tasks:
            raise ValueError(f"MTEB task not found: {task_name}")

        task = tasks[0]
        task.load_data()
        hf_dataset = task.dataset

        if isinstance(hf_dataset, dict):
            data = hf_dataset.get(split) or hf_dataset.get("test") or next(iter(hf_dataset.values()))
        else:
            data = hf_dataset

        texts: list[str] = []
        labels: list[str] | None = None
        queries: list[str] | None = None
        corpus: list[str] | None = None

        feature_names = list(data.features.keys())

        if "sentence1" in feature_names and "sentence2" in feature_names:
            texts = [str(row["sentence1"]) for row in data]
            corpus = [str(row["sentence2"]) for row in data]
            score_key = "score" if "score" in feature_names else "label"
            if score_key in feature_names:
                labels = [str(row[score_key]) for row in data]
        elif "text" in feature_names:
            texts = [str(row["text"]) for row in data]
            if "label" in feature_names:
                labels = [str(row["label"]) for row in data]
        elif "query" in feature_names:
            queries = [str(row["query"]) for row in data]
            if "positive" in feature_names:
                corpus = [str(row["positive"]) for row in data]
            elif "corpus" in feature_names:
                corpus = [str(row["corpus"]) for row in data]
        else:
            texts = [str(row) for row in data]

        return Dataset(
            texts=texts,
            labels=labels,
            queries=queries,
            corpus=corpus,
            metadata={
                "task_name": task_name,
                "task_type": task.metadata.type,
                "languages": task.metadata.languages,
                "split": split,
                "num_rows": len(data),
            },
        )

    def list_available(self) -> list[str]:
        import mteb
        return [t.metadata.name for t in mteb.get_tasks(languages=["eng"])]
