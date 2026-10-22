from __future__ import annotations

from datetime import datetime
from pathlib import Path

from harness.config import HarnessConfig
from harness.eval.evaluator import run_evaluation
from harness.factories import (
    create_dataset_provider,
    create_embedding_model,
    create_experiment_tracker,
    create_hpc_backend,
    create_vector_store,
)
from harness.report.generator import generate_report
from harness.scoring.model_score import compute_model_scores
from harness.scoring.storage_score import compute_storage_scores
from harness.interfaces import Report


class Pipeline:
    def __init__(self, cfg: HarnessConfig) -> None:
        self.cfg = cfg

    def run_full(self) -> list[Path]:
        self._run_doctor()

        embedder = create_embedding_model(self.cfg.embedding)
        provider = create_dataset_provider(self.cfg.evaluation)
        eval_result = run_evaluation(embedder, provider, self.cfg.evaluation)

        hpc = create_hpc_backend(self.cfg.hpc)
        pack_dir = Path("./dist")
        pack_dir.mkdir(parents=True, exist_ok=True)
        pack_path = hpc.pack(Path("."), pack_dir / "hpc-pack.tar.gz")
        job_id = hpc.submit(Path(self.cfg.hpc.remote_dir), {})
        hpc_result_path = hpc.fetch(job_id, Path("./results/hpc"))

        store = create_vector_store(self.cfg.vector_store)
        store.connect()
        store.create_collection(
            self.cfg.vector_store.collection_name,
            self.cfg.vector_store.collection_dimension,
            self.cfg.vector_store.collection_metric,
        )
        sample_texts = [f"sample document {i}" for i in range(50)]
        vectors = embedder.encode(sample_texts)
        store.store(vectors, [{"id": i, "text": t} for i, t in enumerate(sample_texts)])

        model_scores = compute_model_scores(eval_result)
        query_vectors = embedder.encode(["query 1", "query 2"])
        storage_scores = compute_storage_scores(store, query_vectors)

        tracker = create_experiment_tracker(self.cfg.tracking)
        run_id = tracker.start_run(f"full-{datetime.now():%Y%m%d-%H%M%S}")
        for s in model_scores:
            tracker.log_metric(f"model/{s.task}", s.score)
        for s in storage_scores:
            tracker.log_metric(f"storage/{s.metric}", s.score)
        tracker.end_run()

        report = Report(
            model_scores=model_scores,
            storage_scores=storage_scores,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
        )

        store.close()
        return generate_report(report, self.cfg.report)

    def run_baseline(self) -> list[Path]:
        self._run_doctor()

        embedder = create_embedding_model(self.cfg.embedding)
        provider = create_dataset_provider(self.cfg.evaluation)
        eval_result = run_evaluation(embedder, provider, self.cfg.evaluation)

        store = create_vector_store(self.cfg.vector_store)
        store.connect()
        store.create_collection(
            self.cfg.vector_store.collection_name,
            self.cfg.vector_store.collection_dimension,
            self.cfg.vector_store.collection_metric,
        )
        sample_texts = [f"baseline document {i}" for i in range(20)]
        vectors = embedder.encode(sample_texts)
        store.store(vectors, [{"id": i} for i in range(len(sample_texts))])

        model_scores = compute_model_scores(eval_result)
        query_vectors = embedder.encode(["baseline query"])
        storage_scores = compute_storage_scores(store, query_vectors)

        tracker = create_experiment_tracker(self.cfg.tracking)
        run_id = tracker.start_run(f"baseline-{datetime.now():%Y%m%d-%H%M%S}")
        for s in model_scores:
            tracker.log_metric(f"model/{s.task}", s.score)
        tracker.end_run()

        report = Report(
            model_scores=model_scores,
            storage_scores=storage_scores,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
        )

        store.close()
        return generate_report(report, self.cfg.report)

    def _run_doctor(self) -> None:
        from harness.doctor.checks import run_all_checks
        results = run_all_checks(self.cfg)
        fatal_failures = [r for r in results if r.fatal and not r.passed]
        if fatal_failures:
            msg = "; ".join(f"{r.name}: {r.message}" for r in fatal_failures)
            raise SystemExit(f"Pre-flight checks failed: {msg}")
