from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, cfg: HarnessConfig) -> None:
        self.cfg = cfg
        logger.info("[Pipeline] 初始化完成")

    def run_full(self) -> list[Path]:
        logger.info("[Pipeline.run_full] ===== 进入 =====")
        
        logger.info("[Pipeline.run_full] 执行 doctor 检查...")
        self._run_doctor()
        logger.info("[Pipeline.run_full] doctor 检查完成")

        logger.info("[Pipeline.run_full] 创建 embedding 模型...")
        embedder = create_embedding_model(self.cfg.embedding)
        logger.info("[Pipeline.run_full] embedding 模型创建完成")
        
        logger.info("[Pipeline.run_full] 创建数据集 provider...")
        provider = create_dataset_provider(self.cfg.evaluation)
        logger.info("[Pipeline.run_full] 数据集 provider 创建完成")
        
        logger.info("[Pipeline.run_full] 运行评测...")
        eval_result = run_evaluation(embedder, provider, self.cfg.evaluation)
        logger.info("[Pipeline.run_full] 评测完成")

        logger.info("[Pipeline.run_full] 创建 HPC 后端...")
        hpc = create_hpc_backend(self.cfg.hpc)
        logger.info("[Pipeline.run_full] HPC 后端创建完成")
        
        logger.info("[Pipeline.run_full] 打包项目...")
        pack_dir = Path("./dist")
        pack_dir.mkdir(parents=True, exist_ok=True)
        pack_path = hpc.pack(Path("."), pack_dir / "hpc-pack.tar.gz")
        logger.info("[Pipeline.run_full] 打包完成: %s", pack_path)
        
        logger.info("[Pipeline.run_full] 提交 HPC 任务...")
        job_id = hpc.submit(Path(self.cfg.hpc.remote_dir), {})
        logger.info("[Pipeline.run_full] 任务已提交, job_id=%s", job_id)
        
        logger.info("[Pipeline.run_full] 获取 HPC 结果...")
        hpc_result_path = hpc.fetch(job_id, Path("./results/hpc"))
        logger.info("[Pipeline.run_full] HPC 结果获取完成: %s", hpc_result_path)

        logger.info("[Pipeline.run_full] 创建向量存储...")
        store = create_vector_store(self.cfg.vector_store)
        logger.info("[Pipeline.run_full] 向量存储创建完成")
        
        logger.info("[Pipeline.run_full] 连接向量存储...")
        store.connect()
        logger.info("[Pipeline.run_full] 向量存储连接成功")
        
        logger.info("[Pipeline.run_full] 创建集合...")
        store.create_collection(
            self.cfg.vector_store.collection_name,
            self.cfg.vector_store.collection_dimension,
            self.cfg.vector_store.collection_metric,
        )
        logger.info("[Pipeline.run_full] 集合创建完成")
        
        logger.info("[Pipeline.run_full] 生成样本数据并编码...")
        sample_texts = [f"sample document {i}" for i in range(50)]
        vectors = embedder.encode(sample_texts)
        logger.info("[Pipeline.run_full] 编码完成, 向量数=%d", len(vectors))
        
        logger.info("[Pipeline.run_full] 存储向量...")
        store.store(vectors, [{"id": i, "text": t} for i, t in enumerate(sample_texts)])
        logger.info("[Pipeline.run_full] 向量存储完成")

        logger.info("[Pipeline.run_full] 计算模型分数...")
        model_scores = compute_model_scores(eval_result)
        logger.info("[Pipeline.run_full] 模型分数计算完成, 任务数=%d", len(model_scores))
        
        logger.info("[Pipeline.run_full] 编码查询向量...")
        query_vectors = embedder.encode(["query 1", "query 2"])
        logger.info("[Pipeline.run_full] 查询向量编码完成")
        
        logger.info("[Pipeline.run_full] 计算存储分数...")
        storage_scores = compute_storage_scores(store, query_vectors)
        logger.info("[Pipeline.run_full] 存储分数计算完成, 指标数=%d", len(storage_scores))

        logger.info("[Pipeline.run_full] 创建实验追踪器...")
        tracker = create_experiment_tracker(self.cfg.tracking)
        logger.info("[Pipeline.run_full] 实验追踪器创建完成")
        
        logger.info("[Pipeline.run_full] 开始运行记录...")
        run_id = tracker.start_run(f"full-{datetime.now():%Y%m%d-%H%M%S}")
        logger.info("[Pipeline.run_full] 运行已启动, run_id=%s", run_id)
        
        logger.info("[Pipeline.run_full] 记录模型分数...")
        for s in model_scores:
            tracker.log_metric(f"model/{s.task}", s.score)
        logger.info("[Pipeline.run_full] 模型分数记录完成")
        
        logger.info("[Pipeline.run_full] 记录存储分数...")
        for s in storage_scores:
            tracker.log_metric(f"storage/{s.metric}", s.score)
        logger.info("[Pipeline.run_full] 存储分数记录完成")
        
        logger.info("[Pipeline.run_full] 结束运行记录...")
        tracker.end_run()
        logger.info("[Pipeline.run_full] 运行记录已结束")

        logger.info("[Pipeline.run_full] 生成报告...")
        report = Report(
            model_scores=model_scores,
            storage_scores=storage_scores,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
        )
        logger.info("[Pipeline.run_full] 报告对象创建完成")

        logger.info("[Pipeline.run_full] 关闭向量存储...")
        store.close()
        logger.info("[Pipeline.run_full] 向量存储已关闭")
        
        logger.info("[Pipeline.run_full] 生成报告文件...")
        report_paths = generate_report(report, self.cfg.report)
        logger.info("[Pipeline.run_full] 报告生成完成, 文件数=%d", len(report_paths))
        logger.info("[Pipeline.run_full] ===== 退出 =====")
        return report_paths

    def run_baseline(self) -> list[Path]:
        logger.info("[Pipeline.run_baseline] ===== 进入 =====")
        
        logger.info("[Pipeline.run_baseline] 执行 doctor 检查...")
        self._run_doctor()
        logger.info("[Pipeline.run_baseline] doctor 检查完成")

        logger.info("[Pipeline.run_baseline] 创建 embedding 模型...")
        embedder = create_embedding_model(self.cfg.embedding)
        logger.info("[Pipeline.run_baseline] embedding 模型创建完成")
        
        logger.info("[Pipeline.run_baseline] 创建数据集 provider...")
        provider = create_dataset_provider(self.cfg.evaluation)
        logger.info("[Pipeline.run_baseline] 数据集 provider 创建完成")
        
        logger.info("[Pipeline.run_baseline] 运行评测...")
        eval_result = run_evaluation(embedder, provider, self.cfg.evaluation)
        logger.info("[Pipeline.run_baseline] 评测完成")

        logger.info("[Pipeline.run_baseline] 创建向量存储...")
        store = create_vector_store(self.cfg.vector_store)
        logger.info("[Pipeline.run_baseline] 向量存储创建完成")
        
        logger.info("[Pipeline.run_baseline] 连接向量存储...")
        store.connect()
        logger.info("[Pipeline.run_baseline] 向量存储连接成功")
        
        logger.info("[Pipeline.run_baseline] 创建集合...")
        store.create_collection(
            self.cfg.vector_store.collection_name,
            self.cfg.vector_store.collection_dimension,
            self.cfg.vector_store.collection_metric,
        )
        logger.info("[Pipeline.run_baseline] 集合创建完成")
        
        logger.info("[Pipeline.run_baseline] 生成样本数据并编码...")
        sample_texts = [f"baseline document {i}" for i in range(20)]
        vectors = embedder.encode(sample_texts)
        logger.info("[Pipeline.run_baseline] 编码完成, 向量数=%d", len(vectors))
        
        logger.info("[Pipeline.run_baseline] 存储向量...")
        store.store(vectors, [{"id": i} for i in range(len(sample_texts))])
        logger.info("[Pipeline.run_baseline] 向量存储完成")

        logger.info("[Pipeline.run_baseline] 计算模型分数...")
        model_scores = compute_model_scores(eval_result)
        logger.info("[Pipeline.run_baseline] 模型分数计算完成, 任务数=%d", len(model_scores))
        
        logger.info("[Pipeline.run_baseline] 编码查询向量...")
        query_vectors = embedder.encode(["baseline query"])
        logger.info("[Pipeline.run_baseline] 查询向量编码完成")
        
        logger.info("[Pipeline.run_baseline] 计算存储分数...")
        storage_scores = compute_storage_scores(store, query_vectors)
        logger.info("[Pipeline.run_baseline] 存储分数计算完成, 指标数=%d", len(storage_scores))

        logger.info("[Pipeline.run_baseline] 创建实验追踪器...")
        tracker = create_experiment_tracker(self.cfg.tracking)
        logger.info("[Pipeline.run_baseline] 实验追踪器创建完成")
        
        logger.info("[Pipeline.run_baseline] 开始运行记录...")
        run_id = tracker.start_run(f"baseline-{datetime.now():%Y%m%d-%H%M%S}")
        logger.info("[Pipeline.run_baseline] 运行已启动, run_id=%s", run_id)
        
        logger.info("[Pipeline.run_baseline] 记录模型分数...")
        for s in model_scores:
            tracker.log_metric(f"model/{s.task}", s.score)
        logger.info("[Pipeline.run_baseline] 模型分数记录完成")
        
        logger.info("[Pipeline.run_baseline] 结束运行记录...")
        tracker.end_run()
        logger.info("[Pipeline.run_baseline] 运行记录已结束")

        logger.info("[Pipeline.run_baseline] 生成报告...")
        report = Report(
            model_scores=model_scores,
            storage_scores=storage_scores,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
        )
        logger.info("[Pipeline.run_baseline] 报告对象创建完成")

        logger.info("[Pipeline.run_baseline] 关闭向量存储...")
        store.close()
        logger.info("[Pipeline.run_baseline] 向量存储已关闭")
        
        logger.info("[Pipeline.run_baseline] 生成报告文件...")
        report_paths = generate_report(report, self.cfg.report)
        logger.info("[Pipeline.run_baseline] 报告生成完成, 文件数=%d", len(report_paths))
        logger.info("[Pipeline.run_baseline] ===== 退出 =====")
        return report_paths

    def _run_doctor(self) -> None:
        logger.info("[Pipeline._run_doctor] ===== 进入 =====")
        from harness.doctor.checks import run_all_checks
        logger.info("[Pipeline._run_doctor] 执行所有检查...")
        results = run_all_checks(self.cfg)
        logger.info("[Pipeline._run_doctor] 检查完成, 检查项数=%d", len(results))
        
        fatal_failures = [r for r in results if r.fatal and not r.passed]
        if fatal_failures:
            msg = "; ".join(f"{r.name}: {r.message}" for r in fatal_failures)
            logger.error("[Pipeline._run_doctor] 致命检查失败: %s", msg)
            raise SystemExit(f"Pre-flight checks failed: {msg}")
        logger.info("[Pipeline._run_doctor] ===== 退出 =====")
