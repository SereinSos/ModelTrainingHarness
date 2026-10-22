from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from harness.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness", description="Model Training Evaluation Harness")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML (default: config/harness.yaml or $HARNESS_CONFIG)")
    parser.add_argument("--log", choices=["on", "off"], default="off", help="Enable detailed logging (default: off)")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("doctor", help="Run pre-flight checks")
    sub.add_parser("eval", help="Run local evaluation")
    sub.add_parser("pack", help="Package for HPC upload")
    sub.add_parser("submit", help="Submit to HPC")
    sub.add_parser("fetch", help="Fetch HPC results")
    sub.add_parser("baseline", help="Run baseline scoring (skip HPC)")
    sub.add_parser("run", help="Execute full pipeline")
    sub.add_parser("registry", help="List registered backends")

    args = parser.parse_args(argv)

    # Configure logging based on --log flag
    if args.log == "on":
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.command is None:
        parser.print_help()
        return 0

    cfg = load_config(args.config)

    if args.command == "registry":
        return _cmd_registry()

    if args.command == "doctor":
        return _cmd_doctor(cfg)

    if args.command == "eval":
        return _cmd_eval(cfg)

    if args.command == "run":
        return _cmd_run(cfg)

    if args.command == "baseline":
        return _cmd_baseline(cfg)

    if args.command == "pack":
        return _cmd_pack(cfg)

    if args.command == "submit":
        return _cmd_submit(cfg)

    if args.command == "fetch":
        return _cmd_fetch(cfg)

    parser.print_help()
    return 0


def _cmd_registry() -> int:
    from harness.factories import list_registries
    for category, names in list_registries().items():
        print(f"{category}: {', '.join(names) if names else '(none)'}")
    return 0


def _cmd_doctor(cfg) -> int:
    from harness.doctor.checks import run_all_checks
    results = run_all_checks(cfg)
    all_pass = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}: {r.message}")
        if not r.passed and r.fatal:
            all_pass = False
    if all_pass:
        print("\nAll checks passed.")
    else:
        print("\nSome checks failed.", file=sys.stderr)
    return 0 if all_pass else 1


def _cmd_eval(cfg) -> int:
    from harness.factories import create_dataset_provider, create_embedding_model
    from harness.eval.evaluator import run_evaluation

    embedder = create_embedding_model(cfg.embedding)
    provider = create_dataset_provider(cfg.evaluation)
    result = run_evaluation(embedder, provider, cfg.evaluation)

    print(f"Model: {result.model_name}")
    print(f"Elapsed: {result.elapsed_seconds:.1f}s")
    for task, score in result.scores.items():
        print(f"  {task}: {score:.4f}")
    return 0


def _cmd_run(cfg) -> int:
    from harness.pipeline import Pipeline
    pipeline = Pipeline(cfg)
    report_path = pipeline.run_full()
    print(f"Pipeline complete. Report: {report_path}")
    return 0


def _cmd_baseline(cfg) -> int:
    from harness.pipeline import Pipeline
    pipeline = Pipeline(cfg)
    report_path = pipeline.run_baseline()
    print(f"Baseline complete. Report: {report_path}")
    return 0


def _cmd_pack(cfg) -> int:
    from harness.factories import create_hpc_backend
    hpc = create_hpc_backend(cfg.hpc)
    output = Path("./dist/hpc-pack.tar.gz")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = hpc.pack(Path("."), output)
    print(f"Packed: {result}")
    return 0


def _cmd_submit(cfg) -> int:
    from harness.factories import create_hpc_backend
    hpc = create_hpc_backend(cfg.hpc)
    resources = {
        "nodes": cfg.hpc.resources.nodes,
        "ntasks_per_node": cfg.hpc.resources.ntasks_per_node,
        "gpus": cfg.hpc.resources.gpus,
        "memory": cfg.hpc.resources.memory,
        "time_limit": cfg.hpc.resources.time_limit,
        "partition": cfg.hpc.resources.partition,
    }
    job_id = hpc.submit(Path(cfg.hpc.remote_dir), resources)
    print(f"Submitted job: {job_id}")
    return 0


def _cmd_fetch(cfg) -> int:
    from harness.factories import create_hpc_backend
    hpc = create_hpc_backend(cfg.hpc)
    result = hpc.fetch("latest", Path("./results/hpc"))
    print(f"Fetched results: {result.output_path} (status: {result.status})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
