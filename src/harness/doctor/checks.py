from __future__ import annotations

from harness.config import HarnessConfig
from harness.factories import (
    create_dataset_provider,
    create_embedding_model,
    create_hpc_backend,
    create_vector_store,
)
from harness.interfaces import CheckResult


def run_all_checks(cfg: HarnessConfig) -> list[CheckResult]:
    results: list[CheckResult] = []
    results.append(_check_container(cfg))
    results.append(_check_vector_store(cfg))
    results.append(_check_hpc(cfg))
    results.append(_check_datasets(cfg))
    return results


def _check_container(cfg: HarnessConfig) -> CheckResult:
    rt = cfg.container.runtime
    if rt == "none":
        return CheckResult("container_runtime", True, "Container runtime disabled")
    image = cfg.container.image_path
    if image.exists():
        return CheckResult("container_image", True, f"Found: {image}")
    return CheckResult(
        "container_image",
        False,
        f"Image not found: {image} (runtime={rt})",
        fatal=False,
    )


def _check_vector_store(cfg: HarnessConfig) -> CheckResult:
    try:
        store = create_vector_store(cfg.vector_store)
        store.connect()
        store.close()
        return CheckResult("vector_store", True, f"Connected ({cfg.vector_store.type})")
    except Exception as e:
        return CheckResult("vector_store", False, f"Connection failed: {e}", fatal=True)


def _check_hpc(cfg: HarnessConfig) -> CheckResult:
    try:
        hpc = create_hpc_backend(cfg.hpc)
        ok = hpc.check_connectivity()
        if ok:
            return CheckResult("hpc_connectivity", True, f"Backend: {cfg.hpc.backend}")
        return CheckResult("hpc_connectivity", False, "Connectivity check returned False", fatal=False)
    except Exception as e:
        return CheckResult("hpc_connectivity", False, f"Error: {e}", fatal=False)


def _check_datasets(cfg: HarnessConfig) -> CheckResult:
    try:
        provider = create_dataset_provider(cfg.evaluation)
        available = provider.list_available()
        requested = [t.name for t in cfg.evaluation.tasks]
        missing = [t for t in requested if t not in available]
        if missing:
            return CheckResult(
                "datasets",
                False,
                f"Missing tasks: {missing}. Available: {available}",
                fatal=False,
            )
        return CheckResult(
            "datasets",
            True,
            f"All {len(requested)} tasks available ({cfg.evaluation.provider})",
        )
    except Exception as e:
        return CheckResult("datasets", False, f"Error: {e}", fatal=True)
