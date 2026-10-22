from __future__ import annotations

from pathlib import Path

import pytest

from harness.config import load_config
from harness.doctor.checks import run_all_checks, CheckResult


@pytest.fixture
def cfg() -> HarnessConfig:
    return load_config(Path("config/harness.yaml"))


def test_doctor_returns_check_results(cfg) -> None:
    results = run_all_checks(cfg)
    assert len(results) >= 4
    assert all(isinstance(r, CheckResult) for r in results)


def test_doctor_vector_store_passes(cfg) -> None:
    results = run_all_checks(cfg)
    vs_result = next(r for r in results if r.name == "vector_store")
    assert vs_result.passed is True


def test_doctor_hpc_passes(cfg) -> None:
    results = run_all_checks(cfg)
    hpc_result = next(r for r in results if r.name == "hpc_connectivity")
    assert hpc_result.passed is True


def test_doctor_datasets_passes(cfg) -> None:
    results = run_all_checks(cfg)
    ds_result = next(r for r in results if r.name == "datasets")
    assert ds_result.passed is True


def test_doctor_container_nonfatal(cfg) -> None:
    results = run_all_checks(cfg)
    ct_result = next(r for r in results if "container" in r.name)
    assert ct_result.fatal is False
