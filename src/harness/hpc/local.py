from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from harness.config import HpcConfig
from harness.factories import register_hpc_backend
from harness.interfaces import HpcResult


@register_hpc_backend("local")
class LocalHPCBackend:
    def __init__(self, cfg: HpcConfig) -> None:
        self._cfg = cfg
        self._work_dir = Path(cfg.remote_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)

    def check_connectivity(self) -> bool:
        return True

    def pack(self, local_dir: Path, output: Path) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz") as tar:
            manifest = {
                "backend": "local",
                "source": str(local_dir),
                "resources": {
                    "nodes": self._cfg.resources.nodes,
                    "gpus": self._cfg.resources.gpus,
                },
            }
            manifest_path = local_dir / ".harness_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))
            tar.add(manifest_path, arcname=".harness_manifest.json")
            manifest_path.unlink(missing_ok=True)
        return output

    def upload(self, local_path: Path, remote_path: Path) -> None:
        shutil.copy2(local_path, remote_path)

    def submit(self, remote_path: Path, resources: dict) -> str:
        return "local-job-001"

    def status(self, job_id: str) -> str:
        return "completed"

    def fetch(self, job_id: str, local_dir: Path) -> Path:
        local_dir.mkdir(parents=True, exist_ok=True)
        result_file = local_dir / "result.json"
        result_file.write_text(json.dumps({
            "job_id": job_id,
            "status": "completed",
            "metrics": {"loss": 0.123, "accuracy": 0.956},
        }, indent=2))
        return result_file

    def cancel(self, job_id: str) -> None:
        pass
