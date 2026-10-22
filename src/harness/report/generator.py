from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from harness.config import ReportConfig
from harness.interfaces import Report


def generate_report(report: Report, cfg: ReportConfig) -> list[Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths: list[Path] = []

    for fmt in cfg.formats:
        if fmt == "text":
            paths.append(_write_text(report, cfg.output_dir, ts))
        elif fmt == "json":
            paths.append(_write_json(report, cfg.output_dir, ts))

    return paths


def _write_text(report: Report, output_dir: Path, ts: str) -> Path:
    path = output_dir / f"report_{ts}.txt"
    lines = [
        "=" * 60,
        "Model Training Evaluation Report",
        "=" * 60,
        f"Run ID:    {report.run_id}",
        f"Timestamp: {report.timestamp}",
        "",
        "── Model-side Scores ──────────────────────────────",
    ]
    for s in report.model_scores:
        lines.append(f"  {s.task:<40s} {s.score:.4f}")

    lines.append("")
    lines.append("── Storage-side Scores ────────────────────────────")
    for s in report.storage_scores:
        lines.append(f"  {s.metric:<40s} {s.score:.4f}")

    lines.append("")
    lines.append("=" * 60)

    path.write_text("\n".join(lines))
    return path


def _write_json(report: Report, output_dir: Path, ts: str) -> Path:
    path = output_dir / f"report_{ts}.json"
    data = {
        "run_id": report.run_id,
        "timestamp": report.timestamp,
        "model_scores": [
            {"task": s.task, "score": s.score, "details": s.details}
            for s in report.model_scores
        ],
        "storage_scores": [
            {"metric": s.metric, "score": s.score, "details": s.details}
            for s in report.storage_scores
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path
