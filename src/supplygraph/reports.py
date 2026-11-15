"""Write scenario-level exposure metrics and review queues."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_reports(
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "evaluation.json"
    exposures_path = output_dir / "scenario_exposure.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    fields = list(rows[0]) if rows else []
    with exposures_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    return metrics_path, exposures_path
