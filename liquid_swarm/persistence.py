"""Persistence: Save and retrieve swarm run history as JSON files.

Design decisions:
  - JSON files in runs/ directory — zero database dependencies
  - Each run is one file: {run_id}.json
  - Run IDs are timestamp-prefixed for natural sort order
  - Fully portable: no external services, works in Docker, serverless, local
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from liquid_swarm.models import TaskResult


# Default runs directory (relative to project root)
_DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def save_run(
    query: str,
    results: list[TaskResult],
    total_cost: float,
    total_time: float,
    model: str,
    synthesis: str | None = None,
    runs_dir: Path | None = None,
) -> str:
    """Save a completed swarm run to a JSON file.

    Args:
        query: The original user query.
        results: List of worker TaskResults.
        total_cost: Total cost in USD.
        total_time: Total execution time in seconds.
        model: Model ID used for the run.
        synthesis: Optional executive summary text.
        runs_dir: Override directory for storing run files.

    Returns:
        The unique run_id string.
    """
    runs_path = runs_dir or _DEFAULT_RUNS_DIR
    runs_path.mkdir(parents=True, exist_ok=True)

    run_id = f"run-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"

    run_data = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "model": model,
        "total_cost": total_cost,
        "total_time": total_time,
        "worker_count": len(results),
        "success_count": sum(1 for r in results if r.status == "success"),
        "synthesis": synthesis,
        "results": [r.model_dump() for r in results],
    }

    filepath = runs_path / f"{run_id}.json"
    filepath.write_text(json.dumps(run_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return run_id


def list_runs(
    runs_dir: Path | None = None,
    limit: int = 50,
) -> list[dict]:
    """List all saved runs, sorted by timestamp (newest first).

    Returns lightweight metadata — no full results included.

    Args:
        runs_dir: Override directory.
        limit: Maximum number of runs to return.

    Returns:
        List of run metadata dicts.
    """
    runs_path = runs_dir or _DEFAULT_RUNS_DIR
    if not runs_path.exists():
        return []

    runs = []
    for filepath in sorted(runs_path.glob("*.json"), reverse=True):
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            runs.append({
                "run_id": data["run_id"],
                "timestamp": data["timestamp"],
                "query": data["query"],
                "model": data.get("model", "unknown"),
                "total_cost": data["total_cost"],
                "total_time": data["total_time"],
                "worker_count": data.get("worker_count", 0),
                "success_count": data.get("success_count", 0),
            })
        except (json.JSONDecodeError, KeyError):
            continue

        if len(runs) >= limit:
            break

    return runs


def get_run(
    run_id: str,
    runs_dir: Path | None = None,
) -> dict | None:
    """Retrieve a specific run by its ID.

    Args:
        run_id: The unique run identifier.
        runs_dir: Override directory.

    Returns:
        The full run data dict, or None if not found.
    """
    runs_path = runs_dir or _DEFAULT_RUNS_DIR
    filepath = runs_path / f"{run_id}.json"

    if not filepath.exists():
        return None

    try:
        return json.loads(filepath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError):
        return None
