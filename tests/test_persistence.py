"""Tests for Persistence — Run History via JSON files.

TDD: These tests were written BEFORE the implementation.

Given/When/Then Rules:
  - GIVEN completed results → WHEN saved → THEN JSON file created
  - GIVEN 3 saved runs → WHEN listed → THEN sorted newest-first
  - GIVEN a saved run ID → WHEN retrieved → THEN full data returned
  - GIVEN a bad run ID → WHEN retrieved → THEN returns None
"""

import shutil
from pathlib import Path

import pytest

from liquid_swarm.models import TaskResult
from liquid_swarm.persistence import save_run, list_runs, get_run


@pytest.fixture
def tmp_runs_dir(tmp_path):
    """Create a temporary runs directory."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    return runs_dir


@pytest.fixture
def sample_results():
    return [
        TaskResult(
            task_id="worker-000",
            status="success",
            data={"result": "AI market is $200B in 2025."},
            cost_usd=0.0003,
        ),
        TaskResult(
            task_id="worker-001",
            status="success",
            data={"result": "Top players: OpenAI, Google, Meta."},
            cost_usd=0.0003,
        ),
    ]


class TestSaveRun:
    """GIVEN a completed swarm run with results
    WHEN the run is saved
    THEN a JSON file is created in the runs/ directory"""

    def test_save_creates_json_file(self, tmp_runs_dir, sample_results):
        run_id = save_run(
            query="Analyze AI market",
            results=sample_results,
            total_cost=0.0006,
            total_time=5.2,
            model="meta/llama-3.1-8b-instruct",
            runs_dir=tmp_runs_dir,
        )

        assert run_id is not None
        json_files = list(tmp_runs_dir.glob("*.json"))
        assert len(json_files) == 1
        assert run_id in json_files[0].name


class TestListRuns:
    """GIVEN 3 saved runs exist
    WHEN list_runs is called
    THEN all 3 runs are returned sorted by timestamp (newest first)"""

    def test_list_returns_sorted(self, tmp_runs_dir, sample_results):
        import time

        ids = []
        for i in range(3):
            rid = save_run(
                query=f"Query {i}",
                results=sample_results,
                total_cost=0.001 * i,
                total_time=float(i),
                model="test-model",
                runs_dir=tmp_runs_dir,
            )
            ids.append(rid)
            time.sleep(0.05)  # Ensure distinct timestamps

        runs = list_runs(runs_dir=tmp_runs_dir)

        assert len(runs) == 3
        # Newest first
        assert runs[0]["run_id"] == ids[-1]
        assert runs[-1]["run_id"] == ids[0]


class TestGetRun:
    """GIVEN a saved run with ID 'run-abc123'
    WHEN GET is called with that ID
    THEN the full run data is returned"""

    def test_get_returns_full_data(self, tmp_runs_dir, sample_results):
        run_id = save_run(
            query="Test query",
            results=sample_results,
            total_cost=0.0006,
            total_time=3.5,
            model="test-model",
            runs_dir=tmp_runs_dir,
        )

        run = get_run(run_id, runs_dir=tmp_runs_dir)

        assert run is not None
        assert run["run_id"] == run_id
        assert run["query"] == "Test query"
        assert run["total_cost"] == 0.0006
        assert run["total_time"] == 3.5
        assert len(run["results"]) == 2


class TestGetRunNotFound:
    """GIVEN a non-existent run ID
    WHEN get_run is called
    THEN it returns None"""

    def test_nonexistent_run_returns_none(self, tmp_runs_dir):
        result = get_run("nonexistent-id-999", runs_dir=tmp_runs_dir)
        assert result is None
