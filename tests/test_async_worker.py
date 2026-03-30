"""Test: Parallel Async Execution — 50 tasks in ~seconds, not minutes.

Given: A compiled graph and 50 TaskInputs
When:  await graph.ainvoke({"tasks": tasks_50}) is called
Then:  Total elapsed time is < 10 seconds (NOT 50 seconds)
  AND: final_results contains exactly 50 TaskResult objects
  AND: All have status="success"

NOTE: execute_task is mocked here to test ARCHITECTURE (parallelism),
not the API. Real API tests are in test_live_nvidia.py.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from liquid_swarm.models import TaskInput, TaskResult


async def _fast_fake_execute(task, config=None):
    """Deterministic mock: 0.1s sleep, always succeeds."""
    import asyncio
    await asyncio.sleep(0.1)
    return TaskResult(
        task_id=task.task_id,
        status="success",
        data={"result": f"Mock result for {task.task_id}"},
        cost_usd=0.002,
    )


@pytest.mark.asyncio
class TestAsyncParallelExecution:
    """Prove that workers execute in parallel, not sequentially."""

    async def test_fifty_tasks_run_in_parallel(
        self, compiled_graph, fifty_tasks: list[TaskInput],
    ):
        """50 tasks with 0.1s mock each must complete in << 50 seconds."""
        state = {
            "tasks": fifty_tasks,
            "current_task": None,
            "results": [],
            "final_results": [],
            "flagged_results": [],
        }

        with patch("liquid_swarm.nodes.execute_task", side_effect=_fast_fake_execute):
            start = time.perf_counter()
            result = await compiled_graph.ainvoke(state)
            elapsed = time.perf_counter() - start

        # Must be parallel: << 5s (50 × 0.1s sequential = 5s)
        assert elapsed < 5.0, (
            f"Execution took {elapsed:.1f}s — sequential execution detected! "
            f"Expected < 5s for 50 parallel tasks."
        )

        final = result["final_results"]
        assert len(final) == 50
        assert all(isinstance(r, TaskResult) for r in final)
        assert all(r.status == "success" for r in final)

    async def test_ten_tasks_run_fast(
        self, compiled_graph, ten_tasks: list[TaskInput],
    ):
        """10 tasks should complete in ~0.1-0.5 seconds."""
        state = {
            "tasks": ten_tasks,
            "current_task": None,
            "results": [],
            "final_results": [],
            "flagged_results": [],
        }

        with patch("liquid_swarm.nodes.execute_task", side_effect=_fast_fake_execute):
            start = time.perf_counter()
            result = await compiled_graph.ainvoke(state)
            elapsed = time.perf_counter() - start

        assert elapsed < 3.0, (
            f"10 tasks took {elapsed:.1f}s — expected < 3s"
        )
        assert len(result["final_results"]) == 10

    async def test_results_contain_correct_task_ids(
        self, compiled_graph, fifty_tasks: list[TaskInput],
    ):
        """Every input task must produce a result with the same task_id."""
        state = {
            "tasks": fifty_tasks,
            "current_task": None,
            "results": [],
            "final_results": [],
            "flagged_results": [],
        }

        with patch("liquid_swarm.nodes.execute_task", side_effect=_fast_fake_execute):
            result = await compiled_graph.ainvoke(state)

        result_ids = sorted(r.task_id for r in result["final_results"])
        input_ids = sorted(t.task_id for t in fifty_tasks)

        assert result_ids == input_ids
