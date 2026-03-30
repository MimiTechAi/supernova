"""Test: Timeout & Hard-Kill — 1 dead worker, 49 survivors.

Given: A worker with an execute_task that sleeps 30s (> 15s timeout)
  AND: 49 other normal workers
When:  await graph.ainvoke({"tasks": tasks_50}) is called
Then:  final_results + flagged_results = 50 entries total
  AND: Exactly 1 entry has status="timeout"
  AND: The other 49 have status="success"
  AND: No asyncio.TimeoutError propagates upward
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from liquid_swarm.models import TaskInput, TaskResult


@pytest.mark.asyncio
class TestTimeoutAndHardKill:
    """Prove that a timed-out worker doesn't kill its siblings."""

    async def test_one_timeout_forty_nine_survivors(
        self, compiled_graph, fifty_tasks: list[TaskInput],
    ):
        """1 worker times out, 49 succeed. Graph stays alive."""
        doomed_task_id = fifty_tasks[5].task_id  # Worker #5 will hang

        async def slow_or_normal(task, config=None):
            """Mock: task #5 sleeps 30s (will timeout), others sleep 0.1s."""
            import asyncio
            if task.task_id == doomed_task_id:
                await asyncio.sleep(30)  # Will be killed by wait_for
                return TaskResult(
                    task_id=task.task_id,
                    status="success",
                    data={"result": "should not appear"},
                )
            else:
                await asyncio.sleep(0.1)  # Fast completion
                return TaskResult(
                    task_id=task.task_id,
                    status="success",
                    data={"result": f"Completed {task.task_id}"},
                    cost_usd=0.002,
                )

        with patch("liquid_swarm.nodes.execute_task", side_effect=slow_or_normal):
            state = {
                "tasks": fifty_tasks,
                "current_task": None,
                "results": [],
                "final_results": [],
                "flagged_results": [],
            }
            result = await compiled_graph.ainvoke(state)

        # Raw results from workers (before reduce filtering)
        raw_results = result["results"]
        assert len(raw_results) == 50, f"Expected 50 raw results, got {len(raw_results)}"

        # Check final + flagged
        final = result["final_results"]
        timeout_results = [r for r in final if r.status == "timeout"]
        success_results = [r for r in final if r.status == "success"]

        assert len(timeout_results) == 1, (
            f"Expected exactly 1 timeout, got {len(timeout_results)}"
        )
        assert timeout_results[0].task_id == doomed_task_id
        assert len(success_results) == 49

    async def test_timeout_returns_fallback_pydantic(
        self, compiled_graph,
    ):
        """A timed-out worker must return a valid TaskResult, not crash."""
        doomed_task = TaskInput(task_id="slow-one", query="Will timeout")

        async def always_hang(task, config=None):
            import asyncio
            await asyncio.sleep(999)

        with patch("liquid_swarm.nodes.execute_task", side_effect=always_hang):
            state = {
                "tasks": [doomed_task],
                "current_task": None,
                "results": [],
                "final_results": [],
                "flagged_results": [],
            }
            # Must NOT raise — timeout is caught internally
            result = await compiled_graph.ainvoke(state)

        final = result["final_results"]
        assert len(final) == 1
        assert final[0].status == "timeout"
        assert isinstance(final[0], TaskResult)
        assert "timed out" in final[0].data.get("error", "").lower()

    async def test_multiple_timeouts_dont_crash_graph(
        self, compiled_graph, ten_tasks: list[TaskInput],
    ):
        """Even if ALL workers timeout, graph completes without exception."""
        async def all_hang(task, config=None):
            import asyncio
            await asyncio.sleep(999)

        with patch("liquid_swarm.nodes.execute_task", side_effect=all_hang):
            state = {
                "tasks": ten_tasks,
                "current_task": None,
                "results": [],
                "final_results": [],
                "flagged_results": [],
            }
            result = await compiled_graph.ainvoke(state)

        final = result["final_results"]
        assert len(final) == 10
        assert all(r.status == "timeout" for r in final)
