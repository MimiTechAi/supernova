"""Test: Rate-Limit Protection — max 10 concurrent workers.

Given: 50 tasks and API semaphore set to 10
When:  The graph is executed
Then:  At no point do more than 10 workers run simultaneously
  AND: Proof via an atomic concurrent counter in the mock

Architecture: Uses asyncio.Semaphore in the worker_node to gate
execute_task calls. LangGraph's Send-based fan-out dispatches all
workers in one superstep, but the semaphore limits actual API concurrency.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from liquid_swarm.models import TaskInput, TaskResult
from liquid_swarm.nodes import set_api_semaphore


@pytest.mark.asyncio
class TestRateLimitProtection:
    """Prove that the API semaphore limits parallel execution."""

    async def test_max_ten_concurrent_workers(
        self, compiled_graph, fifty_tasks: list[TaskInput],
    ):
        """At most 10 workers hit the API at the same time."""
        # Configure the semaphore to 10
        set_api_semaphore(10)

        current_concurrent = 0
        max_concurrent_seen = 0
        lock = asyncio.Lock()

        async def counting_execute(task, config=None, **kwargs):
            nonlocal current_concurrent, max_concurrent_seen

            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent_seen:
                    max_concurrent_seen = current_concurrent

            await asyncio.sleep(0.5)

            async with lock:
                current_concurrent -= 1

            return TaskResult(
                task_id=task.task_id,
                status="success",
                data={"result": "rate limit test"},
                cost_usd=0.001,
            )

        with patch("liquid_swarm.nodes.execute_task", side_effect=counting_execute):
            state = {
                "tasks": fifty_tasks,
                "current_task": None,
                "results": [],
                "final_results": [],
                "flagged_results": [],
                "global_context": None,
                "strategy_plan": None,
            }
            result = await compiled_graph.ainvoke(state)

        assert len(result["final_results"]) == 50
        assert max_concurrent_seen <= 10, (
            f"Rate limit breached! Saw {max_concurrent_seen} concurrent workers, "
            f"expected max 10."
        )
        assert max_concurrent_seen > 1, (
            f"Only {max_concurrent_seen} concurrent worker seen — no parallelism?"
        )

        # Reset semaphore to default for other tests
        set_api_semaphore(10)
