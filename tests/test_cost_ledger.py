"""Test: Cost Ledger Auditing — assert total_cost_usd <= 0.50.

Given: 50 tasks, each worker returns cost_usd=0.002 (mocked)
When:  The graph completes a full cycle
Then:  sum(r.cost_usd for r in final_results) == 0.10
  AND: assert total_cost <= 0.50

NOTE: execute_task is mocked for deterministic cost assertions.
Real API cost tracking is verified in test_live_nvidia.py.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from liquid_swarm.models import TaskInput, TaskResult


async def _cost_tracking_execute(task, config=None):
    """Mock with fixed cost for deterministic assertions."""
    import asyncio
    await asyncio.sleep(0.05)
    return TaskResult(
        task_id=task.task_id,
        status="success",
        data={"result": f"Cost test for {task.task_id}"},
        cost_usd=0.002,
    )


@pytest.mark.asyncio
class TestCostLedgerAuditing:
    """Prove that costs are tracked and bounded."""

    async def test_total_cost_under_fifty_cents(
        self, compiled_graph, fifty_tasks: list[TaskInput],
    ):
        """50 workers at $0.002 each = $0.10 total. Must be <= $0.50."""
        state = {
            "tasks": fifty_tasks,
            "current_task": None,
            "results": [],
            "final_results": [],
            "flagged_results": [],
        }
        with patch("liquid_swarm.nodes.execute_task", side_effect=_cost_tracking_execute):
            result = await compiled_graph.ainvoke(state)
        final = result["final_results"]

        total_cost = sum(r.cost_usd for r in final)

        assert total_cost <= 0.50, (
            f"BUDGET EXCEEDED! Total cost: ${total_cost:.4f} > $0.50. "
            f"Worker prompts are too verbose — optimize or degrade model."
        )

    async def test_cost_per_worker_is_tracked(
        self, compiled_graph, ten_tasks: list[TaskInput],
    ):
        """Every worker must report its cost."""
        state = {
            "tasks": ten_tasks,
            "current_task": None,
            "results": [],
            "final_results": [],
            "flagged_results": [],
        }
        with patch("liquid_swarm.nodes.execute_task", side_effect=_cost_tracking_execute):
            result = await compiled_graph.ainvoke(state)
        final = result["final_results"]

        for r in final:
            assert r.cost_usd >= 0.0, f"Negative cost for {r.task_id}"
            assert isinstance(r.cost_usd, float)

    async def test_exact_cost_calculation(
        self, compiled_graph, fifty_tasks: list[TaskInput],
    ):
        """50 workers × $0.002 = $0.10 exactly."""
        state = {
            "tasks": fifty_tasks,
            "current_task": None,
            "results": [],
            "final_results": [],
            "flagged_results": [],
        }
        with patch("liquid_swarm.nodes.execute_task", side_effect=_cost_tracking_execute):
            result = await compiled_graph.ainvoke(state)
        final = result["final_results"]

        total_cost = sum(r.cost_usd for r in final)
        expected_cost = 50 * 0.002  # $0.10

        assert abs(total_cost - expected_cost) < 0.001, (
            f"Cost mismatch: ${total_cost:.4f} != ${expected_cost:.4f}"
        )
