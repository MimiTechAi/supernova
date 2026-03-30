"""Test: The Assassin Node — Red Teaming the Reduce Phase.

Given: 50 worker results, including:
       - 48 valid (market_share: 25%)
       - 1 with market_share: "150%" (mathematically impossible)
       - 1 with market_share: "-5%" (negative, impossible)
When:  The reduce_node processes these results
Then:  Valid results remain intact
  AND: Invalid results are flagged
  AND: No exception crashes the graph
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from liquid_swarm.models import TaskInput, TaskResult
from liquid_swarm.nodes import reduce_node


class TestAssassinNode:
    """Red-teaming tests for the data validation pipeline."""

    def test_valid_market_share_passes(self):
        """Normal values (0-100%) must pass validation."""
        result = TaskResult(
            task_id="valid-1",
            status="success",
            data={"market_share": 25.0},
        )
        assert result.data["market_share"] == 25.0

    def test_impossible_market_share_150_percent_raises(self):
        """150% market share is mathematically impossible."""
        with pytest.raises(ValidationError, match="mathematically impossible"):
            TaskResult(
                task_id="rogue-1",
                status="success",
                data={"market_share": 150.0},
            )

    def test_negative_market_share_raises(self):
        """-5% market share is impossible."""
        with pytest.raises(ValidationError, match="mathematically impossible"):
            TaskResult(
                task_id="rogue-2",
                status="success",
                data={"market_share": -5.0},
            )

    def test_string_percentage_150_raises(self):
        """String '150%' must also be caught."""
        with pytest.raises(ValidationError, match="mathematically impossible"):
            TaskResult(
                task_id="rogue-3",
                status="success",
                data={"market_share": "150%"},
            )

    def test_boundary_zero_passes(self):
        """0% is valid (edge case)."""
        result = TaskResult(
            task_id="edge-0",
            status="success",
            data={"market_share": 0.0},
        )
        assert result.data["market_share"] == 0.0

    def test_boundary_100_passes(self):
        """100% is valid (monopoly)."""
        result = TaskResult(
            task_id="edge-100",
            status="success",
            data={"market_share": 100.0},
        )
        assert result.data["market_share"] == 100.0

    def test_no_market_share_field_passes(self):
        """Data without market_share should pass unaffected."""
        result = TaskResult(
            task_id="normal",
            status="success",
            data={"revenue": 1000000},
        )
        assert result.data["revenue"] == 1000000


class TestReduceNodeFiltering:
    """Test the reduce_node's ability to filter assassin data."""

    def test_reduce_filters_rogue_results(self):
        """Reduce must separate valid from invalid results."""
        valid_results = [
            TaskResult(
                task_id=f"valid-{i}",
                status="success",
                data={"market_share": 25.0},
            )
            for i in range(48)
        ]

        # Rogue results: construct with valid data, then inject bad data
        rogue_1 = TaskResult.model_construct(
            task_id="rogue-150",
            status="success",
            data={"market_share": 150.0},
            cost_usd=0.002,
        )
        rogue_2 = TaskResult.model_construct(
            task_id="rogue-neg",
            status="success",
            data={"market_share": -5.0},
            cost_usd=0.002,
        )

        all_results = valid_results + [rogue_1, rogue_2]

        state = {
            "tasks": [],
            "current_task": None,
            "results": all_results,
            "final_results": [],
            "flagged_results": [],
        }

        output = reduce_node(state)
        valid_out = output["final_results"]
        flagged_out = output["flagged_results"]

        assert len(valid_out) == 48, f"Expected 48 valid, got {len(valid_out)}"
        assert len(flagged_out) == 2, f"Expected 2 flagged, got {len(flagged_out)}"

    def test_reduce_does_not_crash_on_rogue_data(self):
        """Graph must survive assassin data without exception."""
        rogue = TaskResult.model_construct(
            task_id="assassin",
            status="success",
            data={"market_share": 999.0},
            cost_usd=0.0,
        )

        state = {
            "tasks": [],
            "current_task": None,
            "results": [rogue],
            "final_results": [],
            "flagged_results": [],
        }

        # Must not raise
        output = reduce_node(state)
        assert len(output["flagged_results"]) == 1
        assert output["flagged_results"][0].status == "flagged"


@pytest.mark.asyncio
class TestAssassinEndToEnd:
    """End-to-end test: rogue worker in a live graph."""

    async def test_rogue_worker_in_live_graph(self, compiled_graph):
        """A rogue worker's output is flagged in the final results."""
        tasks = [
            TaskInput(task_id="normal-1", query="Normal analysis"),
            TaskInput(task_id="normal-2", query="Normal analysis"),
            TaskInput(task_id="rogue", query="Rogue analysis"),
        ]

        async def rogue_execute(task, config=None):
            import asyncio
            await asyncio.sleep(0.1)
            if task.task_id == "rogue":
                # Return result with invalid data by bypassing validator
                return TaskResult.model_construct(
                    task_id=task.task_id,
                    status="success",
                    data={"market_share": 150.0},
                    cost_usd=0.002,
                )
            return TaskResult(
                task_id=task.task_id,
                status="success",
                data={"market_share": 25.0},
                cost_usd=0.002,
            )

        with patch("liquid_swarm.nodes.execute_task", side_effect=rogue_execute):
            state = {
                "tasks": tasks,
                "current_task": None,
                "results": [],
                "final_results": [],
                "flagged_results": [],
            }
            result = await compiled_graph.ainvoke(state)

        flagged = result["flagged_results"]
        assert len(flagged) == 1, f"Expected 1 flagged result, got {len(flagged)}"
