"""Integration Test: Real NVIDIA NIM API Calls — NO MOCKS.

These tests call the actual NVIDIA NIM API.
They prove the entire pipeline works end-to-end:
  TaskInput → Graph → NVIDIA API → TaskResult → Reduce → FinalResults

Prerequisite: NVIDIA_API_KEY must be set in .env or as an environment variable.
"""

from __future__ import annotations

import os
import time

import pytest

from liquid_swarm.config import NVIDIA_API_KEY, SwarmConfig, ModelTier
from liquid_swarm.graph import build_swarm_graph
from liquid_swarm.models import TaskInput, TaskResult
from liquid_swarm.nodes import execute_task, set_api_semaphore


# Skip all tests in this module if no API key is configured
pytestmark = pytest.mark.skipif(
    not NVIDIA_API_KEY,
    reason="NVIDIA_API_KEY not set — skipping live API tests",
)


@pytest.mark.asyncio
class TestLiveNvidiaAPI:
    """Real API calls against NVIDIA NIM. Zero mocks."""

    async def test_single_task_returns_real_llm_output(self):
        """A single task receives a real LLM response."""
        task = TaskInput(
            task_id="live-001",
            query="What is Tesla's current market share in the European electric vehicle market? Answer in one sentence.",
        )

        result = await execute_task(task)

        assert isinstance(result, TaskResult)
        assert result.status == "success"
        assert result.task_id == "live-001"
        assert len(result.data["result"]) > 20, (
            f"LLM output too short: '{result.data['result']}'"
        )
        assert result.data["prompt_tokens"] > 0
        assert result.data["completion_tokens"] > 0
        assert result.data["latency_seconds"] > 0
        assert result.cost_usd > 0

        print(f"\n{'='*60}")
        print(f"  LIVE LLM OUTPUT (single task)")
        print(f"  Model: {result.data['model']}")
        print(f"  Tokens: {result.data['prompt_tokens']} in / {result.data['completion_tokens']} out")
        print(f"  Latency: {result.data['latency_seconds']}s")
        print(f"  Cost: ${result.cost_usd:.4f}")
        print(f"  Response: {result.data['result'][:200]}")
        print(f"{'='*60}")

    async def test_three_tasks_parallel_real_api(self):
        """3 tasks in parallel against the real API — full graph cycle."""
        set_api_semaphore(5)

        graph = build_swarm_graph()
        tasks = [
            TaskInput(task_id="real-001", query="Name the top 3 cloud providers worldwide by market share. Numbers only."),
            TaskInput(task_id="real-002", query="What does a GPU server cost per hour on AWS on average? One sentence."),
            TaskInput(task_id="real-003", query="How large is the global AI market in USD in 2025? One sentence."),
        ]

        state = {
            "tasks": tasks,
            "current_task": None,
            "results": [],
            "final_results": [],
            "flagged_results": [],
        }

        t0 = time.perf_counter()
        result = await graph.ainvoke(state)
        elapsed = time.perf_counter() - t0

        final = result["final_results"]
        flagged = result["flagged_results"]

        assert len(final) == 3, f"Expected 3 results, got {len(final)}"
        assert len(flagged) == 0, f"Unexpected flagged results: {flagged}"
        assert all(r.status == "success" for r in final)

        total_cost = sum(r.cost_usd for r in final)

        print(f"\n{'='*60}")
        print(f"  LIVE GRAPH CYCLE — 3 real tasks parallel")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"{'='*60}")
        for r in final:
            print(f"\n  [{r.task_id}]")
            print(f"  Model: {r.data.get('model', 'n/a')}")
            print(f"  Latency: {r.data.get('latency_seconds', 'n/a')}s")
            print(f"  Output: {r.data['result'][:150]}...")
        print(f"{'='*60}")

        # Parallel: 3 tasks should complete in < time of 3 sequential calls
        assert elapsed < 15.0, f"Too slow: {elapsed:.1f}s — sequential execution?"

    async def test_budget_vs_standard_model(self):
        """A/B Test: Budget (8B) vs Standard (70B) — real quality differences."""
        task = TaskInput(
            task_id="ab-test",
            query="Explain the difference between GPU and TPU in one sentence.",
        )

        budget_config = SwarmConfig(model_tier=ModelTier.BUDGET, max_tokens=100)
        standard_config = SwarmConfig(model_tier=ModelTier.STANDARD, max_tokens=100)

        result_budget = await execute_task(task, config=budget_config)
        result_standard = await execute_task(task, config=standard_config)

        assert result_budget.status == "success"
        assert result_standard.status == "success"

        # Both should produce real output
        assert len(result_budget.data["result"]) > 10
        assert len(result_standard.data["result"]) > 10

        # Cost should differ
        assert result_budget.cost_usd < result_standard.cost_usd

        print(f"\n{'='*60}")
        print(f"  A/B MODEL COMPARISON")
        print(f"  Budget  ({budget_config.model_id}):")
        print(f"    Cost: ${result_budget.cost_usd:.4f}")
        print(f"    Latency: {result_budget.data['latency_seconds']}s")
        print(f"    Output: {result_budget.data['result'][:150]}")
        print(f"  Standard ({standard_config.model_id}):")
        print(f"    Cost: ${result_standard.cost_usd:.4f}")
        print(f"    Latency: {result_standard.data['latency_seconds']}s")
        print(f"    Output: {result_standard.data['result'][:150]}")
        print(f"{'='*60}")

    async def test_ten_tasks_swarm_live(self):
        """10 real tasks — the mini-swarm against NVIDIA NIM."""
        set_api_semaphore(5)  # 5 concurrent to be safe with rate limits

        graph = build_swarm_graph()

        markets = [
            "Cloud Computing", "Artificial Intelligence", "Cybersecurity",
            "Blockchain", "Edge Computing", "Quantum Computing",
            "Robotics", "5G Infrastructure", "Digital Health", "FinTech",
        ]

        tasks = [
            TaskInput(
                task_id=f"swarm-{i:03d}",
                query=f"How large is the global {market} market in USD in 2025? Answer in one sentence with a concrete number.",
            )
            for i, market in enumerate(markets)
        ]

        state = {
            "tasks": tasks,
            "current_task": None,
            "results": [],
            "final_results": [],
            "flagged_results": [],
        }

        t0 = time.perf_counter()
        result = await graph.ainvoke(state)
        elapsed = time.perf_counter() - t0

        final = result["final_results"]
        flagged = result["flagged_results"]

        success = [r for r in final if r.status == "success"]
        failed = [r for r in final if r.status != "success"]

        total_cost = sum(r.cost_usd for r in final)

        print(f"\n{'='*60}")
        print(f"  🚀 LIVE SWARM — 10 real tasks against NVIDIA NIM")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Success: {len(success)}/{len(final)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Flagged: {len(flagged)}")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"{'='*60}")
        for r in success:
            print(f"  [{r.task_id}] {r.data['result'][:120]}")
        print(f"{'='*60}")

        # At least 8 of 10 should succeed (allow for transient API errors)
        assert len(success) >= 8, (
            f"Too many failures: {len(success)}/10 succeeded. "
            f"Failures: {[f.data for f in failed]}"
        )
        assert total_cost < 1.0, f"Cost too high: ${total_cost:.4f}"
