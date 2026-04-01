"""BDD Step Definitions for the Liquid Swarm feature scenarios."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from pytest_bdd import given, parsers, scenario, then, when

from liquid_swarm.graph import build_swarm_graph
from liquid_swarm.models import TaskInput, TaskResult


# ── Scenarios ────────────────────────────────────────────────────────────────

@scenario("../features/swarm.feature", "Full graph cycle 1 → 50 → 1")
def test_full_cycle():
    pass


@scenario(
    "../features/swarm.feature",
    "A single rogue worker does not crash the swarm",
)
def test_rogue_worker():
    pass


@scenario("../features/swarm.feature", "Timeout of a worker isolates the damage")
def test_timeout_isolation():
    pass


# ── Shared Context ──────────────────────────────────────────────────────────

class SwarmContext:
    """Mutable context object shared between BDD steps."""

    def __init__(self):
        self.tasks: list[TaskInput] = []
        self.graph = build_swarm_graph()
        self.result: dict | None = None
        self.rogue_task_id: str | None = None
        self.hanging_task_id: str | None = None


@pytest.fixture
def ctx():
    return SwarmContext()


# ── Given Steps ─────────────────────────────────────────────────────────────

@given(
    parsers.parse("a swarm with {n:d} analysis tasks"),
    target_fixture="ctx",
)
def given_swarm_with_n_tasks(n: int):
    ctx = SwarmContext()
    ctx.tasks = [
        TaskInput(task_id=f"task-{i:03d}", query=f"Analyze segment {i}")
        for i in range(n)
    ]
    return ctx


@given("all APIs are mocked with zero cost")
def given_mocked_apis(ctx: SwarmContext):
    pass


@given(
    parsers.parse(
        "worker {n:d} returns impossible data with market_share {pct:d} percent"
    ),
)
def given_rogue_worker(ctx: SwarmContext, n: int, pct: int):
    ctx.rogue_task_id = ctx.tasks[n].task_id


@given(parsers.parse("worker {n:d} hangs for {seconds:d} seconds"))
def given_hanging_worker(ctx: SwarmContext, n: int, seconds: int):
    ctx.hanging_task_id = ctx.tasks[n].task_id


# ── When Steps ──────────────────────────────────────────────────────────────

@when("the Big Bang is ignited")
def when_big_bang(ctx: SwarmContext):
    state = {
        "tasks": ctx.tasks,
        "current_task": None,
        "results": [],
        "final_results": [],
        "flagged_results": [],
        "global_context": None,
        "strategy_plan": None,
    }

    rogue_id = ctx.rogue_task_id
    hanging_id = ctx.hanging_task_id

    async def custom_execute(task, config=None, global_context="", strategy_plan="", **kwargs):
        if rogue_id and task.task_id == rogue_id:
            await asyncio.sleep(0.1)
            return TaskResult.model_construct(
                task_id=task.task_id,
                status="success",
                data={"market_share": 150.0},
                cost_usd=0.002,
            )
        elif hanging_id and task.task_id == hanging_id:
            await asyncio.sleep(999)  # Far exceeds WORKER_TIMEOUT_SECONDS (3s in tests)
            return TaskResult(
                task_id=task.task_id,
                status="success",
                data={"result": "unreachable"},
            )
        else:
            await asyncio.sleep(0.1)
            return TaskResult(
                task_id=task.task_id,
                status="success",
                data={"result": f"Completed {task.task_id}"},
                cost_usd=0.002,
            )

    async def bootstrap_mock(s):
        return {"global_context": "Mock global context for tests."}

    async def thinker_mock(s):
        return {"strategy_plan": "Mock strategy: execute all tasks with maximum precision."}

    # Apply patches SYNCHRONOUSLY before entering the event loop so they
    # remain active for the entire async execution (asyncio.run creates a
    # fresh loop, patch context is thread-local so it survives the switch).
    # WORKER_TIMEOUT_SECONDS is patched to 3s so hanging workers time out fast.
    with patch("liquid_swarm.nodes.execute_task", side_effect=custom_execute), \
         patch("liquid_swarm.nodes.WORKER_TIMEOUT_SECONDS", 3), \
         patch("liquid_swarm.graph.bootstrap_node", side_effect=bootstrap_mock), \
         patch("liquid_swarm.graph.archivar_node", side_effect=lambda s: {}), \
         patch("liquid_swarm.graph.thinker_node", side_effect=thinker_mock):
        graph = build_swarm_graph()

        async def run():
            return await graph.ainvoke(state)

        # Use new_event_loop to avoid any pytest-asyncio event loop conflicts
        loop = asyncio.new_event_loop()
        try:
            ctx.result = loop.run_until_complete(run())
        finally:
            loop.close()


# ── Then Steps ──────────────────────────────────────────────────────────────

@then(parsers.parse("exactly {n:d} parallel workers are spawned"))
def then_n_workers_spawned(ctx: SwarmContext, n: int):
    # Workers produced n results in the accumulator
    assert len(ctx.result["results"]) == n


@then(parsers.parse("all {n:d} workers return a valid TaskResult"))
def then_all_valid(ctx: SwarmContext, n: int):
    final = ctx.result["final_results"]
    assert all(isinstance(r, TaskResult) for r in final)


@then("the reduce phase aggregates to a single report")
def then_single_report(ctx: SwarmContext):
    assert "final_results" in ctx.result


@then(parsers.parse("the total cost is {cost} USD"))
def then_total_cost(ctx: SwarmContext, cost: str):
    total = sum(r.cost_usd for r in ctx.result["final_results"])
    expected = float(cost)
    assert abs(total - expected) < 0.01, f"Cost: ${total:.4f} != ${expected}"


@then(parsers.parse("the final report contains exactly {n:d} valid results"))
def then_n_valid_results(ctx: SwarmContext, n: int):
    valid = ctx.result["final_results"]
    assert len(valid) == n, f"Expected {n} valid, got {len(valid)}"


@then(parsers.parse("{n:d} result is flagged as invalid"))
def then_n_flagged(ctx: SwarmContext, n: int):
    flagged = ctx.result["flagged_results"]
    assert len(flagged) == n, f"Expected {n} flagged, got {len(flagged)}"


@then(parsers.parse("the final report contains {n:d} results"))
def then_total_results(ctx: SwarmContext, n: int):
    total = len(ctx.result["final_results"]) + len(ctx.result.get("flagged_results", []))
    assert total == n, f"Expected {n} total, got {total}"


@then(parsers.parse("worker {n:d} has status timeout"))
def then_worker_timeout(ctx: SwarmContext, n: int):
    task_id = f"task-{n:03d}"
    all_results = ctx.result["final_results"]
    matching = [r for r in all_results if r.task_id == task_id]
    assert len(matching) == 1
    assert matching[0].status == "timeout"


@then(parsers.parse("the other {n:d} workers have status success"))
def then_others_success(ctx: SwarmContext, n: int):
    success = [r for r in ctx.result["final_results"] if r.status == "success"]
    assert len(success) == n, f"Expected {n} success, got {len(success)}"
