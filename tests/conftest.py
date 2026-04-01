"""Shared test fixtures for the Liquid Swarm test suite."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from liquid_swarm.graph import build_swarm_graph
from liquid_swarm.models import TaskInput
from liquid_swarm.nodes import set_api_semaphore


@pytest.fixture(autouse=True)
def reset_semaphore():
    """Reset the API semaphore before each test to prevent leaking state."""
    set_api_semaphore(50)  # High default for tests — no rate limiting unless explicit
    yield
    set_api_semaphore(50)


async def _mock_bootstrap(state):
    """Fast bootstrap mock — no ChromaDB or embedding calls in unit tests."""
    return {"global_context": "Test context: no prior knowledge needed."}


async def _mock_thinker(state):
    """Fast thinker mock — no LLM call in unit tests."""
    return {"strategy_plan": "Test strategy: execute all tasks precisely."}


async def _mock_archivar(state):
    """Fast archivar mock — no ChromaDB writes in unit tests."""
    return {}


@pytest.fixture
def compiled_graph():
    """A freshly compiled swarm graph with bootstrap/thinker/archivar mocked.

    Unit tests should only test worker execution and graph routing, not
    LLM calls in bootstrap/thinker which require real API keys.
    """
    with patch("liquid_swarm.graph.bootstrap_node", side_effect=_mock_bootstrap), \
         patch("liquid_swarm.graph.thinker_node", side_effect=_mock_thinker), \
         patch("liquid_swarm.graph.archivar_node", side_effect=_mock_archivar):
        yield build_swarm_graph()


def make_state(tasks: list[TaskInput]) -> dict:
    """Create a properly typed SwarmState dict for tests."""
    return {
        "tasks": tasks,
        "current_task": None,
        "results": [],
        "final_results": [],
        "flagged_results": [],
        "global_context": None,
        "strategy_plan": None,
    }


@pytest.fixture
def make_tasks():
    """Factory fixture: create N TaskInput objects."""
    def _make(n: int) -> list[TaskInput]:
        return [
            TaskInput(
                task_id=f"task-{i:03d}",
                query=f"Analyze market segment {i}",
            )
            for i in range(n)
        ]
    return _make


@pytest.fixture
def ten_tasks(make_tasks) -> list[TaskInput]:
    """10 pre-built TaskInput objects."""
    return make_tasks(10)


@pytest.fixture
def fifty_tasks(make_tasks) -> list[TaskInput]:
    """50 pre-built TaskInput objects."""
    return make_tasks(50)
