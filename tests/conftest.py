"""Shared test fixtures for the Liquid Swarm test suite."""

from __future__ import annotations

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


@pytest.fixture
def compiled_graph():
    """A freshly compiled swarm graph for each test."""
    return build_swarm_graph()


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
