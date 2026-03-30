"""Test: Big Bang Routing — 10 Tasks → 10 Send objects.

Given: A state with exactly 10 TaskInputs
When:  route_to_workers(state) is called
Then:  Returns a list of exactly 10 Send objects
  AND: Each Send has node="worker_node"
  AND: Each Send has a dict with key "current_task"
  AND: Each current_task matches one of the original 10 TaskInputs
"""

from __future__ import annotations

from langgraph.types import Send

from liquid_swarm.models import TaskInput
from liquid_swarm.nodes import route_to_workers


class TestRouteToWorkers:
    """Unit tests for the Big Bang routing function."""

    def test_returns_exactly_n_send_objects(self, ten_tasks: list[TaskInput]):
        """10 tasks in state → exactly 10 Send objects out."""
        state = {"tasks": ten_tasks, "current_task": None, "results": []}
        result = route_to_workers(state)

        assert isinstance(result, list)
        assert len(result) == 10
        assert all(isinstance(s, Send) for s in result)

    def test_send_objects_target_worker_node(self, ten_tasks: list[TaskInput]):
        """Every Send must target 'worker_node'."""
        state = {"tasks": ten_tasks, "current_task": None, "results": []}
        result = route_to_workers(state)

        for send_obj in result:
            assert send_obj.node == "worker_node"

    def test_send_payload_contains_current_task(self, ten_tasks: list[TaskInput]):
        """Each Send payload must be a dict with 'current_task' key."""
        state = {"tasks": ten_tasks, "current_task": None, "results": []}
        result = route_to_workers(state)

        for send_obj in result:
            assert "current_task" in send_obj.arg
            assert isinstance(send_obj.arg["current_task"], TaskInput)

    def test_each_task_is_dispatched_exactly_once(self, ten_tasks: list[TaskInput]):
        """Every original task must appear exactly once in the Send payloads."""
        state = {"tasks": ten_tasks, "current_task": None, "results": []}
        result = route_to_workers(state)

        dispatched_ids = [s.arg["current_task"].task_id for s in result]
        original_ids = [t.task_id for t in ten_tasks]

        assert sorted(dispatched_ids) == sorted(original_ids)

    def test_zero_tasks_returns_empty_list(self):
        """Edge case: 0 tasks → 0 Send objects (no crash)."""
        state = {"tasks": [], "current_task": None, "results": []}
        result = route_to_workers(state)

        assert result == []

    def test_single_task_returns_single_send(self):
        """Edge case: 1 task → exactly 1 Send object."""
        task = TaskInput(task_id="solo", query="Solo analysis")
        state = {"tasks": [task], "current_task": None, "results": []}
        result = route_to_workers(state)

        assert len(result) == 1
        assert result[0].arg["current_task"].task_id == "solo"

    def test_five_hundred_tasks_scales(self, make_tasks):
        """Stress test: 500 tasks → 500 Send objects (the McKinsey killer)."""
        tasks = make_tasks(500)
        state = {"tasks": tasks, "current_task": None, "results": []}
        result = route_to_workers(state)

        assert len(result) == 500
