"""Graph State definition using TypedDict with Reducers.

LangGraph requires TypedDict (not Pydantic BaseModel) for states that use
reducer functions like operator.add for fan-in aggregation. Pydantic models
would cause InvalidUpdateError during parallel fan-in.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from liquid_swarm.models import TaskInput, TaskResult


class SwarmState(TypedDict):
    """Strictly typed graph state. No Dict[str, Any] permitted.

    Attributes:
        tasks: The N input tasks for the Big Bang fan-out.
        current_task: Set per-worker via Send() payload. Each worker
                      receives its own task through this key.
        results: Reducer-backed list that automatically appends results
                 from parallel workers during fan-in. Uses operator.add
                 so that concurrent state updates merge safely.
        final_results: Non-reducer output from reduce_node (valid results).
        flagged_results: Non-reducer output from reduce_node (invalid results).
    """

    tasks: list[TaskInput]
    current_task: TaskInput | None
    results: Annotated[list[TaskResult], operator.add]
    final_results: list[TaskResult]
    flagged_results: list[TaskResult]
    global_context: str | None
    strategy_plan: str | None


