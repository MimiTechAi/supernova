"""Strict Pydantic models for the Liquid Swarm data pipeline.

Every data flow in the system is typed via these models.
No Dict[str, Any] is permitted anywhere in the codebase.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TaskInput(BaseModel):
    """A single analysis task dispatched to a micro-agent (drone)."""

    task_id: str = Field(description="Unique identifier for this sub-task")
    query: str = Field(
        description="The exact analysis question for this micro-agent. "
        "Example: 'Analyze Tesla's market share in Europe Q4 2025.'"
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        description="LLM model to use. Cheap default for cost control.",
    )


class TaskResult(BaseModel):
    """Result returned by a single worker drone."""

    task_id: str
    status: str = Field(
        description="One of: 'success', 'timeout', 'error', 'flagged'"
    )
    data: dict[str, object] = Field(
        default_factory=dict,
        description="Structured analysis output from the worker.",
    )
    cost_usd: float = Field(
        default=0.0,
        description="Cost in USD for this worker's LLM calls.",
    )

    @field_validator("data")
    @classmethod
    def validate_no_impossible_values(
        cls, v: dict[str, object],
    ) -> dict[str, object]:
        """Assassin-Node Protection: reject mathematically impossible values.

        This validator fires on every TaskResult construction, acting as
        an inline red-teaming guard. If a rogue worker returns nonsense
        like market_share=150%, this raises ValueError and the reduce node
        will catch it and flag the result.
        """
        if "market_share" in v:
            share = v["market_share"]
            if isinstance(share, str):
                share = float(str(share).replace("%", ""))
            if isinstance(share, (int, float)):
                if not (0 <= float(share) <= 100):
                    raise ValueError(
                        f"market_share={share}% is mathematically impossible "
                        f"(must be 0-100)"
                    )
        return v


class FinalReport(BaseModel):
    """Aggregated output of the entire swarm run."""

    valid_results: list[TaskResult] = Field(default_factory=list)
    flagged_results: list[TaskResult] = Field(default_factory=list)
    total_cost_usd: float = Field(default=0.0)
    task_count: int = Field(default=0)
    success_rate: float = Field(
        default=0.0,
        description="Percentage of workers that completed successfully.",
    )
