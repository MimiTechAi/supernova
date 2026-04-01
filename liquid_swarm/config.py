"""Swarm configuration: NVIDIA NIM model selection, cost control, serverless prep.

Design decisions:
  - ModelTier enum maps to real NVIDIA NIM model IDs.
  - API key loaded from environment variable (NVIDIA_API_KEY).
  - All config is injectable, no global mutable state.
  - Worker business logic (execute_task) accepts SwarmConfig as a parameter,
    keeping it decoupled from LangGraph for serverless portability.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Load .env if python-dotenv is available ──────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional — key can be set via shell export


# ── NVIDIA NIM API Configuration ────────────────────────────────────────────

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")


class ModelTier(str, Enum):
    """Model selection tier for A/B testing cost optimization.

    Maps to real NVIDIA NIM model endpoints:
      BUDGET  → Llama 3.1 8B   (~$0.0003/call, fast, for bulk tasks)
      STANDARD → Llama 3.1 70B  (~$0.002/call, balanced)
      PREMIUM  → Nemotron 70B   (~$0.005/call, highest quality)
    """

    BUDGET = "meta/llama-3.1-8b-instruct"
    STANDARD = "meta/llama-3.1-70b-instruct"
    PREMIUM = "nvidia/llama-3.1-nemotron-70b-instruct"


# Cost lookup per model tier (estimated USD per call)
MODEL_COST: dict[ModelTier, float] = {
    ModelTier.BUDGET: 0.0003,
    ModelTier.STANDARD: 0.002,
    ModelTier.PREMIUM: 0.005,
}


class SwarmConfig(BaseModel):
    """Configuration for a swarm execution run.

    Attributes:
        model_tier: Which NVIDIA NIM model to use for workers.
        max_concurrency: Max parallel workers (rate-limit protection).
        worker_timeout_seconds: Hard kill timeout per worker.
        max_tokens: Maximum tokens for LLM response.
        temperature: LLM sampling temperature (0=deterministic, 1=creative).
    """

    model_tier: ModelTier = Field(
        default=ModelTier.BUDGET,
        description="Start cheap (8B). Only upgrade if quality drops.",
    )
    max_concurrency: int = Field(
        default=10,
        description="Semaphore limit. Prevents HTTP 429 from NVIDIA NIM.",
    )
    worker_timeout_seconds: int = Field(
        default=30,
        description="asyncio.wait_for timeout per worker in seconds.",
    )
    max_tokens: int = Field(
        default=512,
        description="Max tokens per LLM response. Keep low for cost control.",
    )
    temperature: float = Field(
        default=0.2,
        description="Low for factual analysis. Higher for creative tasks.",
    )

    @property
    def cost_per_call(self) -> float:
        """Lookup cost based on model tier."""
        return MODEL_COST[self.model_tier]

    @property
    def model_id(self) -> str:
        """The full NVIDIA NIM model identifier."""
        return self.model_tier.value

    def get(self, key: str, default: Any = None) -> Any:
        """Compatibility for LangSmith @traceable which expects config to be a dict."""
        return getattr(self, key, default)

    def to_langgraph_config(self) -> dict[str, dict[str, int]]:
        """Convert to LangGraph-native config dict for ainvoke().

        Usage:
            config = SwarmConfig()
            result = await graph.ainvoke(state, config=config.to_langgraph_config())
        """
        return {
            "configurable": {
                "max_concurrency": self.max_concurrency,
            }
        }
