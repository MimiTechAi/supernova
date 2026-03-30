"""Graph nodes: Router (Big Bang), Worker (Drone), Reduce (Aggregation).

Architecture:
  START ─→ [route_to_workers] ─→ N × worker_node ─→ reduce_node ─→ END

Critical design decisions:
  1. worker_node MUST catch TimeoutError internally. If it bubbles up,
     LangGraph cancels ALL sibling workers in the same superstep.
  2. execute_task uses httpx to call NVIDIA NIM (OpenAI-compatible API).
     Zero LangGraph imports → 1:1 portable to Modal.com @app.function.
  3. reduce_node filters invalid results (Assassin protection) by
     re-validating through Pydantic, catching ValidationError gracefully.
"""

import asyncio
import time

import httpx
from pydantic import ValidationError

from langgraph.types import Send

from liquid_swarm.config import NVIDIA_API_BASE, NVIDIA_API_KEY, SwarmConfig
from liquid_swarm.models import TaskInput, TaskResult
from liquid_swarm.state import SwarmState

WORKER_TIMEOUT_SECONDS = 30

# Rate-limit semaphore: limits concurrent API calls.
# Default: 10 concurrent workers (prevents HTTP 429).
# Use set_api_semaphore() to reconfigure for tests or different environments.
_api_semaphore: asyncio.Semaphore = asyncio.Semaphore(10)


def set_api_semaphore(max_concurrent: int) -> None:
    """Configure the API rate-limit semaphore.

    Args:
        max_concurrent: Maximum number of workers hitting the API simultaneously.
    """
    global _api_semaphore
    _api_semaphore = asyncio.Semaphore(max_concurrent)


# ── Big Bang Router (Fan-Out) ────────────────────────────────────────────────

def route_to_workers(state: SwarmState) -> list[Send]:
    """Conditional edge: spawns exactly N Send objects for N tasks.

    Each Send delivers a state-patch dict with the key 'current_task'
    set to one TaskInput. The worker_node receives the full SwarmState
    updated with this patch.
    """
    return [
        Send("worker_node", {"current_task": task})
        for task in state["tasks"]
    ]


# ── Async Worker Node (Drone) ───────────────────────────────────────────────

async def worker_node(state: SwarmState) -> dict[str, list[TaskResult]]:
    """Single micro-agent. Async for true parallelism via LangGraph supersteps.

    CRITICAL: TimeoutError is caught HERE, not propagated. If it escapes,
    LangGraph's _panic_or_proceed cancels all 49 sibling workers.
    """
    task: TaskInput | None = state.get("current_task")
    if task is None:
        return {"results": [TaskResult(
            task_id="unknown",
            status="error",
            data={"error": "No current_task in state"},
        )]}

    try:
        async with _api_semaphore:
            result = await asyncio.wait_for(
                execute_task(task),
                timeout=WORKER_TIMEOUT_SECONDS,
            )
        return {"results": [result]}
    except asyncio.TimeoutError:
        return {"results": [TaskResult(
            task_id=task.task_id,
            status="timeout",
            data={"error": f"Worker timed out after {WORKER_TIMEOUT_SECONDS}s"},
        )]}
    except Exception as exc:
        return {"results": [TaskResult(
            task_id=task.task_id,
            status="error",
            data={"error": str(exc)},
        )]}


# ── Task Execution (Real NVIDIA NIM API — serverless-portable) ──────────────

async def execute_task(
    task: TaskInput,
    config: SwarmConfig | None = None,
) -> TaskResult:
    """Execute analytical work by calling NVIDIA NIM API.

    This function has ZERO LangGraph imports, making it directly portable
    to Modal.com @app.function or any other serverless container runtime.

    Uses the OpenAI-compatible /v1/chat/completions endpoint on NVIDIA NIM.

    Args:
        task: The analysis task to execute.
        config: Optional swarm configuration for model selection.

    Returns:
        A validated TaskResult with the LLM analysis output.
    """
    cfg = config or SwarmConfig()

    system_prompt = (
        "You are a precise market analyst. Respond concisely, with structure and "
        "concrete numbers when possible. Maximum 3 sentences."
    )

    payload = {
        "model": cfg.model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.query},
        ],
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    t0 = time.perf_counter()

    async with httpx.AsyncClient(timeout=25.0) as client:
        response = await client.post(
            f"{NVIDIA_API_BASE}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()

    elapsed = time.perf_counter() - t0
    body = response.json()

    # Extract the LLM response
    llm_output = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    return TaskResult(
        task_id=task.task_id,
        status="success",
        data={
            "result": llm_output,
            "model": cfg.model_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_seconds": round(elapsed, 3),
        },
        cost_usd=cfg.cost_per_call,
    )


# ── Reduce Node (Aggregation + Assassin Filtering) ─────────────────────────

def reduce_node(state: SwarmState) -> dict[str, list[TaskResult]]:
    """Aggregate worker results. Filter invalid data (Red Teaming).

    Re-validates each TaskResult through Pydantic. Results that fail
    validation (e.g. market_share > 100%) are separated into flagged_results
    instead of crashing the graph.

    IMPORTANT: Writes to 'final_results' and 'flagged_results', NOT 'results'.
    The 'results' key uses operator.add reducer — writing back to it would
    double the entries.
    """
    valid: list[TaskResult] = []
    flagged: list[TaskResult] = []

    for result in state.get("results", []):
        try:
            # Re-validate through Pydantic to catch assassin data
            TaskResult.model_validate(result.model_dump())
            valid.append(result)
        except (ValidationError, ValueError):
            # Assassin data detected — flag but don't crash
            flagged_result = TaskResult(
                task_id=result.task_id,
                status="flagged",
                data={"original_data": str(result.data), "reason": "validation_failed"},
                cost_usd=result.cost_usd,
            )
            flagged.append(flagged_result)

    return {"final_results": valid, "flagged_results": flagged}
