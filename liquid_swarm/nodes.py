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

from pydantic import ValidationError, BaseModel, Field
from langgraph.types import Send
from langsmith import traceable

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from liquid_swarm.config import SwarmConfig
from liquid_swarm.providers import get_provider_config, LLMProvider
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


from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class WorkerSubState(TypedDict):
    """State for the worker's internal dynamic mesh loop."""
    task: TaskInput
    attempts: int
    result: dict[str, object] | None
    error: str | None
    global_context: str

async def generate_node(state: WorkerSubState) -> dict:
    task = state["task"]
    try:
        res = await execute_task(task, global_context=state.get("global_context", ""))
        return {"result": res.data, "attempts": state["attempts"] + 1, "error": None}
    except Exception as exc:
        return {"error": str(exc), "attempts": state["attempts"] + 1}

def evaluate_edge(state: WorkerSubState) -> str:
    """Decide if the worker should self-correct or finish."""
    if state["error"] or not state["result"]:
        return "generate" if state["attempts"] < 3 else "end"
    
    # Check confidence
    conf_str = state["result"].get("confidence", "")
    try:
        conf = int("".join(c for c in conf_str if c.isdigit()))
        if conf < 80 and state["attempts"] < 3:
            # Tell LLM to self-correct by mutating the query
            state["task"].query += f" (Note: previous attempt only got {conf}% confidence. BE MORE PRECISE.)"
            return "generate"
    except ValueError:
        pass
    
    return "end"

# Compile the Worker Subgraph (Dynamic Mesh)
worker_builder = StateGraph(WorkerSubState)
worker_builder.add_node("generate", generate_node)
worker_builder.add_edge(START, "generate")
worker_builder.add_conditional_edges("generate", evaluate_edge, {"generate": "generate", "end": END})
compiled_worker_mesh = worker_builder.compile()

async def worker_node(state: SwarmState) -> dict[str, list[TaskResult]]:
    """Single micro-agent powered by an internal self-correcting mesh graph."""
    task: TaskInput | None = state.get("current_task")
    global_context = state.get("global_context", "")
    if task is None:
        return {"results": [TaskResult(task_id="unknown", status="error", data={"error": "No current_task"})]}

    try:
        async with _api_semaphore:
            # We run the sub-graph mesh to accomplish self-correction iteratively
            final_sub_state = await compiled_worker_mesh.ainvoke(
                {"task": task, "attempts": 0, "result": None, "error": None, "global_context": global_context}
            )
            
        if final_sub_state["error"]:
            res = TaskResult(
                task_id=task.task_id, 
                status="error", 
                data={"error": final_sub_state["error"]}
            )
        else:
            res = TaskResult(
                task_id=task.task_id, 
                status="success", 
                data=final_sub_state["result"] or {}
            )
        return {"results": [res]}

    except asyncio.TimeoutError:
        return {"results": [TaskResult(task_id=task.task_id, status="timeout", data={"error": "Timed out"})]}
    except Exception as exc:
        return {"results": [TaskResult(task_id=task.task_id, status="error", data={"error": str(exc)})]}


# ── Task Execution (Real NVIDIA NIM API — serverless-portable) ──────────────

class LLMOutput(BaseModel):
    """The structured output strictly expected from the LLM."""
    result: str = Field(description="The concise analysis text.")
    confidence: int = Field(description="Confidence percentage between 0 and 100.")
    market_share: float | None = Field(default=None, description="Extracted market share if applicable.")

def get_llm(cfg: SwarmConfig):
    provider_config = get_provider_config()
    model_name = provider_config.default_model or cfg.model_id
    
    if provider_config.provider in (LLMProvider.NVIDIA, LLMProvider.OPENAI):
        return ChatOpenAI(
            api_key=provider_config.api_key,
            base_url=provider_config.base_url,
            model=model_name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    elif provider_config.provider == LLMProvider.ANTHROPIC:
        return ChatAnthropic(
            api_key=provider_config.api_key,
            base_url=provider_config.base_url,
            model=model_name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    elif provider_config.provider == LLMProvider.OLLAMA:
        return ChatOllama(
            base_url=provider_config.base_url,
            model=model_name,
            temperature=cfg.temperature,
            format="json",
        )
    return None

from langchain_core.messages import ToolMessage
from liquid_swarm.tools import web_search_tool

@traceable(name="execute_task")
async def execute_task(
    task: TaskInput,
    config: SwarmConfig | None = None,
    global_context: str = ""
) -> TaskResult:
    """Execute analytical work using LangChain ChatModels, Tool calling, and Structured Outputs."""
    if isinstance(config, dict):
        config = SwarmConfig(**config)
    cfg = config or SwarmConfig()
    provider_config = get_provider_config()

    context_str = f"GLOBAL KNOWLEDGE BASE:\n{global_context}\n\n" if global_context else ""
    system_prompt = (
        "You are a precise market analyst. Respond concisely, with structure and "
        "concrete numbers when possible. Maximum 3 sentences. Rate your confidence.\n\n"
        f"{context_str}"
        "IMPORTANT: If facts are outdated or you lack current data, use the web_search_tool."
    )

    llm = get_llm(cfg)
    llm_with_tools = llm.bind_tools([web_search_tool])
    structured_llm = llm.with_structured_output(LLMOutput)

    t0 = time.perf_counter()
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=task.query)
    ]
    
    # 1. Ask LLM to think/use tools
    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)
    
    # 2. If tools are requested, execute them
    if getattr(response, "tool_calls", None):
        for tc in response.tool_calls:
            if tc["name"] == "web_search_tool":
                tool_res = await web_search_tool.ainvoke(tc["args"])
                messages.append(ToolMessage(content=tool_res, tool_call_id=tc["id"]))
    
    # 3. Request final structured format
    messages.append(HumanMessage("Now format our findings into the requested JSON structured output format."))
    final_output: LLMOutput = await structured_llm.ainvoke(messages)

    elapsed = time.perf_counter() - t0
    
    return TaskResult(
        task_id=task.task_id,
        status="success",
        data={
            "result": final_output.result,
            "confidence": f"[CONFIDENCE: {final_output.confidence}%]",
            "market_share": final_output.market_share,
            "model": provider_config.default_model or cfg.model_id,
            "latency_seconds": round(elapsed, 3),
        },
        cost_usd=cfg.cost_per_call,
    )


# HTTP status codes that indicate transient failures (worth retrying)
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503}


async def execute_task_with_retry(
    task: TaskInput,
    config: SwarmConfig | None = None,
    max_retries: int = 1,
) -> TaskResult:
    """Execute a task with retry logic for transient failures.

    Retries on transient HTTP errors (429, 500, 502, 503) with exponential
    backoff. Permanent errors (401, 403, 404, 422) fail immediately.

    This function wraps execute_task — zero LangGraph imports, fully
    portable to serverless runtimes.

    Args:
        task: The analysis task to execute.
        config: Optional swarm configuration.
        max_retries: Number of retry attempts for transient errors.

    Returns:
        A TaskResult with status 'success' or 'error'.
    """
    last_error: str = ""

    for attempt in range(1 + max_retries):
        try:
            result = await execute_task(task, config)
            return result
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            last_error = f"HTTP {status_code}: {exc}"

            # Permanent errors — don't retry
            if status_code not in _TRANSIENT_STATUS_CODES:
                return TaskResult(
                    task_id=task.task_id,
                    status="error",
                    data={"error": last_error, "retries": attempt},
                    cost_usd=0.0,
                )

            # Transient error — retry with backoff if attempts remain
            if attempt < max_retries:
                backoff = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s, ...
                await asyncio.sleep(backoff)
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (2 ** attempt))

    # All retries exhausted
    return TaskResult(
        task_id=task.task_id,
        status="error",
        data={"error": last_error, "retries": max_retries},
        cost_usd=0.0,
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
