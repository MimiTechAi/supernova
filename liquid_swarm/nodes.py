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
# Created lazily per event loop to be compatible with asyncio.run() in tests.
_max_concurrent: int = 10
_api_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: object | None = None  # track which event loop owns the semaphore


def _get_semaphore() -> asyncio.Semaphore:
    """Return the API semaphore, creating a fresh one if the event loop changed."""
    global _api_semaphore, _semaphore_loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _api_semaphore is None or current_loop is not _semaphore_loop:
        _api_semaphore = asyncio.Semaphore(_max_concurrent)
        _semaphore_loop = current_loop

    return _api_semaphore


def set_api_semaphore(max_concurrent: int) -> None:
    """Configure the API rate-limit semaphore.

    Args:
        max_concurrent: Maximum number of workers hitting the API simultaneously.
    """
    global _max_concurrent, _api_semaphore
    _max_concurrent = max_concurrent
    _api_semaphore = None  # Force recreation on next use


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


@traceable(name="thinker_node")
async def thinker_node(state: SwarmState) -> dict:
    """O1/Claude 3.7 Reasoning Node. Think critically before execution."""
    tasks = state.get("tasks", [])
    if not tasks:
        return {"strategy_plan": ""}

    task_queries = "\n".join(f"- {t.query}" for t in tasks)
    global_context = state.get("global_context", "")
    
    prompt = (
        "You are the master Thinker Agent (similar to o1 / Claude 3.7 reasoning core). "
        "Before we dispatch the following disjoint tasks to parallel workers, "
        "write a dense, highly analytical execution strategy. "
        "Consider potential cross-dependencies, missing information vectors, and the global context. "
        "Wrap your reasoning in <thought_process> tags, then provide the final Strategy Directives."
        f"\n\nGLOBAL CONTEXT:\n{global_context}"
        f"\n\nTASKS TO EXECUTE:\n{task_queries}"
    )

    cfg = SwarmConfig()
    llm = get_llm(cfg)

    if not llm:
        return {"strategy_plan": "No valid LLM configured for Thinker."}

    messages = [
        SystemMessage(content="You are the elite reasoning core of the Liquid Swarm."),
        HumanMessage(content=prompt)
    ]
    
    response = await llm.ainvoke(messages)
    return {"strategy_plan": str(response.content)}


from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class WorkerSubState(TypedDict):
    """State for the worker's internal dynamic mesh loop."""
    task: TaskInput
    attempts: int
    result: dict[str, object] | None
    cost_usd: float  # Propagate cost from execute_task through the mesh
    error: str | None
    global_context: str
    strategy_plan: str

async def generate_node(state: WorkerSubState) -> dict:
    task = state["task"]
    try:
        res = await execute_task(
            task,
            global_context=state.get("global_context", ""),
            strategy_plan=state.get("strategy_plan", ""),
        )
        return {
            "result": res.data,
            "cost_usd": res.cost_usd,
            "attempts": state["attempts"] + 1,
            "error": None,
        }
    except Exception as exc:
        return {"error": str(exc), "attempts": state["attempts"] + 1, "cost_usd": 0.0}

def evaluate_edge(state: WorkerSubState) -> str:
    """Decide if the worker should self-correct or finish."""
    if state["error"] or not state["result"]:
        return "generate" if state["attempts"] < 3 else "end"
    
    # Check confidence — re-queue with a self-correction hint if too low
    conf_str = str(state["result"].get("confidence", ""))
    try:
        conf = int("".join(c for c in conf_str if c.isdigit()))
        if conf < 80 and state["attempts"] < 3:
            # Rebuild task with appended self-correction hint (Pydantic models are immutable)
            original = state["task"]
            state["task"] = original.model_copy(update={
                "query": original.query + f" (Note: previous attempt confidence={conf}%. Be more precise and cite sources.)"
            })
            return "generate"
    except (ValueError, TypeError):
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
    strategy_plan = state.get("strategy_plan", "")
    if task is None:
        return {"results": [TaskResult(task_id="unknown", status="error", data={"error": "No current_task"})]}

    try:
        async with _get_semaphore():
            # Run the self-correcting mesh; hard-kill after WORKER_TIMEOUT_SECONDS
            final_sub_state = await asyncio.wait_for(
                compiled_worker_mesh.ainvoke(
                    {
                        "task": task,
                        "attempts": 0,
                        "result": None,
                        "cost_usd": 0.0,
                        "error": None,
                        "global_context": global_context,
                        "strategy_plan": strategy_plan,
                    }
                ),
                timeout=WORKER_TIMEOUT_SECONDS,
            )

        cost = final_sub_state.get("cost_usd", 0.0) or 0.0
        if final_sub_state.get("error"):
            res = TaskResult(
                task_id=task.task_id,
                status="error",
                data={"error": final_sub_state["error"]},
                cost_usd=cost,
            )
        else:
            raw_data = final_sub_state["result"] or {}
            try:
                res = TaskResult(
                    task_id=task.task_id,
                    status="success",
                    data=raw_data,
                    cost_usd=cost,
                )
            except ValidationError:
                # Data contains impossible values (e.g. market_share > 100%).
                # Preserve it unmodified so reduce_node can detect and flag it.
                res = TaskResult.model_construct(
                    task_id=task.task_id,
                    status="success",
                    data=raw_data,
                    cost_usd=cost,
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
    global_context: str = "",
    strategy_plan: str = ""
) -> TaskResult:
    """Execute analytical work using LangChain ChatModels, Tool calling, and Structured Outputs."""
    if isinstance(config, dict):
        config = SwarmConfig(**config)
    cfg = config or SwarmConfig()
    provider_config = get_provider_config()

    context_str = f"GLOBAL KNOWLEDGE BASE:\n{global_context}\n\n" if global_context else ""
    strategy_str = f"OVERARCHING STRATEGY DIRECTIVES (Follow these):\n{strategy_plan}\n\n" if strategy_plan else ""
    
    system_prompt = (
        "You are a precise market analyst. Respond concisely, with structure and "
        "concrete numbers when possible. Maximum 3 sentences. Rate your confidence.\n\n"
        f"{context_str}"
        f"{strategy_str}"
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
    if getattr(response, "tool_calls", None):
        gathered_context = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                gathered_context.append(f"Search Result: {msg.content}")
            elif getattr(msg, "content", None) and not getattr(msg, "tool_calls", None):
                gathered_context.append(str(msg.content))
        
        flat_prompt = (
            "Here is the context gathered from the web and previous steps:\n"
            f"{' '.join(gathered_context)}\n\n"
            f"Original Query: {task.query}\n\n"
            "Format the findings into the requested JSON structured output format. "
            "WARNING: You MUST output ONLY a raw JSON object with EXACTLY these keys: "
            "'result' (string), 'confidence' (integer 0-100), and 'market_share' (float or null). "
            "Do not include markdown formatting (```json), and do not include any trailing conversational text."
        )
        final_messages = [SystemMessage(content=system_prompt), HumanMessage(content=flat_prompt)]
    else:
        messages.append(HumanMessage(
            "Now format our findings into the requested JSON structured output format. "
            "WARNING: You MUST output ONLY a raw JSON object with EXACTLY these keys: "
            "'result' (string), 'confidence' (integer 0-100), and 'market_share' (float or null). "
            "Do not include markdown formatting (```json), and do not include any trailing conversational text."
        ))
        final_messages = messages
        
    try:
        final_output: LLMOutput = await structured_llm.ainvoke(final_messages)
    except Exception:
        # Structured-output parsing failed (model returned non-JSON).
        # Fall back to a plain LLM call and wrap the raw text.
        raw_response = await llm.ainvoke(final_messages)
        raw_text = str(getattr(raw_response, "content", raw_response)).strip()
        final_output = LLMOutput(
            result=raw_text[:1000] if raw_text else "INSUFFICIENT DATA",
            confidence=40,
            market_share=None,
        )

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
