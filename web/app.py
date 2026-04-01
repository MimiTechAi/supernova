"""Supernova — Live Web UI Backend.

FastAPI server with SSE streaming for real-time swarm visualization.
Each worker result is streamed to the frontend as it arrives.
Supports multiple LLM providers: NVIDIA NIM, OpenAI, Anthropic, Ollama.

Copyright 2026 MiMi Tech Ai UG, Bad Liebenzell, Germany.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_swarm.models import TaskInput, TaskResult
from liquid_swarm.providers import LLMProvider, ProviderConfig, get_provider_config
from liquid_swarm.synthesis import synthesize_results
from liquid_swarm.persistence import save_run, list_runs, get_run
from liquid_swarm.ledger import record_run, get_ledger_summary
from liquid_swarm.web_search import (
    SearchEngine,
    SearchCache,
    get_search_engine,
    build_search_context,
    parse_sources,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from contextlib import asynccontextmanager
from liquid_swarm.graph import build_swarm_graph
from langchain_core.runnables import RunnableConfig

# Postgres checkpointer is optional — only available when psycopg binary is installed
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver as _AsyncPostgresSaver
    _POSTGRES_AVAILABLE = True
except ImportError:
    _AsyncPostgresSaver = None  # type: ignore[assignment,misc]
    _POSTGRES_AVAILABLE = False

app = FastAPI(title="Supernova — Command Center")


# ── API Key Authentication Middleware ────────────────────────────────────────

@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    """Protect /api/ endpoints with API key authentication."""
    swarm_keys_env = os.environ.get("SWARM_API_KEYS", "")

    if not swarm_keys_env:
        return await call_next(request)

    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    valid_keys = {k.strip() for k in swarm_keys_env.split(",") if k.strip()}

    api_key = request.headers.get("X-Api-Key")
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={"detail": "API key required. Set X-Api-Key header."},
        )

    if api_key not in valid_keys:
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid API key."},
        )

    return await call_next(request)


# Serve static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@asynccontextmanager
async def get_memory_saver():
    postgres_url = os.environ.get("POSTGRES_URL")
    if postgres_url and _POSTGRES_AVAILABLE and _AsyncPostgresSaver is not None:
        async with _AsyncPostgresSaver.from_conn_string(postgres_url) as memory:
            await memory.setup()
            yield memory
    else:
        async with AsyncSqliteSaver.from_conn_string("supernova_checkpoints.db") as memory:
            yield memory


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI."""
    html_path = static_dir / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/config")
async def get_config():
    """Return available providers, models and default config."""
    provider_cfg = get_provider_config()
    return {
        "provider": provider_cfg.provider.value,
        "models": provider_cfg.available_models,
        "default_model": provider_cfg.default_model,
        "has_api_key": bool(provider_cfg.api_key),
        "available_providers": [
            {"id": p.value, "name": p.value.upper()} for p in LLMProvider
        ],
    }


def _build_provider_config(model_id: str = "", provider_str: str = "") -> ProviderConfig:
    """Build a ProviderConfig from request params, falling back to env vars."""
    base_cfg = get_provider_config()

    # Allow override from request
    if provider_str:
        try:
            base_cfg.provider = LLMProvider(provider_str.lower())
        except ValueError:
            pass

    if model_id:
        base_cfg.default_model = model_id

    return base_cfg


async def _llm_call(
    provider_cfg: ProviderConfig,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.2,
    stream: bool = False,
) -> dict:
    """Unified LLM API call that works with any provider.

    All supported providers use the OpenAI-compatible /v1/chat/completions
    endpoint. Returns the raw response dict.
    """
    payload = {
        "model": provider_cfg.default_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    headers = provider_cfg.get_headers()

    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            f"{provider_cfg.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()

    return resp.json()


async def _llm_call_stream(
    provider_cfg: ProviderConfig,
    messages: list[dict],
    max_tokens: int = 768,
    temperature: float = 0.3,
) -> AsyncGenerator[str, None]:
    """Streaming LLM call — yields tokens as they arrive."""
    payload = {
        "model": provider_cfg.default_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    headers = provider_cfg.get_headers()

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{provider_cfg.base_url}/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


async def _generate_subtasks(
    main_query: str,
    num_tasks: int,
    provider_cfg: ProviderConfig,
    system_prompt_context: str = "",
) -> list[str]:
    """Use the LLM to break a main query into N sub-tasks."""
    role_context = f"\nContext: Workers will act as: {system_prompt_context}" if system_prompt_context else ""

    if num_tasks > 10:
        decomposition_strategy = (
            f"Since we have {num_tasks} agents, you MUST use a hierarchical or dimensional strategy (such as PESTEL, SWOT, or technical architectural layers) to divide the topic into broad distinct categories first.\n"
            f"Then, assign individual specific questions within those categories to completely eliminate overlap and redundancy. Ensure maximum breadth and depth."
        )
    else:
        decomposition_strategy = "Ensure that the tasks are distinct and cover different aspects of the main problem without redundant overlap."

    prompt = (
        f"You are a master project manager. Break the following task into exactly "
        f"{num_tasks} specific, independent sub-tasks for parallel analysis.\n\n"
        f"Main task: {main_query}{role_context}\n\n"
        f"STRATEGY: {decomposition_strategy}\n\n"
        f"Reply ONLY with a numbered list (1. ... 2. ... etc.) containing the final {num_tasks} tasks, "
        f"without introduction, explanation, or category headers. Every line starting with a number should be a concrete "
        f"analysis question that a single analyst can answer using web search."
    )

    body = await _llm_call(
        provider_cfg,
        messages=[
            {"role": "system", "content": "You are a precise project manager."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.3,
    )

    text = body["choices"][0]["message"]["content"]

    # Parse numbered list
    lines = [
        line.strip()
        for line in text.strip().split("\n")
        if line.strip() and line.strip()[0].isdigit()
    ]

    subtasks = []
    for line in lines[:num_tasks]:
        cleaned = line.lstrip("0123456789.)- ").strip()
        if cleaned:
            subtasks.append(cleaned)

    while len(subtasks) < num_tasks:
        subtasks.append(f"Analyze aspect {len(subtasks)+1} of: {main_query}")

    return subtasks[:num_tasks]


async def _execute_single_task(
    task: TaskInput,
    semaphore: asyncio.Semaphore,
    provider_cfg: ProviderConfig,
    system_prompt: str = "",
    max_tokens: int = 512,
    temperature: float = 0.2,
    cost_budget_remaining: float | None = None,
    search_engine: SearchEngine | None = None,
    search_cache: SearchCache | None = None,
    main_query: str = "",
) -> TaskResult:
    """Execute a single task with semaphore rate limiting and optional web search."""
    hallucination_guard = (
        "\n\nCRITICAL RULE: Evaluate the search context before answering. If the provided facts "
        "do not contain enough concrete information to answer the question, you MUST reply "
        "exactly with 'INSUFFICIENT DATA'. Do not invent or guess facts."
        "\nYour [CONFIDENCE: X%] score must strictly reflect the density of concrete numbers and cited sources."
    )
    
    worker_prompt = system_prompt or (
        "You are a precise analyst. Respond with structure "
        "and concrete data. Maximum 4-5 sentences."
    )
    worker_prompt += hallucination_guard

    # Budget guard: if remaining budget is 0 or negative, skip
    cost_per_call = provider_cfg.get_model_cost(provider_cfg.default_model)
    if cost_budget_remaining is not None and cost_budget_remaining < cost_per_call:
        return TaskResult(
            task_id=task.task_id,
            status="error",
            data={"error": "Budget exceeded — worker skipped"},
            cost_usd=0.0,
        )

    t0 = time.perf_counter()

    try:
        async with semaphore:
            # Phase 1: Web Search (if enabled)
            search_results_data = []
            worker_input = task.query
            
            if search_engine is not None:
                # Check cache first
                cached = search_cache.get(task.query) if search_cache else None
                if cached is not None:
                    search_results = cached
                else:
                    search_results = await search_engine.search(task.query)
                    if search_cache:
                        search_cache.put(task.query, search_results)
                
                # Format context
                context_prompt = build_search_context(search_results)
                worker_input = f"{context_prompt}\n\nORIGINAL USER QUERY / CONTENT:\n{main_query}\n\nYOUR SPECIFIC SUB-TASK:\n{task.query}"
                
                # Save raw sources for UI
                search_results_data = [
                    {"title": r.title, "url": r.url, "domain": r.source}
                    for r in search_results
                ]
            
            else:
                # If no search, just supply the original context
                worker_input = f"ORIGINAL USER QUERY / CONTENT:\n{main_query}\n\nYOUR SPECIFIC SUB-TASK:\n{task.query}"

            # Phase 2: LLM Call
            body = await _llm_call(
                provider_cfg,
                messages=[
                    {"role": "system", "content": worker_prompt},
                    {"role": "user", "content": worker_input},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

        elapsed = time.perf_counter() - t0
        llm_output = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})

        # Parse confidence and sources
        confidence = _parse_confidence(llm_output)
        cited_sources = parse_sources(llm_output) if search_engine else []
        
        # Mark used sources
        final_sources = []
        for src in search_results_data:
            is_used = any(src["url"] == cited for cited in cited_sources)
            final_sources.append({**src, "used_by_llm": is_used})

        return TaskResult(
            task_id=task.task_id,
            status="success",
            data={
                "result": llm_output,
                "model": provider_cfg.default_model,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "latency_seconds": round(elapsed, 3),
                "confidence": confidence,
                "sources": final_sources,
                "search_query": task.query if search_engine else None,
            },
            cost_usd=cost_per_call,
        )
    except asyncio.TimeoutError:
        return TaskResult(
            task_id=task.task_id,
            status="timeout",
            data={"error": "Worker timed out"},
        )
    except Exception as exc:
        return TaskResult(
            task_id=task.task_id,
            status="error",
            data={"error": str(exc)},
        )


def _parse_confidence(text: str) -> int | None:
    """Extract [CONFIDENCE: XX] tag from LLM output."""
    import re
    match = re.search(r'\[CONFIDENCE:\s*(\d+)\s*%?\]', text, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        return max(0, min(100, val))
    return None


# ── Prompt Presets ───────────────────────────────────────────────────────────

PROMPT_PRESETS = {
    "market_analyst": {
        "name": "Market Analyst",
        "icon": "📊",
        "prompt": (
            "You are a precise market analyst. Respond with structure "
            "and concrete numbers when possible. Maximum 4-5 sentences. "
            "End your response with [CONFIDENCE: X%] where X is your "
            "confidence level (0-100) based on data availability."
        ),
    },
    "code_reviewer": {
        "name": "Code Reviewer",
        "icon": "💻",
        "prompt": (
            "You are a senior software engineer performing code review. "
            "Focus on: bugs, security issues, performance, best practices. "
            "Be specific and actionable. Maximum 4-5 sentences. "
            "End with [CONFIDENCE: X%] based on code clarity."
        ),
    },
    "research_scientist": {
        "name": "Research Scientist",
        "icon": "🔬",
        "prompt": (
            "You are a research scientist. Analyze with academic rigor. "
            "Cite sources where possible, evaluate evidence quality. "
            "Distinguish between established facts and hypotheses. Maximum 4-5 sentences. "
            "End with [CONFIDENCE: X%] based on evidence strength."
        ),
    },
    "legal_analyst": {
        "name": "Legal Analyst",
        "icon": "⚖️",
        "prompt": (
            "You are a legal analyst. Identify relevant regulations, "
            "compliance risks, and legal implications. Be precise about "
            "jurisdictions. Maximum 4-5 sentences. "
            "End with [CONFIDENCE: X%] based on regulatory clarity."
        ),
    },
    "futurist": {
        "name": "Predictive Futurist",
        "icon": "🔮",
        "prompt": (
            "You are a predictive futurist specialized in extrapolating current "
            "data into future trends. Analyze the provided current context and "
            "predict outcomes 3-5 years into the future. Identify structural shifts, "
            "emerging risks, and high-probability scenarios. Maximum 4-5 sentences. "
            "End with [CONFIDENCE: X%] based on data stability."
        ),
    },
    "custom": {
        "name": "Custom",
        "icon": "✏️",
        "prompt": "",
    },
}


@app.get("/api/prompts")
async def get_prompts():
    """Return available system prompt presets."""
    return {
        "presets": [
            {"id": k, "name": v["name"], "icon": v["icon"], "prompt": v["prompt"]}
            for k, v in PROMPT_PRESETS.items()
        ]
    }


@app.post("/api/ignite")
async def ignite_swarm(request: Request):
    """Start the swarm and stream results via SSE."""
    body = await request.json()
    main_query = body.get("query", "")
    num_tasks = min(max(int(body.get("num_tasks", 5)), 1), 250)
    model_id = body.get("model", "")
    provider_str = body.get("provider", "")
    system_prompt_id = body.get("system_prompt", "market_analyst")
    custom_prompt = body.get("custom_prompt", "")
    cost_budget = body.get("cost_budget", None)
    web_search_enabled = body.get("web_search_enabled", True)

    # Resolve provider
    provider_cfg = _build_provider_config(model_id, provider_str)

    # Resolve system prompt
    if system_prompt_id == "custom" and custom_prompt:
        system_prompt = custom_prompt
    elif system_prompt_id in PROMPT_PRESETS:
        system_prompt = PROMPT_PRESETS[system_prompt_id]["prompt"]
    else:
        system_prompt = PROMPT_PRESETS["market_analyst"]["prompt"]

    cost_per_call = provider_cfg.get_model_cost(provider_cfg.default_model)
    semaphore = asyncio.Semaphore(10)
    
    search_engine = get_search_engine() if web_search_enabled else None
    search_cache = SearchCache() if web_search_enabled else None

    async def event_stream():
        t_total = time.perf_counter()

        # Pre-run cost estimate
        estimated_cost = cost_per_call * (num_tasks + 2)  # workers + decomp + synthesis
        yield _sse_event("cost_estimate", {
            "estimated_cost": round(estimated_cost, 6),
            "cost_per_worker": cost_per_call,
            "budget": cost_budget,
        })

        # Phase 1: Generate subtasks
        yield _sse_event("phase", {"phase": "decomposing", "message": "Decomposing task into sub-tasks..."})

        try:
            subtasks = await _generate_subtasks(main_query, num_tasks, provider_cfg, system_prompt)
        except Exception as e:
            yield _sse_event("error", {"message": f"Error during task decomposition: {e}"})
            return

        tasks = [
            TaskInput(task_id=f"worker-{i:03d}", query=q)
            for i, q in enumerate(subtasks)
        ]

        # Phase 2: Send task list to frontend
        yield _sse_event("tasks", {
            "tasks": [{"id": t.task_id, "query": t.query} for t in tasks],
            "model": provider_cfg.default_model,
            "provider": provider_cfg.provider.value,
            "cost_per_call": cost_per_call,
        })

        # Phase 3: Execute all tasks in parallel via LangGraph Mesh
        yield _sse_event("phase", {"phase": "executing", "message": f"Igniting {len(tasks)} workers via LangGraph..."})

        thread_id = f"run-{int(time.time())}"
        config = RunnableConfig(configurable={"thread_id": thread_id})
        
        state_input = {
            "tasks": tasks,
            "current_task": None,
            "results": [],
            "final_results": [],
            "flagged_results": [],
            "global_context": None,
            "strategy_plan": None,
        }
        
        completed = 0
        total_cost = 0.0
        all_results: list[TaskResult] = []
        completed_ids = set()

        # Stream directly from the graph
        async with get_memory_saver() as memory:
            swarm_graph = build_swarm_graph(checkpointer=memory)
            async for event in swarm_graph.astream(state_input, config=config, stream_mode="updates"):
                # worker_node returns dict like {"results": [TaskResult(...)]}
                if "worker_node" in event:
                    new_results = event["worker_node"].get("results", [])
                    for res in new_results:
                        if res.task_id not in completed_ids:
                            completed_ids.add(res.task_id)
                            completed += 1
                            total_cost += res.cost_usd
                            all_results.append(res)
                            
                            # Convert TaskResult to dict for JSON serialization
                            data_dict = {
                                "result": res.data.get("result", ""),
                                "confidence": res.data.get("confidence", ""),
                                "latency_seconds": res.data.get("latency_seconds", 0),
                                "model": res.data.get("model", ""),
                            }

                            yield _sse_event("result", {
                                "task_id": res.task_id,
                                "status": res.status,
                                "data": data_dict,
                                "cost_usd": res.cost_usd,
                                "completed": completed,
                                "total": len(tasks),
                            })

        # Graph hits the 'interrupt_before=["reduce_node"]' breakpoint and pauses.
        yield _sse_event("paused", {
            "thread_id": thread_id, 
            "message": "Workers completed. Awaiting Human-in-the-Loop review."
        })
        # We STOP the ignition stream here. The frontend will hit /api/approve to continue.
        return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/api/approve")
async def approve_run(request: Request):
    """Resume the paused LangGraph after human approval."""
    body = await request.json()
    thread_id = body.get("thread_id")
    provider_str = body.get("provider", "openai")
    model_id = body.get("model", "")
    
    if not thread_id:
        return JSONResponse(status_code=400, content={"detail": "Missing thread_id"})
        
    provider_cfg = _build_provider_config(model_id, provider_str)
    cost_per_call = provider_cfg.get_model_cost(provider_cfg.default_model)
    config = RunnableConfig(configurable={"thread_id": thread_id})

    async def approve_stream():
        t_approve_start = time.perf_counter()
        yield _sse_event("phase", {"phase": "synthesizing", "message": "Human approved. Proceeding to Reducer & Synthesis..."})
        
        # Resume the graph
        async with get_memory_saver() as memory:
            swarm_graph = build_swarm_graph(checkpointer=memory)
            async for event in swarm_graph.astream(None, config=config, stream_mode="updates"):
                pass # We just run reduce_node and archivar_node to the end
                
            final_state = (await swarm_graph.aget_state(config)).values
            all_results = final_state.get("final_results", []) + final_state.get("flagged_results", [])
        total_cost = sum(r.cost_usd for r in all_results)

        synthesis_text = ""
        try:
            successful = [r for r in all_results if r.status == "success" and r.data.get("result") 
                          and r.data.get("result") != 'INSUFFICIENT DATA']

            if not successful:
                synthesis_text = "No successful worker results available for synthesis."
                yield _sse_event("synthesis", {"summary": synthesis_text, "cost_usd": 0.0})
            else:
                findings = [f"Finding {i}: {r.data['result']}" for i, r in enumerate(successful, 1)]
                context_for_final = "\n\n".join(findings)

                synth_prompt = (
                    "You are a senior analyst. Below are findings from parallel agents. "
                    "Synthesize them into a coherent executive summary.\n\n"
                    "Requirements:\n"
                    "- Start with a one-sentence headline\n"
                    "- Combine and deduplicate insights\n"
                    "- Keep it under 200 words\n"
                    "- Use structured formatting\n\n"
                    f"--- DATA ---\n\n{context_for_final}\n\n--- END DATA ---\n\n"
                    "Executive Summary:"
                )

                messages = [
                    {"role": "system", "content": "You are a senior analyst creating executive summaries."},
                    {"role": "user", "content": synth_prompt},
                ]

                # Token-by-token streaming
                async for token in _llm_call_stream(provider_cfg, messages):
                    synthesis_text += token
                    yield _sse_event("synthesis_token", {"token": token})

                synthesis_cost = cost_per_call
                total_cost += synthesis_cost
                yield _sse_event("synthesis_complete", {
                    "summary": synthesis_text,
                    "cost_usd": synthesis_cost,
                })

        except Exception as e:
            synthesis_text = f"Synthesis unavailable: {e}"
            yield _sse_event("synthesis", {"summary": synthesis_text, "cost_usd": 0.0})

        # Phase 5: Persistence — actually save the run
        total_time = round(time.perf_counter() - t_approve_start, 3)
        success_count = sum(1 for r in all_results if r.status == "success")
        run_id = None
        try:
            run_id = save_run(
                query=body.get("query", thread_id),
                results=all_results,
                total_cost=total_cost,
                total_time=total_time,
                model=provider_cfg.default_model,
                synthesis=synthesis_text,
            )
            # Record in cost ledger
            record_run(
                run_id=run_id,
                query=body.get("query", thread_id),
                model=provider_cfg.default_model,
                worker_count=len(all_results),
                success_count=success_count,
                cost_usd=total_cost,
                duration_sec=total_time,
            )
        except Exception:
            pass  # Persistence failure should never crash the response

        yield _sse_event("complete", {
            "total_time": total_time,
            "total_cost": round(total_cost, 6),
            "total_tasks": len(all_results),
            "success_count": success_count,
            "run_id": run_id,
        })

    return StreamingResponse(
        approve_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Persistence API ─────────────────────────────────────────────────────────

@app.get("/api/runs")
async def api_list_runs():
    """List all saved swarm runs (newest first)."""
    return list_runs()


@app.get("/api/runs/{run_id}")
async def api_get_run(run_id: str):
    """Retrieve a specific run by ID."""
    run = get_run(run_id)
    if run is None:
        return JSONResponse(status_code=404, content={"detail": "Run not found"})
    return run


# ── Cost Ledger API ──────────────────────────────────────────────────────────

@app.get("/api/ledger")
async def api_ledger():
    """Return cumulative cost ledger summary with daily/weekly/monthly rollups."""
    return get_ledger_summary()


# ── Export API ───────────────────────────────────────────────────────────────

@app.get("/api/export/{run_id}")
async def api_export_run(run_id: str, fmt: str = "json"):
    """Export a completed run as JSON or Markdown.

    Query params:
        fmt: 'json' (default) or 'md'
    """
    from fastapi.responses import PlainTextResponse
    run = get_run(run_id)
    if run is None:
        return JSONResponse(status_code=404, content={"detail": "Run not found"})

    if fmt == "md":
        results = run.get("results", [])
        lines = [
            f"# Supernova Report\n",
            f"**Query:** {run.get('query', '')}\n",
            f"**Date:** {run.get('timestamp', '')[:19].replace('T', ' ')} UTC\n",
            f"**Model:** {run.get('model', '')} | **Workers:** {run.get('worker_count', 0)} | "
            f"**Cost:** ${run.get('total_cost', 0):.6f} | **Time:** {run.get('total_time', 0):.2f}s\n",
        ]
        synthesis = run.get("synthesis")
        if synthesis:
            lines += [f"\n## Executive Summary\n\n{synthesis}\n"]
        lines.append("\n## Worker Results\n")
        for r in results:
            data = r.get("data", {})
            lines.append(f"\n### {r.get('task_id', '?')}\n")
            lines.append(f"**Status:** {r.get('status', '?')} | "
                         f"**Confidence:** {data.get('confidence', 'N/A')} | "
                         f"**Latency:** {data.get('latency_seconds', 0)}s\n\n")
            lines.append(str(data.get("result", "")) + "\n")
            sources = data.get("sources", [])
            if sources:
                lines.append("\n**Sources:**\n")
                for s in sources:
                    used = " (cited)" if s.get("used_by_llm") else ""
                    lines.append(f"- [{s.get('title', s.get('domain', ''))}]({s.get('url', '')}){used}\n")
        content = "\n".join(lines)
        return PlainTextResponse(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{run_id}.md"'},
        )

    # Default: JSON
    from fastapi.responses import Response
    return Response(
        content=json.dumps(run, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.json"'},
    )


# ── Health & Metrics ─────────────────────────────────────────────────────────

_start_time = time.time()
_request_count: int = 0
_error_count: int = 0


@app.get("/health")
async def health():
    """Health check endpoint for load balancers and uptime monitors."""
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "version": "1.0.0",
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible plain-text metrics endpoint."""
    from fastapi.responses import PlainTextResponse
    uptime = round(time.time() - _start_time, 1)
    ledger = get_ledger_summary()
    lines = [
        "# HELP supernova_uptime_seconds Server uptime in seconds",
        "# TYPE supernova_uptime_seconds gauge",
        f"supernova_uptime_seconds {uptime}",
        "",
        "# HELP supernova_total_runs Total completed swarm runs",
        "# TYPE supernova_total_runs counter",
        f"supernova_total_runs {ledger['total_runs']}",
        "",
        "# HELP supernova_total_cost_usd Cumulative cost in USD",
        "# TYPE supernova_total_cost_usd counter",
        f"supernova_total_cost_usd {ledger['total_cost_usd']}",
        "",
        "# HELP supernova_daily_cost_usd Cost in USD (last 24h)",
        "# TYPE supernova_daily_cost_usd gauge",
        f"supernova_daily_cost_usd {ledger['daily_cost_usd']}",
        "",
        "# HELP supernova_weekly_cost_usd Cost in USD (last 7 days)",
        "# TYPE supernova_weekly_cost_usd gauge",
        f"supernova_weekly_cost_usd {ledger['weekly_cost_usd']}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")


# ── WebSocket — unified real-time channel ────────────────────────────────────

@app.websocket("/ws/swarm")
async def ws_swarm(websocket: WebSocket):
    """WebSocket endpoint — single persistent channel for ignite + approve flow.

    Messages received (JSON):
      { "type": "ignite", "query": "...", "num_tasks": 5, ... }
      { "type": "approve", "thread_id": "...", "model": "..." }

    Messages sent (JSON):
      { "event": "<phase|tasks|result|paused|synthesis_token|...>", "data": {...} }
    """
    await websocket.accept()

    async def ws_send(event: str, data: dict) -> None:
        await websocket.send_text(json.dumps({"event": event, "data": data}, ensure_ascii=False))

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "ignite":
                main_query = msg.get("query", "")
                num_tasks = min(max(int(msg.get("num_tasks", 5)), 1), 250)
                model_id = msg.get("model", "")
                provider_str = msg.get("provider", "")
                system_prompt_id = msg.get("system_prompt", "market_analyst")
                custom_prompt = msg.get("custom_prompt", "")
                cost_budget = msg.get("cost_budget", None)
                web_search_enabled = msg.get("web_search_enabled", True)

                provider_cfg = _build_provider_config(model_id, provider_str)
                if system_prompt_id == "custom" and custom_prompt:
                    system_prompt = custom_prompt
                elif system_prompt_id in PROMPT_PRESETS:
                    system_prompt = PROMPT_PRESETS[system_prompt_id]["prompt"]
                else:
                    system_prompt = PROMPT_PRESETS["market_analyst"]["prompt"]

                cost_per_call = provider_cfg.get_model_cost(provider_cfg.default_model)

                # Cost estimate
                estimated_cost = cost_per_call * (num_tasks + 2)
                await ws_send("cost_estimate", {
                    "estimated_cost": round(estimated_cost, 6),
                    "cost_per_worker": cost_per_call,
                    "budget": cost_budget,
                })

                # Phase 1: Decompose
                await ws_send("phase", {"phase": "decomposing", "message": "Decomposing task into sub-tasks..."})
                try:
                    subtasks = await _generate_subtasks(main_query, num_tasks, provider_cfg, system_prompt)
                except Exception as e:
                    await ws_send("error", {"message": f"Decomposition failed: {e}"})
                    continue

                tasks_list = [
                    TaskInput(task_id=f"worker-{i:03d}", query=q)
                    for i, q in enumerate(subtasks)
                ]
                await ws_send("tasks", {
                    "tasks": [{"id": t.task_id, "query": t.query} for t in tasks_list],
                    "model": provider_cfg.default_model,
                    "provider": provider_cfg.provider.value,
                    "cost_per_call": cost_per_call,
                })

                # Phase 2: Execute via LangGraph
                await ws_send("phase", {"phase": "executing", "message": f"Igniting {len(tasks_list)} workers via LangGraph..."})

                thread_id = f"run-{int(time.time())}"
                run_config = RunnableConfig(configurable={"thread_id": thread_id})
                state_input = {
                    "tasks": tasks_list,
                    "current_task": None,
                    "results": [],
                    "final_results": [],
                    "flagged_results": [],
                    "global_context": None,
                    "strategy_plan": None,
                }

                completed = 0
                completed_ids: set[str] = set()

                async with get_memory_saver() as memory:
                    swarm_graph = build_swarm_graph(checkpointer=memory)
                    async for event in swarm_graph.astream(state_input, config=run_config, stream_mode="updates"):
                        if "worker_node" in event:
                            for res in event["worker_node"].get("results", []):
                                if res.task_id not in completed_ids:
                                    completed_ids.add(res.task_id)
                                    completed += 1
                                    await ws_send("result", {
                                        "task_id": res.task_id,
                                        "status": res.status,
                                        "data": {
                                            "result": res.data.get("result", ""),
                                            "confidence": res.data.get("confidence", ""),
                                            "latency_seconds": res.data.get("latency_seconds", 0),
                                            "model": res.data.get("model", ""),
                                        },
                                        "cost_usd": res.cost_usd,
                                        "completed": completed,
                                        "total": len(tasks_list),
                                    })

                await ws_send("paused", {
                    "thread_id": thread_id,
                    "message": "Workers completed. Awaiting Human-in-the-Loop review.",
                })

            elif msg_type == "approve":
                thread_id = msg.get("thread_id", "")
                model_id = msg.get("model", "")
                provider_cfg = _build_provider_config(model_id, "")
                cost_per_call = provider_cfg.get_model_cost(provider_cfg.default_model)
                run_config = RunnableConfig(configurable={"thread_id": thread_id})

                await ws_send("phase", {"phase": "synthesizing", "message": "Human approved. Proceeding to Reducer & Synthesis..."})

                async with get_memory_saver() as memory:
                    swarm_graph = build_swarm_graph(checkpointer=memory)
                    async for _ in swarm_graph.astream(None, config=run_config, stream_mode="updates"):
                        pass
                    final_state = (await swarm_graph.aget_state(run_config)).values
                    all_results = final_state.get("final_results", []) + final_state.get("flagged_results", [])

                total_cost = sum(r.cost_usd for r in all_results)
                synthesis_text = ""
                try:
                    successful = [
                        r for r in all_results
                        if r.status == "success" and r.data.get("result") and r.data.get("result") != "INSUFFICIENT DATA"
                    ]
                    if not successful:
                        synthesis_text = "No successful worker results available for synthesis."
                        await ws_send("synthesis", {"summary": synthesis_text, "cost_usd": 0.0})
                    else:
                        findings = [f"Finding {i}: {r.data['result']}" for i, r in enumerate(successful, 1)]
                        synth_prompt = (
                            "You are a senior analyst. Synthesize these parallel findings into a concise executive summary.\n\n"
                            "Requirements: Start with a one-sentence headline. Combine and deduplicate insights. "
                            "Keep it under 200 words. Use structured formatting.\n\n"
                            f"--- DATA ---\n\n{chr(10).join(findings)}\n\n--- END DATA ---\n\nExecutive Summary:"
                        )
                        messages = [
                            {"role": "system", "content": "You are a senior analyst creating executive summaries."},
                            {"role": "user", "content": synth_prompt},
                        ]
                        async for token in _llm_call_stream(provider_cfg, messages):
                            synthesis_text += token
                            await ws_send("synthesis_token", {"token": token})
                        total_cost += cost_per_call
                        await ws_send("synthesis_complete", {"summary": synthesis_text, "cost_usd": cost_per_call})
                except Exception as e:
                    synthesis_text = f"Synthesis unavailable: {e}"
                    await ws_send("synthesis", {"summary": synthesis_text, "cost_usd": 0.0})

                success_count = sum(1 for r in all_results if r.status == "success")
                run_id = None
                try:
                    run_id = save_run(
                        query=thread_id,
                        results=all_results,
                        total_cost=total_cost,
                        total_time=0.0,
                        model=provider_cfg.default_model,
                        synthesis=synthesis_text,
                    )
                    record_run(
                        run_id=run_id,
                        query=thread_id,
                        model=provider_cfg.default_model,
                        worker_count=len(all_results),
                        success_count=success_count,
                        cost_usd=total_cost,
                        duration_sec=0.0,
                    )
                except Exception:
                    pass

                await ws_send("complete", {
                    "total_time": 0.0,
                    "total_cost": round(total_cost, 6),
                    "total_tasks": len(all_results),
                    "success_count": success_count,
                    "run_id": run_id,
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"event": "error", "data": {"message": str(e)}}))
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
