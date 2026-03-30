"""Liquid Swarm — Live Web UI Backend.

FastAPI server with SSE streaming for real-time swarm visualization.
Each worker result is streamed to the frontend as it arrives.

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

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_swarm.config import (
    NVIDIA_API_BASE,
    NVIDIA_API_KEY,
    MODEL_COST,
    ModelTier,
    SwarmConfig,
)
from liquid_swarm.models import TaskInput, TaskResult
from liquid_swarm.nodes import set_api_semaphore
from liquid_swarm.synthesis import synthesize_results
from liquid_swarm.persistence import save_run, list_runs, get_run

app = FastAPI(title="Liquid Swarm — Live UI")


# ── API Key Authentication Middleware ────────────────────────────────────────
# When SWARM_API_KEYS env var is set (comma-separated), all /api/ endpoints
# require a valid X-Api-Key header. When not set, auth is disabled (dev mode).

@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    """Protect /api/ endpoints with API key authentication."""
    swarm_keys_env = os.environ.get("SWARM_API_KEYS", "")

    # If no API keys configured → open access (dev mode)
    if not swarm_keys_env:
        return await call_next(request)

    # Only protect /api/ routes
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    # Parse valid keys (comma-separated)
    valid_keys = {k.strip() for k in swarm_keys_env.split(",") if k.strip()}

    api_key = request.headers.get("X-Api-Key")
    if not api_key:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=401,
            content={"detail": "API key required. Set X-Api-Key header."},
        )

    if api_key not in valid_keys:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid API key."},
        )

    return await call_next(request)


# Serve static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI."""
    html_path = static_dir / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/config")
async def get_config():
    """Return available models and default config."""
    return {
        "models": [
            {"id": tier.value, "name": tier.name, "cost": MODEL_COST[tier]}
            for tier in ModelTier
        ],
        "has_api_key": bool(NVIDIA_API_KEY),
    }


async def _generate_subtasks(
    main_query: str,
    num_tasks: int,
    config: SwarmConfig,
) -> list[str]:
    """Use the LLM to break a main query into N sub-tasks."""
    prompt = (
        f"You are a project manager. Break the following task into exactly "
        f"{num_tasks} specific, independent sub-tasks for parallel analysis.\n\n"
        f"Main task: {main_query}\n\n"
        f"Reply ONLY with a numbered list (1. ... 2. ... etc.), "
        f"without introduction or explanation. Each sub-task should be a concrete "
        f"analysis question that a single analyst can answer."
    )

    payload = {
        "model": config.model_id,
        "messages": [
            {"role": "system", "content": "You are a precise project manager."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.3,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{NVIDIA_API_BASE}/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()

    body = resp.json()
    text = body["choices"][0]["message"]["content"]

    # Parse numbered list
    lines = [
        line.strip()
        for line in text.strip().split("\n")
        if line.strip() and line.strip()[0].isdigit()
    ]

    # Clean up numbering
    subtasks = []
    for line in lines[:num_tasks]:
        # Remove "1. ", "2) ", etc.
        cleaned = line.lstrip("0123456789.)- ").strip()
        if cleaned:
            subtasks.append(cleaned)

    # Fallback if parsing fails
    while len(subtasks) < num_tasks:
        subtasks.append(f"Analyze aspect {len(subtasks)+1} of: {main_query}")

    return subtasks[:num_tasks]


async def _execute_single_task(
    task: TaskInput,
    semaphore: asyncio.Semaphore,
    config: SwarmConfig,
) -> TaskResult:
    """Execute a single task with semaphore rate limiting."""
    payload = {
        "model": config.model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise market analyst. Respond with structure "
                    "and concrete numbers. Maximum 4-5 sentences."
                ),
            },
            {"role": "user", "content": task.query},
        ],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    t0 = time.perf_counter()

    try:
        async with semaphore:
            async with httpx.AsyncClient(timeout=25.0) as client:
                resp = await client.post(
                    f"{NVIDIA_API_BASE}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()

        elapsed = time.perf_counter() - t0
        body = resp.json()
        llm_output = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})

        return TaskResult(
            task_id=task.task_id,
            status="success",
            data={
                "result": llm_output,
                "model": config.model_id,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "latency_seconds": round(elapsed, 3),
            },
            cost_usd=config.cost_per_call,
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


@app.post("/api/ignite")
async def ignite_swarm(request: Request):
    """Start the swarm and stream results via SSE."""
    body = await request.json()
    main_query = body.get("query", "")
    num_tasks = min(max(int(body.get("num_tasks", 5)), 1), 50)
    model_tier_str = body.get("model_tier", "BUDGET")

    try:
        model_tier = ModelTier[model_tier_str]
    except KeyError:
        model_tier = ModelTier.BUDGET

    config = SwarmConfig(
        model_tier=model_tier,
        max_tokens=512,
        temperature=0.2,
    )

    semaphore = asyncio.Semaphore(10)

    async def event_stream():
        t_total = time.perf_counter()

        # Phase 1: Generate subtasks
        yield _sse_event("phase", {"phase": "decomposing", "message": "Decomposing task into sub-tasks..."})

        try:
            subtasks = await _generate_subtasks(main_query, num_tasks, config)
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
            "model": config.model_id,
            "cost_per_call": config.cost_per_call,
        })

        # Phase 3: Execute all tasks in parallel, stream each result
        yield _sse_event("phase", {"phase": "executing", "message": f"Igniting {len(tasks)} workers..."})

        completed = 0
        total_cost = 0.0
        all_results: list[TaskResult] = []

        async def run_and_stream(task: TaskInput):
            """Run a single task and return its result."""
            return await _execute_single_task(task, semaphore, config)

        # Create all task coroutines
        coros = [run_and_stream(t) for t in tasks]

        # Use asyncio.as_completed to stream results as they finish
        for future in asyncio.as_completed(coros):
            result = await future
            completed += 1
            total_cost += result.cost_usd
            all_results.append(result)

            yield _sse_event("result", {
                "task_id": result.task_id,
                "status": result.status,
                "data": result.data,
                "cost_usd": result.cost_usd,
                "completed": completed,
                "total": len(tasks),
            })

        # Phase 4: Synthesis — combine all results into executive summary
        yield _sse_event("phase", {"phase": "synthesizing", "message": "Synthesizing executive summary..."})

        try:
            synthesis = await synthesize_results(all_results, config)
            synthesis_cost = config.cost_per_call
            total_cost += synthesis_cost
            yield _sse_event("synthesis", {
                "summary": synthesis,
                "cost_usd": synthesis_cost,
            })
        except Exception as e:
            yield _sse_event("synthesis", {
                "summary": f"Synthesis unavailable: {e}",
                "cost_usd": 0.0,
            })

        # Phase 5: Completion + Persistence
        elapsed = time.perf_counter() - t_total

        # Auto-save run to history
        synthesis_text = synthesis if 'synthesis' in dir() else None
        try:
            run_id = save_run(
                query=main_query,
                results=all_results,
                total_cost=round(total_cost, 6),
                total_time=round(elapsed, 2),
                model=config.model_id,
                synthesis=synthesis_text,
            )
        except Exception:
            run_id = None

        yield _sse_event("complete", {
            "total_time": round(elapsed, 2),
            "total_cost": round(total_cost, 6),
            "total_tasks": len(tasks),
            "success_count": completed,
            "run_id": run_id,
        })

    return StreamingResponse(
        event_stream(),
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
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": "Run not found"})
    return run


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
