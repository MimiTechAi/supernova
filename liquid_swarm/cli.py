"""Supernova CLI — Run the swarm from your terminal.

Usage:
    python -m supernova run "Analyze the global AI chip market in 2026"
    python -m supernova run "..." --workers 10 --provider openai --model gpt-4o
    python -m supernova run "..." --no-search
    python -m supernova serve              # Start the web UI
    python -m supernova serve --port 8080
    python -m supernova runs               # List recent runs
    python -m supernova runs --last 10

Environment variables (or .env file):
    LLM_PROVIDER   = nvidia | openai | anthropic | ollama
    LLM_API_KEY    = your-api-key
    LLM_MODEL      = model override (optional)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_dotenv():
    """Load .env from the current working directory or project root."""
    try:
        from dotenv import load_dotenv
        # Try cwd first, then walk up to find .env
        for p in [Path.cwd(), *Path.cwd().parents]:
            env_file = p / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                return
    except ImportError:
        pass


def _color(text: str, code: int) -> str:
    """ANSI color if stdout is a TTY."""
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text


def _green(t: str) -> str: return _color(t, 32)
def _yellow(t: str) -> str: return _color(t, 33)
def _cyan(t: str) -> str: return _color(t, 36)
def _bold(t: str) -> str: return _color(t, 1)
def _red(t: str) -> str: return _color(t, 31)


def _print_banner():
    banner = r"""
  ███████╗██╗   ██╗██████╗ ███████╗██████╗ ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗
  ██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗████╗  ██║██╔═══██╗██║   ██║██╔══██╗
  ███████╗██║   ██║██████╔╝█████╗  ██████╔╝██╔██╗ ██║██║   ██║██║   ██║███████║
  ╚════██║██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║
  ███████║╚██████╔╝██║     ███████╗██║  ██║██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║
  ╚══════╝ ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝
    """
    print(_cyan(banner))
    print(_bold("  Massively Parallel AI Agent Orchestration — github.com/mimitechai/supernova"))
    print()


# ── run command ───────────────────────────────────────────────────────────────

async def _cmd_run(args: argparse.Namespace):
    """Execute a swarm run and stream results to stdout."""
    _ensure_dotenv()

    # Late imports so CLI starts fast without loading all deps
    from liquid_swarm.models import TaskInput
    from liquid_swarm.providers import get_provider_config, LLMProvider, ProviderConfig
    from liquid_swarm.web_search import get_search_engine, SearchCache, build_search_context
    from liquid_swarm.persistence import save_run

    # Provider setup
    provider_cfg = get_provider_config()
    if args.provider:
        try:
            provider_cfg.provider = LLMProvider(args.provider.lower())
        except ValueError:
            print(_red(f"Unknown provider: {args.provider}. Use: nvidia, openai, anthropic, ollama"))
            sys.exit(1)
    if args.model:
        provider_cfg.default_model = args.model

    num_workers = args.workers
    query = args.query
    web_search = not args.no_search

    _print_banner()
    print(_bold(f"Query: ") + query)
    print(_bold(f"Workers: ") + str(num_workers))
    print(_bold(f"Provider: ") + f"{provider_cfg.provider.value} / {provider_cfg.default_model}")
    print(_bold(f"Web Search: ") + ("enabled" if web_search else "disabled"))
    print()

    # Task decomposition via LLM
    print(_yellow("Decomposing task into sub-tasks..."))
    t0 = time.perf_counter()

    import httpx
    decomp_prompt = (
        f"Break the following task into exactly {num_workers} specific, independent sub-tasks "
        f"for parallel analysis.\n\nMain task: {query}\n\n"
        f"Reply ONLY with a numbered list (1. ... 2. ... etc.)."
    )
    payload = {
        "model": provider_cfg.default_model,
        "messages": [
            {"role": "system", "content": "You are a precise project manager."},
            {"role": "user", "content": decomp_prompt},
        ],
        "max_tokens": 2048,
        "temperature": 0.3,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{provider_cfg.base_url}/chat/completions",
                json=payload,
                headers=provider_cfg.get_headers(),
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        print(_red(f"Task decomposition failed: {exc}"))
        sys.exit(1)

    lines = [l.strip() for l in text.strip().split("\n") if l.strip() and l.strip()[0].isdigit()]
    subtasks = [l.lstrip("0123456789.)- ").strip() for l in lines[:num_workers]]
    while len(subtasks) < num_workers:
        subtasks.append(f"Analyze aspect {len(subtasks) + 1} of: {query}")

    tasks = [TaskInput(task_id=f"worker-{i:03d}", query=q) for i, q in enumerate(subtasks)]

    print(_green(f"Decomposed into {len(tasks)} sub-tasks:"))
    for t in tasks:
        print(f"  [{t.task_id}] {t.query[:90]}{'...' if len(t.query) > 90 else ''}")
    print()

    # Execute workers in parallel
    print(_yellow(f"Igniting {len(tasks)} parallel workers..."))
    semaphore = asyncio.Semaphore(min(10, len(tasks)))
    search_engine = get_search_engine() if web_search else None
    search_cache = SearchCache() if web_search else None

    async def run_worker(task: TaskInput) -> dict:
        cost_per_call = provider_cfg.get_model_cost(provider_cfg.default_model)
        worker_t0 = time.perf_counter()
        try:
            async with semaphore:
                worker_input = task.query
                if search_engine:
                    cached = search_cache.get(task.query) if search_cache else None
                    results = cached or await search_engine.search(task.query)
                    if search_cache and not cached:
                        search_cache.put(task.query, results)
                    worker_input = f"{build_search_context(results)}\n\nSUB-TASK: {task.query}"

                body_payload = {
                    "model": provider_cfg.default_model,
                    "messages": [
                        {"role": "system", "content": (
                            "You are a precise analyst. Respond with structure and concrete data. "
                            "Maximum 4-5 sentences. End with [CONFIDENCE: X%]."
                        )},
                        {"role": "user", "content": worker_input},
                    ],
                    "max_tokens": 512,
                    "temperature": 0.2,
                }
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        f"{provider_cfg.base_url}/chat/completions",
                        json=body_payload,
                        headers=provider_cfg.get_headers(),
                    )
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]

            elapsed = time.perf_counter() - worker_t0
            return {
                "task_id": task.task_id,
                "query": task.query,
                "result": content,
                "status": "success",
                "cost_usd": cost_per_call,
                "latency": round(elapsed, 2),
            }
        except Exception as exc:
            elapsed = time.perf_counter() - worker_t0
            return {
                "task_id": task.task_id,
                "query": task.query,
                "result": str(exc),
                "status": "error",
                "cost_usd": 0.0,
                "latency": round(elapsed, 2),
            }

    worker_tasks = [run_worker(t) for t in tasks]
    results_raw = []
    completed = 0
    for coro in asyncio.as_completed(worker_tasks):
        r = await coro
        results_raw.append(r)
        completed += 1
        status_icon = _green("✓") if r["status"] == "success" else _red("✗")
        print(f"  {status_icon} [{r['task_id']}] {r['latency']}s — {r['query'][:70]}...")

    print()
    success_count = sum(1 for r in results_raw if r["status"] == "success")
    total_cost = sum(r["cost_usd"] for r in results_raw)
    total_time = round(time.perf_counter() - t0, 2)

    # Print results
    print(_bold("=" * 70))
    print(_bold("WORKER RESULTS"))
    print(_bold("=" * 70))
    for r in results_raw:
        if r["status"] == "success":
            print(f"\n{_cyan(r['task_id'])}: {r['query'][:60]}...")
            print(r["result"])
            print()

    # Synthesis
    print(_yellow("Synthesizing executive summary..."))
    successful = [r for r in results_raw if r["status"] == "success"]
    if successful:
        findings = "\n\n".join(f"Finding {i}: {r['result']}" for i, r in enumerate(successful, 1))
        synth_payload = {
            "model": provider_cfg.default_model,
            "messages": [
                {"role": "system", "content": "You are a senior analyst creating executive summaries."},
                {"role": "user", "content": (
                    f"Synthesize these findings into a 200-word executive summary:\n\n{findings}"
                )},
            ],
            "max_tokens": 512,
            "temperature": 0.3,
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{provider_cfg.base_url}/chat/completions",
                    json=synth_payload,
                    headers=provider_cfg.get_headers(),
                )
                resp.raise_for_status()
                synthesis = resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            synthesis = f"Synthesis failed: {exc}"

        print(_bold("=" * 70))
        print(_bold("EXECUTIVE SUMMARY"))
        print(_bold("=" * 70))
        print(synthesis)
        print()
    else:
        synthesis = "All workers failed."

    # Stats
    print(_bold("=" * 70))
    print(f"Workers: {success_count}/{len(tasks)} succeeded")
    print(f"Total Cost: ${total_cost:.6f}")
    print(f"Total Time: {total_time}s")

    # Save run
    if not args.no_save:
        from liquid_swarm.models import TaskResult
        task_results = [
            TaskResult(
                task_id=r["task_id"],
                status=r["status"],
                data={"result": r["result"], "query": r["query"], "latency_seconds": r["latency"]},
                cost_usd=r["cost_usd"],
            )
            for r in results_raw
        ]
        try:
            run_id = save_run(
                query=query,
                results=task_results,
                total_cost=total_cost,
                total_time=total_time,
                model=provider_cfg.default_model,
                synthesis=synthesis,
            )
            print(f"Run saved: {_green(run_id)}")
        except Exception as exc:
            print(_yellow(f"Could not save run: {exc}"))

    # Export if requested
    if args.output:
        out_path = Path(args.output)
        if out_path.suffix == ".json":
            out_path.write_text(
                json.dumps({
                    "query": query,
                    "synthesis": synthesis,
                    "results": results_raw,
                    "stats": {"workers": len(tasks), "success": success_count,
                               "cost_usd": total_cost, "time_seconds": total_time},
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            # Markdown
            md_lines = [
                f"# Supernova Report\n",
                f"**Query:** {query}\n",
                f"**Date:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}\n",
                f"**Model:** {provider_cfg.default_model} | **Workers:** {len(tasks)} | **Cost:** ${total_cost:.6f}\n",
                f"\n## Executive Summary\n\n{synthesis}\n",
                f"\n## Worker Results\n",
            ]
            for r in results_raw:
                md_lines.append(f"\n### {r['task_id']}: {r['query']}\n")
                md_lines.append(f"**Status:** {r['status']} | **Latency:** {r['latency']}s\n\n")
                md_lines.append(r["result"] + "\n")
            out_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"Exported to: {_green(str(out_path))}")


# ── serve command ─────────────────────────────────────────────────────────────

def _cmd_serve(args: argparse.Namespace):
    """Start the Supernova web UI."""
    _ensure_dotenv()
    _print_banner()
    print(_bold(f"Starting Supernova Web UI on http://0.0.0.0:{args.port}"))
    print()
    try:
        import uvicorn
        uvicorn.run(
            "web.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    except ImportError:
        print(_red("uvicorn not installed. Run: pip install uvicorn[standard]"))
        sys.exit(1)


# ── runs command ──────────────────────────────────────────────────────────────

def _cmd_runs(args: argparse.Namespace):
    """List recent swarm runs."""
    _ensure_dotenv()
    from liquid_swarm.persistence import list_runs
    runs = list_runs(limit=args.last)
    if not runs:
        print("No runs found.")
        return

    print(_bold(f"{'RUN ID':<35} {'DATE':<22} {'WORKERS':>8} {'COST':>10}  QUERY"))
    print("-" * 100)
    for r in runs:
        date = r["timestamp"][:19].replace("T", " ")
        query_short = r["query"][:45] + ("..." if len(r["query"]) > 45 else "")
        cost = f"${r['total_cost']:.6f}"
        workers = f"{r['success_count']}/{r['worker_count']}"
        print(f"{r['run_id']:<35} {date:<22} {workers:>8} {cost:>10}  {query_short}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="supernova",
        description="Supernova — Massively Parallel AI Agent Orchestration",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # run
    run_parser = subparsers.add_parser("run", help="Execute a swarm run")
    run_parser.add_argument("query", help="The main analysis query")
    run_parser.add_argument("-w", "--workers", type=int, default=5, help="Number of parallel workers (default: 5)")
    run_parser.add_argument("-p", "--provider", default="", help="LLM provider: nvidia|openai|anthropic|ollama")
    run_parser.add_argument("-m", "--model", default="", help="Model override (e.g. gpt-4o)")
    run_parser.add_argument("--no-search", action="store_true", help="Disable web search augmentation")
    run_parser.add_argument("--no-save", action="store_true", help="Do not persist the run to disk")
    run_parser.add_argument("-o", "--output", default="", help="Export report to file (.json or .md)")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the web UI")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable hot-reload (dev mode)")

    # runs
    runs_parser = subparsers.add_parser("runs", help="List recent swarm runs")
    runs_parser.add_argument("--last", type=int, default=20, help="Number of runs to show (default: 20)")

    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(_cmd_run(args))
    elif args.command == "serve":
        _cmd_serve(args)
    elif args.command == "runs":
        _cmd_runs(args)
    else:
        _print_banner()
        parser.print_help()


if __name__ == "__main__":
    main()
