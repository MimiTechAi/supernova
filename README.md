<div align="center">

# 🌌 SUPERNOVA

### Massively Parallel AI Agent Orchestration

**Decompose → Ignite N Agents → Ground with Web Search → Synthesize**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/supernova-swarm.svg)](https://pypi.org/project/supernova-swarm/)
[![Status](https://img.shields.io/badge/status-production_ready-success.svg)]()
[![LangGraph](https://img.shields.io/badge/built_on-LangGraph-6366f1.svg)](https://github.com/langchain-ai/langgraph)
[![Providers](https://img.shields.io/badge/providers-OpenAI%20%7C%20Anthropic%20%7C%20NVIDIA%20%7C%20Ollama-orange.svg)]()
[![CHANGELOG](https://img.shields.io/badge/changelog-v1.0.0-informational.svg)](CHANGELOG.md)

<br />

<img src="assets/intro-demo.webp" width="100%" alt="Supernova Architecture in Action" />

<br />

*A complex query enters. A supernova ignites — dozens of specialized AI agents explode outward in parallel,
each grounded in real-time web search. Results collapse back inward into a single, precise executive summary.*

</div>

---

## TL;DR — One Command Demo

```bash
# With Ollama (free, no API key)
pip install supernova-swarm
supernova run "Analyze the global AI chip market in 2026" --provider ollama --workers 5

# With OpenAI
OPENAI_API_KEY=sk-... supernova run "Analyze the global AI chip market" --workers 8

# Start the Web UI
supernova serve  # → http://localhost:8000
```

---

## Why Supernova?

Single LLM calls hit hard limits: context windows, response time, and hallucination rate all degrade as complexity grows.
Supernova solves this with a **map-reduce agent architecture** — break the problem, solve it in parallel, merge the truth.

| Feature | Single LLM | Supernova |
|---------|-----------|-----------|
| Complex multi-faceted queries | Struggles | Native — decomposed into N tasks |
| Hallucination rate | High | Reduced via real-time web search (SAG) |
| Response speed (complex queries) | Sequential | **Parallel** — N workers run simultaneously |
| Context window limit | ~200k tokens | Unlimited — each worker has its own context |
| Provider flexibility | Locked-in | OpenAI · Anthropic · NVIDIA · Ollama |
| Cost control | None | Hard budget guards + per-run cost ledger |
| Memory across runs | None | ChromaDB vector memory (bootstrapped each run) |

---

## vs. Other Agent Frameworks

| | **Supernova** | AutoGen | CrewAI | LangGraph Vanilla |
|---|---|---|---|---|
| Parallel execution | ✅ Native fan-out | ❌ Sequential by default | ⚠️ Limited | ⚠️ Manual |
| Real-time web search | ✅ Built-in (DDG + Tavily) | ❌ Plugin only | ❌ Plugin only | ❌ Manual |
| Provider agnostic | ✅ 4 providers | ⚠️ OpenAI-first | ⚠️ OpenAI-first | ✅ |
| Self-correcting workers | ✅ Internal mesh graph | ❌ | ❌ | ❌ |
| Human-in-the-Loop | ✅ LangGraph HITL | ⚠️ Basic | ❌ | ✅ |
| Budget guard | ✅ Hard stop | ❌ | ❌ | ❌ |
| Assassin Node (data validation) | ✅ Pydantic re-validation | ❌ | ❌ | ❌ |
| Vector memory across runs | ✅ ChromaDB | ❌ | ❌ | ❌ |
| Production web UI | ✅ FastAPI + SSE | ❌ | ❌ | ❌ |
| CLI | ✅ `supernova run` | ❌ | ❌ | ❌ |
| 200+ parallel workers | ✅ Tested | ❌ | ❌ | ⚠️ |

---

## Architecture

```
User Query
    │
    ▼
bootstrap_node ──── ChromaDB: retrieve 3 most relevant past findings
    │
    ▼
thinker_node ─────── O1-style reasoning: analyze tasks, generate strategy directives
    │
    ▼ route_to_workers (fan-out: N × Send())
    │
   ┌─────────────────────────────────────────────┐
   │  worker_node × N  (fully parallel)          │
   │  ┌─────────────────────────────────────┐    │
   │  │  generate_node                      │    │
   │  │    1. DuckDuckGo / Tavily search    │    │
   │  │    2. Deep page scraping            │    │
   │  │    3. LLM call with tool binding    │    │
   │  │    4. Structured output (Pydantic)  │    │
   │  └──────────────┬──────────────────────┘    │
   │                 │ evaluate_edge              │
   │                 │ (confidence < 80% → retry, max 3×)
   │  ┌──────────────▼──────────────────────┐    │
   │  │  TaskResult (status, confidence,    │    │
   │  │  data, cost_usd, latency)           │    │
   │  └─────────────────────────────────────┘    │
   └──────────────────────┬──────────────────────┘
                          │ fan-in (operator.add reducer)
                          ▼
                    reduce_node ─── Assassin Node: Pydantic re-validation
                          │         flags impossible data (market_share > 100%)
                          ▼
                    archivar_node ── persist to ChromaDB
                          │
                          ▼
                        END ── Executive Summary (streaming)
```

### Key Nodes Explained

**🧠 Thinker Node** — Before any worker runs, a high-capability LLM reasons about the full task list,
identifying cross-dependencies and generating strategic directives. Workers receive this strategy
as additional context, improving coherence across independent results.

**⚔️ Assassin Node (reduce_node)** — Every TaskResult is re-validated through Pydantic after the
worker returns it. Results containing mathematically impossible values (e.g., `market_share=150%`)
are silently flagged into `flagged_results` instead of crashing the graph or polluting synthesis.
This is inline red-teaming at the data layer.

**🗂️ Archivar Node** — Successful findings are embedded and stored in ChromaDB with timestamps.
On the next run, the Bootstrap Node retrieves the 3 most relevant past findings as context,
giving the swarm memory across sessions.

---

## Quick Start

### Option A: CLI (Recommended)

```bash
pip install supernova-swarm

# Free local demo — no API key needed (requires Ollama)
ollama pull llama3.1:8b
supernova run "What are the top 5 AI breakthroughs of 2026?" \
  --provider ollama --workers 5

# With OpenAI
export OPENAI_API_KEY=sk-...
supernova run "Analyze the global EV battery supply chain" \
  --provider openai --model gpt-4o --workers 8 --output report.md

# Start web UI
supernova serve
```

### Option B: Web UI

```bash
git clone https://github.com/mimitechai/supernova.git
cd supernova

# Install
pip install -e .
# or: make install

# Configure
cp .env.example .env
# Edit .env with your API keys

# Start
make serve
# → http://localhost:8000
```

### Option C: Docker

```bash
git clone https://github.com/mimitechai/supernova.git
cd supernova
cp .env.example .env   # edit with your keys
docker-compose up -d
# → http://localhost:8000
```

---

## CLI Reference

```
supernova run "query" [OPTIONS]

  -w, --workers INT      Number of parallel agents (default: 5, max: 250)
  -p, --provider STR     nvidia | openai | anthropic | ollama
  -m, --model STR        Model override (e.g. gpt-4o, claude-opus-4)
  --no-search            Disable real-time web search
  --no-save              Don't persist the run to disk
  -o, --output FILE      Export report (.json or .md)

supernova serve [OPTIONS]
  --host STR             Bind host (default: 0.0.0.0)
  --port INT             Port (default: 8000)
  --reload               Hot-reload (dev mode)

supernova runs [OPTIONS]
  --last INT             Number of runs to show (default: 20)
```

---

## Configuration

```env
# .env — copy from .env.example

# Provider (nvidia | openai | anthropic | ollama)
LLM_PROVIDER=openai
LLM_API_KEY=sk-...

# Optional: model override
LLM_MODEL=gpt-4o-mini

# Optional: use Tavily for more reliable search
TAVILY_API_KEY=tvly-...

# Optional: budget guard (0 = unlimited)
SWARM_BUDGET_USD=1.00

# Optional: protect web endpoints with API key
SWARM_API_KEYS=your-secret-key

# Optional: LangSmith tracing
LANGSMITH_API_KEY=lsv2_...
LANGCHAIN_TRACING_V2=true
```

---

## Providers

| Provider | Model Examples | Cost | Notes |
|----------|---------------|------|-------|
| **NVIDIA NIM** | Llama 3.1 8B/70B, Nemotron | $0.0003–$0.005/call | Fast, cheap |
| **OpenAI** | gpt-4o-mini, gpt-4o, o3-mini | $0.0003–$0.01/call | Most capable |
| **Anthropic** | Claude 3.5 Haiku, Sonnet 4, Opus 4 | $0.001–$0.015/call | Best for reasoning |
| **Ollama** | Llama 3.2 3B, Llama 3.1 8B, Mistral 7B | Free | Local, no internet needed |

Switch providers with one env var: `LLM_PROVIDER=anthropic LLM_API_KEY=sk-ant-...`

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/config` | GET | Provider config + available models |
| `/api/prompts` | GET | System prompt presets |
| `/api/ignite` | POST | Start swarm (SSE stream) |
| `/api/approve` | POST | Resume after Human-in-the-Loop review |
| `/api/runs` | GET | List all saved runs |
| `/api/runs/{id}` | GET | Get specific run |
| `/api/export/{id}` | GET | Download run as JSON or Markdown |
| `/api/ledger` | GET | Cost ledger (daily/weekly/monthly) |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus-compatible metrics |

---

## Advanced Usage

### Human-in-the-Loop (HITL)

The swarm pauses after all workers complete and before synthesis. Review individual results, then approve:

```python
# POST /api/ignite → returns thread_id when paused
# Review results in the UI
# POST /api/approve {"thread_id": "...", "provider": "openai"}
```

### Programmatic Python API

```python
import asyncio
from liquid_swarm.models import TaskInput
from liquid_swarm.graph import build_swarm_graph

async def main():
    graph = build_swarm_graph()  # no checkpointer = no HITL pause
    result = await graph.ainvoke({
        "tasks": [
            TaskInput(task_id="t1", query="What is the current global LLM market size?"),
            TaskInput(task_id="t2", query="Who are the top 5 AI chip manufacturers in 2026?"),
            TaskInput(task_id="t3", query="What are the main regulatory risks for AI companies?"),
        ],
        "current_task": None,
        "results": [],
        "final_results": [],
        "flagged_results": [],
        "global_context": None,
        "strategy_plan": None,
    })
    for r in result["final_results"]:
        print(f"[{r.task_id}] {r.data['result'][:200]}")

asyncio.run(main())
```

### Background Daemon (Autonomous Monitoring)

```python
from liquid_swarm.daemon import start_daemon
# Starts a background watchdog that runs scheduled swarm jobs every N seconds
# Jobs stored in supernova_jobs.db (SQLite)
start_daemon()
```

---

## Prompt Presets

| Preset | Use Case |
|--------|----------|
| **Market Analyst** | Market sizing, competitive landscape, financial analysis |
| **Research Scientist** | Academic literature, evidence evaluation, hypothesis testing |
| **Code Reviewer** | Bugs, security vulnerabilities, performance, best practices |
| **Legal Analyst** | Regulatory compliance, jurisdiction-specific risks |
| **Predictive Futurist** | 3–5 year trend extrapolation, structural shifts |
| **Custom** | Define your own system prompt |

---

## Testing

```bash
make test           # Unit tests only (no API keys needed)
make test-cov       # With coverage HTML report → htmlcov/
make test-live      # All tests including live API calls (needs keys)
```

Coverage target: **85%** (enforced via pytest-cov).
Test suite: **17 test files**, BDD scenarios (pytest-bdd), async-first (pytest-asyncio).

---

## Deployment

### Docker Compose (Recommended)

```bash
docker-compose up -d --build
# Web UI: http://localhost:8000
# Prometheus metrics: http://localhost:8000/metrics
```

Volumes:
- `./runs/` — JSON run history
- `./supernova_memory.db/` — ChromaDB vector memory
- `./supernova_checkpoints.db` — LangGraph HITL checkpoints
- `./supernova_ledger.db` — Cost ledger

### PostgreSQL Checkpoints (Production)

```env
POSTGRES_URL=postgresql+asyncpg://user:pass@host:5432/supernova
```

### Kubernetes

Supernova workers are stateless — the only shared state is the SQLite/PostgreSQL checkpointer.
Scale horizontally with standard HPA on CPU/memory. Each pod is independent.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, guidelines, and good first issues.

```bash
# Quick contribution setup
git clone https://github.com/mimitechai/supernova.git && cd supernova
make install-dev
make test
```

---

## Roadmap

- [ ] Hugging Face Space (live hosted demo)
- [ ] PDF export via `fpdf2`
- [ ] Multi-provider race mode (run same query on 3 providers, compare)
- [ ] Prometheus + Grafana dashboard template
- [ ] `pip install supernova-swarm` → PyPI publish
- [ ] LangGraph Cloud deployment support
- [ ] gRPC streaming API

---

<div align="center">

**[Documentation](https://github.com/mimitechai/supernova#readme)** ·
**[Issues](https://github.com/mimitechai/supernova/issues)** ·
**[Discussions](https://github.com/mimitechai/supernova/discussions)** ·
**[CHANGELOG](CHANGELOG.md)**

<br />

<sub>Built with love by <a href="https://mimitech.ai">MiMi Tech Ai UG</a>, Bad Liebenzell, Germany.</sub><br />
<sub>Copyright © 2026. Distributed under the <a href="LICENSE">Apache License 2.0</a>.</sub>

</div>
