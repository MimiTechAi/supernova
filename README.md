# Liquid Swarm

**Massively parallel AI agent orchestration via LangGraph & NVIDIA NIM.**

> At rest, the system exists as a single code node. Running cost: zero.  
> When a massive task arrives — like a global market analysis — we ignite a *cognitive supernova*.  
> Our algorithm slices the problem and spawns hundreds of isolated, hyper-specialized micro-agents in milliseconds.  
> They swarm out, complete hundreds of sub-tasks in parallel, validate the hard data, and self-destruct immediately after.  
> The system scales from 1 to 500 workers and back to 1 — in a single breath.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA-NIM_Powered-76b900.svg)](https://build.nvidia.com/)

---

## Architecture

```
START ──[route_to_workers]──► N × worker_node ──► reduce_node ──► END
              │                      │                  │
         Conditional Edge      Parallel Fan-Out     Fan-In Aggregation
         (N Send objects)      (Semaphore-gated)    (Assassin filtering)
```

**Key Design Decisions:**

| Feature | Implementation |
|---|---|
| **Fan-Out** | LangGraph `Send()` — one per sub-task, true parallel supersteps |
| **Rate Limiting** | `asyncio.Semaphore` — prevents HTTP 429 from NVIDIA NIM |
| **Fault Isolation** | `asyncio.wait_for()` — timeout kills 1 worker, not 49 siblings |
| **Red Teaming** | Pydantic `field_validator` — rejects impossible values (e.g. market_share > 100%) |
| **LLM Backend** | NVIDIA NIM (OpenAI-compatible) — Llama 3.1 8B/70B, Nemotron 70B |
| **Portability** | `execute_task()` has zero LangGraph imports — 1:1 portable to Modal.com |

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [NVIDIA NIM API key](https://build.nvidia.com/)

### Installation

```bash
git clone https://github.com/mimitechai/liquid-swarm.git
cd liquid-swarm
uv sync --all-extras
```

### Configuration

Create a `.env` file in the project root:

```bash
NVIDIA_API_KEY=nvapi-your-key-here
```

### Run the Web UI

```bash
uv run python -m web.app
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Run Tests

```bash
# Unit & architecture tests (no API key needed)
uv run pytest -v

# Including live API tests (requires NVIDIA_API_KEY)
uv run pytest -v --tb=short
```

## Model Tiers

| Tier | Model | Cost/Call | Use Case |
|---|---|---|---|
| ⚡ **Budget** | Llama 3.1 8B | ~$0.0003 | Bulk tasks, high volume |
| 🔥 **Standard** | Llama 3.1 70B | ~$0.002 | Balanced quality/cost |
| 💎 **Premium** | Nemotron 70B | ~$0.005 | Highest quality analysis |

## Project Structure

```
liquid-swarm/
├── liquid_swarm/
│   ├── config.py      # NVIDIA NIM model tiers, cost tables, SwarmConfig
│   ├── graph.py       # LangGraph StateGraph: build & compile
│   ├── models.py      # Pydantic models: TaskInput, TaskResult, FinalReport
│   ├── nodes.py       # Graph nodes: router, worker, reduce (+ execute_task)
│   └── state.py       # TypedDict with Annotated reducers for fan-in
├── web/
│   ├── app.py         # FastAPI backend with SSE streaming
│   └── static/
│       └── index.html # Real-time swarm visualization dashboard
├── tests/
│   ├── features/      # Gherkin BDD scenarios
│   ├── step_defs/     # pytest-bdd step definitions
│   ├── test_routing.py
│   ├── test_timeout.py
│   ├── test_rate_limit.py
│   ├── test_assassin_node.py
│   ├── test_async_worker.py
│   ├── test_cost_ledger.py
│   └── test_live_nvidia.py  # Real API integration tests
├── LICENSE            # Apache 2.0
├── pyproject.toml
└── README.md
```

## Test Coverage

The test suite covers all critical architecture invariants:

| Test | What it proves |
|---|---|
| `test_routing` | 10 tasks → exactly 10 `Send` objects, 500 tasks scales |
| `test_timeout` | 1 dead worker, 49 survivors — no `TimeoutError` propagation |
| `test_rate_limit` | Semaphore caps concurrent API calls at configured limit |
| `test_assassin_node` | Impossible values (market_share > 100%) flagged, not crashed |
| `test_async_worker` | Parallel workers produce correct results via mocked execution |
| `test_cost_ledger` | Cost tracking is deterministic and accurate |
| `test_live_nvidia` | End-to-end with real NVIDIA NIM API (skipped without key) |
| BDD scenarios | Full graph cycles: 1→50→1, rogue workers, timeout isolation |

```bash
uv run pytest --cov=liquid_swarm --cov-report=term-missing
# Target: >90% coverage
```

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run the test suite (`uv run pytest`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

Copyright 2026 [MiMi Tech Ai UG](https://mimitech.ai), Bad Liebenzell, Germany.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
