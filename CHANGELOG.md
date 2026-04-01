# Changelog

All notable changes to Supernova are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [1.0.0] — 2026-04-01

### Added
- **CLI**: `supernova run "query" --workers 5 --provider openai --output report.md`
  - Full terminal interface with colored output, real-time worker streaming
  - Markdown and JSON export via `--output file.md` / `--output file.json`
  - `supernova serve` to start web UI, `supernova runs` to list history
- **Persistent Cost Ledger** (`liquid_swarm/ledger.py`)
  - SQLite-backed cross-run cost tracking with daily/weekly/monthly rollups
  - Budget alert via `SWARM_BUDGET_USD` env var
  - `/api/ledger` endpoint for dashboard consumption
- **Export API**: `GET /api/export/{run_id}?fmt=json|md` — download any run as JSON or Markdown
- **Health & Metrics endpoints**: `GET /health` and `GET /metrics` (Prometheus-compatible)
- **Provider-aware Memory** (`liquid_swarm/memory.py`)
  - Automatic embedding backend selection: OpenAI → Ollama → HuggingFace → FakeEmbeddings
  - No longer crashes for non-OpenAI providers
- **Background Daemon** (`liquid_swarm/daemon.py`)
  - Proactive watchdog that runs scheduled swarm jobs autonomously
  - SQLite-backed cron job scheduler (`supernova_jobs.db`)
- **Makefile** with `make demo`, `make serve`, `make test`, `make docker-run` targets
- **CONTRIBUTING.md** with architecture overview and contribution guidelines
- Python 3.13 support declared in classifiers

### Fixed
- **Critical**: `synthesis.py` was hardcoded to NVIDIA API — now uses multi-provider system
- **Critical**: `save_run()` was never called in `/api/approve` — run history was not persisted
- **Critical**: `state_input` in `ignite_swarm` was missing required SwarmState keys (`final_results`, `flagged_results`, `global_context`, `strategy_plan`) — caused LangGraph state errors
- **Critical**: `deep_scrape_url()` regex bug — `r'\\s+'` matched literal `\s` instead of whitespace
- `web/app.py`: `total_time` in `complete` event always returned 0 (was a placeholder)
- `nodes.py`: removed stale development comment `# Late-bound lookup to 152`
- `asyncio.gather(*scrape_tasks)` in web_search now uses `return_exceptions=True` to prevent one failed scrape from aborting all others

### Improved
- **Web Search Retry**: DuckDuckGo and Tavily now retry up to 3 times with exponential backoff (1s, 2s, 4s) on transient failures
- `pyproject.toml`: added `[project.scripts]` entry for `supernova` CLI command, `hatchling` build backend, `ruff`/`mypy` dev deps, improved classifiers

---

## [0.9.0] — 2026-03-28

### Added
- Phase 3: Real-time web search via DuckDuckGo (SAG — Search-Augmented Generation)
- ChromaDB long-term vector memory (bootstrap + archivar nodes)
- Deep page scraping for richer search context
- Thinker Node (O1/Claude 3.7-style reasoning before worker dispatch)
- Predictive Futurist role preset
- WebP demo recordings in README

### Fixed
- Robustified architecture for 200+ parallel workers
- Worker cascade failure prevention (TimeoutError caught internally)

---

## [0.8.0] — 2026-03-20

### Added
- Multi-provider support: NVIDIA NIM, OpenAI, Anthropic, Ollama
- Human-in-the-Loop (HITL) checkpointing via LangGraph interrupt
- Worker self-correction mesh (internal sub-graph, up to 3 retry attempts)
- Confidence scoring `[CONFIDENCE: X%]` with re-querying on low scores
- Budget guard: per-call cost tracking + budget enforcement
- Assassin Node: Pydantic re-validation in reduce_node filters impossible data
- LangSmith tracing via `@traceable`
- PostgreSQL checkpointer support (production deployments)
- Token-by-token synthesis streaming via SSE

---

## [0.5.0] — 2026-03-10

### Added
- Initial LangGraph StateGraph with fan-out / fan-in architecture
- NVIDIA NIM provider (OpenAI-compatible API)
- FastAPI web server with SSE streaming
- Real-time dark-theme web UI
- Task decomposition via LLM
- Parallel worker execution with Semaphore rate limiting
- SQLite checkpointing
- JSON run persistence

[Unreleased]: https://github.com/mimitechai/supernova/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/mimitechai/supernova/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/mimitechai/supernova/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/mimitechai/supernova/compare/v0.5.0...v0.8.0
[0.5.0]: https://github.com/mimitechai/supernova/releases/tag/v0.5.0
