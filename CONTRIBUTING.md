# Contributing to Supernova

Thank you for your interest in contributing! Supernova is an open-source project and we welcome contributions of all kinds.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Good First Issues](#good-first-issues)

---

## Code of Conduct

Be kind, inclusive, and constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

---

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/supernova.git`
3. **Create a branch**: `git checkout -b feat/my-feature`
4. **Make changes**, run tests, commit
5. **Open a Pull Request** against `main`

---

## Development Setup

```bash
# 1. Clone and enter
git clone https://github.com/mimitechai/supernova.git
cd supernova

# 2. Create virtualenv (Python 3.12+)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install with dev extras
make install-dev
# or: pip install -e ".[dev,memory-local]"

# 4. Copy and configure .env
make env
# Edit .env with your API keys
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | No | `nvidia` \| `openai` \| `anthropic` \| `ollama` (default: `nvidia`) |
| `LLM_API_KEY` | Yes* | Your provider's API key (*not needed for Ollama) |
| `NVIDIA_API_KEY` | No | NVIDIA NIM key (legacy, still works) |
| `TAVILY_API_KEY` | No | Tavily search key (falls back to DuckDuckGo) |
| `OPENAI_API_KEY` | No | Used for ChromaDB embeddings if available |
| `LANGSMITH_API_KEY` | No | Enable LangSmith tracing |

---

## Running Tests

```bash
# Unit tests only (no API keys needed)
make test

# With coverage report
make test-cov

# All tests including live API calls
make test-live
```

Tests are in `tests/`. New features should come with tests. Aim to keep coverage above 85%.

---

## How to Contribute

### Bug Reports

Open a [Bug Report](https://github.com/mimitechai/supernova/issues/new?template=bug_report.md) with:
- What you expected vs. what happened
- Minimal reproduction steps
- Python version, OS, and relevant `.env` settings (never share API keys!)

### Feature Requests

Open a [Feature Request](https://github.com/mimitechai/supernova/issues/new?template=feature_request.md) with:
- The use case / problem you're solving
- Proposed solution or approach

### New LLM Providers

Add a new entry to `_PROVIDER_DEFAULTS` in `liquid_swarm/providers.py` and add corresponding tests in `tests/test_providers.py`.

### New Worker Roles / Prompt Presets

Add to `PROMPT_PRESETS` in `web/app.py`. Keep prompts concise, actionable, and role-specific.

---

## Pull Request Guidelines

- **One concern per PR** — don't mix bug fixes with new features
- **Tests required** — all new code needs test coverage
- **No breaking changes** to public APIs without discussion in an issue first
- **Update CHANGELOG.md** under `[Unreleased]` with your changes
- PR title format: `feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`, `test: ...`

---

## Good First Issues

Look for issues labeled [`good first issue`](https://github.com/mimitechai/supernova/issues?q=label%3A%22good+first+issue%22) — these are well-scoped tasks that don't require deep context.

Ideas for good contributions:
- Add a new prompt preset (Security Analyst, Medical Researcher, etc.)
- Add tests for an untested module
- Improve error messages with more context
- Add a new export format (CSV, PDF via fpdf2)
- Translate the README to another language

---

## Architecture Overview

```
START → bootstrap_node (ChromaDB memory)
      → thinker_node (O1-style reasoning strategy)
      → route_to_workers (fan-out: N × Send())
      → worker_node × N (parallel, self-correcting mesh)
             ↓ (fan-in via operator.add reducer)
      → reduce_node (Assassin-node: Pydantic re-validation)
      → archivar_node (persist findings to ChromaDB)
      → END
```

Key files:
- `liquid_swarm/nodes.py` — core execution logic
- `liquid_swarm/graph.py` — LangGraph StateGraph builder
- `liquid_swarm/state.py` — SwarmState TypedDict
- `liquid_swarm/web_search.py` — search engine abstraction
- `web/app.py` — FastAPI + SSE streaming server
- `liquid_swarm/cli.py` — CLI interface

---

Questions? Open a [Discussion](https://github.com/mimitechai/supernova/discussions) or file an issue.
