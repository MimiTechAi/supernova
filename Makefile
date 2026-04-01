.DEFAULT_GOAL := help
PYTHON        := python3
PIP           := pip3

# ── Help ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  ███████╗██╗   ██╗██████╗ ███████╗██████╗ ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗"
	@echo "  ╚════██╝╚═╝   ╚═╝╚═════╝ ╚══════╝╚═════╝ ╚══╝  ╚══╝ ╚═════╝ ╚═════╝ ╚════╝"
	@echo ""
	@echo "  Massively Parallel AI Agent Orchestration"
	@echo ""
	@echo "  Usage: make <target>"
	@echo ""
	@echo "  Setup:"
	@echo "    install         Install all dependencies"
	@echo "    install-dev     Install with dev extras"
	@echo "    env             Copy .env.example to .env (edit before use)"
	@echo ""
	@echo "  Run:"
	@echo "    demo            Run a local demo with Ollama (no API key needed)"
	@echo "    demo-openai     Run a demo with OpenAI (needs OPENAI_API_KEY)"
	@echo "    demo-nvidia     Run a demo with NVIDIA NIM (needs NVIDIA_API_KEY)"
	@echo "    serve           Start the web UI at http://localhost:8000"
	@echo "    serve-dev       Start with hot-reload (development)"
	@echo ""
	@echo "  Tests:"
	@echo "    test            Run all tests (skips live API tests)"
	@echo "    test-cov        Run tests with coverage report"
	@echo "    test-live       Run all tests including live API calls"
	@echo ""
	@echo "  Code Quality:"
	@echo "    lint            Run ruff linter"
	@echo "    format          Auto-format with ruff"
	@echo "    typecheck       Run mypy type checks"
	@echo ""
	@echo "  Docker:"
	@echo "    docker-build    Build Docker image"
	@echo "    docker-run      Run in Docker (set .env first)"
	@echo "    docker-stop     Stop Docker containers"
	@echo ""
	@echo "  Utils:"
	@echo "    runs            List recent swarm runs"
	@echo "    clean           Remove .pyc, __pycache__, .pytest_cache"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
.PHONY: install
install:
	$(PIP) install -e .

.PHONY: install-dev
install-dev:
	$(PIP) install -e ".[dev,memory-local]"

.PHONY: env
env:
	@if [ ! -f .env ]; then cp .env.example .env && echo ".env created. Edit it with your API keys."; \
	else echo ".env already exists."; fi

# ── Run ───────────────────────────────────────────────────────────────────────
.PHONY: demo
demo: ## Local demo — requires Ollama running locally
	@echo "Starting Supernova demo with Ollama (make sure Ollama is running: ollama serve)"
	@echo "Pulling llama3.1:8b if not already available..."
	-ollama pull llama3.1:8b
	LLM_PROVIDER=ollama $(PYTHON) -m liquid_swarm run \
		"What are the most important AI developments in 2026?" \
		--workers 4 --no-search

.PHONY: demo-openai
demo-openai: ## Demo with OpenAI (needs OPENAI_API_KEY in .env or env)
	LLM_PROVIDER=openai $(PYTHON) -m liquid_swarm run \
		"What are the most important AI developments in 2026?" \
		--workers 5

.PHONY: demo-nvidia
demo-nvidia: ## Demo with NVIDIA NIM (needs NVIDIA_API_KEY in .env or env)
	LLM_PROVIDER=nvidia $(PYTHON) -m liquid_swarm run \
		"What are the most important AI developments in 2026?" \
		--workers 5

.PHONY: serve
serve: ## Start the Supernova web UI
	$(PYTHON) -m liquid_swarm serve

.PHONY: serve-dev
serve-dev: ## Start with hot-reload (development mode)
	$(PYTHON) -m liquid_swarm serve --reload

# ── Tests ─────────────────────────────────────────────────────────────────────
.PHONY: test
test: ## Run unit tests (no live API calls)
	$(PYTHON) -m pytest tests/ -m "not live" -v

.PHONY: test-cov
test-cov: ## Run tests with coverage HTML report
	$(PYTHON) -m pytest tests/ -m "not live" \
		--cov=liquid_swarm \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		-v
	@echo "Coverage report: htmlcov/index.html"

.PHONY: test-live
test-live: ## Run ALL tests including live API calls (needs API keys)
	$(PYTHON) -m pytest tests/ -v

# ── Code Quality ──────────────────────────────────────────────────────────────
.PHONY: lint
lint:
	$(PYTHON) -m ruff check liquid_swarm/ web/

.PHONY: format
format:
	$(PYTHON) -m ruff format liquid_swarm/ web/
	$(PYTHON) -m ruff check --fix liquid_swarm/ web/

.PHONY: typecheck
typecheck:
	$(PYTHON) -m mypy liquid_swarm/ --ignore-missing-imports

# ── Docker ────────────────────────────────────────────────────────────────────
.PHONY: docker-build
docker-build:
	docker build -t supernova:latest .

.PHONY: docker-run
docker-run:
	docker-compose up -d
	@echo "Supernova running at http://localhost:8000"

.PHONY: docker-stop
docker-stop:
	docker-compose down

# ── Utils ─────────────────────────────────────────────────────────────────────
.PHONY: runs
runs: ## List recent swarm runs
	$(PYTHON) -m liquid_swarm runs

.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Cleaned."
