# ── Stage 1: Build ──────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv for fast, deterministic dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project metadata first (cache layer for dependencies)
COPY pyproject.toml ./

# Install dependencies only (no source code yet — maximizes cache hits)
RUN uv pip install --system --no-cache -r pyproject.toml

# ── Stage 2: Runtime ────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY liquid_swarm/ ./liquid_swarm/
COPY web/ ./web/

# Create runs directory for persistence
RUN mkdir -p /app/runs

# Non-root user for security
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/config').raise_for_status()" || exit 1

# Run with uvicorn
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
