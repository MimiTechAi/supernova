"""Persistent Cost Ledger — tracks cumulative spend across all swarm runs.

Uses SQLite for zero-dependency persistence. Provides per-run and
aggregate cost tracking with monthly rollups.

Design:
  - One row per completed run
  - Aggregate queries for daily/weekly/monthly totals
  - Budget alert threshold configurable via SWARM_BUDGET_USD env var
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DB_PATH = Path(os.environ.get("SUPERNOVA_LEDGER_DB", "supernova_ledger.db"))
_BUDGET_USD = float(os.environ.get("SWARM_BUDGET_USD", "0"))  # 0 = unlimited


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_ledger() -> None:
    """Initialize the ledger database (idempotent)."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cost_ledger (
                run_id        TEXT PRIMARY KEY,
                timestamp     TEXT NOT NULL,
                query         TEXT NOT NULL,
                model         TEXT NOT NULL,
                worker_count  INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                cost_usd      REAL NOT NULL DEFAULT 0.0,
                duration_sec  REAL NOT NULL DEFAULT 0.0
            )
        """)
        conn.commit()


def record_run(
    run_id: str,
    query: str,
    model: str,
    worker_count: int,
    success_count: int,
    cost_usd: float,
    duration_sec: float,
) -> None:
    """Record a completed run in the ledger.

    Args:
        run_id: Unique run identifier.
        query: The original user query.
        model: Model ID used.
        worker_count: Total workers spawned.
        success_count: Workers that succeeded.
        cost_usd: Total cost of this run.
        duration_sec: Wall-clock time in seconds.
    """
    init_ledger()
    try:
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cost_ledger
                    (run_id, timestamp, query, model, worker_count, success_count, cost_usd, duration_sec)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    query,
                    model,
                    worker_count,
                    success_count,
                    cost_usd,
                    duration_sec,
                ),
            )
            conn.commit()

        # Budget alert
        if _BUDGET_USD > 0:
            total = get_total_spend()
            if total >= _BUDGET_USD:
                logger.warning(
                    f"[Ledger] BUDGET EXCEEDED: ${total:.4f} >= ${_BUDGET_USD:.2f}. "
                    "Set SWARM_BUDGET_USD=0 to disable this guard."
                )
    except Exception as exc:
        logger.warning(f"[Ledger] Failed to record run {run_id}: {exc}")


def get_total_spend(since_days: int | None = None) -> float:
    """Return total spend in USD, optionally limited to last N days."""
    init_ledger()
    try:
        with _get_conn() as conn:
            if since_days:
                from datetime import timedelta
                cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
                row = conn.execute(
                    "SELECT SUM(cost_usd) FROM cost_ledger WHERE timestamp >= ?", (cutoff,)
                ).fetchone()
            else:
                row = conn.execute("SELECT SUM(cost_usd) FROM cost_ledger").fetchone()
        return float(row[0] or 0.0)
    except Exception:
        return 0.0


def get_ledger_summary() -> dict:
    """Return a summary of all spending for the dashboard."""
    init_ledger()
    try:
        with _get_conn() as conn:
            total = conn.execute("SELECT SUM(cost_usd), COUNT(*) FROM cost_ledger").fetchone()
            daily = conn.execute(
                "SELECT SUM(cost_usd) FROM cost_ledger WHERE timestamp >= date('now', '-1 day')"
            ).fetchone()
            weekly = conn.execute(
                "SELECT SUM(cost_usd) FROM cost_ledger WHERE timestamp >= date('now', '-7 days')"
            ).fetchone()
            monthly = conn.execute(
                "SELECT SUM(cost_usd) FROM cost_ledger WHERE timestamp >= date('now', '-30 days')"
            ).fetchone()
            recent = conn.execute(
                "SELECT run_id, timestamp, query, model, cost_usd, duration_sec, "
                "success_count, worker_count FROM cost_ledger "
                "ORDER BY timestamp DESC LIMIT 20"
            ).fetchall()

        return {
            "total_cost_usd": round(float(total[0] or 0), 6),
            "total_runs": int(total[1] or 0),
            "daily_cost_usd": round(float(daily[0] or 0), 6),
            "weekly_cost_usd": round(float(weekly[0] or 0), 6),
            "monthly_cost_usd": round(float(monthly[0] or 0), 6),
            "budget_usd": _BUDGET_USD,
            "budget_used_pct": round((float(total[0] or 0) / _BUDGET_USD * 100), 1) if _BUDGET_USD > 0 else None,
            "recent_runs": [dict(r) for r in recent],
        }
    except Exception as exc:
        logger.warning(f"[Ledger] Summary query failed: {exc}")
        return {
            "total_cost_usd": 0.0,
            "total_runs": 0,
            "daily_cost_usd": 0.0,
            "weekly_cost_usd": 0.0,
            "monthly_cost_usd": 0.0,
            "budget_usd": _BUDGET_USD,
            "budget_used_pct": None,
            "recent_runs": [],
        }
