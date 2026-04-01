"""Proactive background daemon for the Supernova Swarm.

Allows the swarm to wake up autonomously and monitor specific topics
or run predefined tasks continuously in the background.
"""

import asyncio
import logging
from datetime import datetime, timezone
import sqlite3

from liquid_swarm.models import TaskInput
from liquid_swarm.graph import build_swarm_graph

logger = logging.getLogger("supernova.daemon")
logging.basicConfig(level=logging.INFO)

DB_PATH = "supernova_jobs.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cron_jobs (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                schedule_interval_sec INTEGER NOT NULL,
                last_run_at TEXT,
                status TEXT DEFAULT 'active'
            )
        """)
        conn.commit()


async def run_daemon_job(job_id: str, query: str):
    logger.info(f"[Daemon] Waking up Swarm for job {job_id}: {query}")
    try:
        # Build an ephemeral graph WITHOUT checkpointer, so it runs
        # straight through the mesh without pausing for Human-in-The-Loop
        swarm_graph = build_swarm_graph(None)
        
        state_input = {
            "tasks": [TaskInput(task_id=f"daemon_{job_id}_1", query=query)],
            "current_task": None,
            "results": []
        }
        
        # This triggers: bootstrap -> thinker -> route -> worker -> reduce -> archivar
        # The Archivar will automatically persist the results!
        await swarm_graph.ainvoke(state_input)
        
        logger.info(f"[Daemon] Job {job_id} complete! Results woven into memory.")
        
    except Exception as e:
        logger.error(f"[Daemon] Job {job_id} failed: {e}")


async def _daemon_loop():
    logger.info("[Daemon] Starting Swarm Proactive Watchdog...")
    
    # Wait a bit before first execution to let the app fully initialize
    await asyncio.sleep(5)
    
    while True:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                jobs = conn.execute("SELECT * FROM cron_jobs WHERE status = 'active'").fetchall()
                
            now = datetime.now(timezone.utc)
            
            for job in jobs:
                should_run = False
                last_run = job["last_run_at"]
                
                if last_run:
                    last_dt = datetime.fromisoformat(last_run)
                    elapsed = (now - last_dt).total_seconds()
                    if elapsed >= job["schedule_interval_sec"]:
                        should_run = True
                else:
                    should_run = True
                    
                if should_run:
                    # Update last run lock
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute(
                            "UPDATE cron_jobs SET last_run_at = ? WHERE id = ?",
                            (now.isoformat(), job["id"])
                        )
                    # Spin off background task -> fully non-blocking
                    asyncio.create_task(run_daemon_job(job["id"], job["query"]))
                    
        except Exception as e:
            logger.error(f"[DaemonLoop] Error: {e}")
            
        await asyncio.sleep(60)


_daemon_task = None

def start_daemon():
    global _daemon_task
    init_db()
    
    # Insert a sample watchdog job if none exists
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM cron_jobs").fetchone()[0]
        if count == 0:
            conn.execute(
                "INSERT INTO cron_jobs (id, query, schedule_interval_sec) VALUES (?, ?, ?)",
                ("market_watch_1", "What is the latest global market sentiment on AI and LLMs?", 86400)
            )
            
    _daemon_task = asyncio.create_task(_daemon_loop())

def stop_daemon():
    global _daemon_task
    if _daemon_task:
        _daemon_task.cancel()
