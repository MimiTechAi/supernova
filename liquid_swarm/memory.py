"""Swarm Long-Term Memory via ChromaDB.

Uses vector similarity search to retrieve relevant past findings
and persist new ones across runs.

Provider-aware embedding strategy:
  - If OPENAI_API_KEY or LLM_PROVIDER=openai → OpenAIEmbeddings
  - If LLM_PROVIDER=ollama → OllamaEmbeddings (nomic-embed-text)
  - Fallback → sentence-transformers via HuggingFaceEmbeddings (no key needed)
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime

from langchain_chroma import Chroma
from liquid_swarm.state import SwarmState

logger = logging.getLogger(__name__)


def get_embedding_function():
    """Return the best available embedding function based on environment.

    Resolution order:
      1. OpenAI (if OPENAI_API_KEY set or LLM_PROVIDER=openai)
      2. Ollama (if LLM_PROVIDER=ollama)
      3. HuggingFace sentence-transformers (free, local, no key needed)
    """
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    openai_key = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("LLM_API_KEY", "")

    # 1. OpenAI embeddings (best quality, needs key)
    if openai_key and provider in ("openai", "nvidia", ""):
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(api_key=openai_key)
        except Exception:
            pass

    # 2. Ollama local embeddings (free, needs Ollama running)
    if provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
            base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434")
            return OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)
        except Exception:
            pass

    # 3. HuggingFace sentence-transformers (free, local, always works)
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception:
        pass

    # 4. Last resort: fake embeddings (no-op, memory won't be meaningful
    #    but won't crash the graph)
    from langchain_core.embeddings import FakeEmbeddings
    logger.warning("[Memory] No embedding backend found — using FakeEmbeddings. "
                   "Set OPENAI_API_KEY or install langchain-community for real memory.")
    return FakeEmbeddings(size=384)


def get_vector_store() -> Chroma:
    """Return the ChromaDB vector store with the best available embeddings."""
    return Chroma(
        collection_name="supernova_global_memory",
        embedding_function=get_embedding_function(),
        persist_directory="supernova_memory.db",
    )


async def bootstrap_node(state: SwarmState) -> dict:
    """Retrieve relevant past context from long-term vector memory.

    Queries ChromaDB with the first task's query to surface related
    findings from previous swarm runs. Injected into SwarmState as
    global_context for the Thinker and Workers.
    """
    if not state.get("tasks"):
        return {"global_context": "No tasks provided."}

    query_hint = state["tasks"][0].query

    try:
        vs = get_vector_store()
        results = await vs.asimilarity_search(query_hint, k=3)
        if results:
            global_context = "\n---\n".join(r.page_content for r in results)
        else:
            global_context = "No prior knowledge available (first run)."
    except Exception as exc:
        # Graceful degradation — memory failure must never block execution
        logger.warning(f"[Bootstrap] Memory retrieval failed: {exc}")
        global_context = f"No prior knowledge available (memory unavailable: {exc})."

    return {"global_context": global_context}


async def archivar_node(state: SwarmState) -> dict:
    """Persist successful findings to the long-term vector database.

    Only saves results with status='success' and non-empty result text.
    Metadata includes task_id and timestamp for later filtering.
    """
    final_results = state.get("final_results", [])
    if not final_results:
        return {}

    docs: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for r in final_results:
        if r.status == "success" and r.data.get("result"):
            text = str(r.data["result"])
            if text and text != "INSUFFICIENT DATA":
                docs.append(text)
                metadatas.append({
                    "task_id": r.task_id,
                    "timestamp": datetime.now().isoformat(),
                    "cost_usd": str(r.cost_usd),
                })
                ids.append(str(uuid.uuid4()))

    if not docs:
        return {}

    try:
        vs = get_vector_store()
        await vs.aadd_texts(texts=docs, metadatas=metadatas, ids=ids)
        logger.info(f"[Archivar] Persisted {len(docs)} findings to memory.")
    except Exception as exc:
        logger.warning(f"[Archivar] Failed to save memory: {exc}")

    return {}
