"""Synthesis module: Combine worker results into an executive summary.

After all workers complete, one final LLM call takes all successful results
and produces a coherent, structured executive summary.

This module has ZERO LangGraph imports — portable to any runtime.
Uses the multi-provider system (NVIDIA, OpenAI, Anthropic, Ollama).
"""

from __future__ import annotations

import httpx

from liquid_swarm.config import SwarmConfig
from liquid_swarm.models import TaskResult
from liquid_swarm.providers import get_provider_config


async def synthesize_results(
    results: list[TaskResult],
    config: SwarmConfig | None = None,
) -> str:
    """Synthesize all successful worker results into an executive summary.

    Filters out failed/error results, builds a context string from successful
    analyses, and makes one final LLM call to merge them into a coherent report.

    If no successful results exist, returns a fallback message without calling
    the LLM (saves cost).

    Uses the multi-provider system — works with NVIDIA, OpenAI, Anthropic, Ollama.

    Args:
        results: All worker TaskResults (mixed success/error).
        config: Swarm configuration for model selection.

    Returns:
        A string containing the executive summary.
    """
    cfg = config or SwarmConfig()

    # Filter successful results only (skip INSUFFICIENT DATA responses)
    successful = [
        r for r in results
        if r.status == "success"
        and r.data.get("result")
        and r.data.get("result") != "INSUFFICIENT DATA"
    ]

    # Fallback: no successful results → no LLM call
    if not successful:
        return (
            "No successful worker results available for synthesis. "
            "All workers failed or timed out."
        )

    # Build context from successful analyses
    findings = []
    for i, result in enumerate(successful, 1):
        confidence = result.data.get("confidence", "")
        findings.append(f"Finding {i} {confidence}: {result.data['result']}")
    context = "\n\n".join(findings)

    prompt = (
        "You are a senior analyst. Below are findings from multiple parallel "
        "research agents. Synthesize them into a coherent executive summary.\n\n"
        "Requirements:\n"
        "- Start with a one-sentence headline\n"
        "- Combine and deduplicate insights\n"
        "- Highlight key numbers and trends\n"
        "- Keep it under 200 words\n"
        "- Use structured formatting (bold headers, bullet points)\n\n"
        f"--- FINDINGS ---\n\n{context}\n\n--- END FINDINGS ---\n\n"
        "Executive Summary:"
    )

    # Use the multi-provider system
    provider_cfg = get_provider_config()

    payload = {
        "model": provider_cfg.default_model or cfg.model_id,
        "messages": [
            {
                "role": "system",
                "content": "You are a senior analyst who creates clear, structured executive summaries.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 768,
        "temperature": 0.3,
        "stream": False,
    }

    headers = provider_cfg.get_headers()

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{provider_cfg.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()

    body = response.json()
    return body["choices"][0]["message"]["content"]
