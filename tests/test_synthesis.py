"""Tests for Synthesis Report — final LLM summarization.

TDD: These tests were written BEFORE the implementation.

Given/When/Then Rules:
  - GIVEN completed worker results → THEN synthesize into executive summary
  - GIVEN mixed results (some failed) → THEN only include successful ones
  - GIVEN all results failed → THEN return fallback without LLM call
"""

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from liquid_swarm.models import TaskResult
from liquid_swarm.synthesis import synthesize_results
from liquid_swarm.config import SwarmConfig, ModelTier


@pytest.fixture
def config():
    return SwarmConfig(model_tier=ModelTier.BUDGET)


@pytest.fixture
def successful_results():
    return [
        TaskResult(
            task_id=f"worker-{i:03d}",
            status="success",
            data={"result": f"Analysis result {i} with concrete findings."},
            cost_usd=0.0003,
        )
        for i in range(5)
    ]


@pytest.fixture
def mixed_results():
    return [
        TaskResult(task_id="worker-000", status="success",
                   data={"result": "Market is growing at 15%."}, cost_usd=0.0003),
        TaskResult(task_id="worker-001", status="error",
                   data={"error": "Timeout"}, cost_usd=0.0),
        TaskResult(task_id="worker-002", status="success",
                   data={"result": "Tesla leads with 25%."}, cost_usd=0.0003),
        TaskResult(task_id="worker-003", status="error",
                   data={"error": "HTTP 500"}, cost_usd=0.0),
        TaskResult(task_id="worker-004", status="success",
                   data={"result": "Revenue reached $50B."}, cost_usd=0.0003),
    ]


class TestSynthesisWithAllSuccessful:
    """GIVEN 5 completed worker results with status 'success'
    WHEN the synthesis phase runs
    THEN it returns a single coherent text summary"""

    @pytest.mark.asyncio
    async def test_synthesis_returns_summary(self, successful_results, config):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "choices": [{"message": {"content": "Executive Summary: The AI market is booming."}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 100},
        })

        async def mock_post(*args, **kwargs):
            return mock_response

        with patch("liquid_swarm.synthesis.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            summary = await synthesize_results(successful_results, config)

        assert summary is not None
        assert len(summary) > 0
        assert "Executive Summary" in summary


class TestSynthesisWithMixedResults:
    """GIVEN worker results where 2 out of 5 failed
    WHEN the synthesis phase runs
    THEN only the 3 successful results are included in the synthesis"""

    @pytest.mark.asyncio
    async def test_synthesis_filters_failed(self, mixed_results, config):
        captured_payload = {}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "choices": [{"message": {"content": "Summary of 3 successful analyses."}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        })

        async def mock_post(*args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            return mock_response

        with patch("liquid_swarm.synthesis.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            summary = await synthesize_results(mixed_results, config)

        assert summary is not None
        # Verify the prompt sent to LLM contains only 3 results (not 5)
        user_msg = captured_payload["messages"][-1]["content"]
        assert "Timeout" not in user_msg, "Failed results should be excluded"
        assert "HTTP 500" not in user_msg, "Error results should be excluded"


class TestSynthesisWithAllFailed:
    """GIVEN an empty list of successful results (all failed)
    WHEN the synthesis phase runs
    THEN it returns a fallback message instead of calling the LLM"""

    @pytest.mark.asyncio
    async def test_synthesis_fallback_no_llm_call(self, config):
        all_failed = [
            TaskResult(task_id="worker-000", status="error",
                       data={"error": "Timeout"}, cost_usd=0.0),
            TaskResult(task_id="worker-001", status="timeout",
                       data={"error": "Timeout"}, cost_usd=0.0),
        ]

        with patch("liquid_swarm.synthesis.httpx.AsyncClient") as MockClient:
            summary = await synthesize_results(all_failed, config)
            MockClient.assert_not_called()

        assert "no successful" in summary.lower() or "no results" in summary.lower()
