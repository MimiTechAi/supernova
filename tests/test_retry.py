"""Tests for Worker Retry with Exponential Backoff.

TDD: These tests were written BEFORE the implementation.

Given/When/Then Rules:
  - GIVEN a worker that fails with a transient error → THEN retry 1x
  - GIVEN a worker that succeeds on retry → THEN status is "success"
  - GIVEN a permanent error (401) → THEN no retry
  - GIVEN both attempts fail → THEN cost is 0
"""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest

from liquid_swarm.models import TaskInput, TaskResult
from liquid_swarm.nodes import execute_task_with_retry
from liquid_swarm.config import SwarmConfig, ModelTier


@pytest.fixture
def sample_task():
    return TaskInput(task_id="retry-001", query="Test query for retry")


@pytest.fixture
def config():
    return SwarmConfig(model_tier=ModelTier.BUDGET)


class TestRetryTransientErrors:
    """GIVEN a worker that fails with a transient error (HTTP 500, timeout)
    WHEN the worker executes
    THEN it retries exactly 1 time before marking as 'error'"""

    @pytest.mark.asyncio
    async def test_retries_once_on_http_500(self, sample_task, config):
        """HTTP 500 is transient — should retry once."""
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_fail.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error", request=MagicMock(), response=mock_response_fail
            )
        )

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_response_fail

        with patch("liquid_swarm.nodes.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await execute_task_with_retry(sample_task, config, max_retries=1)

        assert call_count == 2, "Should call API exactly 2 times (1 original + 1 retry)"
        assert result.status == "error"


class TestRetrySuccessOnSecondAttempt:
    """GIVEN a worker that fails on first attempt but succeeds on retry
    WHEN the worker executes
    THEN the final result status is 'success'"""

    @pytest.mark.asyncio
    async def test_succeeds_on_retry(self, sample_task, config):
        """First call fails with 500, second call succeeds."""
        call_count = 0

        mock_response_ok = MagicMock()
        mock_response_ok.status_code = 200
        mock_response_ok.raise_for_status = MagicMock()
        mock_response_ok.json = MagicMock(return_value={
            "choices": [{"message": {"content": "Analysis result"}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 100},
        })

        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_fail.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error", request=MagicMock(), response=mock_response_fail
            )
        )

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_response_fail
            return mock_response_ok

        with patch("liquid_swarm.nodes.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await execute_task_with_retry(sample_task, config, max_retries=1)

        assert result.status == "success"
        assert result.data["result"] == "Analysis result"


class TestNoPermanentRetry:
    """GIVEN a worker that fails with a permanent error (HTTP 401, invalid model)
    WHEN the worker executes
    THEN it does NOT retry and immediately returns 'error'"""

    @pytest.mark.asyncio
    async def test_no_retry_on_401(self, sample_task, config):
        """HTTP 401 is permanent (bad API key) — no retry."""
        call_count = 0

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Unauthorized", request=MagicMock(), response=mock_response
            )
        )

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_response

        with patch("liquid_swarm.nodes.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await execute_task_with_retry(sample_task, config, max_retries=1)

        assert call_count == 1, "Should NOT retry on 401"
        assert result.status == "error"


class TestRetryCostAccounting:
    """GIVEN a worker with retry enabled
    WHEN both attempts fail
    THEN cost is zero (no successful API call)"""

    @pytest.mark.asyncio
    async def test_zero_cost_on_double_failure(self, sample_task, config):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error", request=MagicMock(), response=mock_response
            )
        )

        async def mock_post(*args, **kwargs):
            return mock_response

        with patch("liquid_swarm.nodes.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await execute_task_with_retry(sample_task, config, max_retries=1)

        assert result.cost_usd == 0.0, "Failed attempts should not incur cost"
        assert result.status == "error"
