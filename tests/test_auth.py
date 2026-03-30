"""Tests for API Key Authentication Middleware.

TDD: These tests were written BEFORE the implementation.

Given/When/Then Rules:
  - GIVEN no X-Api-Key header → THEN 401
  - GIVEN invalid X-Api-Key → THEN 401
  - GIVEN valid X-Api-Key → THEN request proceeds
  - GIVEN no SWARM_API_KEYS env var → THEN auth is DISABLED (open access)
"""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


class TestAuthRequired:
    """GIVEN a request to POST /api/ignite WITHOUT X-Api-Key header
    WHEN the server processes the request
    THEN it returns HTTP 401 with message 'API key required'"""

    def test_missing_api_key_returns_401(self):
        with patch.dict(os.environ, {"SWARM_API_KEYS": "secret-key-123"}):
            # Re-import to pick up the env var
            import importlib
            import web.app as app_module
            importlib.reload(app_module)
            client = TestClient(app_module.app)

            response = client.post("/api/ignite", json={
                "query": "Test", "num_tasks": 2, "model_tier": "BUDGET"
            })

        assert response.status_code == 401
        assert "API key required" in response.json()["detail"]


class TestInvalidApiKey:
    """GIVEN a request with an INVALID X-Api-Key header
    WHEN the server processes the request
    THEN it returns HTTP 401 with message 'Invalid API key'"""

    def test_wrong_api_key_returns_401(self):
        with patch.dict(os.environ, {"SWARM_API_KEYS": "correct-key"}):
            import importlib
            import web.app as app_module
            importlib.reload(app_module)
            client = TestClient(app_module.app)

            response = client.post(
                "/api/ignite",
                json={"query": "Test", "num_tasks": 2, "model_tier": "BUDGET"},
                headers={"X-Api-Key": "wrong-key"},
            )

        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]


class TestValidApiKey:
    """GIVEN a request with a VALID X-Api-Key header
    WHEN the server processes the request
    THEN it proceeds (no 401 error)"""

    def test_correct_api_key_passes(self):
        with patch.dict(os.environ, {"SWARM_API_KEYS": "valid-key-456"}):
            import importlib
            import web.app as app_module
            importlib.reload(app_module)
            client = TestClient(app_module.app)

            # We just check it doesn't return 401 — it may fail
            # with a different error (no NVIDIA key), but NOT 401
            response = client.post(
                "/api/ignite",
                json={"query": "Test", "num_tasks": 2, "model_tier": "BUDGET"},
                headers={"X-Api-Key": "valid-key-456"},
            )

        assert response.status_code != 401


class TestAuthDisabledWhenNotConfigured:
    """GIVEN no SWARM_API_KEYS env variable is set
    WHEN any request arrives
    THEN authentication is DISABLED (open access for dev mode)"""

    def test_no_env_var_means_open_access(self):
        env = os.environ.copy()
        env.pop("SWARM_API_KEYS", None)

        with patch.dict(os.environ, env, clear=True):
            import importlib
            import web.app as app_module
            importlib.reload(app_module)
            client = TestClient(app_module.app)

            # Should NOT return 401 when no SWARM_API_KEYS is set
            response = client.post(
                "/api/ignite",
                json={"query": "Test", "num_tasks": 2, "model_tier": "BUDGET"},
            )

        assert response.status_code != 401
