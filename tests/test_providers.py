"""Tests for Multi-Provider Support.

TDD: Tests written BEFORE providers.py exists.

Given/When/Then Rules:
  - GIVEN provider="nvidia" → THEN base_url is NVIDIA NIM
  - GIVEN provider="openai" → THEN base_url is api.openai.com
  - GIVEN provider="ollama" → THEN base_url is localhost:11434
  - GIVEN unknown provider → THEN raise ValueError
  - GIVEN LLM_PROVIDER env var → THEN auto-detect provider
  - GIVEN no env vars → THEN fallback to nvidia with NVIDIA_API_KEY
"""

import os
from unittest.mock import patch

import pytest

from liquid_swarm.providers import (
    LLMProvider,
    ProviderConfig,
    get_provider_config,
)


class TestProviderEnum:
    """GIVEN a provider name string
    WHEN creating a provider
    THEN the correct enum value is returned"""

    def test_nvidia_provider(self):
        assert LLMProvider.NVIDIA.value == "nvidia"

    def test_openai_provider(self):
        assert LLMProvider.OPENAI.value == "openai"

    def test_ollama_provider(self):
        assert LLMProvider.OLLAMA.value == "ollama"

    def test_anthropic_provider(self):
        assert LLMProvider.ANTHROPIC.value == "anthropic"


class TestProviderConfig:
    """GIVEN a provider enum
    WHEN creating ProviderConfig
    THEN correct base_url and auth format are set"""

    def test_nvidia_base_url(self):
        cfg = ProviderConfig(provider=LLMProvider.NVIDIA, api_key="test-key")
        assert "nvidia" in cfg.base_url or "integrate.api" in cfg.base_url

    def test_openai_base_url(self):
        cfg = ProviderConfig(provider=LLMProvider.OPENAI, api_key="sk-test")
        assert "openai.com" in cfg.base_url

    def test_ollama_base_url(self):
        cfg = ProviderConfig(provider=LLMProvider.OLLAMA, api_key="")
        assert "localhost" in cfg.base_url or "127.0.0.1" in cfg.base_url

    def test_anthropic_base_url(self):
        cfg = ProviderConfig(provider=LLMProvider.ANTHROPIC, api_key="sk-ant-test")
        assert "anthropic" in cfg.base_url

    def test_nvidia_auth_header(self):
        cfg = ProviderConfig(provider=LLMProvider.NVIDIA, api_key="nvapi-xxx")
        headers = cfg.get_headers()
        assert headers["Authorization"] == "Bearer nvapi-xxx"

    def test_ollama_no_auth_required(self):
        cfg = ProviderConfig(provider=LLMProvider.OLLAMA, api_key="")
        headers = cfg.get_headers()
        # Ollama doesn't need auth
        assert "Authorization" not in headers or headers.get("Authorization") == ""


class TestProviderModels:
    """GIVEN a provider
    WHEN querying available models
    THEN provider-specific defaults are returned"""

    def test_nvidia_default_models(self):
        cfg = ProviderConfig(provider=LLMProvider.NVIDIA, api_key="test")
        assert len(cfg.available_models) >= 3
        assert any("llama" in m["id"] for m in cfg.available_models)

    def test_openai_default_models(self):
        cfg = ProviderConfig(provider=LLMProvider.OPENAI, api_key="test")
        assert any("gpt" in m["id"] for m in cfg.available_models)

    def test_ollama_default_models(self):
        cfg = ProviderConfig(provider=LLMProvider.OLLAMA, api_key="")
        assert any("llama" in m["id"] for m in cfg.available_models)


class TestAutoDetectProvider:
    """GIVEN environment variables
    WHEN get_provider_config() is called
    THEN the correct provider is auto-detected"""

    def test_explicit_provider_env(self):
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai",
            "LLM_API_KEY": "sk-test123",
        }):
            cfg = get_provider_config()
            assert cfg.provider == LLMProvider.OPENAI

    def test_nvidia_fallback(self):
        env = os.environ.copy()
        env.pop("LLM_PROVIDER", None)
        env["NVIDIA_API_KEY"] = "nvapi-test"
        with patch.dict(os.environ, env, clear=True):
            cfg = get_provider_config()
            assert cfg.provider == LLMProvider.NVIDIA

    def test_custom_model_override(self):
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai",
            "LLM_API_KEY": "sk-test",
            "LLM_MODEL": "gpt-4o-mini",
        }):
            cfg = get_provider_config()
            assert cfg.default_model == "gpt-4o-mini"

    def test_custom_base_url(self):
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai",
            "LLM_API_KEY": "sk-test",
            "LLM_BASE_URL": "https://my-proxy.example.com/v1",
        }):
            cfg = get_provider_config()
            assert cfg.base_url == "https://my-proxy.example.com/v1"
