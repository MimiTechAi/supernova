"""Multi-Provider Support: NVIDIA NIM, OpenAI, Anthropic, Ollama.

Users bring their own API keys. All providers use the OpenAI-compatible
/v1/chat/completions endpoint (or compatible wrappers).

Configuration via environment variables:
  LLM_PROVIDER   = nvidia | openai | anthropic | ollama  (default: nvidia)
  LLM_API_KEY    = your-api-key (or NVIDIA_API_KEY for backward compat)
  LLM_MODEL      = model override (optional)
  LLM_BASE_URL   = custom endpoint override (optional, e.g. for proxies)

Copyright 2026 MiMi Tech Ai UG, Bad Liebenzell, Germany.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import os
from enum import Enum

from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    NVIDIA = "nvidia"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


# ── Provider Defaults ────────────────────────────────────────────────────────

_PROVIDER_DEFAULTS: dict[LLMProvider, dict] = {
    LLMProvider.NVIDIA: {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "models": [
            {"id": "meta/llama-3.1-8b-instruct", "name": "Llama 3.1 8B (Budget)", "cost": 0.0003},
            {"id": "meta/llama-3.1-70b-instruct", "name": "Llama 3.1 70B (Standard)", "cost": 0.002},
            {"id": "nvidia/llama-3.1-nemotron-70b-instruct", "name": "Nemotron 70B (Premium)", "cost": 0.005},
        ],
        "default_model": "meta/llama-3.1-8b-instruct",
        "requires_auth": True,
    },
    LLMProvider.OPENAI: {
        "base_url": "https://api.openai.com/v1",
        "models": [
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini (Budget)", "cost": 0.0003},
            {"id": "gpt-4o", "name": "GPT-4o (Standard)", "cost": 0.005},
            {"id": "o3-mini", "name": "o3-mini (Premium)", "cost": 0.01},
        ],
        "default_model": "gpt-4o-mini",
        "requires_auth": True,
    },
    LLMProvider.ANTHROPIC: {
        "base_url": "https://api.anthropic.com/v1",
        "models": [
            {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku (Budget)", "cost": 0.001},
            {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4 (Standard)", "cost": 0.003},
            {"id": "claude-opus-4-20250514", "name": "Claude Opus 4 (Premium)", "cost": 0.015},
        ],
        "default_model": "claude-3-5-haiku-20241022",
        "requires_auth": True,
    },
    LLMProvider.OLLAMA: {
        "base_url": "http://localhost:11434/v1",
        "models": [
            {"id": "llama3.2:3b", "name": "Llama 3.2 3B (Local)", "cost": 0.0},
            {"id": "llama3.1:8b", "name": "Llama 3.1 8B (Local)", "cost": 0.0},
            {"id": "mistral:7b", "name": "Mistral 7B (Local)", "cost": 0.0},
        ],
        "default_model": "llama3.1:8b",
        "requires_auth": False,
    },
}


class ProviderConfig(BaseModel):
    """Configuration for a specific LLM provider.

    Holds base URL, authentication, available models, and defaults.
    Can be overridden with custom base_url and model via env vars.
    """

    provider: LLMProvider
    api_key: str = ""
    base_url: str = ""
    default_model: str = ""

    def model_post_init(self, __context) -> None:
        """Set defaults from provider lookup if not overridden."""
        defaults = _PROVIDER_DEFAULTS[self.provider]
        if not self.base_url:
            self.base_url = defaults["base_url"]
        if not self.default_model:
            self.default_model = defaults["default_model"]

    @property
    def available_models(self) -> list[dict]:
        """Return available models for this provider."""
        return _PROVIDER_DEFAULTS[self.provider]["models"]

    @property
    def requires_auth(self) -> bool:
        """Whether this provider requires an API key."""
        return _PROVIDER_DEFAULTS[self.provider]["requires_auth"]

    def get_headers(self) -> dict[str, str]:
        """Build HTTP headers for the provider's API.

        Each provider may use a different auth scheme:
          - NVIDIA/OpenAI: Bearer token in Authorization header
          - Anthropic: x-api-key header
          - Ollama: no auth needed (local)
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }

        if self.provider == LLMProvider.ANTHROPIC:
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        elif self.requires_auth and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def get_model_cost(self, model_id: str) -> float:
        """Look up cost for a specific model ID."""
        for model in self.available_models:
            if model["id"] == model_id:
                return model["cost"]
        # Unknown model — return a safe default
        return 0.001


def get_provider_config() -> ProviderConfig:
    """Auto-detect provider from environment variables.

    Resolution order:
      1. LLM_PROVIDER + LLM_API_KEY (explicit)
      2. NVIDIA_API_KEY (backward compat)
      3. Fallback to NVIDIA with empty key

    Optional overrides:
      - LLM_MODEL: override the default model
      - LLM_BASE_URL: override the base URL (e.g. for proxies)
    """
    # Explicit provider
    provider_str = os.environ.get("LLM_PROVIDER", "").strip().lower()
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    model_override = os.environ.get("LLM_MODEL", "").strip()
    base_url_override = os.environ.get("LLM_BASE_URL", "").strip()

    if provider_str:
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            raise ValueError(
                f"Unknown LLM_PROVIDER: '{provider_str}'. "
                f"Supported: {', '.join(p.value for p in LLMProvider)}"
            )
    else:
        # Backward compat: check for NVIDIA_API_KEY
        nvidia_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if nvidia_key:
            provider = LLMProvider.NVIDIA
            api_key = nvidia_key
        else:
            provider = LLMProvider.NVIDIA

    return ProviderConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url_override or "",
        default_model=model_override or "",
    )
