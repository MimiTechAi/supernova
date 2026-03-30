"""Tests for Custom System Prompts.

Given/When/Then Rules:
  - GIVEN preset "code_reviewer" → THEN prompt contains "code review"
  - GIVEN preset "custom" with user text → THEN user text is used
  - GIVEN unknown preset → THEN fallback to market_analyst
"""

import pytest
import importlib


class TestPromptPresets:
    """GIVEN a known preset ID
    WHEN resolving the prompt
    THEN the correct preset prompt is returned"""

    def test_market_analyst_preset(self):
        from web.app import PROMPT_PRESETS
        preset = PROMPT_PRESETS["market_analyst"]
        assert "market analyst" in preset["prompt"].lower()
        assert "CONFIDENCE" in preset["prompt"]

    def test_code_reviewer_preset(self):
        from web.app import PROMPT_PRESETS
        preset = PROMPT_PRESETS["code_reviewer"]
        assert "code review" in preset["prompt"].lower()
        assert "CONFIDENCE" in preset["prompt"]

    def test_research_scientist_preset(self):
        from web.app import PROMPT_PRESETS
        preset = PROMPT_PRESETS["research_scientist"]
        assert "research scientist" in preset["prompt"].lower()

    def test_legal_analyst_preset(self):
        from web.app import PROMPT_PRESETS
        preset = PROMPT_PRESETS["legal_analyst"]
        assert "legal" in preset["prompt"].lower()

    def test_custom_preset_has_empty_prompt(self):
        from web.app import PROMPT_PRESETS
        preset = PROMPT_PRESETS["custom"]
        assert preset["prompt"] == ""

    def test_all_presets_have_icons(self):
        from web.app import PROMPT_PRESETS
        for key, preset in PROMPT_PRESETS.items():
            assert "icon" in preset, f"Preset {key} missing icon"
            assert len(preset["icon"]) > 0


class TestConfidenceParsing:
    """GIVEN LLM output with [CONFIDENCE: XX%] tag
    WHEN parsing confidence
    THEN the correct integer is extracted"""

    def test_parse_confidence_normal(self):
        from web.app import _parse_confidence
        assert _parse_confidence("Analysis result. [CONFIDENCE: 85%]") == 85

    def test_parse_confidence_no_tag(self):
        from web.app import _parse_confidence
        assert _parse_confidence("Just a normal response.") is None

    def test_parse_confidence_clamped_high(self):
        from web.app import _parse_confidence
        assert _parse_confidence("[CONFIDENCE: 150%]") == 100

    def test_parse_confidence_clamped_low(self):
        from web.app import _parse_confidence
        assert _parse_confidence("[CONFIDENCE: -10%]") is None  # regex won't match negative

    def test_parse_confidence_without_percent(self):
        from web.app import _parse_confidence
        assert _parse_confidence("[CONFIDENCE: 72]") == 72

    def test_parse_confidence_case_insensitive(self):
        from web.app import _parse_confidence
        assert _parse_confidence("[confidence: 90%]") == 90


class TestBudgetGuard:
    """GIVEN a cost_budget_remaining of $0.00
    WHEN a worker tries to execute
    THEN it returns error 'Budget exceeded' without calling the LLM"""

    @pytest.mark.asyncio
    async def test_budget_exceeded_skips_worker(self):
        from liquid_swarm.models import TaskInput
        from liquid_swarm.providers import ProviderConfig, LLMProvider
        from web.app import _execute_single_task

        task = TaskInput(task_id="budget-test", query="Test query")
        provider_cfg = ProviderConfig(
            provider=LLMProvider.NVIDIA,
            api_key="test-key",
        )

        result = await _execute_single_task(
            task=task,
            semaphore=asyncio.Semaphore(10),
            provider_cfg=provider_cfg,
            cost_budget_remaining=0.0,  # No budget left
        )

        assert result.status == "error"
        assert "Budget exceeded" in result.data["error"]
        assert result.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_budget_none_means_unlimited(self):
        """GIVEN no budget (None)
        WHEN worker executes
        THEN budget guard is not triggered (may fail for other reasons)"""
        from liquid_swarm.models import TaskInput
        from liquid_swarm.providers import ProviderConfig, LLMProvider
        from web.app import _execute_single_task

        task = TaskInput(task_id="budget-test-2", query="Test query")
        provider_cfg = ProviderConfig(
            provider=LLMProvider.NVIDIA,
            api_key="fake-key",
        )

        result = await _execute_single_task(
            task=task,
            semaphore=asyncio.Semaphore(10),
            provider_cfg=provider_cfg,
            cost_budget_remaining=None,  # No budget set = unlimited
        )

        # Should NOT be "Budget exceeded" — it will fail for API reasons
        assert "Budget exceeded" not in result.data.get("error", "")


import asyncio
