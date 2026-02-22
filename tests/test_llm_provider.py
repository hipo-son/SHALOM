import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from shalom.core.llm_provider import LLMProvider


class DummyResponse(BaseModel):
    """Simple Pydantic model for testing structured output."""

    answer: str
    score: float = 0.0


class TestLLMProviderInit:
    """Tests for LLMProvider initialization and branching logic."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=False)
    @patch("shalom.core.llm_provider.openai.OpenAI")
    def test_init_openai(self, mock_openai_cls):
        """OpenAI provider initializes with correct client."""
        provider = LLMProvider(provider_type="openai", model_name="gpt-4o")

        assert provider.provider_type == "openai"
        assert provider.model_name == "gpt-4o"
        mock_openai_cls.assert_called_once_with(api_key="sk-test-key")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False)
    @patch("shalom.core.llm_provider.Anthropic")
    def test_init_anthropic(self, mock_anthropic_cls):
        """Anthropic provider initializes with correct client."""
        provider = LLMProvider(provider_type="anthropic", model_name="claude-sonnet-4-6")

        assert provider.provider_type == "anthropic"
        assert provider.model_name == "claude-sonnet-4-6"
        mock_anthropic_cls.assert_called_once_with(api_key="sk-ant-test")

    def test_init_unsupported_provider(self):
        """Unsupported provider type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMProvider(provider_type="gemini", model_name="gemini-pro")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("shalom.core.llm_provider.openai.OpenAI")
    def test_init_case_insensitive(self, mock_openai_cls):
        """Provider type is case-insensitive."""
        provider = LLMProvider(provider_type="OpenAI", model_name="gpt-4o")
        assert provider.provider_type == "openai"


class TestGenerateStructuredOutput:
    """Tests for generate_structured_output dispatch logic."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("shalom.core.llm_provider.openai.OpenAI")
    def test_dispatch_openai(self, mock_openai_cls):
        """OpenAI provider dispatches to _call_openai_structured."""
        provider = LLMProvider(provider_type="openai", model_name="gpt-4o")

        # Mock the internal method
        expected = DummyResponse(answer="test", score=0.5)
        provider._call_openai_structured = MagicMock(return_value=expected)

        result = provider.generate_structured_output(
            system_prompt="sys",
            user_prompt="user",
            response_model=DummyResponse,
            seed=42,
        )
        assert result.answer == "test"
        provider._call_openai_structured.assert_called_once()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False)
    @patch("shalom.core.llm_provider.Anthropic")
    def test_dispatch_anthropic(self, mock_anthropic_cls):
        """Anthropic provider dispatches to _call_anthropic_structured."""
        provider = LLMProvider(provider_type="anthropic", model_name="claude-sonnet-4-6")

        expected = DummyResponse(answer="test", score=0.8)
        provider._call_anthropic_structured = MagicMock(return_value=expected)

        result = provider.generate_structured_output(
            system_prompt="sys",
            user_prompt="user",
            response_model=DummyResponse,
        )
        assert result.answer == "test"
        provider._call_anthropic_structured.assert_called_once()


class TestOpenAIStructured:
    """Tests for _call_openai_structured with mocked OpenAI client."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("shalom.core.llm_provider.openai.OpenAI")
    def test_openai_structured_output_success(self, mock_openai_cls):
        """Successful parsing via beta.chat.completions.parse."""
        provider = LLMProvider(provider_type="openai", model_name="gpt-4o")

        expected = DummyResponse(answer="hydrogen", score=0.9)

        # Mock the nested API call
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.parsed = expected
        mock_response.choices = [mock_choice]
        provider.client.beta.chat.completions.parse.return_value = mock_response

        result = provider._call_openai_structured(
            system_prompt="sys",
            user_prompt="user",
            response_model=DummyResponse,
            seed=42,
        )
        assert result.answer == "hydrogen"
        assert result.score == 0.9

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("shalom.core.llm_provider.openai.OpenAI")
    def test_openai_parsed_none_raises(self, mock_openai_cls):
        """ValueError raised when parsed result is None."""
        provider = LLMProvider(provider_type="openai", model_name="gpt-4o")

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.parsed = None
        mock_response.choices = [mock_choice]
        provider.client.beta.chat.completions.parse.return_value = mock_response

        with pytest.raises(ValueError, match="Parsed result is None"):
            provider._call_openai_structured(
                system_prompt="sys",
                user_prompt="user",
                response_model=DummyResponse,
                seed=42,
            )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("shalom.core.llm_provider.openai.OpenAI")
    def test_openai_fallback_function_calling(self, mock_openai_cls):
        """Falls back to function calling when beta API raises AttributeError."""
        provider = LLMProvider(provider_type="openai", model_name="gpt-4o")

        # beta.chat.completions.parse raises AttributeError (old SDK)
        provider.client.beta.chat.completions.parse.side_effect = AttributeError("no parse")

        # Fallback function call response
        mock_func_call = MagicMock()
        mock_func_call.arguments = '{"answer": "fallback", "score": 0.5}'

        mock_choice = MagicMock()
        mock_choice.message.function_call = mock_func_call

        mock_fallback_response = MagicMock()
        mock_fallback_response.choices = [mock_choice]
        provider.client.chat.completions.create.return_value = mock_fallback_response

        result = provider._call_openai_structured(
            system_prompt="sys",
            user_prompt="user",
            response_model=DummyResponse,
            seed=42,
        )
        assert result.answer == "fallback"
        assert result.score == 0.5

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("shalom.core.llm_provider.openai.OpenAI")
    def test_openai_fallback_missing_function_call(self, mock_openai_cls):
        """ValueError raised when fallback function_call is missing."""
        provider = LLMProvider(provider_type="openai", model_name="gpt-4o")

        provider.client.beta.chat.completions.parse.side_effect = AttributeError("no parse")

        mock_choice = MagicMock()
        mock_choice.message.function_call = None

        mock_fallback_response = MagicMock()
        mock_fallback_response.choices = [mock_choice]
        provider.client.chat.completions.create.return_value = mock_fallback_response

        with pytest.raises(ValueError, match="Function call arguments are missing"):
            provider._call_openai_structured(
                system_prompt="sys",
                user_prompt="user",
                response_model=DummyResponse,
                seed=42,
            )


class TestAnthropicStructured:
    """Tests for _call_anthropic_structured with mocked Anthropic client."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False)
    @patch("shalom.core.llm_provider.Anthropic")
    def test_anthropic_tool_use_success(self, mock_anthropic_cls):
        """Successful extraction from tool_use content block."""
        provider = LLMProvider(provider_type="anthropic", model_name="claude-sonnet-4-6")

        mock_block = MagicMock()
        mock_block.type = "tool_use"
        mock_block.input = {"answer": "oxygen", "score": 0.7}

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        provider.client.messages.create.return_value = mock_response

        result = provider._call_anthropic_structured(
            system_prompt="sys",
            user_prompt="user",
            response_model=DummyResponse,
            seed=42,
        )
        assert result.answer == "oxygen"
        assert result.score == 0.7

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False)
    @patch("shalom.core.llm_provider.Anthropic")
    def test_anthropic_no_tool_use_raises(self, mock_anthropic_cls):
        """ValueError raised when no tool_use block is found."""
        provider = LLMProvider(provider_type="anthropic", model_name="claude-sonnet-4-6")

        mock_block = MagicMock()
        mock_block.type = "text"  # Not tool_use

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        provider.client.messages.create.return_value = mock_response

        with pytest.raises(ValueError, match="Failed to extract structured data"):
            provider._call_anthropic_structured(
                system_prompt="sys",
                user_prompt="user",
                response_model=DummyResponse,
                seed=42,
            )

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False)
    @patch("shalom.core.llm_provider.Anthropic")
    def test_anthropic_empty_response_raises(self, mock_anthropic_cls):
        """ValueError raised when response content is empty."""
        provider = LLMProvider(provider_type="anthropic", model_name="claude-sonnet-4-6")

        mock_response = MagicMock()
        mock_response.content = []
        provider.client.messages.create.return_value = mock_response

        with pytest.raises(ValueError, match="Failed to extract structured data"):
            provider._call_anthropic_structured(
                system_prompt="sys",
                user_prompt="user",
                response_model=DummyResponse,
                seed=42,
            )


# ---------------------------------------------------------------------------
# Usage callback and token extraction tests
# ---------------------------------------------------------------------------


class TestUsageCallback:
    """Tests for _extract_token_count and _report_usage."""

    def test_extract_token_count_first_attr(self):
        """Returns first non-None attribute value."""
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.input_tokens = 200
        result = LLMProvider._extract_token_count(usage, "prompt_tokens", "input_tokens")
        assert result == 100

    def test_extract_token_count_fallback_attr(self):
        """Falls back to second attr when first is None."""
        usage = MagicMock()
        usage.prompt_tokens = None
        usage.input_tokens = 200
        result = LLMProvider._extract_token_count(usage, "prompt_tokens", "input_tokens")
        assert result == 200

    def test_extract_token_count_all_none(self):
        """Returns 0 when all attributes are None."""
        usage = MagicMock()
        usage.prompt_tokens = None
        usage.input_tokens = None
        result = LLMProvider._extract_token_count(usage, "prompt_tokens", "input_tokens")
        assert result == 0

    def test_extract_token_count_missing_attr(self):
        """Returns 0 when attributes don't exist."""
        usage = object()  # No attributes at all
        result = LLMProvider._extract_token_count(usage, "prompt_tokens", "input_tokens")
        assert result == 0

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("shalom.core.llm_provider.openai.OpenAI")
    def test_report_usage_calls_callback(self, mock_openai_cls):
        """_report_usage invokes the callback with correct data."""
        recorded = []
        provider = LLMProvider(
            provider_type="openai", model_name="gpt-4o",
            usage_callback=lambda data: recorded.append(data),
        )

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 500
        mock_usage.completion_tokens = 200
        provider._report_usage(mock_usage)

        assert len(recorded) == 1
        assert recorded[0]["provider"] == "openai"
        assert recorded[0]["model"] == "gpt-4o"
        assert recorded[0]["input_tokens"] == 500
        assert recorded[0]["output_tokens"] == 200

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("shalom.core.llm_provider.openai.OpenAI")
    def test_report_usage_no_callback(self, mock_openai_cls):
        """_report_usage does nothing without callback."""
        provider = LLMProvider(provider_type="openai", model_name="gpt-4o")
        assert provider.usage_callback is None
        # Should not raise
        provider._report_usage(MagicMock())

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("shalom.core.llm_provider.openai.OpenAI")
    def test_report_usage_none_response(self, mock_openai_cls):
        """_report_usage handles None response_usage gracefully."""
        recorded = []
        provider = LLMProvider(
            provider_type="openai", model_name="gpt-4o",
            usage_callback=lambda data: recorded.append(data),
        )
        provider._report_usage(None)
        assert len(recorded) == 0

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("shalom.core.llm_provider.openai.OpenAI")
    def test_report_usage_callback_exception_caught(self, mock_openai_cls):
        """_report_usage catches callback exceptions."""
        def bad_callback(data):
            raise RuntimeError("callback error")

        provider = LLMProvider(
            provider_type="openai", model_name="gpt-4o",
            usage_callback=bad_callback,
        )
        # Should not raise — exception is caught and logged
        provider._report_usage(MagicMock())

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("shalom.core.llm_provider.openai.OpenAI")
    def test_openai_structured_reports_usage(self, mock_openai_cls):
        """_call_openai_structured calls _report_usage."""
        recorded = []
        provider = LLMProvider(
            provider_type="openai", model_name="gpt-4o",
            usage_callback=lambda data: recorded.append(data),
        )

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50

        expected = DummyResponse(answer="test", score=0.5)
        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_choice = MagicMock()
        mock_choice.message.parsed = expected
        mock_response.choices = [mock_choice]
        provider.client.beta.chat.completions.parse.return_value = mock_response

        provider._call_openai_structured("sys", "user", DummyResponse, 42)
        assert len(recorded) == 1
        assert recorded[0]["input_tokens"] == 100
