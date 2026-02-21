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
