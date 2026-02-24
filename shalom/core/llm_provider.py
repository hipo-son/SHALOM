import logging
import os
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, cast

import openai
from anthropic import Anthropic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class LLMProvider:
    """Unified LLM provider abstracting OpenAI and Anthropic APIs.

    Designed for easy backend switching between different LLM providers
    while exposing a single structured-output interface.
    """

    def __init__(
        self,
        provider_type: str = "openai",
        model_name: str = "gpt-4o",
        usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        base_url: Optional[str] = None,
    ):
        self.provider_type = provider_type.lower()
        self.model_name = model_name
        self.usage_callback = usage_callback
        self.base_url = base_url or os.environ.get("SHALOM_LLM_BASE_URL")

        self.client: Union[openai.OpenAI, Anthropic]
        if self.provider_type == "openai":
            client_kwargs: Dict[str, Any] = {}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
                # Local LLM servers (Ollama, vLLM, etc.) don't need a real API key
                client_kwargs["api_key"] = os.environ.get("OPENAI_API_KEY") or "local"
            else:
                client_kwargs["api_key"] = os.environ.get("OPENAI_API_KEY")
            self.client = openai.OpenAI(**client_kwargs)
        elif self.provider_type == "anthropic":
            anthropic_kwargs: Dict[str, Any] = {
                "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            }
            if self.base_url:
                anthropic_kwargs["base_url"] = self.base_url
            self.client = Anthropic(**anthropic_kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider_type}")

    @staticmethod
    def _extract_token_count(usage: Any, *attr_names: str) -> int:
        """None-safe token extraction from usage objects.

        Iterates through attribute names and returns the first non-None value.
        Handles the ``getattr(x, a, 0) or getattr(x, b, 0)`` pitfall where
        ``0 or fallback`` incorrectly falls through to the fallback.
        """
        for attr in attr_names:
            value = getattr(usage, attr, None)
            if value is not None:
                return int(value)
        return 0

    def _report_usage(self, response_usage: Any) -> None:
        """Report token usage to the registered callback, if any."""
        if not self.usage_callback or not response_usage:
            return
        try:
            self.usage_callback({
                "provider": self.provider_type,
                "model": self.model_name,
                "input_tokens": self._extract_token_count(
                    response_usage, "prompt_tokens", "input_tokens",
                ),
                "output_tokens": self._extract_token_count(
                    response_usage, "completion_tokens", "output_tokens",
                ),
            })
        except Exception as e:
            logger.warning("Usage callback failed: %s", e)

    def generate_structured_output(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        seed: Optional[int] = 42,
    ) -> T:
        """Send a prompt to the LLM and receive structured JSON conforming to ``response_model``.

        Automatically injects domain knowledge from `AGENT_GUIDELINES.md` into the system prompt.

        Args:
            system_prompt: System-level instruction for the LLM.
            user_prompt: User-level prompt describing the task.
            response_model: Pydantic model class defining the expected output schema.
            seed: Random seed for reproducibility (OpenAI only). Defaults to 42.

        Returns:
            An instance of ``response_model`` populated with the LLM's response.

        Note:
            Anthropic API does not support the ``seed`` parameter; it is
            silently ignored for reproducibility intent only.
        """
        # Inject central domain guidelines into the system prompt
        guidelines_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "AGENT_GUIDELINES.md"
        )
        try:
            with open(guidelines_path, "r", encoding="utf-8") as f:
                guidelines = f.read()
            injected_system_prompt = f"{guidelines}\n\n=== TASK SPECIFIC INSTRUCTIONS ===\n{system_prompt}"
        except FileNotFoundError:
            logger.warning("AGENT_GUIDELINES.md not found. Proceeding with raw system prompt.")
            injected_system_prompt = system_prompt

        # Audit log the LLM call
        try:
            from shalom.core.audit import log_event
            log_event("llm_call", {
                "provider": self.provider_type,
                "model": self.model_name,
                "base_url": self.base_url,
                "response_model": response_model.__name__,
            })
        except Exception:
            pass  # Audit logging must never block execution

        if self.provider_type == "openai":
            return self._call_openai_structured(injected_system_prompt, user_prompt, response_model, seed)
        elif self.provider_type == "anthropic":
            return self._call_anthropic_structured(injected_system_prompt, user_prompt, response_model, seed)
        raise ValueError(f"Provider {self.provider_type} lacks structured output handling.")

    def _call_openai_structured(
        self, system_prompt: str, user_prompt: str, response_model: Type[T], seed: Optional[int]
    ) -> T:
        """Call OpenAI Structured Outputs via ``beta.chat.completions.parse``.

        Falls back to legacy function-calling for older SDK versions.
        """
        try:
            # Note: Requires openai>=1.40.0 for structured outputs
            client_openai = cast(openai.OpenAI, self.client)
            response = client_openai.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_model,
                temperature=0.0,  # Deterministic temperature for reproducibility
                seed=seed,
            )
            self._report_usage(response.usage)
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Parsed result is None from openai beta completions.")
            return parsed
        except AttributeError:
            # Fallback for older openai versions: Function calling
            schema = response_model.model_json_schema()
            client_openai = cast(openai.OpenAI, self.client)
            fallback_res = client_openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                functions=[
                    {
                        "name": "structured_response",
                        "description": "Output the structured response",
                        "parameters": schema,
                    }
                ],
                function_call={"name": "structured_response"},
                temperature=0.0,
                seed=seed,
            )

            self._report_usage(fallback_res.usage)
            func_call = fallback_res.choices[0].message.function_call
            if func_call is None or func_call.arguments is None:
                raise ValueError("Function call arguments are missing.")

            json_str = func_call.arguments
            return response_model.model_validate_json(json_str)

    def _call_anthropic_structured(
        self, system_prompt: str, user_prompt: str, response_model: Type[T], seed: Optional[int]
    ) -> T:
        """Extract structured data via Anthropic tool-use API."""
        schema = response_model.model_json_schema()

        client_anthropic = cast(Anthropic, self.client)
        response = client_anthropic.messages.create(
            model=self.model_name,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[
                {
                    "name": "structured_response",
                    "description": "Output the structured response",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": "structured_response"},
            temperature=0.0,  # Deterministic temperature for reproducibility
        )

        self._report_usage(response.usage)

        # Parse tool_use result from response
        for block in response.content:
            if block.type == "tool_use":
                return response_model(**block.input)

        raise ValueError("Failed to extract structured data from Anthropic response.")
