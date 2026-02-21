import os
from typing import Type, TypeVar, Optional, Union, cast
from pydantic import BaseModel
import openai
from anthropic import Anthropic

T = TypeVar("T", bound=BaseModel)


class LLMProvider:
    """Unified LLM provider abstracting OpenAI and Anthropic APIs.

    Designed for easy backend switching between different LLM providers
    while exposing a single structured-output interface.
    """

    def __init__(self, provider_type: str = "openai", model_name: str = "gpt-4o"):
        self.provider_type = provider_type.lower()
        self.model_name = model_name

        self.client: Union[openai.OpenAI, Anthropic]
        if self.provider_type == "openai":
            self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif self.provider_type == "anthropic":
            self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {provider_type}")

    def generate_structured_output(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        seed: Optional[int] = 42,
    ) -> T:
        """Send a prompt to the LLM and receive structured JSON conforming to ``response_model``.

        Args:
            system_prompt: System-level instruction for the LLM.
            user_prompt: User-level prompt describing the task.
            response_model: Pydantic model class defining the expected output schema.
            seed: Random seed for reproducibility (OpenAI only). Defaults to 42.

        Returns:
            An instance of ``response_model`` populated with the LLM's response.
        """
        if self.provider_type == "openai":
            return self._call_openai_structured(system_prompt, user_prompt, response_model, seed)
        elif self.provider_type == "anthropic":
            return self._call_anthropic_structured(system_prompt, user_prompt, response_model, seed)
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

        # Parse tool_use result from response
        for block in response.content:
            if block.type == "tool_use":
                return response_model(**block.input)

        raise ValueError("Failed to extract structured data from Anthropic response.")
