import os
import json
from typing import Any, Dict, Type, TypeVar
from pydantic import BaseModel
import openai
from anthropic import Anthropic

T = TypeVar('T', bound=BaseModel)

class LLMProvider:
    """
    LLM API 호출을 추상화하는 기본 프로바이더
    향후 OpenAI, Anthropic 등 다양한 백엔드를 손쉽게 교체할 수 있도록 설계
    """
    def __init__(self, provider_type: str = "openai", model_name: str = "gpt-4o"):
        self.provider_type = provider_type.lower()
        self.model_name = model_name
        
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
        response_model: Type[T]
    ) -> T:
        """
        LLM에게 프롬프트를 전달하고, Pydantic 스키마(response_model)에 맞는 
        구조화된 데이터(JSON)를 반환받는 함수.
        """
        if self.provider_type == "openai":
            return self._call_openai_structured(system_prompt, user_prompt, response_model)
        elif self.provider_type == "anthropic":
            return self._call_anthropic_structured(system_prompt, user_prompt, response_model)

    def _call_openai_structured(
        self, system_prompt: str, user_prompt: str, response_model: Type[T]
    ) -> T:
        """OpenAI의 Structured Outputs 기능 (beta.chat.completions.parse) 사용"""
        try:
            # Note: Requires openai>=1.40.0 for structured outputs
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=response_model,
                temperature=0.2 # 구조화된 답변이므로 온도를 낮춤
            )
            return response.choices[0].message.parsed
        except AttributeError:
            # Fallback for older openai versions: Function calling
            schema = response_model.model_json_schema()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                functions=[{
                    "name": "structured_response",
                    "description": "Output the structured response",
                    "parameters": schema
                }],
                function_call={"name": "structured_response"},
                temperature=0.2
            )
            json_str = response.choices[0].message.function_call.arguments
            return response_model.model_validate_json(json_str)

    def _call_anthropic_structured(
        self, system_prompt: str, user_prompt: str, response_model: Type[T]
    ) -> T:
        """Anthropic API를 통한 구조화된 데이터 추출 (Tool use 활용)"""
        schema = response_model.model_json_schema()
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[{
                "name": "structured_response",
                "description": "Output the structured response",
                "input_schema": schema
            }],
            tool_choice={"type": "tool", "name": "structured_response"},
            temperature=0.2
        )
        
        # 툴 호출 결과를 파싱
        for block in response.content:
            if block.type == "tool_use":
                return response_model(**block.input)
        
        raise ValueError("Failed to extract structured data from Anthropic response.")
