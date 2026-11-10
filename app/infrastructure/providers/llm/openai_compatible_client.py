"""Gemini 등 OpenAI 호환 Chat Completions API adapter."""

import json
from typing import Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from app.contracts.ai_result_document import GeneratedAnswer
from app.core.config import Settings
from app.domain.ai_result.error import AiResultErrorCode, AiResultGenerationError


class LlmClient(Protocol):
    """provider 교체를 위한 최소 인터페이스다."""

    def generate_grounded_answer(self, system_prompt: str, user_prompt: str) -> GeneratedAnswer:
        """근거가 있는 구조화 답변을 생성한다."""


class OpenAiCompatibleLlmClient:
    """환경 변수로 받은 호환 endpoint에 요청한다."""

    def __init__(self, settings: Settings, *, use_embedding_api_key: bool = False) -> None:
        self._settings = settings
        # 의료 채팅만 임베딩 키를 사용한다. 복지 클라이언트는 기본값으로 LLM 키를 쓴다.
        api_key = (
            settings.embedding_api_key
            if use_embedding_api_key
            else settings.llm_api_key or settings.openai_api_key
        )
        self._client = (
            OpenAI(
                api_key=api_key.get_secret_value() if api_key else "",
                base_url=settings.llm_base_url,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )
            if api_key and api_key.get_secret_value()
            else None
        )

    def close(self) -> None:
        """앱 종료 시 연결 풀을 닫는다."""
        if self._client is not None:
            self._client.close()

    def generate_grounded_answer(self, system_prompt: str, user_prompt: str) -> GeneratedAnswer:
        if self._client is None:
            raise AiResultGenerationError(AiResultErrorCode.AI_API_KEY_MISSING, retryable=False)
        try:
            schema = GeneratedAnswer.model_json_schema()
            # gpt-5.6-sol의 strict structured output은 루트 객체에
            # additionalProperties=false를 요구한다.
            def close_objects(node: object) -> None:
                if not isinstance(node, dict):
                    return
                if node.get("type") == "object":
                    node.setdefault("additionalProperties", False)
                for value in node.values():
                    if isinstance(value, dict):
                        close_objects(value)
                    elif isinstance(value, list):
                        for item in value:
                            close_objects(item)

            close_objects(schema)
            request_kwargs = {
                "model": self._settings.llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "grounded_answer",
                        "strict": True,
                        "schema": schema,
                    },
                },
            }
            # Gemini OpenAI 호환 API는 OpenAI 전용 reasoning_effort를 받지 않는다.
            if self._settings.llm_provider != "gemini_openai_compatible":
                request_kwargs["reasoning_effort"] = self._settings.llm_reasoning_effort
                if self._settings.llm_reasoning_effort == "none" and not self._settings.llm_model.startswith("gpt-5"):
                    request_kwargs["temperature"] = self._settings.llm_temperature
            else:
                request_kwargs["temperature"] = self._settings.llm_temperature
            if self._settings.llm_model.startswith("gpt-5"):
                request_kwargs.pop("temperature", None)
            response = self._client.chat.completions.create(**request_kwargs)
        except APITimeoutError:
            raise AiResultGenerationError(AiResultErrorCode.AI_TIMEOUT, retryable=True) from None
        except RateLimitError:
            raise AiResultGenerationError(AiResultErrorCode.AI_RATE_LIMIT, retryable=True) from None
        except AuthenticationError:
            raise AiResultGenerationError(
                AiResultErrorCode.AI_AUTH_FAILED, retryable=False
            ) from None
        except BadRequestError:
            raise AiResultGenerationError(
                AiResultErrorCode.AI_REQUEST_REJECTED, retryable=False
            ) from None
        except APIConnectionError:
            raise AiResultGenerationError(
                AiResultErrorCode.AI_CONNECTION_FAILED, retryable=True
            ) from None
        except APIStatusError:
            raise AiResultGenerationError(
                AiResultErrorCode.AI_GENERATION_FAILED, retryable=True
            ) from None
        content = response.choices[0].message.content
        if not content:
            raise AiResultGenerationError(AiResultErrorCode.AI_RESPONSE_INVALID, retryable=True)
        try:
            normalized = content.strip()
            if normalized.startswith("```json") and normalized.endswith("```"):
                normalized = normalized[7:-3].strip()
            return GeneratedAnswer.model_validate(json.loads(normalized))
        except (json.JSONDecodeError, ValueError):
            raise AiResultGenerationError(
                AiResultErrorCode.AI_RESPONSE_INVALID, retryable=True
            ) from None

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """복지 설명처럼 짧은 일반 텍스트를 생성한다."""
        if (
            self._settings.llm_api_key is None
            and self._settings.openai_api_key is None
        ):
            raise AiResultGenerationError(AiResultErrorCode.AI_API_KEY_MISSING, retryable=False)
        request_kwargs = {
            "model": self._settings.llm_model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        }
        if self._settings.llm_provider != "gemini_openai_compatible":
            request_kwargs["reasoning_effort"] = self._settings.llm_reasoning_effort
            if self._settings.llm_reasoning_effort == "none":
                request_kwargs["temperature"] = self._settings.llm_temperature
        else:
            request_kwargs["temperature"] = self._settings.llm_temperature
        if self._settings.llm_model.startswith("gpt-5"):
            request_kwargs.pop("temperature", None)
        response = self._client.chat.completions.create(**request_kwargs)
        return (response.choices[0].message.content or "").strip()
