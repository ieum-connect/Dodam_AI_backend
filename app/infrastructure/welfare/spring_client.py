"""Spring 백엔드의 복지 카탈로그를 읽는 HTTP 클라이언트."""

from typing import Any

import httpx

from app.core.config import Settings


class WelfareBackendUnavailable(RuntimeError):
    """Spring 복지 API에 연결할 수 없을 때 발생한다."""


class SpringWelfareClient:
    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.spring_backend_base_url.rstrip("/")
        self._timeout = settings.spring_backend_timeout_seconds
        self._token = settings.ai_internal_token.get_secret_value()

    async def list_programs(self, *, query: str | None = None, size: int = 100) -> list[dict[str, Any]]:
        params: dict[str, str | int] = {"size": min(max(size, 1), 100)}
        if query:
            params["query"] = query
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    f"{self._base_url}/api/welfare-programs",
                    params=params,
                    headers={"X-Internal-Token": self._token},
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WelfareBackendUnavailable("Spring 복지 카탈로그를 불러오지 못했습니다.") from exc
        if not isinstance(payload, list):
            raise WelfareBackendUnavailable("Spring 복지 카탈로그 응답 형식이 올바르지 않습니다.")
        return payload

    async def get_program(self, program_id: str) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    f"{self._base_url}/api/welfare-programs/{program_id}",
                    headers={"X-Internal-Token": self._token},
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WelfareBackendUnavailable("Spring 복지 상세정보를 불러오지 못했습니다.") from exc
        if not isinstance(payload, dict):
            raise WelfareBackendUnavailable("Spring 복지 상세 응답 형식이 올바르지 않습니다.")
        return payload
