"""텍스트 목록을 벡터로 바꾸는 임베딩 포트와 결과 형식, 예외를 정의한다.

문서 청크 적재와 RAG 질의 검색이 같은 포트를 쓴다. 둘이 같은 모델과 차원을 써야 거리가 의미 있다.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class TextEmbeddingErrorCode(StrEnum):
    """임베딩이 실패한 이유."""

    MISSING_API_KEY = "MISSING_API_KEY"  # API 키가 설정되지 않음
    TIMEOUT = "TIMEOUT"  # 제한 시간 안에 응답이 없음
    RATE_LIMITED = "RATE_LIMITED"  # 호출 한도 초과(429)
    API_ERROR = "API_ERROR"  # provider 서버 오류(5xx)나 네트워크 오류
    REQUEST_REJECTED = "REQUEST_REJECTED"  # 인증 실패, 잘못된 요청 등 429 외 4xx
    INVALID_RESPONSE = "INVALID_RESPONSE"  # 벡터 개수나 차원이 요청과 맞지 않음


class TextEmbeddingError(Exception):
    """임베딩 실패. 메시지에는 오류 코드만 담고 키와 입력 원문은 담지 않는다."""

    def __init__(self, code: TextEmbeddingErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


@dataclass(frozen=True)
class TextEmbeddings:
    """입력 순서와 같은 순서의 벡터 목록, 실제 모델 이름, 벡터 차원."""

    vectors: list[list[float]]
    model_name: str
    dimensions: int


class TextEmbedder(Protocol):
    """텍스트 목록을 같은 모델·차원의 벡터 목록으로 바꾸는 인터페이스."""

    def embed(self, texts: Sequence[str]) -> TextEmbeddings:
        """입력마다 벡터 하나를 순서대로 돌려준다. 하나라도 실패하면 TextEmbeddingError를 던진다."""
        ...
