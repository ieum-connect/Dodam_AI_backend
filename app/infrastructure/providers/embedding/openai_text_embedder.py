"""OpenAI Embeddings API로 텍스트 벡터를 받는 임베딩 포트 구현."""

import time
from collections.abc import Callable, Sequence

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import SecretStr, ValidationError

from app.application.ports.text_embedder import (
    TextEmbeddingError,
    TextEmbeddingErrorCode,
    TextEmbeddings,
)
from app.core.config import Settings

# 청크는 500토큰 이하라 한 요청이 5만 토큰을 넘지 않는다. API 한도는 입력 2048개, 30만 토큰이다.
_BATCH_SIZE = 100

# 다시 부르면 성공할 수 있는 오류. 나머지는 같은 요청을 다시 보내도 결과가 같다.
_RETRYABLE_CODES = frozenset(
    {
        TextEmbeddingErrorCode.TIMEOUT,
        TextEmbeddingErrorCode.RATE_LIMITED,
        TextEmbeddingErrorCode.API_ERROR,
    }
)


class OpenAITextEmbedder:
    """텍스트를 배치로 나눠 OpenAI에 보내고, 일시적인 실패는 정해진 횟수만큼 다시 부른다.

    API 키는 embed를 부를 때만 확인한다.
    """

    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        model: str,
        dimensions: int,
        timeout_seconds: float,
        max_retries: int,
        backoff_seconds: float,
        batch_size: int = _BATCH_SIZE,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api_key = api_key.get_secret_value() if api_key is not None else ""
        self._model = model
        self._dimensions = dimensions
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._batch_size = batch_size
        self._sleep = sleep
        # 키가 없으면 SDK가 OPENAI_API_KEY 환경 변수를 대신 읽거나 생성 시 실패하므로 만들지 않는다.
        # 재시도는 sleep을 주입할 수 있게 이 클래스가 직접 하고 SDK 재시도는 끈다.
        self._client = (
            OpenAI(
                api_key=self._api_key,
                timeout=timeout_seconds,
                max_retries=0,
                http_client=httpx.Client(transport=transport),
            )
            if self._api_key
            else None
        )

    def embed(self, texts: Sequence[str]) -> TextEmbeddings:
        """입력 순서대로 벡터를 돌려준다. 한 배치라도 끝내 실패하면 아무것도 돌려주지 않는다."""
        if not texts:
            return TextEmbeddings(vectors=[], model_name=self._model, dimensions=self._dimensions)
        if self._client is None:
            raise TextEmbeddingError(TextEmbeddingErrorCode.MISSING_API_KEY)
        vectors: list[list[float]] = []
        model_name = self._model
        for start in range(0, len(texts), self._batch_size):
            batch_vectors, model_name = self._embed_with_retry(
                self._client, list(texts[start : start + self._batch_size])
            )
            vectors.extend(batch_vectors)
        return TextEmbeddings(vectors=vectors, model_name=model_name, dimensions=self._dimensions)

    def close(self) -> None:
        """HTTP 연결을 닫는다. 임베더를 만든 쪽이 프로세스를 끝낼 때 한 번 부른다."""
        if self._client is not None:
            self._client.close()

    def _embed_with_retry(
        self, client: OpenAI, batch: list[str]
    ) -> tuple[list[list[float]], str]:
        """처음 호출 포함 최대 max_retries + 1번 부른다. 대기 시간은 재시도마다 두 배가 된다."""
        attempt = 0
        while True:
            try:
                return self._embed_batch(client, batch)
            except TextEmbeddingError as error:
                if error.code not in _RETRYABLE_CODES or attempt >= self._max_retries:
                    raise
            self._sleep(self._backoff_seconds * 2**attempt)
            attempt += 1

    def _embed_batch(self, client: OpenAI, batch: list[str]) -> tuple[list[list[float]], str]:
        """배치 하나를 한 번 호출하고 개수와 차원을 확인한다."""
        # SDK 예외는 요청 객체(키가 든 헤더, 입력 원문)를 들고 있으므로 from None으로 끊는다.
        try:
            response = client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
                encoding_format="float",
            )
        except APITimeoutError:
            raise TextEmbeddingError(TextEmbeddingErrorCode.TIMEOUT) from None
        except RateLimitError:
            raise TextEmbeddingError(TextEmbeddingErrorCode.RATE_LIMITED) from None
        except (InternalServerError, APIConnectionError):
            raise TextEmbeddingError(TextEmbeddingErrorCode.API_ERROR) from None
        except APIStatusError:
            raise TextEmbeddingError(TextEmbeddingErrorCode.REQUEST_REJECTED) from None
        except (APIError, ValidationError, ValueError, TypeError):
            raise TextEmbeddingError(TextEmbeddingErrorCode.INVALID_RESPONSE) from None
        try:
            data = sorted(response.data, key=lambda item: item.index)
            vectors = [list(item.embedding) for item in data]
        except (AttributeError, TypeError):
            raise TextEmbeddingError(TextEmbeddingErrorCode.INVALID_RESPONSE) from None
        if len(vectors) != len(batch) or any(len(v) != self._dimensions for v in vectors):
            raise TextEmbeddingError(TextEmbeddingErrorCode.INVALID_RESPONSE)
        # 응답의 model은 실제로 쓴 모델 이름이다. 없으면 요청한 이름을 쓴다.
        return vectors, response.model or self._model


def create_text_embedder(
    settings: Settings, transport: httpx.BaseTransport | None = None
) -> OpenAITextEmbedder:
    """EMBEDDING_* 설정으로 임베더를 만든다. 이때는 API 키가 없어도 된다."""
    return OpenAITextEmbedder(
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.embedding_timeout_seconds,
        max_retries=settings.embedding_max_retries,
        backoff_seconds=settings.embedding_retry_backoff_seconds,
        transport=transport,
    )
