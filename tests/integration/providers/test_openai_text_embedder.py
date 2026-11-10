"""OpenAI 텍스트 임베더 테스트. httpx.MockTransport로 대신해 실제 API를 부르지 않는다."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.application.ports.text_embedder import TextEmbeddingError, TextEmbeddingErrorCode
from app.core.config import Settings
from app.infrastructure.providers.embedding import create_text_embedder
from app.infrastructure.providers.embedding.openai_text_embedder import OpenAITextEmbedder

API_KEY = "sk-embedding-test-secret"
OTHER_KEY = "sk-other-feature-secret"
SOURCE_TEXT = "합성 혈색소 13.5 g/dL"
DIMENSIONS = 768
RESPONSE_MODEL = "text-embedding-3-small"

Handler = Callable[[httpx.Request], httpx.Response]


def vector(seed: int, dimensions: int = DIMENSIONS) -> list[float]:
    """입력 번호를 첫 값에 담은 합성 벡터를 만든다. 순서 확인에 쓴다."""
    return [float(seed)] + [0.0] * (dimensions - 1)


def ok_response(
    request: httpx.Request, *, reverse: bool = False, **overrides: Any
) -> httpx.Response:
    """요청 입력마다 벡터 하나를 담은 성공 응답을 만든다. 첫 값은 입력 문자열 끝의 번호다."""
    inputs = json.loads(request.content)["input"]
    data = [
        {"object": "embedding", "index": index, "embedding": vector(int(text.split()[-1]))}
        for index, text in enumerate(inputs)
    ]
    if reverse:
        data.reverse()
    body = {
        "object": "list",
        "data": data,
        "model": RESPONSE_MODEL,
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }
    body.update(overrides)
    return httpx.Response(200, json=body)


class Recorder:
    """요청을 기록하고, 정해 둔 응답이나 예외를 순서대로 돌려주는 가짜 전송 계층."""

    def __init__(self, *steps: Handler) -> None:
        self.requests: list[httpx.Request] = []
        self._steps = list(steps)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """다음 단계가 있으면 그것을, 없으면 성공 응답을 쓴다."""
        self.requests.append(request)
        step = self._steps.pop(0) if self._steps else ok_response
        return step(request)


def status(code: int) -> Handler:
    """정해진 상태 코드로 실패하는 단계."""
    return lambda request: httpx.Response(code, json={"error": {"message": "합성 오류"}})


def raises(error: type[httpx.HTTPError]) -> Handler:
    """전송 중 httpx 예외를 던지는 단계."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise error("합성 전송 오류", request=request)

    return handler


def make_embedder(
    recorder: Recorder, sleeps: list[float], *, api_key: str | None = API_KEY, **overrides: Any
) -> OpenAITextEmbedder:
    """가짜 전송 계층과 기록용 sleep을 쓰는 임베더를 만든다."""
    options: dict[str, Any] = {
        "api_key": SecretStr(api_key) if api_key is not None else None,
        "model": "text-embedding-3-small",
        "dimensions": DIMENSIONS,
        "timeout_seconds": 5,
        "max_retries": 2,
        "backoff_seconds": 1.0,
        "transport": httpx.MockTransport(recorder),
        "sleep": sleeps.append,
    }
    options.update(overrides)
    return OpenAITextEmbedder(**options)


def texts(count: int) -> list[str]:
    """번호가 붙은 합성 문장 목록."""
    return [f"합성 문장 {number}" for number in range(count)]


def test_splits_into_batches_and_keeps_input_order() -> None:
    """250개를 100개씩 3번 보내고, 응답 index가 뒤집혀도 입력 순서대로 돌려준다."""
    recorder = Recorder(*[lambda r: ok_response(r, reverse=True)] * 3)
    sleeps: list[float] = []

    result = make_embedder(recorder, sleeps, batch_size=100).embed(texts(250))

    assert [len(json.loads(r.content)["input"]) for r in recorder.requests] == [100, 100, 50]
    assert [v[0] for v in result.vectors] == [float(n) for n in range(250)]
    assert all(len(v) == DIMENSIONS for v in result.vectors)
    assert result.model_name == RESPONSE_MODEL
    assert result.dimensions == DIMENSIONS
    assert sleeps == []


def test_request_uses_model_dimensions_and_embedding_key() -> None:
    """모델, 차원, float 형식, 입력을 보내고 임베딩 전용 키를 헤더에 쓴다."""
    recorder = Recorder()

    make_embedder(recorder, []).embed(["합성 문장 0"])

    request = recorder.requests[0]
    assert request.url.path.endswith("/embeddings")
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert json.loads(request.content) == {
        "model": "text-embedding-3-small",
        "input": ["합성 문장 0"],
        "dimensions": DIMENSIONS,
        "encoding_format": "float",
    }


@pytest.mark.parametrize(
    "failure",
    [
        status(429),
        status(500),
        status(503),
        raises(httpx.ReadTimeout),
        raises(httpx.ConnectError),
    ],
    ids=["429", "500", "503", "timeout", "connection"],
)
def test_retries_transient_failure_then_succeeds(failure: Handler) -> None:
    """일시적인 실패는 한 번 기다렸다가 다시 불러 성공한다."""
    recorder = Recorder(failure)
    sleeps: list[float] = []

    result = make_embedder(recorder, sleeps).embed(["합성 문장 7"])

    assert len(recorder.requests) == 2
    assert sleeps == [1.0]
    assert result.vectors[0][0] == 7.0


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (status(429), TextEmbeddingErrorCode.RATE_LIMITED),
        (status(502), TextEmbeddingErrorCode.API_ERROR),
        (raises(httpx.ReadTimeout), TextEmbeddingErrorCode.TIMEOUT),
    ],
    ids=["429", "502", "timeout"],
)
def test_gives_up_after_max_retries(failure: Handler, code: TextEmbeddingErrorCode) -> None:
    """계속 실패하면 처음 포함 3번 부르고, 1초·2초 기다린 뒤 마지막 오류를 올린다."""
    recorder = Recorder(failure, failure, failure)
    sleeps: list[float] = []

    with pytest.raises(TextEmbeddingError) as error:
        make_embedder(recorder, sleeps).embed(["합성 문장 0"])

    assert error.value.code == code
    assert len(recorder.requests) == 3
    assert sleeps == [1.0, 2.0]


def test_later_batch_failure_fails_whole_call() -> None:
    """둘째 배치가 끝내 실패하면 첫 배치 벡터도 돌려주지 않고 예외를 올린다."""
    recorder = Recorder(ok_response, status(429), status(429), status(429))

    with pytest.raises(TextEmbeddingError) as error:
        make_embedder(recorder, [], batch_size=2).embed(texts(3))

    assert error.value.code == TextEmbeddingErrorCode.RATE_LIMITED
    assert len(recorder.requests) == 4


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(code: int) -> None:
    """429 외 4xx는 다시 불러도 같으므로 한 번만 부르고 REQUEST_REJECTED로 실패한다."""
    recorder = Recorder(status(code))
    sleeps: list[float] = []

    with pytest.raises(TextEmbeddingError) as error:
        make_embedder(recorder, sleeps).embed(["합성 문장 0"])

    assert error.value.code == TextEmbeddingErrorCode.REQUEST_REJECTED
    assert len(recorder.requests) == 1
    assert sleeps == []


def wrong_dimensions(request: httpx.Request) -> httpx.Response:
    """767차원 벡터를 돌려주는 응답."""
    return ok_response(
        request, data=[{"object": "embedding", "index": 0, "embedding": vector(0, 767)}]
    )


def missing_vector(request: httpx.Request) -> httpx.Response:
    """입력 2개에 벡터 1개만 돌려주는 응답."""
    return ok_response(
        request, data=[{"object": "embedding", "index": 0, "embedding": vector(0)}]
    )


def not_json(request: httpx.Request) -> httpx.Response:
    """JSON이 아닌 본문."""
    return httpx.Response(200, text="not json", headers={"content-type": "application/json"})


@pytest.mark.parametrize(
    ("failure", "inputs"),
    [(wrong_dimensions, 1), (missing_vector, 2), (not_json, 1)],
    ids=["wrong-dimensions", "missing-vector", "not-json"],
)
def test_rejects_invalid_response(failure: Handler, inputs: int) -> None:
    """차원·개수가 다르거나 본문이 JSON이 아니면 다시 부르지 않고 INVALID_RESPONSE로 실패한다."""
    recorder = Recorder(failure)
    sleeps: list[float] = []

    with pytest.raises(TextEmbeddingError) as error:
        make_embedder(recorder, sleeps).embed(texts(inputs))

    assert error.value.code == TextEmbeddingErrorCode.INVALID_RESPONSE
    assert len(recorder.requests) == 1
    assert sleeps == []


@pytest.mark.parametrize("api_key", [None, ""])
def test_missing_key_fails_without_request(api_key: str | None) -> None:
    """키가 없으면 요청을 보내지 않고 MISSING_API_KEY로 실패한다."""
    recorder = Recorder()

    with pytest.raises(TextEmbeddingError) as error:
        make_embedder(recorder, [], api_key=api_key).embed(["합성 문장 0"])

    assert error.value.code == TextEmbeddingErrorCode.MISSING_API_KEY
    assert recorder.requests == []


def test_empty_input_sends_no_request() -> None:
    """입력이 없으면 요청 없이 빈 결과를 돌려준다. 키가 없어도 실패하지 않는다."""
    recorder = Recorder()

    result = make_embedder(recorder, [], api_key=None).embed([])

    assert result.vectors == []
    assert recorder.requests == []


def test_error_does_not_carry_key_or_source_text() -> None:
    """예외의 메시지, repr, 연결된 원인에 키와 입력 원문이 없다."""
    recorder = Recorder(status(400))

    with pytest.raises(TextEmbeddingError) as error:
        make_embedder(recorder, []).embed([SOURCE_TEXT])

    rendered = f"{error.value!s} {error.value!r}"
    assert API_KEY not in rendered
    assert SOURCE_TEXT not in rendered
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ is True


def make_settings(**values: Any) -> Settings:
    """.env를 읽지 않는 설정."""
    return Settings(
        _env_file=None,
        database_url="unused",
        ai_internal_token="token",
        storage_local_root="root",
        **values,
    )


EMBEDDING_ENV_NAMES = [
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_TIMEOUT_SECONDS",
    "EMBEDDING_MAX_RETRIES",
    "EMBEDDING_RETRY_BACKOFF_SECONDS",
]


def test_create_ignores_other_feature_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENAI_API_KEY, LLM_API_KEY만 있으면 그 키로 부르지 않고 MISSING_API_KEY로 실패한다."""
    for name in EMBEDDING_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", OTHER_KEY)
    monkeypatch.setenv("LLM_API_KEY", OTHER_KEY)
    recorder = Recorder()
    settings = make_settings()

    with pytest.raises(TextEmbeddingError) as error:
        create_text_embedder(settings, httpx.MockTransport(recorder)).embed(["합성 문장 0"])

    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == DIMENSIONS
    assert error.value.code == TextEmbeddingErrorCode.MISSING_API_KEY
    assert recorder.requests == []


def test_create_uses_embedding_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """EMBEDDING_* 환경 변수의 키와 모델로 요청한다."""
    for name in EMBEDDING_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EMBEDDING_API_KEY", API_KEY)
    monkeypatch.setenv("EMBEDDING_MODEL", "synthetic-embedding-model")
    monkeypatch.setenv("EMBEDDING_MAX_RETRIES", "0")
    recorder = Recorder(status(429))

    embedder = create_text_embedder(make_settings(), httpx.MockTransport(recorder))
    with pytest.raises(TextEmbeddingError):
        embedder.embed(["합성 문장 0"])

    request = recorder.requests[0]
    assert len(recorder.requests) == 1
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert json.loads(request.content)["model"] == "synthetic-embedding-model"
