"""재색인 유스케이스 단위 테스트. 청크 저장소와 임베더는 가짜로 대신한다. 데이터는 합성이다."""

import logging
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

import pytest

from app.application.ports.document_chunk_repository import IndexTarget
from app.application.ports.text_embedder import (
    TextEmbeddingError,
    TextEmbeddingErrorCode,
    TextEmbeddings,
)
from app.application.use_cases.index_document_chunks import (
    ChunkIndexOutcome,
    ChunkIndexResult,
    ChunkIndexSkipReason,
    index_document_chunks,
)
from app.domain.document.document_analysis import AnalysisStatus
from app.domain.document.document_chunk import DocumentChunk, PageText
from app.domain.document.document_page import PageOcrStatus

DOCUMENT_ID = UUID("11111111-2222-3333-4444-555555555555")
ANALYSIS_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
PAGE_ID = UUID("99999999-8888-7777-6666-555555555555")
SOURCE_TEXT = "합성 재색인 원문"
SUCCEEDED_TARGET = IndexTarget(ANALYSIS_ID, AnalysisStatus.SUCCEEDED, has_chunks=False)


class FakeChunkRepository:
    """호출을 기록하는 가짜 청크 저장소. 정해 둔 대상·원문·교체 결과·예외를 쓴다."""

    def __init__(
        self,
        target: IndexTarget | None = SUCCEEDED_TARGET,
        pages: list[PageText] | None = None,
        replaced: bool = True,
        target_error: Exception | None = None,
        replace_error: Exception | None = None,
    ) -> None:
        self.calls: list[tuple] = []
        self.target = target
        self.pages = (
            [PageText(PAGE_ID, 1, PageOcrStatus.SUCCEEDED, SOURCE_TEXT)] if pages is None else pages
        )
        self.replaced = replaced
        self.target_error = target_error
        self.replace_error = replace_error

    def find_index_target(self, document_id: UUID) -> IndexTarget | None:
        """호출을 기록하고 정해 둔 대상을 돌려준다."""
        self.calls.append(("find_index_target", document_id))
        if self.target_error is not None:
            raise self.target_error
        return self.target

    def find_page_texts(self, document_id: UUID) -> list[PageText]:
        """호출을 기록하고 정해 둔 원문 페이지를 돌려준다."""
        self.calls.append(("find_page_texts", document_id))
        return self.pages

    def replace_chunks(
        self,
        document_id: UUID,
        analysis_id: UUID,
        chunks: Sequence[DocumentChunk],
        embeddings: TextEmbeddings,
        created_at: datetime,
    ) -> bool:
        """받은 id, 청크 원문, 벡터를 기록한다."""
        self.calls.append(
            (
                "replace_chunks",
                document_id,
                analysis_id,
                [chunk.content for chunk in chunks],
                embeddings.vectors,
            )
        )
        if self.replace_error is not None:
            raise self.replace_error
        return self.replaced

    def call_names(self) -> list[str]:
        """기록한 호출의 이름만 순서대로 돌려준다."""
        return [call[0] for call in self.calls]


class FakeEmbedder:
    """받은 텍스트를 기록하고 입력마다 합성 벡터 하나를 돌려주는 가짜 임베더."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[list[str]] = []
        self.error = error

    def embed(self, texts: Sequence[str]) -> TextEmbeddings:
        """호출을 기록한다. 정해 둔 예외가 있으면 던진다."""
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        return TextEmbeddings(
            vectors=[[float(index)] for index in range(len(texts))],
            model_name="fake-embedding",
            dimensions=1,
        )


def run(repository: FakeChunkRepository, embedder: FakeEmbedder | None = None) -> ChunkIndexResult:
    """가짜 포트로 유스케이스를 한 번 실행한다."""
    return index_document_chunks(
        DOCUMENT_ID, chunk_repository=repository, embedder=embedder or FakeEmbedder()
    )


def test_indexes_latest_succeeded_analysis() -> None:
    """원문을 읽어 청크 순서대로 임베딩하고 최신 분석 id로 교체한다."""
    repository = FakeChunkRepository()
    embedder = FakeEmbedder()

    result = run(repository, embedder)

    assert result == ChunkIndexResult(
        ChunkIndexOutcome.INDEXED, analysis_id=ANALYSIS_ID, chunk_count=1
    )
    assert repository.call_names() == ["find_index_target", "find_page_texts", "replace_chunks"]
    assert embedder.calls == [[SOURCE_TEXT]]
    assert repository.calls[-1] == (
        "replace_chunks",
        DOCUMENT_ID,
        ANALYSIS_ID,
        [SOURCE_TEXT],
        [[0.0]],
    )


@pytest.mark.parametrize(
    "status", [AnalysisStatus.PENDING, AnalysisStatus.RUNNING, AnalysisStatus.FAILED]
)
def test_skips_when_latest_analysis_not_succeeded(status: AnalysisStatus) -> None:
    """최신 분석이 SUCCEEDED가 아니면 원문을 읽지도 임베딩하지도 않는다."""
    repository = FakeChunkRepository(target=IndexTarget(ANALYSIS_ID, status, has_chunks=False))
    embedder = FakeEmbedder()

    result = run(repository, embedder)

    assert result == ChunkIndexResult(
        ChunkIndexOutcome.SKIPPED,
        analysis_id=ANALYSIS_ID,
        skip_reason=ChunkIndexSkipReason.ANALYSIS_NOT_SUCCEEDED,
    )
    assert repository.call_names() == ["find_index_target"]
    assert embedder.calls == []


def test_skips_deleted_or_missing_document() -> None:
    """삭제됐거나 없는 문서(대상 None)는 아무것도 하지 않는다."""
    repository = FakeChunkRepository(target=None)
    embedder = FakeEmbedder()

    result = run(repository, embedder)

    assert result == ChunkIndexResult(
        ChunkIndexOutcome.SKIPPED, skip_reason=ChunkIndexSkipReason.NOT_FOUND
    )
    assert repository.call_names() == ["find_index_target"]
    assert embedder.calls == []


def test_skips_already_indexed_analysis_without_embedding() -> None:
    """최신 분석에 청크가 이미 있으면 임베딩 비용을 쓰지 않고 건너뛴다."""
    repository = FakeChunkRepository(
        target=IndexTarget(ANALYSIS_ID, AnalysisStatus.SUCCEEDED, has_chunks=True)
    )
    embedder = FakeEmbedder()

    result = run(repository, embedder)

    assert result == ChunkIndexResult(
        ChunkIndexOutcome.SKIPPED,
        analysis_id=ANALYSIS_ID,
        skip_reason=ChunkIndexSkipReason.ALREADY_INDEXED,
    )
    assert repository.call_names() == ["find_index_target"]
    assert embedder.calls == []


def test_embedding_failure_returns_failed_without_replacing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """임베딩이 실패하면 교체하지 않고 오류 코드를 돌려준다. 로그와 결과에 원문이 없다."""
    repository = FakeChunkRepository()
    embedder = FakeEmbedder(error=TextEmbeddingError(TextEmbeddingErrorCode.RATE_LIMITED))

    with caplog.at_level(logging.INFO):
        result = run(repository, embedder)

    assert result == ChunkIndexResult(
        ChunkIndexOutcome.FAILED, analysis_id=ANALYSIS_ID, error_code="RATE_LIMITED"
    )
    assert "replace_chunks" not in repository.call_names()
    assert "TextEmbeddingError(RATE_LIMITED)" in caplog.text
    assert SOURCE_TEXT not in caplog.text
    assert SOURCE_TEXT not in repr(result)


def test_superseded_replacement_is_skipped() -> None:
    """교체 직전에 문서가 삭제됐거나 더 새 분석이 생겨 교체가 False면 건너뜀으로 돌려준다."""
    result = run(FakeChunkRepository(replaced=False))

    assert result == ChunkIndexResult(
        ChunkIndexOutcome.SKIPPED,
        analysis_id=ANALYSIS_ID,
        skip_reason=ChunkIndexSkipReason.SUPERSEDED,
    )


@pytest.mark.parametrize(
    ("arguments", "analysis_id"),
    [
        ({"replace_error": ValueError(SOURCE_TEXT)}, ANALYSIS_ID),
        # 대상 조회에서 실패하면 분석 id를 아직 모른다.
        ({"target_error": OSError(SOURCE_TEXT)}, None),
    ],
    ids=["replace_fails", "target_lookup_fails"],
)
def test_unexpected_error_returns_failed(
    arguments: dict[str, Exception], analysis_id: UUID | None, caplog: pytest.LogCaptureFixture
) -> None:
    """저장소 예외는 올리지 않고 UNEXPECTED_ERROR로 돌려준다. 로그에는 클래스 이름만 남는다."""
    with caplog.at_level(logging.ERROR):
        result = run(FakeChunkRepository(**arguments))

    assert result == ChunkIndexResult(
        ChunkIndexOutcome.FAILED, analysis_id=analysis_id, error_code="UNEXPECTED_ERROR"
    )
    assert next(iter(arguments.values())).__class__.__name__ in caplog.text
    assert SOURCE_TEXT not in caplog.text


def test_no_text_replaces_with_empty_list_without_embedding() -> None:
    """원문이 없어 청크가 0개면 임베딩을 부르지 않고 빈 목록으로 교체한다."""
    repository = FakeChunkRepository(pages=[PageText(PAGE_ID, 1, PageOcrStatus.SKIPPED, None)])
    embedder = FakeEmbedder()

    result = run(repository, embedder)

    assert result == ChunkIndexResult(
        ChunkIndexOutcome.INDEXED, analysis_id=ANALYSIS_ID, chunk_count=0
    )
    assert embedder.calls == []
    assert repository.calls[-1] == ("replace_chunks", DOCUMENT_ID, ANALYSIS_ID, [], [])
