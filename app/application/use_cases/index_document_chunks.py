"""저장된 페이지 원문으로 문서를 청크·임베딩으로 색인하는 유스케이스.

분석 직후 색인(analyze_document)과 임베딩 실패 뒤 재색인(스크립트, 재시도 API)이 같은 코드를 쓴다.
VLM은 다시 부르지 않고, 문서와 분석의 상태는 바꾸지 않는다. HTTP와 무관하다.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from app.application.ports.document_chunk_repository import DocumentChunkRepository
from app.application.ports.text_embedder import TextEmbedder, TextEmbeddingError, TextEmbeddings
from app.domain.document.document_analysis import AnalysisStatus
from app.pipelines.documents.steps.chunk_pages import chunk_pages

logger = logging.getLogger(__name__)

UNEXPECTED_ERROR_CODE = "UNEXPECTED_ERROR"


class ChunkIndexOutcome(StrEnum):
    """색인 한 번의 결과 종류."""

    INDEXED = "INDEXED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


class ChunkIndexSkipReason(StrEnum):
    """색인을 건너뛴 이유."""

    NOT_FOUND = "NOT_FOUND"  # 문서가 없거나 삭제됐거나 분석 행이 없다.
    ANALYSIS_NOT_SUCCEEDED = "ANALYSIS_NOT_SUCCEEDED"
    ALREADY_INDEXED = "ALREADY_INDEXED"  # 최신 분석에 청크가 이미 있다.
    SUPERSEDED = "SUPERSEDED"  # 교체 직전에 문서가 삭제됐거나 더 새 분석이 생겼다.


@dataclass(frozen=True)
class ChunkIndexResult:
    """색인 한 번의 결과. 원문은 담지 않는다.

    error_code는 임베딩 오류 코드 값이고, 그 밖의 예외는 UNEXPECTED_ERROR다.
    """

    outcome: ChunkIndexOutcome
    analysis_id: UUID | None = None
    chunk_count: int = 0
    skip_reason: ChunkIndexSkipReason | None = None
    error_code: str | None = None


def index_document_chunks(
    document_id: UUID, *, chunk_repository: DocumentChunkRepository, embedder: TextEmbedder
) -> ChunkIndexResult:
    """삭제되지 않은 문서의 최신 분석이 SUCCEEDED이고 아직 색인 전일 때만 다시 색인한다."""
    try:
        target = chunk_repository.find_index_target(document_id)
    except Exception as error:
        _log_failure(document_id, None, error)
        return ChunkIndexResult(ChunkIndexOutcome.FAILED, error_code=UNEXPECTED_ERROR_CODE)
    if target is None:
        return _skipped(document_id, None, ChunkIndexSkipReason.NOT_FOUND)
    if target.status != AnalysisStatus.SUCCEEDED:
        reason = ChunkIndexSkipReason.ANALYSIS_NOT_SUCCEEDED
        return _skipped(document_id, target.analysis_id, reason)
    if target.has_chunks:
        # 같은 분석을 다시 교체하면 유일 키 위반이므로, 임베딩 비용을 쓰기 전에 거른다.
        # ponytail: 동시 재색인 둘이 함께 통과하면 늦은 쪽이 유일 키 위반으로 FAILED가 된다(롤백되어
        # 데이터는 안전). 막으려면 replace_chunks 잠금 안에서 이 분석의 청크 존재를 확인한다.
        return _skipped(document_id, target.analysis_id, ChunkIndexSkipReason.ALREADY_INDEXED)
    return index_analysis_chunks(
        document_id, target.analysis_id, chunk_repository=chunk_repository, embedder=embedder
    )


def index_analysis_chunks(
    document_id: UUID,
    analysis_id: UUID,
    *,
    chunk_repository: DocumentChunkRepository,
    embedder: TextEmbedder,
) -> ChunkIndexResult:
    """페이지 원문을 청크로 나누고 임베딩해 이전 분석의 청크·임베딩과 교체한다.

    예외를 올리지 않고 결과로 돌려준다.
    분석 직후 호출에서 이미 저장한 분석 결과가 FAILED로 바뀌지 않게 하려는 것이다.
    """
    try:
        chunks = chunk_pages(chunk_repository.find_page_texts(document_id))
        # 임베딩 호출은 오래 걸리므로 DB 트랜잭션 밖에서 끝낸다.
        embeddings = (
            embedder.embed([chunk.content for chunk in chunks])
            if chunks
            else TextEmbeddings(vectors=[], model_name="", dimensions=0)
        )
        # 청크가 없어도 교체해야 이전 분석의 임베딩이 검색에 남지 않는다.
        replaced = chunk_repository.replace_chunks(
            document_id, analysis_id, chunks, embeddings, datetime.now(UTC)
        )
    except Exception as error:
        _log_failure(document_id, analysis_id, error)
        code = error.code.value if isinstance(error, TextEmbeddingError) else UNEXPECTED_ERROR_CODE
        return ChunkIndexResult(ChunkIndexOutcome.FAILED, analysis_id=analysis_id, error_code=code)
    if not replaced:
        return _skipped(document_id, analysis_id, ChunkIndexSkipReason.SUPERSEDED)
    logger.info(
        "청크 색인 완료: document_id=%s, analysis_id=%s, chunks=%d",
        document_id,
        analysis_id,
        len(chunks),
    )
    return ChunkIndexResult(
        ChunkIndexOutcome.INDEXED, analysis_id=analysis_id, chunk_count=len(chunks)
    )


def _skipped(
    document_id: UUID, analysis_id: UUID | None, reason: ChunkIndexSkipReason
) -> ChunkIndexResult:
    """건너뛴 이유를 info 로그로 남기고 결과로 돌려준다."""
    logger.info(
        "청크 색인 건너뜀: document_id=%s, analysis_id=%s, reason=%s",
        document_id,
        analysis_id,
        reason,
    )
    return ChunkIndexResult(
        ChunkIndexOutcome.SKIPPED, analysis_id=analysis_id, skip_reason=reason
    )


def _log_failure(document_id: UUID, analysis_id: UUID | None, error: Exception) -> None:
    """실패를 error 로그로 남긴다. 메시지에 원문이 있을 수 있어 클래스 이름과 오류 코드만 쓴다."""
    logger.error(
        "청크 색인 실패, 분석 결과는 유지합니다: document_id=%s, analysis_id=%s, error=%s%s",
        document_id,
        analysis_id,
        type(error).__name__,
        f"({error.code})" if isinstance(error, TextEmbeddingError) else "",
    )
