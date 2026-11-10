"""청킹할 페이지 원문을 읽고 청크와 그 임베딩을 저장하는 포트를 정의한다."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.application.ports.text_embedder import TextEmbeddings
from app.domain.document.document_analysis import AnalysisStatus
from app.domain.document.document_chunk import DocumentChunk, PageText


@dataclass(frozen=True)
class IndexTarget:
    """재색인 판단에 쓰는 문서의 최신 분석 상태, 청크 수, 임베딩이 없는 청크 수."""

    analysis_id: UUID
    status: AnalysisStatus
    chunk_count: int
    missing_embedding_count: int


@dataclass(frozen=True)
class ChunkText:
    """임베딩을 채울 청크의 id와 본문."""

    chunk_id: UUID
    content: str


class DocumentChunkRepository(Protocol):
    """페이지 원문 조회와 청크 저장 인터페이스. 메서드 하나가 트랜잭션 하나다.

    청크의 임베딩은 source_type이 DOCUMENT_CHUNK인 행만 센다. 한 청크에 그런 임베딩이
    두 개 이상인 경우는 다루지 않는다.
    """

    def find_page_texts(self, document_id: UUID) -> list[PageText]:
        """문서의 페이지 원문을 page_no 순서로 읽는다."""
        ...

    def find_index_target(self, document_id: UUID) -> IndexTarget | None:
        """삭제되지 않은 문서의 최신 분석을 읽는다. 문서나 분석 행이 없으면 None이다."""
        ...

    def find_unindexed_document_ids(self) -> list[UUID]:
        """삭제되지 않았고 최신 분석이 SUCCEEDED이며 원문 있는 페이지가 있는 문서 중,
        최신 분석의 청크가 없거나 임베딩이 없는 청크가 있는 문서 id 목록."""
        ...

    def find_chunks_without_embedding(self, analysis_id: UUID) -> list[ChunkText]:
        """분석의 청크 중 임베딩이 없는 것을 chunk_index 순서로 읽는다."""
        ...

    def add_missing_embeddings(
        self,
        document_id: UUID,
        analysis_id: UUID,
        chunks: Sequence[ChunkText],
        embeddings: TextEmbeddings,
        created_at: datetime,
    ) -> bool:
        """받은 청크 중 아직 임베딩이 없는 청크에만 임베딩을 넣는다. 청크는 지우지 않는다.

        문서가 삭제됐거나 analysis_id가 최신 분석이 아니면 아무것도 바꾸지 않고 False를 반환한다.
        그 사이 다른 실행이 이미 채운 청크는 건너뛰고 True를 반환한다.
        청크 수와 벡터 수가 다르면 ValueError를 올린다.
        """
        ...

    def replace_chunks(
        self,
        document_id: UUID,
        analysis_id: UUID,
        chunks: Sequence[DocumentChunk],
        embeddings: TextEmbeddings,
        created_at: datetime,
    ) -> bool:
        """새 청크·임베딩을 넣고 이전 analysis의 것을 한 트랜잭션으로 지운다.

        문서가 삭제됐거나 analysis_id가 최신 분석이 아니면 아무것도 바꾸지 않고 False를 반환한다.
        삭제 판정은 deleted_at과 status를 둘 다 본다.
        이전 analysis 청크 중 하나라도 인용되어 있으면 이전 청크는 모두 남는다.
        이전 임베딩은 항상 지운다.
        넣다가 실패하면 전부 롤백하고 예외를 올린다.
        """
        ...

    def mark_indexing(self, document_id: UUID, started_at: datetime) -> bool:
        """삭제되지 않은 문서의 색인 상태를 INDEXING으로 바꾼다."""
        ...

    def mark_index_failed(self, document_id: UUID, error_code: str) -> None:
        """기존 임베딩이 없을 때만 FAILED로 바꾸고 오류 코드를 기록한다."""
        ...
