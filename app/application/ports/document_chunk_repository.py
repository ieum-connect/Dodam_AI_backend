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
    """재색인 판단에 쓰는 문서의 최신 분석 상태와 청크 유무."""

    analysis_id: UUID
    status: AnalysisStatus
    has_chunks: bool


class DocumentChunkRepository(Protocol):
    """페이지 원문 조회와 청크 저장 인터페이스. 메서드 하나가 트랜잭션 하나다."""

    def find_page_texts(self, document_id: UUID) -> list[PageText]:
        """문서의 페이지 원문을 page_no 순서로 읽는다."""
        ...

    def find_index_target(self, document_id: UUID) -> IndexTarget | None:
        """삭제되지 않은 문서의 최신 분석을 읽는다. 문서나 분석 행이 없으면 None이다."""
        ...

    def find_unindexed_document_ids(self) -> list[UUID]:
        """최신 분석이 SUCCEEDED이고 원문이 있는데 DOCUMENT_CHUNK 임베딩이 없는 문서 id 목록."""
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
