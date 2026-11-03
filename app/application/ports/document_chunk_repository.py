"""청킹할 페이지 원문을 읽고 청크와 그 임베딩을 저장하는 포트를 정의한다."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.application.ports.text_embedder import TextEmbeddings
from app.domain.document.document_chunk import DocumentChunk, PageText


class DocumentChunkRepository(Protocol):
    """페이지 원문 조회와 청크 저장 인터페이스. 메서드 하나가 트랜잭션 하나다."""

    def find_page_texts(self, document_id: UUID) -> list[PageText]:
        """문서의 페이지 원문을 page_no 순서로 읽는다."""
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
        인용된 이전 청크는 남기고 그 임베딩만 지운다. 넣다가 실패하면 전부 롤백하고 예외를 올린다.
        """
        ...
