"""청킹할 페이지 원문을 읽고 document_chunks에 청크를 저장하는 포트를 정의한다."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.domain.document.document_chunk import DocumentChunk, PageText


class DocumentChunkRepository(Protocol):
    """페이지 원문 조회와 청크 저장 인터페이스. 메서드 하나가 트랜잭션 하나다."""

    def find_page_texts(self, document_id: UUID) -> list[PageText]:
        """문서의 페이지 원문을 page_no 순서로 읽는다."""
        ...

    def save_chunks(
        self, analysis_id: UUID, chunks: Sequence[DocumentChunk], created_at: datetime
    ) -> None:
        """청크를 모두 넣는다. (analysis_id, chunk_index)가 충돌하면 전부 취소하고 예외를 올린다."""
        ...
