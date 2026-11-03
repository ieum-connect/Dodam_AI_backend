"""document_pages에서 원문을 읽고 document_chunks에 청크를 쓰는 SQLAlchemy 구현."""

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import insert, select
from sqlalchemy.orm import Session, sessionmaker

from app.domain.document.document_chunk import DocumentChunk, PageText
from app.domain.document.document_page import PageOcrStatus
from app.infrastructure.db.models.document_chunk import DocumentChunkModel
from app.infrastructure.db.models.document_page import DocumentPageModel


class SqlAlchemyDocumentChunkRepository:
    """메서드마다 session을 열고 끝날 때 commit한다. 메서드 하나가 트랜잭션 하나다."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def find_page_texts(self, document_id: UUID) -> list[PageText]:
        """문서의 페이지 원문을 page_no 순서로 읽는다."""
        with self._session_factory() as session:
            rows = session.execute(
                select(
                    DocumentPageModel.id,
                    DocumentPageModel.page_no,
                    DocumentPageModel.ocr_status,
                    DocumentPageModel.ocr_text,
                )
                .where(DocumentPageModel.document_id == document_id)
                .order_by(DocumentPageModel.page_no)
            ).all()
        return [
            PageText(
                page_id=row.id,
                page_no=row.page_no,
                ocr_status=PageOcrStatus(row.ocr_status),
                ocr_text=row.ocr_text,
            )
            for row in rows
        ]

    def save_chunks(
        self, analysis_id: UUID, chunks: Sequence[DocumentChunk], created_at: datetime
    ) -> None:
        """청크를 한 트랜잭션으로 넣는다. 충돌하면 이 트랜잭션만 롤백하고 예외를 올린다."""
        if not chunks:
            return
        with self._session_factory.begin() as session:
            session.execute(
                insert(DocumentChunkModel),
                [
                    {
                        "id": uuid4(),
                        "analysis_id": analysis_id,
                        "page_id": chunk.page_id,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "bounding_box": chunk.bounding_box,
                        "token_count": chunk.token_count,
                        "created_at": created_at,
                    }
                    for chunk in chunks
                ],
            )
