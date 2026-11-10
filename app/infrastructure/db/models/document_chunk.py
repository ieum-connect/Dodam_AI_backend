"""Spring Flyway가 만든 document_chunks 테이블 매핑. V2를 따른다."""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import ForeignKey, Integer, Text, UniqueConstraint, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column

# ORM bulk INSERT가 FK 대상 테이블을 찾을 수 있게 그 매핑을 함께 불러온다.
from app.infrastructure.db.models import document, document_analysis, document_page  # noqa: F401
from app.infrastructure.db.models.base import Base


class DocumentChunkModel(Base):
    """분석 한 번에서 나온 청크 하나. (analysis_id, chunk_index)가 유일하다."""

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("analysis_id", "chunk_index", name="document_chunks_analysis_index_key"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    analysis_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("document_analyses.id", name="document_chunks_analysis_fk")
    )
    page_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("document_pages.id", name="document_chunks_page_fk")
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    # 박스가 없으면 JSON null이 아니라 SQL NULL로 넣는다.
    bounding_box: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True))
    token_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True, precision=6), server_default=func.current_timestamp()
    )
