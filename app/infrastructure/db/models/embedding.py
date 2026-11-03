"""Spring Flyway가 만든 embeddings 테이블 매핑. V2를 따른다."""

from datetime import datetime
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, String, Uuid, func
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.models.base import Base

SOURCE_TYPE_DOCUMENT_CHUNK = "DOCUMENT_CHUNK"
EMBEDDING_DIMENSIONS = 768


class EmbeddingModel(Base):
    """청크나 건강 기록 하나의 벡터. source_id는 FK 없이 원본 행 id를 가리킨다."""

    __tablename__ = "embeddings"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    source_type: Mapped[str] = mapped_column(String(30))
    source_id: Mapped[UUID] = mapped_column(Uuid)
    # patients, visits는 매핑하지 않아 FK를 적지 않는다. 그래서 대상 모델 import도 필요 없다.
    patient_id: Mapped[UUID] = mapped_column(Uuid)
    visit_id: Mapped[UUID | None] = mapped_column(Uuid)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    model_name: Mapped[str] = mapped_column(String(100))
    dimensions: Mapped[int] = mapped_column(Integer, server_default=str(EMBEDDING_DIMENSIONS))
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True, precision=6), server_default=func.current_timestamp()
    )
