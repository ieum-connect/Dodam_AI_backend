"""의료 문서 벡터 검색 내부 API 계약."""

from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class DocumentEvidenceSearchRequest(BaseModel):
    request_id: UUID
    patient_id: UUID
    actor_user_id: UUID
    question: str = Field(min_length=1, max_length=2000)
    allowed_document_ids: list[UUID] = Field(min_length=1, max_length=20)
    limit: int = Field(default=12, ge=1, le=50)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("질문은 공백만으로 구성할 수 없습니다.")
        return normalized


class DocumentEvidenceItem(BaseModel):
    citation_id: UUID
    source_type: str = "DOCUMENT_CHUNK"
    document_id: UUID
    chunk_id: UUID
    page_no: int | None = None
    source_text: str
    similarity: float
    document_status: str


class DocumentEvidenceWarning(BaseModel):
    code: str
    document_id: UUID


class DocumentEvidenceSearchResponse(BaseModel):
    request_id: UUID
    model_name: str
    dimensions: int
    searchable_document_ids: list[UUID]
    items: list[DocumentEvidenceItem]
    warnings: list[DocumentEvidenceWarning]
