"""문서 청킹의 입력과 출력 형식. 출력 필드는 document_chunks 컬럼과 같다."""

from dataclasses import dataclass
from uuid import UUID

from app.domain.document.document_page import PageOcrStatus


@dataclass(frozen=True)
class PageText:
    """청킹할 페이지 한 장의 원문. document_pages 한 행에서 읽는다."""

    page_id: UUID
    page_no: int
    ocr_status: PageOcrStatus
    ocr_text: str | None


@dataclass(frozen=True)
class DocumentChunk:
    """페이지 원문을 나눈 조각 하나.

    chunk_index는 같은 분석 안에서 0부터 연속된다. token_count는 content를 cl100k_base로 센 값이다.
    bounding_box는 아직 채우지 않아 항상 None이다.
    """

    chunk_index: int
    content: str
    page_id: UUID
    token_count: int
    bounding_box: dict[str, int] | None = None
