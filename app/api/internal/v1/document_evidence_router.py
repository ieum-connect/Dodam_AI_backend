"""Spring 내부 호출용 의료 문서 벡터 검색 API."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.api.internal.v1.ai_result_document_router import require_internal_token
from app.contracts.document_evidence import (
    DocumentEvidenceSearchRequest,
    DocumentEvidenceSearchResponse,
)

router = APIRouter(prefix="/internal/document-evidence", tags=["document evidence"])


@router.post("/search", response_model=DocumentEvidenceSearchResponse)
def search_document_evidence(
    request: DocumentEvidenceSearchRequest,
    _: Annotated[None, Depends(require_internal_token)],
    http_request: Request,
) -> DocumentEvidenceSearchResponse:
    return http_request.app.state.document_evidence_search.search(request)
