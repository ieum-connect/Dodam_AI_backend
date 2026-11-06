"""질문을 임베딩하고 허용된 의료 문서 청크만 검색한다."""

from app.contracts.document_evidence import (
    DocumentEvidenceSearchRequest,
    DocumentEvidenceSearchResponse,
    DocumentEvidenceWarning,
)
from app.core.config import Settings
from app.infrastructure.medical_rag.document_evidence_repository import (
    DocumentEvidenceRepository,
)
from app.infrastructure.medical_rag.embedding_provider import MedicalRagEmbeddingProvider


class SearchDocumentEvidenceUseCase:
    def __init__(
        self,
        settings: Settings,
        provider: MedicalRagEmbeddingProvider,
        repository: DocumentEvidenceRepository,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._repository = repository

    def search(
        self, request: DocumentEvidenceSearchRequest
    ) -> DocumentEvidenceSearchResponse:
        query_embedding = self._provider.embed_query(request.question)
        items = self._repository.search(
            patient_id=request.patient_id,
            allowed_document_ids=request.allowed_document_ids,
            query_embedding=query_embedding,
            model_name=self._provider.model,
            dimensions=self._provider.dimensions,
            limit=min(request.limit, self._settings.medical_rag_search_limit),
        )
        warning_document_ids = {
            item.document_id
            for item in items
            if item.document_status == "NEEDS_REVIEW"
        }
        return DocumentEvidenceSearchResponse(
            request_id=request.request_id,
            model_name=self._provider.model,
            dimensions=self._provider.dimensions,
            items=items,
            warnings=[
                DocumentEvidenceWarning(
                    code="PARTIAL_DOCUMENT_ANALYSIS", document_id=document_id
                )
                for document_id in sorted(warning_document_ids, key=str)
            ],
        )
