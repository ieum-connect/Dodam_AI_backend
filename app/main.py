"""FastAPI 앱을 만들고 상태 확인 API를 등록한다."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.error_handlers import register_error_handlers
from app.api.internal.v1.ai_result_document_router import router as ai_result_document_router
from app.api.internal.v1.document_analysis_router import router as document_analysis_router
from app.api.internal.v1.document_evidence_router import router as document_evidence_router
from app.api.internal.v1.welfare_router import router as welfare_router
from app.application.use_cases.generate_result_document import GenerateResultDocumentUseCase
from app.core.config import get_settings
from app.infrastructure.providers.llm.openai_compatible_client import OpenAiCompatibleLlmClient
from app.application.use_cases.search_document_evidence import SearchDocumentEvidenceUseCase
from app.infrastructure.db.session import get_session_factory
from app.infrastructure.medical_rag.document_evidence_repository import DocumentEvidenceRepository
from app.infrastructure.medical_rag.embedding_provider import MedicalRagEmbeddingProvider


@asynccontextmanager
async def lifespan(app: FastAPI):
    """LLM 클라이언트를 앱 인스턴스마다 한 번 생성하고 종료 시 닫는다."""
    client = OpenAiCompatibleLlmClient(get_settings())
    app.state.ai_result_document_use_case = GenerateResultDocumentUseCase(client)
    app.state.document_evidence_search = SearchDocumentEvidenceUseCase(
        get_settings(),
        MedicalRagEmbeddingProvider(get_settings()),
        DocumentEvidenceRepository(get_session_factory()),
    )
    try:
        yield
    finally:
        client.close()


app = FastAPI(title="Dodam AI Backend", lifespan=lifespan)
register_error_handlers(app)
app.include_router(ai_result_document_router)
app.include_router(document_analysis_router)
app.include_router(document_evidence_router)
app.include_router(welfare_router)


@app.get("/health")
def health() -> dict[str, str]:
    """서버가 살아 있는지 확인한다."""
    return {"status": "ok"}
