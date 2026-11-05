"""Authenticated Spring upload -> background VLM extraction -> pending tasks."""

import logging
from contextlib import ExitStack
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.api.internal.v1.ai_result_document_router import require_internal_token
from app.application.services.retrying_page_extractor import RetryingPageExtractor
from app.application.use_cases.analyze_document import analyze_document
from app.core.config import get_settings
from app.domain.common.error import ApiError, CommonErrorCode
from app.infrastructure.db.repositories.document_analysis_repository import (
    SqlAlchemyDocumentAnalysisRepository,
)
from app.infrastructure.db.repositories.document_chunk_repository import (
    SqlAlchemyDocumentChunkRepository,
)
from app.infrastructure.db.session import get_session_factory
from app.infrastructure.providers.embedding import create_text_embedder
from app.infrastructure.providers.llm.openai_page_extractor import create_page_extractor
from app.infrastructure.storage.local_document_storage import LocalDocumentStorage

router = APIRouter(prefix="/internal/documents", tags=["Document analysis"])
logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    documentId: UUID
    objectKey: str = Field(min_length=1, max_length=500)
    mimeType: str = Field(min_length=1, max_length=100)
    requestId: str | None = Field(default=None, max_length=64)


def run_analysis(document_id: UUID) -> None:
    settings = get_settings()
    try:
        storage = LocalDocumentStorage(settings.storage_local_root)
        factory = get_session_factory()
        repository = SqlAlchemyDocumentAnalysisRepository(factory)
        with ExitStack() as resources:
            inner = create_page_extractor(settings)
            resources.callback(inner.close)
            embedder = create_text_embedder(settings)
            resources.callback(embedder.close)
            analyze_document(
                document_id,
                reader=storage,
                writer=storage,
                repository=repository,
                page_repository=repository,
                chunk_repository=SqlAlchemyDocumentChunkRepository(factory),
                embedder=embedder,
                extractor=RetryingPageExtractor(
                    inner,
                    max_retries=settings.openai_vlm_max_retries,
                    backoff_seconds=settings.openai_vlm_retry_backoff_seconds,
                ),
                render_dpi=settings.document_render_dpi,
                max_pages=settings.document_max_pages,
                max_page_side_px=settings.document_max_page_side_px,
            )
    except Exception as error:
        logger.error(
            "Background document analysis failed: document=%s error=%s",
            document_id,
            type(error).__name__,
        )


@router.post("/{document_id}/analyze", status_code=202)
def start_analysis(
    document_id: UUID,
    body: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    _: Annotated[None, Depends(require_internal_token)],
) -> dict[str, str]:
    if document_id != body.documentId:
        raise ApiError(CommonErrorCode.INVALID_REQUEST)
    # Never use a supplied object key to choose a file: the worker reads ownership/storage from DB.
    background_tasks.add_task(run_analysis, document_id)
    return {"status": "accepted", "documentId": str(document_id)}
