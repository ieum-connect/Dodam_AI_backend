"""복지 AI 내부 API의 기본 상태 엔드포인트."""

from fastapi import APIRouter, Depends, Request

from app.core.config import Settings, get_settings
from app.infrastructure.welfare.spring_client import SpringWelfareClient
from app.domain.welfare.policy_chunk import chunk_program
from app.infrastructure.welfare.embedding_store import WelfareEmbeddingStore
from app.application.welfare.search_policy import search
from app.contracts.welfare import WelfareExplanationRequest, WelfareSearchRequest
from app.application.welfare.explain_recommendation import explain

embedding_store = WelfareEmbeddingStore()

router = APIRouter(prefix="/api/v1/welfare", tags=["welfare-ai"])


@router.get("/ready")
def welfare_ready() -> dict[str, str]:
    """복지 AI 모듈이 로드되었는지 확인한다."""
    return {"status": "ready", "mode": "rules-first"}


def welfare_client(settings: Settings = Depends(get_settings)) -> SpringWelfareClient:
    return SpringWelfareClient(settings)


@router.get("/programs")
async def list_programs(client: SpringWelfareClient = Depends(welfare_client)) -> dict[str, object]:
    """Spring DB에 동기화된 실제 복지서비스만 반환한다."""
    programs = await client.list_programs()
    return {"items": programs, "count": len(programs), "source": "spring-welfare-catalog"}


@router.post("/chunks")
async def chunk_programs(client: SpringWelfareClient = Depends(welfare_client)) -> dict[str, object]:
    programs = await client.list_programs()
    chunks = [chunk for program in programs for chunk in chunk_program(program)]
    return {"items": [chunk.__dict__ for chunk in chunks], "count": len(chunks), "source": "spring-welfare-catalog"}


@router.post("/embeddings/sync")
async def sync_embeddings(client: SpringWelfareClient = Depends(welfare_client)) -> dict[str, object]:
    programs = await client.list_programs()
    chunks = [chunk for program in programs for chunk in chunk_program(program)]
    changed = embedding_store.upsert(chunks)
    return {"changed": len(changed), "total": len(embedding_store.all()), "model": "content-hash-v1"}


@router.post("/search")
async def search_programs(request: WelfareSearchRequest) -> dict[str, object]:
    return {"items": search(embedding_store, request.query, request.limit), "source": "welfare-policy-chunks"}


@router.post("/explanations")
def explain_recommendation(request: WelfareExplanationRequest):
    return explain(request)
