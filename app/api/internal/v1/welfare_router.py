"""복지 AI 내부 API의 기본 상태 엔드포인트."""

from fastapi import APIRouter, Depends, Request

from app.core.config import Settings, get_settings
from app.infrastructure.welfare.spring_client import SpringWelfareClient

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
