"""복지 AI 내부 API의 기본 상태 엔드포인트."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/welfare", tags=["welfare-ai"])


@router.get("/ready")
def welfare_ready() -> dict[str, str]:
    """복지 AI 모듈이 로드되었는지 확인한다."""
    return {"status": "ready", "mode": "rules-first"}
