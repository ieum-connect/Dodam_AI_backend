"""예방접종 일정 계산 내부 API. Spring만 호출하며 프론트가 직접 부르지 않는다."""

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.api.internal.v1.ai_result_document_router import require_internal_token
from app.contracts.vaccination import VaccinationScheduleRequest
from app.domain.vaccination.engine import compute_schedule
from app.domain.vaccination.error import VaccinationError
from app.domain.vaccination.schedule_data import load_schedule

# import 시점에 읽어 데이터가 깨졌으면 서버 기동이 실패하게 한다.
load_schedule()

router = APIRouter(prefix="/internal/vaccinations", tags=["vaccination"])


@router.post("/schedule")
def calculate_schedule(
    request: VaccinationScheduleRequest,
    _: Annotated[None, Depends(require_internal_token)],
) -> dict[str, Any]:
    """생년월일·기준일·체크한 차수로 접종 묶음과 다음 접종을 계산한다."""
    return compute_schedule(
        load_schedule(), request.birth_date, request.as_of, request.completed_doses
    )


async def handle_vaccination_error(_: Request, error: VaccinationError) -> JSONResponse:
    """도메인 오류를 공통 오류와 같은 code·message·timestamp 모양으로 반환한다."""
    return JSONResponse(
        status_code=error.status_code,
        content={
            "code": error.code.value,
            "message": error.message,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )
