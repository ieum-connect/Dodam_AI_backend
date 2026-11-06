from app.contracts.welfare import WelfareExplanationRequest, WelfareExplanationResponse


def explain(request: WelfareExplanationRequest) -> WelfareExplanationResponse:
    recommendation = request.recommendation
    summary = f"{request.program.name}은(는) 현재 입력한 정보 기준 신청 가능성을 확인해볼 수 있는 서비스입니다."
    return WelfareExplanationResponse(
        summary=summary,
        matched_reasons=recommendation.matched_conditions,
        additional_checks=recommendation.missing_conditions,
        source_evidence=request.evidence,
        ai_generated=False,
    )
