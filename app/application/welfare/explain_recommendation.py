from app.contracts.welfare import WelfareExplanationRequest, WelfareExplanationResponse
from app.infrastructure.providers.llm.openai_compatible_client import OpenAiCompatibleLlmClient


def explain(request: WelfareExplanationRequest, llm: OpenAiCompatibleLlmClient | None = None) -> WelfareExplanationResponse:
    recommendation = request.recommendation
    summary = f"{request.program.name}은(는) 현재 입력한 정보 기준 신청 가능성을 확인해볼 수 있는 서비스입니다."
    ai_generated = False
    if llm is not None:
        prompt = (
            "다음은 복지 추천 결과입니다. 정책 내용이나 구비서류를 새로 만들지 말고, "
            "입력된 일치 조건과 추가 확인 조건만 사용해 추천 이유를 한 문장으로 설명하세요. "
            f"서비스명: {request.program.name}; 일치 조건: {request.recommendation.matched_conditions}; "
            f"추가 확인 조건: {request.recommendation.missing_conditions}"
        )
        try:
            generated = llm.generate_text(
                "당신은 복지 추천 이유만 간결하게 설명하는 도우미입니다. 자격을 확정하지 마세요.", prompt
            )
            if generated:
                summary, ai_generated = generated, True
        except Exception:
            pass
    return WelfareExplanationResponse(
        summary=summary,
        matched_reasons=recommendation.matched_conditions,
        additional_checks=recommendation.missing_conditions,
        source_evidence=request.evidence,
        ai_generated=ai_generated,
    )
