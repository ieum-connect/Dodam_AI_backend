from app.contracts.welfare import WelfareRecommendationRequest


def recommendation_mode(request: WelfareRecommendationRequest) -> str:
    if request.proactive and request.storage_consent:
        return "PROACTIVE"
    if request.additional_profile:
        return "SECOND_PASS"
    return "FIRST_PASS"
