"""복지 AI 내부 API 요청·응답 계약."""

from pydantic import BaseModel, Field

from app.domain.welfare.recommendation_context import WelfareProfile, WelfareRecommendation


class WelfareProgramInput(BaseModel):
    program_id: str
    external_service_id: str | None = None
    name: str
    purpose_summary: str | None = None
    purpose_detail: str | None = None
    target: str | None = None
    selection_criteria: str | None = None
    benefit: str | None = None
    application_method: str | None = None
    required_documents: str | None = None
    agency_name: str | None = None
    detail_url: str | None = None
    online_application_url: str | None = None
    modified_at: str | None = None


class WelfareSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    profile: WelfareProfile | None = None
    limit: int = Field(default=10, ge=1, le=50)


class WelfareRecommendationRequest(BaseModel):
    patient_id: str
    profile: WelfareProfile
    additional_profile: dict[str, str | int | bool | None] = Field(default_factory=dict)
    storage_consent: bool = False
    proactive: bool = False


class WelfareSearchResult(BaseModel):
    program_id: str
    name: str
    score: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)


class WelfareExplanationRequest(BaseModel):
    program: WelfareProgramInput
    recommendation: WelfareRecommendation
    evidence: list[str] = Field(default_factory=list, max_length=20)


class WelfareExplanationResponse(BaseModel):
    summary: str
    matched_reasons: list[str]
    additional_checks: list[str]
    source_evidence: list[str]
    ai_generated: bool = True
