"""Spring 추천 결과와 AI 설명 생성에 사용하는 순수 계약."""

from pydantic import BaseModel, Field


class WelfareProfile(BaseModel):
    age_months: int | None = Field(default=None, ge=0, le=240)
    region_code: str | None = None
    household_size: int | None = Field(default=None, ge=1, le=30)
    income_band: str | None = None


class WelfareRecommendation(BaseModel):
    program_id: str
    name: str
    score: float = Field(ge=0, le=100)
    status: str
    matched_conditions: list[str] = Field(default_factory=list)
    missing_conditions: list[str] = Field(default_factory=list)
