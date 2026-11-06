"""Spring 추천 결과와 AI 설명 생성에 사용하는 순수 계약."""

from pydantic import BaseModel, Field


class WelfareProfile(BaseModel):
    age_months: int | None = Field(default=None, ge=0, le=240)
    region_code: str | None = None
    household_size: int | None = Field(default=None, ge=1, le=30)
    income_band: str | None = None


class WelfareCondition(BaseModel):
    """추천 엔진이 판정한 조건 하나. 설명은 이 판정 외의 내용을 만들지 않는다."""

    key: str
    label: str
    status: str = Field(pattern="^(MET|UNMET|UNKNOWN)$")
    weight: float = 0
    message: str = ""

    @property
    def status_label(self) -> str:
        return {"MET": "충족", "UNMET": "미충족", "UNKNOWN": "확인필요"}[self.status]


class WelfareChildContext(BaseModel):
    """설명에 사용할 아이·가구 요약. 저장되지 않은 항목은 None으로 온다."""

    child_name: str | None = None
    age_label: str | None = None
    age_months: int | None = Field(default=None, ge=0, le=300)
    region_label: str | None = None
    income_band_label: str | None = None
    household_size: int | None = Field(default=None, ge=1, le=30)
    child_count: int | None = Field(default=None, ge=0, le=30)
    dual_income: bool | None = None
    single_parent: bool | None = None
    multicultural: bool | None = None
    disability: bool | None = None
    childcare_status: str | None = None
    profile_saved: bool = False


class WelfareRecommendation(BaseModel):
    program_id: str
    name: str
    score: float = Field(ge=0, le=100)
    status: str
    region_scope: str | None = None
    region_label: str | None = None
    matched_conditions: list[str] = Field(default_factory=list)
    missing_conditions: list[str] = Field(default_factory=list)
    conditions: list[WelfareCondition] = Field(default_factory=list)
