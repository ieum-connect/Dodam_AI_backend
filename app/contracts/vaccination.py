"""예방접종 일정 계산 내부 API 계약. Spring과 맞춰 camelCase JSON을 쓴다."""

import re
from datetime import date
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from pydantic.alias_generators import to_camel

DoseId = Annotated[str, StringConstraints(min_length=1, max_length=50)]
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class VaccinationScheduleRequest(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, extra="forbid")

    birth_date: date
    as_of: date
    completed_doses: list[DoseId] = Field(default_factory=list, max_length=200)

    @field_validator("birth_date", "as_of", mode="before")
    @classmethod
    def require_iso_date_string(cls, value: Any) -> Any:
        """'YYYY-MM-DD' 문자열만 받는다. 숫자(타임스탬프)·시각 포함 문자열은 거절한다."""
        if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
            raise ValueError("날짜는 YYYY-MM-DD 형식이어야 합니다.")
        return value
