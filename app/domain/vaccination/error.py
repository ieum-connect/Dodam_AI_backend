"""예방접종 일정 계산의 도메인 오류다. 공통 오류와 같은 JSON 모양, 400으로 내려 준다."""

from enum import StrEnum


class VaccinationErrorCode(StrEnum):
    """예방접종 계산에서 Spring에 전달하는 오류 코드다."""

    BIRTH_AFTER_AS_OF = "VACCINATION_001"
    UNKNOWN_DOSE_ID = "VACCINATION_002"


_MESSAGE_BY_CODE: dict[VaccinationErrorCode, str] = {
    VaccinationErrorCode.BIRTH_AFTER_AS_OF: "생년월일이 기준일보다 늦습니다.",
    VaccinationErrorCode.UNKNOWN_DOSE_ID: "알 수 없는 접종 차수가 포함되어 있습니다.",
}


class VaccinationError(Exception):
    """입력 값이 일정 규칙과 맞지 않을 때 쓰는 오류다. 메시지에 입력 원문을 넣지 않는다."""

    status_code = 400

    def __init__(self, code: VaccinationErrorCode) -> None:
        super().__init__(code.value)
        self.code = code

    @property
    def message(self) -> str:
        return _MESSAGE_BY_CODE[self.code]
