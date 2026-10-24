"""의료 판단·처방 변경 질문을 보수적으로 차단한다."""

from app.contracts.ai_result_document import QuestionType


class SafetyCheck:
    """규칙 기반 1차 안전 검사다. LLM은 차단 뒤 호출하지 않는다."""

    _medical_decision_keywords = (
        "진단해",
        "진단을 내려",
        "진단 내려",
        "병인가",
        "병이야",
        "약을 끊",
        "약 끊",
        "약을 바꿔",
        "약 바꿔",
        "용량을 늘",
        "용량 늘",
        "용량을 줄",
        "치료 방법",
        "치료를 선택",
        "응급실",
        "응급인가",
        "병원 가야",
        "해열제 먹여",
        "약 먹여",
        "괜찮은가",
        "괜찮을까",
    )

    _urgent_keywords = (
        "숨을 못",
        "숨 못",
        "호흡 곤란",
        "의식이 없",
        "의식 없",
        "경련",
        "입술이 파래",
        "청색증",
        "심한 출혈",
    )

    def classify(self, question: str) -> QuestionType | None:
        """의료 판단으로 보이면 즉시 차단 유형을 반환한다."""
        normalized = question.replace(" ", "").lower()
        has_medical_decision_keyword = any(
            keyword.replace(" ", "") in normalized
            for keyword in (*self._medical_decision_keywords, *self._urgent_keywords)
        )
        if has_medical_decision_keyword:
            return QuestionType.MEDICAL_DECISION_REQUEST
        return None

    @classmethod
    def user_message(cls, question: str = "") -> str:
        normalized = question.replace(" ", "").lower()
        if any(keyword.replace(" ", "") in normalized for keyword in cls._urgent_keywords):
            return (
                "아이에게 호흡 곤란, 의식 저하, 경련, 청색증 또는 심한 출혈이 있다면 "
                "AI 답변을 기다리지 말고 즉시 119 또는 가까운 응급실에 연락해 주세요."
            )
        return (
            "도담이는 아이의 의료기록을 쉽게 정리해드릴 수 있지만 진단, 약 변경, "
            "치료 결정은 할 수 없어요. 담당 소아청소년과 의료진과 확인해 주세요."
        )
