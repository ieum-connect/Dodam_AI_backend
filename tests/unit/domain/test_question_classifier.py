"""도담의 부모 친화적인 질문 분류 규칙을 검증한다."""

import pytest

from app.contracts.ai_result_document import QuestionType
from app.domain.ai_result.question_classifier import QuestionClassifier


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("우리 아이 검사 결과가 무슨 뜻이야?", QuestionType.DOCUMENT_EASY_EXPLANATION),
        ("도담아 이 기록을 알려줘", QuestionType.DOCUMENT_EASY_EXPLANATION),
        ("다음 예방접종 일정이 언제야?", QuestionType.FOLLOW_UP_SCHEDULE),
        ("처방받은 약을 요약해줘", QuestionType.MEDICATION_SUMMARY),
    ],
)
def test_classifies_common_parent_questions(question: str, expected: QuestionType) -> None:
    assert QuestionClassifier().classify(question, None) is expected
