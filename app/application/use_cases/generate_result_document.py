"""근거 기반 상세 설명·비교 문서 생성 유스케이스다."""

import json

from app.contracts.ai_result_document import (
    GeneratedAnswer,
    GenerateResultDocumentRequest,
    GenerateResultDocumentResponse,
    QuestionType,
)
from app.domain.ai_result.error import AiResultGenerationError
from app.domain.ai_result.question_classifier import QuestionClassifier
from app.domain.ai_result.safety_check import SafetyCheck
from app.infrastructure.providers.llm.openai_compatible_client import LlmClient


class GenerateResultDocumentUseCase:
    """LLM 호출 전후에 안전성과 citation 무결성을 강제한다."""

    def __init__(
        self,
        llm_client: LlmClient,
        safety_checks: SafetyCheck | None = None,
        classifier: QuestionClassifier | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._safety_checks = safety_checks or SafetyCheck()
        self._classifier = classifier or QuestionClassifier()

    def generate(self, request: GenerateResultDocumentRequest) -> GenerateResultDocumentResponse:
        blocked_type = self._safety_checks.classify(request.question)
        if blocked_type:
            return GenerateResultDocumentResponse(
                request_id=request.request_id,
                question_type=blocked_type,
                blocked=True,
                block_reason="MEDICAL_DECISION_REQUEST",
                user_message=self._safety_checks.user_message(request.question),
            )

        question_type = self._classifier.classify(request.question, request.question_type_hint)
        if question_type is QuestionType.UNSUPPORTED:
            return GenerateResultDocumentResponse(
                request_id=request.request_id,
                question_type=question_type,
                blocked=True,
                block_reason="UNSUPPORTED_QUESTION",
                user_message=(
                    "아이의 검사 결과 비교, 진료 문서 설명, 처방 기록, 다음 진료 일정처럼 "
                    "의료기록에서 확인 가능한 질문으로 바꿔 주세요."
                ),
            )
        if not request.evidence:
            return GenerateResultDocumentResponse(
                request_id=request.request_id,
                question_type=question_type,
                blocked=False,
                result_status="INSUFFICIENT",
                user_message="아이의 의료문서 또는 건강기록에서 답변 근거를 찾지 못했어요.",
            )

        try:
            answer = self._llm_client.generate_grounded_answer(
                self._system_prompt(question_type),
                self._user_prompt(request, question_type),
            )
        except AiResultGenerationError as error:
            return GenerateResultDocumentResponse(
                request_id=request.request_id,
                question_type=question_type,
                blocked=False,
                failed=True,
                error_code=error.code.value,
                retryable=error.retryable,
                user_message=error.user_message(),
            )
        return self._validated_response(request, question_type, answer)

    @staticmethod
    def _system_prompt(question_type: QuestionType) -> str:
        return f"""당신은 가족 건강 서비스 '도담'의 영유아·어린이 의료기록 설명 도우미입니다.
보호자가 불안하지 않도록 따뜻하고 차분한 존댓말을 사용하되, 중요한 정보는 모호하게 표현하지 마세요.
질문 유형은 {question_type.value}입니다.
제공된 근거에 없는 사실, 진단 확정, 약 변경·중단, 치료 결정을 절대 생성하지 마세요.
아이의 나이·체중·발달 단계가 근거에 없으면 추측하지 마세요.
성인 정상 범위를 소아 기준처럼 적용하지 마세요.
복용량, 예방접종 일정, 응급 여부는 근거만으로 확정하지 마세요.
담당 소아청소년과에 확인하도록 안내하세요.
수치·단위·날짜는 제공된 값 그대로 사용하세요.
답변은 보호자가 휴대폰에서 빠르게 읽을 수 있도록 짧고 명확하게 작성하세요.
가능하면 '한눈에 보기 → 기록에서 확인된 내용 → 보호자가 다음에 확인할 점'의 흐름으로 구성하세요.
전문용어가 나오면 바로 뒤에 일상적인 표현으로 풀어 쓰세요. 겁을 주거나 과도하게 안심시키지 마세요.
사용자를 '엄마', '아빠'로 추정하지 말고 '보호자님' 또는 주어 없는 문장을 사용하세요.
도담이 실제 진료를 대신하지 않는다는 사실을 자연스럽게 안내하되 같은 경고를 반복하지 마세요.
답변의 모든 문장은 citation_ids로 제공된 근거를 하나 이상 인용해야 합니다.
반드시 title, answer, sentences 배열을 가진 JSON만 반환하세요.
sentences 원소는 text와 citation_ids 배열을 가져야 합니다."""

    @staticmethod
    def _user_prompt(request: GenerateResultDocumentRequest, question_type: QuestionType) -> str:
        evidence = [source.model_dump(mode="json") for source in request.evidence]
        return json.dumps(
            {
                "question": request.question,
                "questionType": question_type.value,
                "evidence": evidence,
                "instruction": (
                    "도담의 아이 건강 기록 화면에 표시할 답변입니다. 근거를 충분히 활용하되 "
                    "문서에 적힌 내용과 계산 가능한 비교만 다루고, 보호자가 이해하기 쉬운 "
                    "짧은 문장으로 설명하세요."
                ),
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _validated_response(
        request: GenerateResultDocumentRequest,
        question_type: QuestionType,
        answer: GeneratedAnswer,
    ) -> GenerateResultDocumentResponse:
        allowed = {source.citation_id for source in request.evidence}
        sentences = [
            sentence
            for sentence in answer.sentences
            if sentence.citation_ids and set(sentence.citation_ids).issubset(allowed)
        ]
        if not sentences:
            return GenerateResultDocumentResponse(
                request_id=request.request_id,
                question_type=question_type,
                blocked=False,
                result_status="INSUFFICIENT",
                user_message="생성된 답변에서 원문 근거를 확인하지 못했어요.",
            )
        result_status = "COMPLETE" if len(sentences) == len(answer.sentences) else "PARTIAL"
        return GenerateResultDocumentResponse(
            request_id=request.request_id,
            question_type=question_type,
            blocked=False,
            result_status=result_status,
            title=answer.title,
            answer="\n\n".join(sentence.text for sentence in sentences),
            sentences=sentences,
        )
