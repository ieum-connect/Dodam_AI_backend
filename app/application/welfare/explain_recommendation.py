"""추천 사유 설명 생성.

추천 점수와 조건 판정은 Spring 추천 엔진이 결정하고, 이 모듈은 그 판정을 문장으로 풀어쓴다.
설명이 다른 수치를 만들어내면 화면의 일치율과 어긋나므로, 생성 결과에 허용되지 않은 백분율이
섞이면 규칙 기반 문장으로 되돌린다.
"""

import re

from app.contracts.welfare import (
    WelfareExplanationRequest,
    WelfareExplanationResponse,
    WelfareProgramInput,
)
from app.domain.welfare.recommendation_context import (
    WelfareChildContext,
    WelfareCondition,
)
from app.infrastructure.providers.llm.openai_compatible_client import OpenAiCompatibleLlmClient

SYSTEM_PROMPT = (
    "당신은 정부 복지서비스 추천 결과를 보호자에게 설명하는 도우미입니다. "
    "주어진 판정 결과와 사업 원문에 있는 내용만 사용하고, 없는 내용은 절대 만들지 마세요. "
    "자격을 확정하지 말고 최종 확인은 신청기관에서 해야 한다는 점을 지키세요."
)
WRITING_RULES = (
    "작성 규칙:\n"
    "1) 3~4문장으로 쓰세요.\n"
    "2) 첫 문장에서 일치율 수치를 그대로 인용하고, 어떤 조건이 맞아서 추천되었는지 구체적으로 쓰세요.\n"
    "3) 두 번째 문장에서는 이 사업이 실제로 무엇을 지원하는지 지원내용 원문에 근거해 쓰세요.\n"
    "4) 마지막 문장에서는 '확인필요'나 '미충족'으로 표시된 조건을 그대로 안내하세요. 없으면 신청 전 기관 확인만 안내하세요.\n"
    "5) 위에 주어진 일치율 외의 백분율이나 금액을 새로 만들지 마세요.\n"
    "6) 보호자에게 말하듯 존댓말로 쓰고, 조건 이름은 주어진 표현을 사용하세요."
)
MAX_SOURCE_LENGTH = 600
PERCENT_PATTERN = re.compile(r"(\d{1,3})\s*%")


def explain(request: WelfareExplanationRequest, llm: OpenAiCompatibleLlmClient | None = None) -> WelfareExplanationResponse:
    recommendation = request.recommendation
    conditions = recommendation.conditions
    fallback = _rule_based_summary(request)
    summary, ai_generated = fallback, False

    if llm is not None:
        try:
            generated = llm.generate_text(SYSTEM_PROMPT, _build_prompt(request))
        except Exception:
            generated = ""
        cleaned = _sanitize(generated, request)
        if cleaned:
            summary, ai_generated = cleaned, True

    return WelfareExplanationResponse(
        summary=summary,
        matched_reasons=_messages(conditions, "MET") or recommendation.matched_conditions,
        additional_checks=_messages(conditions, "UNKNOWN", "UNMET") or recommendation.missing_conditions,
        source_evidence=request.evidence,
        score=recommendation.score,
        status=recommendation.status,
        ai_generated=ai_generated,
    )


def _build_prompt(request: WelfareExplanationRequest) -> str:
    recommendation = request.recommendation
    sections = [
        "[아이·가구 정보]\n" + _describe_child(request.child),
        "[사업 정보]\n" + _describe_program(request.program, recommendation.region_label),
        "[추천 엔진 판정]\n" + _describe_judgement(request),
        WRITING_RULES,
    ]
    return "\n\n".join(sections)


def _describe_child(child: WelfareChildContext | None) -> str:
    if child is None or not child.profile_saved:
        rows = [] if child is None else _child_rows(child)
        rows.append("가족정보 저장: 안 함 (소득·가구 조건은 판정하지 못함)")
        return "\n".join(f"- {row}" for row in rows)
    return "\n".join(f"- {row}" for row in _child_rows(child))


def _child_rows(child: WelfareChildContext) -> list[str]:
    flags = {
        "맞벌이": child.dual_income,
        "한부모": child.single_parent,
        "다문화": child.multicultural,
        "장애": child.disability,
    }
    rows = []
    if child.age_label:
        rows.append(f"아이 연령: {child.age_label}")
    if child.region_label:
        rows.append(f"거주지역: {child.region_label}")
    if child.income_band_label:
        rows.append(f"소득구간: {child.income_band_label}")
    if child.household_size:
        rows.append(f"가구원 수: {child.household_size}명")
    if child.child_count:
        rows.append(f"자녀 수: {child.child_count}명")
    if child.childcare_status:
        rows.append(f"양육 형태: {child.childcare_status}")
    selected = [name for name, value in flags.items() if value]
    rows.append("선택한 가구 요건: " + (", ".join(selected) if selected else "없음"))
    return rows


def _describe_program(program: WelfareProgramInput, region_label: str | None) -> str:
    rows = [f"서비스명: {program.name}"]
    if region_label:
        rows.append(f"신청 가능 지역: {region_label}")
    if program.agency_name:
        rows.append(f"소관기관: {program.agency_name}")
    for label, value in (
        ("지원대상 원문", program.target or program.selection_criteria),
        ("지원내용 원문", program.benefit),
        ("구비서류", program.required_documents),
    ):
        if value:
            rows.append(f"{label}: {_truncate(value)}")
    return "\n".join(f"- {row}" for row in rows)


def _describe_judgement(request: WelfareExplanationRequest) -> str:
    recommendation = request.recommendation
    rows = [
        f"조건 일치율: {_percent(recommendation.score)}% (이 수치를 그대로 인용하세요)",
        f"상태: {'조건을 대부분 충족' if recommendation.status == 'ELIGIBLE' else '추가 확인 필요'}",
    ]
    if recommendation.conditions:
        rows.append("조건별 판정:")
        rows += [
            f"  · [{condition.status_label}] {condition.label}: {condition.message}"
            for condition in recommendation.conditions
        ]
    else:
        rows += [f"  · [충족] {item}" for item in recommendation.matched_conditions]
        rows += [f"  · [확인필요] {item}" for item in recommendation.missing_conditions]
    return "\n".join(f"- {row}" if not row.startswith("  ") else row for row in rows)


def _rule_based_summary(request: WelfareExplanationRequest) -> str:
    recommendation = request.recommendation
    met = _messages(recommendation.conditions, "MET") or recommendation.matched_conditions
    pending = _messages(recommendation.conditions, "UNKNOWN", "UNMET") or recommendation.missing_conditions
    percent = _percent(recommendation.score)
    first = f"{recommendation.name}은(는) 입력한 정보와 조건 일치율이 {percent}%예요."
    if met:
        first += " " + _join_sentences(met)
    middle = ""
    benefit = request.program.benefit or request.program.target
    if benefit:
        middle = f" 지원 내용은 다음과 같아요: {_truncate(benefit, 160)}."
    last = (
        " " + _join_sentences(pending) + " 신청 전에 신청기관에서 최종 자격을 확인해 주세요."
        if pending
        else " 신청 전에 신청기관에서 최종 자격을 확인해 주세요."
    )
    return (first + middle + last).strip()


def _sanitize(text: str, request: WelfareExplanationRequest) -> str:
    """일치율과 어긋나는 수치를 만들어낸 설명은 버린다."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    allowed = {_percent(request.recommendation.score)}
    source = " ".join(
        value
        for value in (
            request.program.target,
            request.program.selection_criteria,
            request.program.benefit,
            request.program.purpose_summary,
            request.program.purpose_detail,
            " ".join(condition.message for condition in request.recommendation.conditions),
        )
        if value
    )
    allowed |= {int(match) for match in PERCENT_PATTERN.findall(source)}
    return "" if any(int(match) not in allowed for match in PERCENT_PATTERN.findall(cleaned)) else cleaned


def _messages(conditions: list[WelfareCondition], *statuses: str) -> list[str]:
    return [condition.message for condition in conditions if condition.status in statuses and condition.message]


def _join_sentences(messages: list[str]) -> str:
    return " ".join(message if message.endswith(("요.", "다.", ".")) else f"{message}." for message in messages[:3])


def _truncate(value: str, limit: int = MAX_SOURCE_LENGTH) -> str:
    collapsed = " ".join(value.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "..."


def _percent(score: float) -> int:
    return int(round(score))
