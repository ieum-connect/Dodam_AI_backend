"""Candidate generation must preserve evidence and never guess appointment times."""

from datetime import timedelta
from uuid import uuid4

import pytest

from app.domain.document.action_candidates import build_candidates
from app.domain.document.page_extraction import ExtractedItem, PageExtraction


def extraction(kind="SCHEDULE", value="2026-12-03T10:00", source="2026년 12월 3일 오전 10시 검사"):
    return PageExtraction(
        document_type="DISCHARGE_GUIDE",
        raw_text=source,
        items=[
            ExtractedItem(
                item_type=kind,
                normalized_name="다음 검사",
                raw_value=source,
                normalized_value=value,
                unit=None,
                reference_range=None,
                confidence=0.95,
                source_text=source,
                source_box=None,
            )
        ],
    )


def test_explicit_korean_time_and_stable_ids_across_pages():
    document_id = uuid4()
    page = extraction()
    rows = build_candidates(document_id, {1: page, 2: page})
    assert len(rows) == 1
    assert rows[0].type == "TEST"
    assert rows[0].due_at.utcoffset() == timedelta(hours=9)
    assert rows[0].id == build_candidates(document_id, {3: page})[0].id


def test_missing_date_relative_date_and_dosage_never_become_appointments():
    for kind, value in [
        ("SCHEDULE", "2026-12-03"),
        ("SCHEDULE", "2주 뒤"),
        ("SCHEDULE", "2026-99-99T10:00"),
        ("MEDICATION", "하루 3회"),
    ]:
        assert build_candidates(uuid4(), {1: extraction(kind, value)})[0].due_at is None


def test_non_action_and_ungrounded_evidence_are_ignored():
    assert not build_candidates(uuid4(), {1: extraction("LAB_VALUE", "13.5")})
    page = extraction().model_copy(update={"raw_text": "이 문서에는 검사 일정이 없습니다."})
    assert not build_candidates(uuid4(), {1: page})


def test_document_notices_do_not_hide_a_clinical_caution():
    sources = [
        "실제 환자 기록이나 의료 지시가 아닙니다.",
        "AI 모델 응답을 대체한 연동 테스트에만 사용합니다.",
        "검사 전 8시간 동안 금식하세요.",
    ]
    page = extraction("CAUTION", None, sources[0]).model_copy(
        update={
            "raw_text": "\n".join(sources),
            "items": [
                extraction("CAUTION", None, source).items[0].model_copy(
                    update={"normalized_name": None}
                )
                for source in sources
            ],
        }
    )
    rows = build_candidates(uuid4(), {1: page})
    assert [(row.type, row.title) for row in rows] == [("CAUTION", sources[2])]


def test_normalized_date_cannot_invent_missing_year_or_time():
    for source in [
        "12월 3일 오전 10시 검사",
        "2026년 12월 3일 검사",
        "2026년 12월 4일 오전 10시 검사",
    ]:
        assert build_candidates(uuid4(), {1: extraction(source=source)})[0].due_at is None


def test_model_added_timezone_does_not_shift_a_korean_appointment():
    candidate = build_candidates(uuid4(), {1: extraction(value="2026-12-03T10:00Z")})[0]
    assert candidate.due_at.hour == 10
    assert candidate.due_at.utcoffset() == timedelta(hours=9)


@pytest.mark.parametrize(
    "source",
    [
        "2026.12.3 오전 10시 검사, 2026.12.17 재진",
        "2026년 12월 3일 10시간 금식 후 검사",
        "2026년 12월 3일 오전 10시 검사 또는 오후 2시 검사",
        "2026년 12월 3일 10:00-11:00 검사",
    ],
)
def test_ambiguous_dates_or_times_and_durations_do_not_set_due_at(source):
    assert build_candidates(uuid4(), {1: extraction(source=source)})[0].due_at is None
