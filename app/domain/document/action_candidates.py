"""Turn grounded VLM extraction into unapproved tasks, never into schedules."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid5
from zoneinfo import ZoneInfo

from app.domain.document.page_extraction import ExtractedItemType, PageExtraction

# Narrow fallback for document-only notices that may still be labeled as cautions.
_DOCUMENT_NOTICE = re.compile(
    r"(?:실제(?:환자기록|진료기록)(?:(?:이나|또는)의료지시)?(?:이|가)?아닙니다"
    r"|(?:AI모델응답을대체한)?(?:연동|통합)?테스트(?:용으로|에)만사용합니다)[.!。]?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ActionCandidate:
    id: UUID
    type: str
    title: str
    description: str
    due_at: datetime | None


def _explicit_datetime(value: str | None, source: str) -> datetime | None:
    # Date-only and relative dates deliberately require guardian input.
    if not value or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?", value
    ):
        return None
    try:
        parsed = datetime.fromisoformat(value)
        dates = list(
            re.finditer(r"(?<!\d)(\d{4})[년./-]\s*(\d{1,2})[월./-]\s*(\d{1,2})(?!\d)", source)
        )
        # An attached ISO offset is a timezone. A dash between prose times is ambiguous.
        clock_source = re.sub(r"(T\d{2}:\d{2}(?::\d{2})?)[+-]\d{2}:\d{2}(?!\d)", r"\1", source)
        times = list(
            re.finditer(
                r"(?<!\d)(?:(오전|오후)\s*)?(\d{1,2})(?::(\d{2})(?!\d)|시(?!간)(?:\s*(\d{1,2})분)?)",
                clock_source,
            )
        )
        if len(dates) != 1 or len(times) != 1:
            return None
        meridiem, source_hour, colon_minute, korean_minute = times[0].groups()
        hour = int(source_hour)
        if meridiem:
            if not 1 <= hour <= 12:
                return None
            hour = hour % 12 + (12 if meridiem == "오후" else 0)
        source_values = (
            *map(int, dates[0].groups()),
            hour,
            int(colon_minute or korean_minute or 0),
        )
        if source_values != (parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute):
            return None
        source_has_zone = re.search(r"(?:Z\b|UTC\b|[+-]\d{2}:\d{2})", source)
        if source_has_zone:
            return parsed if parsed.tzinfo else None
        # A model-added Z/offset is not evidence of a timezone in a Korean document.
        return parsed.replace(tzinfo=ZoneInfo("Asia/Seoul"))
    except ValueError:
        return None


def build_candidates(
    document_id: UUID, pages: Mapping[int, PageExtraction]
) -> list[ActionCandidate]:
    """Only document-grounded schedules, medication and cautions become candidates."""
    candidates: dict[UUID, ActionCandidate] = {}
    for page_no, extraction in sorted(pages.items()):
        for item in extraction.items:
            if item.item_type not in {
                ExtractedItemType.SCHEDULE,
                ExtractedItemType.MEDICATION,
                ExtractedItemType.CAUTION,
            }:
                continue
            # Reject hallucinated evidence. Whitespace may differ between OCR and items.
            source = "".join(item.source_text.split())
            if not source or source not in "".join(extraction.raw_text.split()):
                continue
            if item.item_type == ExtractedItemType.CAUTION and _DOCUMENT_NOTICE.fullmatch(source):
                continue
            kind = item.item_type.value
            if kind == "SCHEDULE":
                kind = (
                    "TEST"
                    if re.search(r"검사|검진|test|exam", item.source_text, re.I)
                    else "REVISIT"
                )
            title = (item.normalized_name or item.source_text).strip()[:200]
            fingerprint = f"{item.item_type}:{source}:{item.normalized_value or ''}"
            candidate_id = uuid5(document_id, fingerprint)
            candidates.setdefault(
                candidate_id,
                ActionCandidate(
                    id=candidate_id,
                    type=kind,
                    title=title,
                    description=f"원문 {page_no}쪽: {item.source_text}",
                    # Medication normalized_value is dosage, not an appointment date.
                    due_at=_explicit_datetime(item.normalized_value, item.source_text)
                    if item.item_type == ExtractedItemType.SCHEDULE
                    else None,
                ),
            )
    return list(candidates.values())
