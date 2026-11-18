"""Turn grounded VLM extraction into unapproved tasks, never into schedules."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid5
from zoneinfo import ZoneInfo

from app.domain.document.page_extraction import DocumentType, ExtractedItemType, PageExtraction

# Narrow fallback for document-only notices that may still be labeled as cautions.
_DOCUMENT_NOTICE = re.compile(
    r"(?:실제(?:환자기록|진료기록)(?:(?:이나|또는)의료지시)?(?:이|가)?아닙니다"
    r"|(?:AI모델응답을대체한)?(?:연동|통합)?테스트(?:용으로|에)만사용합니다)[.!。]?",
    re.IGNORECASE,
)
_NUMBERED_ROW = re.compile(r"^\s*[1-9]\d*[.)]?\s+[^\W\d_]")
_ROW_END = re.compile(r"^\s*(?:[·•↳]|[-*]\s)")


@dataclass(frozen=True)
class ActionCandidate:
    id: UUID
    type: str
    title: str
    description: str
    due_at: datetime | None


def _grounded_table_row(source_lines: list[str], raw_text: str) -> bool:
    """Verify column-ordered cells against one uniquely anchored prescription row."""
    if len(source_lines) < 2 or not _NUMBERED_ROW.match(source_lines[0]):
        return False
    anchor = " ".join(source_lines[0].split())
    raw_lines = [" ".join(line.split()) for line in raw_text.splitlines()]
    starts = [
        i for i, line in enumerate(raw_lines)
        if line == anchor or line.startswith(anchor + " ")
    ]
    if len(starts) != 1:
        return False
    start = starts[0]
    cells = [raw_lines[start][len(anchor):]]
    for line in raw_lines[start + 1:]:
        if not line or _NUMBERED_ROW.match(line) or _ROW_END.match(line):
            break
        cells.append(line)
    row = " ".join(cells)
    for line in source_lines[1:]:
        cell = " ".join(line.split())
        match = re.search(r"(?<![\w.])" + re.escape(cell) + r"(?![\w.])", row)
        if match is None:
            return False
        row = row[:match.start()] + "\n" + row[match.end():]
    return True


def _grounded_medication_evidence(source_text: str, raw_text: str) -> bool:
    """Verify a verbatim passage, a numbered table row, or separated lines in page order."""
    raw = "".join(raw_text.split())
    source = "".join(source_text.split())
    if source and source in raw:
        return True
    source_lines = [line for line in source_text.splitlines() if line.strip()]
    if _grounded_table_row(source_lines, raw_text):
        return True
    lines = ["".join(line.split()) for line in source_lines]
    if len(lines) < 2 or any(len(line) < 8 for line in lines):
        return False
    cursor = 0
    for line in lines:
        position = raw.find(line, cursor)
        if position < 0:
            return False
        cursor = position + len(line)
    return True


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
    document_id: UUID,
    pages: Mapping[int, PageExtraction],
    *,
    declared_doc_type: str | None = None,
) -> list[ActionCandidate]:
    """Require prescription context for medication and original text for every candidate."""
    candidates: dict[UUID, ActionCandidate] = {}
    for page_no, extraction in sorted(pages.items()):
        prescription_page = extraction.document_type == DocumentType.PRESCRIPTION
        if declared_doc_type == "PRESCRIPTION":
            prescription_page = (
                prescription_page or extraction.document_type == DocumentType.UNKNOWN
            )
        elif declared_doc_type not in (None, "UNKNOWN"):
            prescription_page = False
        for item in extraction.items:
            if item.item_type not in {
                ExtractedItemType.SCHEDULE,
                ExtractedItemType.MEDICATION,
                ExtractedItemType.CAUTION,
            }:
                continue
            if item.item_type == ExtractedItemType.MEDICATION and not prescription_page:
                continue
            # Reject hallucinated evidence. Whitespace may differ between OCR and items.
            source = "".join(item.source_text.split())
            grounded = bool(source) and source in "".join(extraction.raw_text.split())
            if item.item_type == ExtractedItemType.MEDICATION:
                grounded = _grounded_medication_evidence(item.source_text, extraction.raw_text)
            if not grounded:
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
