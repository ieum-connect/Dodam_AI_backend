"""페이지 원문을 슬라이딩 윈도우 청크로 나누는 문서 파이프라인 단계.

DB, 설정, 외부 API를 import하지 않는 순수 함수다. 같은 입력이면 항상 같은 결과를 반환한다.
tiktoken은 첫 호출 때 인코딩 파일만 내려받는다. 문서 원문은 밖으로 보내지 않는다.
"""

import re
from collections.abc import Sequence
from functools import lru_cache

import tiktoken

from app.domain.document.document_chunk import DocumentChunk, PageText
from app.domain.document.document_page import PageOcrStatus

MAX_TOKENS = 500
MIN_TOKENS = 300
OVERLAP_TOKENS = 60
# 단위가 이 크기 이하면 다음 단위가 들어가지 않아 끊긴 청크는 MIN_TOKENS 안팎 이상이 된다.
MAX_UNIT_TOKENS = MAX_TOKENS - MIN_TOKENS

_ENCODING_NAME = "cl100k_base"
# 검사표는 행이 의미 단위라 줄바꿈에서 먼저 자른다. 문장 끝은 점 앞이 숫자가 아니고 뒤에 공백이
# 있을 때만 자른다. 그래서 "11.2 g/dL"과 "1. 항목"은 잘리지 않는다.
_UNIT_BOUNDARY = re.compile(r"\n\s*|(?<=[^\d\s][.!?。])\s+")
_WHITESPACE = re.compile(r"\s+")

type _Span = tuple[int, int]


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_ENCODING_NAME)


def count_tokens(text: str) -> int:
    """글의 토큰 수를 센다. 원문에 특수 토큰 모양의 글자가 있어도 일반 글자로 센다."""
    return len(_encoding().encode_ordinary(text))


def chunk_pages(pages: Sequence[PageText]) -> list[DocumentChunk]:
    """페이지 번호 순서로 원문을 청크로 나눈다. 원문이 없거나 SKIPPED인 페이지는 건너뛴다.

    MAX_TOKENS 이하인 원문은 쪼개지 않는다. 더 긴 원문을 나눈 청크는 MAX_TOKENS를 넘지 않고,
    페이지의 마지막 청크만 MIN_TOKENS보다 짧을 수 있다. 청크는 페이지를 넘어 합치지 않는다.
    """
    chunks: list[DocumentChunk] = []
    for page in sorted(pages, key=lambda page: page.page_no):
        if page.ocr_status == PageOcrStatus.SKIPPED or not page.ocr_text:
            continue
        for content in _split_text(page.ocr_text.strip()):
            chunks.append(
                DocumentChunk(
                    chunk_index=len(chunks),
                    content=content,
                    page_id=page.page_id,
                    token_count=count_tokens(content),
                )
            )
    return chunks


def _split_text(text: str) -> list[str]:
    if not text:
        return []
    if count_tokens(text) <= MAX_TOKENS:
        return [text]
    units = _split_units(text)
    windows: list[str] = []
    first = 0
    while True:
        last = _last_unit_in_window(text, units, first)
        windows.append(text[units[first][0] : units[last][1]].strip())
        if last == len(units) - 1:
            return windows
        first = _overlap_start(text, units, first, last)


def _split_units(text: str) -> list[_Span]:
    """글을 MAX_UNIT_TOKENS 이하 단위로 나눈다. 큰 단위는 공백, 그다음 글자 위치에서 자른다."""
    units: list[_Span] = []
    for unit in _spans(text, 0, len(text), _UNIT_BOUNDARY):
        if count_tokens(text[unit[0] : unit[1]]) <= MAX_UNIT_TOKENS:
            units.append(unit)
            continue
        for word in _spans(text, *unit, _WHITESPACE):
            if count_tokens(text[word[0] : word[1]]) <= MAX_UNIT_TOKENS:
                units.append(word)
            else:
                units.extend(_split_by_characters(text, *word))
    return units


def _spans(text: str, start: int, end: int, boundary: re.Pattern[str]) -> list[_Span]:
    spans: list[_Span] = []
    for match in boundary.finditer(text, start, end):
        if text[start : match.start()].strip():
            spans.append((start, match.start()))
        start = match.end()
    if text[start:end].strip():
        spans.append((start, end))
    return spans


def _split_by_characters(text: str, start: int, end: int) -> list[_Span]:
    """글자 위치에서 자른다. 토큰 ID로 자르면 한글 글자 중간에서 끊겨 깨진 글자가 생길 수 있다."""
    spans: list[_Span] = []
    while start < end:
        # 토큰 수가 글자 수에 꼭 비례하지는 않으므로, 실제로 세어 확인한 위치만 low에 넣는다.
        low, high = start + 1, end
        while low < high:
            middle = (low + high + 1) // 2
            if count_tokens(text[start:middle]) <= MAX_UNIT_TOKENS:
                low = middle
            else:
                high = middle - 1
        spans.append((start, low))
        start = low
    return spans


def _last_unit_in_window(text: str, units: Sequence[_Span], first: int) -> int:
    # 토큰 수는 구간별 합과 다를 수 있어 합산하지 않고 실제 구간을 다시 센다.
    last = first
    while last + 1 < len(units) and (
        count_tokens(text[units[first][0] : units[last + 1][1]]) <= MAX_TOKENS
    ):
        last += 1
    return last


def _overlap_start(text: str, units: Sequence[_Span], first: int, last: int) -> int:
    """다음 윈도우의 시작 단위를 고른다. 단위를 자르지 않아 겹침이 OVERLAP_TOKENS보다 클 수 있다."""
    start = last
    while (
        start > first + 1
        and count_tokens(text[units[start][0] : units[last][1]]) < OVERLAP_TOKENS
    ):
        start -= 1
    # 다음 윈도우에 새 단위가 반드시 하나 들어가야 끝난다.
    while count_tokens(text[units[start][0] : units[last + 1][1]]) > MAX_TOKENS:
        start += 1
    return start
