"""페이지 원문 청킹 단계 단위 테스트. 실제 cl100k_base로 세고, 원문은 모두 합성 데이터다."""

import ast
import re
from pathlib import Path
from uuid import UUID

import pytest

from app.domain.document.document_chunk import DocumentChunk, PageText
from app.domain.document.document_page import PageOcrStatus
from app.pipelines.documents.steps import chunk_pages as chunk_pages_module
from app.pipelines.documents.steps.chunk_pages import (
    MAX_TOKENS,
    MIN_TOKENS,
    OVERLAP_TOKENS,
    chunk_pages,
    count_tokens,
)

PAGE_ID_1 = UUID("00000000-0000-0000-0000-000000000001")
PAGE_ID_2 = UUID("00000000-0000-0000-0000-000000000002")
PAGE_ID_3 = UUID("00000000-0000-0000-0000-000000000003")
LAB_ROWS = [f"항목{i}(T{i}) {i}.5 g/dL  [참고치 {i}.0~{i + 1}.0]" for i in range(1, 201)]
LAB_TEXT = "\n".join(LAB_ROWS)


def make_page(
    text: str | None,
    page_no: int = 1,
    page_id: UUID = PAGE_ID_1,
    status: PageOcrStatus = PageOcrStatus.SUCCEEDED,
) -> PageText:
    """합성 원문으로 페이지 입력을 만든다."""
    return PageText(page_id=page_id, page_no=page_no, ocr_status=status, ocr_text=text)


def text_of_tokens(token_count: int) -> str:
    """토큰 수가 정확히 token_count인 글을 만든다."""
    text = "a" + " a" * (token_count - 1)
    assert count_tokens(text) == token_count
    return text


def overlap_of(previous: str, following: str) -> str:
    """앞 청크의 끝과 다음 청크의 시작이 겹치는 가장 긴 글을 찾는다."""
    for size in range(min(len(previous), len(following)), 0, -1):
        if previous.endswith(following[:size]):
            return following[:size]
    return ""


@pytest.mark.parametrize(
    "pages",
    [
        [],
        [make_page(None)],
        [make_page("")],
        [make_page(" \n\t ")],
        [make_page(None, status=PageOcrStatus.FAILED)],
        [make_page(None, status=PageOcrStatus.SKIPPED)],
        [make_page("원문이 남아 있는 건너뛴 페이지", status=PageOcrStatus.SKIPPED)],
    ],
    ids=["no_pages", "none", "empty", "whitespace", "failed", "skipped", "skipped_with_text"],
)
def test_pages_without_usable_text_make_no_chunks(pages: list[PageText]) -> None:
    """원문이 없거나 공백뿐이거나 SKIPPED인 페이지는 청크를 만들지 않는다."""
    assert chunk_pages(pages) == []


def test_short_text_becomes_one_chunk() -> None:
    """최소 크기보다 짧은 글은 앞뒤 공백만 빼고 그대로 1개 청크가 된다."""
    text = "혈색소 13.5 g/dL\n참고치 12.0~16.0"

    assert chunk_pages([make_page(f"  {text}\n")]) == [
        DocumentChunk(
            chunk_index=0,
            content=text,
            page_id=PAGE_ID_1,
            token_count=count_tokens(text),
            bounding_box=None,
        )
    ]


@pytest.mark.parametrize("token_count", [MIN_TOKENS - 1, MIN_TOKENS, MAX_TOKENS])
def test_text_up_to_max_tokens_is_not_split(token_count: int) -> None:
    """최대 크기 이하의 글은 쪼개지 않는다."""
    chunks = chunk_pages([make_page(text_of_tokens(token_count))])

    assert [chunk.token_count for chunk in chunks] == [token_count]


def test_text_over_max_tokens_is_split() -> None:
    """최대 크기를 1토큰만 넘어도 나누고, 어떤 청크도 최대 크기를 넘지 않는다."""
    chunks = chunk_pages([make_page(text_of_tokens(MAX_TOKENS + 1))])

    assert len(chunks) >= 2
    assert all(chunk.token_count <= MAX_TOKENS for chunk in chunks)


def test_long_text_chunks_stay_within_window() -> None:
    """긴 글의 청크는 300~500 토큰이고 마지막만 짧을 수 있다. token_count는 실제 값과 같다."""
    chunks = chunk_pages([make_page(LAB_TEXT)])

    assert len(chunks) > 2
    assert all(chunk.token_count == count_tokens(chunk.content) for chunk in chunks)
    assert all(chunk.token_count <= MAX_TOKENS for chunk in chunks)
    assert all(chunk.token_count >= MIN_TOKENS for chunk in chunks[:-1])


def test_neighbor_chunks_overlap_on_whole_rows() -> None:
    """이웃 청크는 겹침 길이 이상 겹치고, 검사표 행은 잘리지 않는다."""
    chunks = chunk_pages([make_page(LAB_TEXT)])

    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert count_tokens(overlap_of(previous.content, following.content)) >= OVERLAP_TOKENS
    assert all(line in LAB_ROWS for chunk in chunks for line in chunk.content.split("\n"))


def test_every_row_is_kept_in_order() -> None:
    """겹친 행을 빼고 이으면 원문의 모든 행이 순서대로 나온다."""
    chunks = chunk_pages([make_page(LAB_TEXT)])

    lines = [line for chunk in chunks for line in chunk.content.split("\n")]

    assert list(dict.fromkeys(lines)) == LAB_ROWS


def test_multiple_pages_are_chunked_separately_in_page_order() -> None:
    """페이지를 넘어 합치지 않고, 입력 순서가 뒤섞여도 page_no 순서로 번호를 매긴다."""
    chunks = chunk_pages(
        [
            make_page("둘째 페이지", page_no=2, page_id=PAGE_ID_2),
            make_page("첫째 페이지", page_no=1, page_id=PAGE_ID_1),
        ]
    )

    assert [(c.chunk_index, c.content, c.page_id) for c in chunks] == [
        (0, "첫째 페이지", PAGE_ID_1),
        (1, "둘째 페이지", PAGE_ID_2),
    ]


def test_chunk_index_is_continuous_across_pages() -> None:
    """건너뛴 페이지가 중간에 있어도 chunk_index는 0부터 연속된다."""
    chunks = chunk_pages(
        [
            make_page(LAB_TEXT, page_no=1, page_id=PAGE_ID_1),
            make_page(None, page_no=2, page_id=PAGE_ID_2, status=PageOcrStatus.SKIPPED),
            make_page(LAB_TEXT, page_no=3, page_id=PAGE_ID_3),
        ]
    )

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.page_id for chunk in chunks} == {PAGE_ID_1, PAGE_ID_3}


def test_same_input_gives_same_result() -> None:
    """같은 입력이면 항상 같은 결과다."""
    pages = [make_page(LAB_TEXT)]

    assert chunk_pages(pages) == chunk_pages(pages)


def test_splits_only_at_sentence_ends_not_inside_numbers() -> None:
    """한 줄 글은 문장 끝에서만 자른다. 소수점, 범위, 목록 번호에서는 자르지 않는다."""
    sentences = [f"{i}. 혈색소는 {i}.5 g/dL이고 참고치는 12.0~16.0 입니다." for i in range(1, 201)]

    chunks = chunk_pages([make_page(" ".join(sentences))])

    assert len(chunks) > 1
    for chunk in chunks:
        assert re.fullmatch(r"\d+\. 혈색소.*입니다\.", chunk.content)


def test_long_sentence_is_split_at_whitespace() -> None:
    """단위 상한을 넘는 문장은 공백에서 자르므로 단어가 잘리지 않고, 창 크기도 지킨다."""
    words = [f"단어{i}" for i in range(1, 501)]

    chunks = chunk_pages([make_page(" ".join(words))])

    assert all(word in words for chunk in chunks for word in chunk.content.split())
    assert all(chunk.token_count <= MAX_TOKENS for chunk in chunks)
    assert all(chunk.token_count >= MIN_TOKENS for chunk in chunks[:-1])
    assert count_tokens(overlap_of(chunks[0].content, chunks[1].content)) >= OVERLAP_TOKENS


def test_text_without_whitespace_is_split_without_broken_characters() -> None:
    """공백 없는 글은 글자 위치에서 잘라 깨진 글자가 없고, 잘라도 원문 일부를 그대로 담는다."""
    text = "헤모글로빈당화혈색소콜레스테롤" * 200

    chunks = chunk_pages([make_page(text)])

    assert len(chunks) > 1
    assert all("�" not in chunk.content for chunk in chunks)
    assert all(chunk.content in text for chunk in chunks)
    assert all(chunk.token_count <= MAX_TOKENS for chunk in chunks)
    assert text.startswith(chunks[0].content)
    assert text.endswith(chunks[-1].content)


def test_special_token_text_is_treated_as_plain_text() -> None:
    """원문에 특수 토큰 모양의 글자가 있어도 예외 없이 일반 글자로 다룬다."""
    text = "합성 원문 <|endoftext|> 합성 원문"

    assert [chunk.content for chunk in chunk_pages([make_page(text)])] == [text]


def test_module_has_no_db_settings_or_external_api_imports() -> None:
    """청킹 모듈은 DB, 설정, 외부 API를 import하지 않는다."""
    tree = ast.parse(Path(chunk_pages_module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    forbidden = ("sqlalchemy", "psycopg", "openai", "httpx", "app.core", "app.infrastructure")
    assert not [name for name in imported if name.startswith(forbidden)]
