"""document_chunks 저장소 테스트. 실제 PostgreSQL에 쓰고 테스트가 끝나면 롤백한다."""

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Row, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.domain.document.document_chunk import DocumentChunk, PageText
from app.domain.document.document_page import PageOcrStatus
from app.infrastructure.db.repositories.document_chunk_repository import (
    SqlAlchemyDocumentChunkRepository,
)

CREATED_AT = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)


@pytest.fixture
def page_id(db_session: Session, document_id: UUID) -> UUID:
    """청크가 가리킬 페이지 행을 하나 만든다."""
    return insert_page(db_session, document_id, 1, PageOcrStatus.SUCCEEDED, "합성 원문")


def insert_page(
    session: Session, document_id: UUID, page_no: int, status: PageOcrStatus, ocr_text: str | None
) -> UUID:
    """document_pages에 합성 행을 넣고 id를 돌려준다."""
    page_id = uuid4()
    session.execute(
        text(
            "INSERT INTO document_pages (id, document_id, page_no, ocr_status, ocr_text)"
            " VALUES (:id, :document_id, :page_no, :status, :ocr_text)"
        ),
        {
            "id": page_id,
            "document_id": document_id,
            "page_no": page_no,
            "status": status,
            "ocr_text": ocr_text,
        },
    )
    return page_id


def make_chunks(page_id: UUID, *indexes: int) -> list[DocumentChunk]:
    """받은 번호마다 합성 청크를 하나씩 만든다."""
    return [
        DocumentChunk(
            chunk_index=index, content=f"합성 청크 {index}", page_id=page_id, token_count=index + 10
        )
        for index in indexes
    ]


def fetch_chunks(session: Session, analysis_id: UUID) -> Sequence[Row]:
    """분석의 청크를 chunk_index 순서로 읽는다."""
    return session.execute(
        text(
            "SELECT analysis_id, page_id, chunk_index, content, token_count, created_at,"
            " bounding_box IS NULL AS box_is_sql_null"
            " FROM document_chunks WHERE analysis_id = :id ORDER BY chunk_index"
        ),
        {"id": analysis_id},
    ).all()


def fetch_analysis_status(session: Session, analysis_id: UUID) -> str:
    """분석 작업의 상태를 읽는다."""
    return session.execute(
        text("SELECT status FROM document_analyses WHERE id = :id"), {"id": analysis_id}
    ).scalar_one()


def test_save_inserts_all_fields(
    db_session: Session,
    session_factory: sessionmaker[Session],
    analysis_id: UUID,
    page_id: UUID,
) -> None:
    """청크의 모든 필드를 저장하고, bounding_box는 JSON null이 아닌 SQL NULL로 넣는다."""
    repository = SqlAlchemyDocumentChunkRepository(session_factory)

    repository.save_chunks(analysis_id, make_chunks(page_id, 0, 1), CREATED_AT)

    rows = fetch_chunks(db_session, analysis_id)
    assert [tuple(row) for row in rows] == [
        (analysis_id, page_id, 0, "합성 청크 0", 10, CREATED_AT, True),
        (analysis_id, page_id, 1, "합성 청크 1", 11, CREATED_AT, True),
    ]


def test_empty_chunks_insert_nothing(
    db_session: Session, session_factory: sessionmaker[Session], analysis_id: UUID
) -> None:
    """청크 목록이 비어 있으면 아무 행도 만들지 않는다."""
    SqlAlchemyDocumentChunkRepository(session_factory).save_chunks(analysis_id, [], CREATED_AT)

    assert fetch_chunks(db_session, analysis_id) == []


def test_find_page_texts_returns_only_this_document_in_page_order(
    db_session: Session, session_factory: sessionmaker[Session], document_id: UUID
) -> None:
    """이 문서의 페이지만 page_no 순서로 읽고, 상태와 원문을 그대로 돌려준다."""
    second = insert_page(db_session, document_id, 2, PageOcrStatus.SKIPPED, None)
    first = insert_page(db_session, document_id, 1, PageOcrStatus.SUCCEEDED, "합성 원문")

    pages = SqlAlchemyDocumentChunkRepository(session_factory).find_page_texts(document_id)

    assert pages == [
        PageText(first, 1, PageOcrStatus.SUCCEEDED, "합성 원문"),
        PageText(second, 2, PageOcrStatus.SKIPPED, None),
    ]
    assert SqlAlchemyDocumentChunkRepository(session_factory).find_page_texts(uuid4()) == []


def test_duplicate_chunk_index_rolls_back_only_that_transaction(
    db_session: Session,
    session_factory: sessionmaker[Session],
    analysis_id: UUID,
    page_id: UUID,
) -> None:
    """같은 번호를 다시 저장하면 IntegrityError가 나고 먼저 저장한 행과 분석 상태는 그대로다."""
    repository = SqlAlchemyDocumentChunkRepository(session_factory)
    repository.save_chunks(analysis_id, make_chunks(page_id, 0), CREATED_AT)
    before = fetch_chunks(db_session, analysis_id)
    status_before = fetch_analysis_status(db_session, analysis_id)

    with pytest.raises(IntegrityError):
        repository.save_chunks(
            analysis_id,
            [DocumentChunk(chunk_index=0, content="덮어쓰기 금지", page_id=page_id, token_count=1)],
            CREATED_AT,
        )

    assert fetch_chunks(db_session, analysis_id) == before
    assert fetch_analysis_status(db_session, analysis_id) == status_before


def test_partial_conflict_leaves_no_new_rows(
    db_session: Session,
    session_factory: sessionmaker[Session],
    analysis_id: UUID,
    page_id: UUID,
) -> None:
    """새 번호와 중복 번호가 섞인 저장은 전체가 롤백되어 새 번호도 남지 않는다."""
    repository = SqlAlchemyDocumentChunkRepository(session_factory)
    repository.save_chunks(analysis_id, make_chunks(page_id, 1), CREATED_AT)

    with pytest.raises(IntegrityError):
        repository.save_chunks(analysis_id, make_chunks(page_id, 0, 1, 2), CREATED_AT)

    assert [row.chunk_index for row in fetch_chunks(db_session, analysis_id)] == [1]


def test_same_index_in_other_analysis_keeps_previous_chunks(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    analysis_id: UUID,
    page_id: UUID,
) -> None:
    """다른 분석에는 같은 번호를 저장할 수 있고, 이전 분석의 청크는 그대로 남는다."""
    other_analysis_id = uuid4()
    db_session.execute(
        text(
            "INSERT INTO document_analyses (id, document_id, parser_version)"
            " VALUES (:id, :document_id, 'v1')"
        ),
        {"id": other_analysis_id, "document_id": document_id},
    )
    repository = SqlAlchemyDocumentChunkRepository(session_factory)
    repository.save_chunks(analysis_id, make_chunks(page_id, 0), CREATED_AT)
    before = fetch_chunks(db_session, analysis_id)

    repository.save_chunks(other_analysis_id, make_chunks(page_id, 0), CREATED_AT)

    assert fetch_chunks(db_session, analysis_id) == before
    assert [row.chunk_index for row in fetch_chunks(db_session, other_analysis_id)] == [0]
