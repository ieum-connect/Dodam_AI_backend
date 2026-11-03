"""document_chunks 저장소 테스트. 실제 PostgreSQL에 쓰고 테스트가 끝나면 롤백한다."""

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Row, select, text
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session, sessionmaker

from app.application.ports.text_embedder import TextEmbeddings
from app.domain.document.document_chunk import DocumentChunk, PageText
from app.domain.document.document_page import PageOcrStatus
from app.infrastructure.db.models.embedding import EmbeddingModel
from app.infrastructure.db.repositories.document_chunk_repository import (
    SqlAlchemyDocumentChunkRepository,
)

CREATED_AT = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
MODEL_NAME = "text-embedding-3-small"
DIMENSIONS = 768


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


def make_vector(seed: float, dimensions: int = DIMENSIONS) -> list[float]:
    """첫 값에 seed를 담은 합성 벡터."""
    return [seed] + [0.0] * (dimensions - 1)


def make_embeddings(*vectors: list[float]) -> TextEmbeddings:
    """받은 벡터를 담은 합성 임베딩 결과."""
    return TextEmbeddings(vectors=list(vectors), model_name=MODEL_NAME, dimensions=DIMENSIONS)


def insert_analysis(session: Session, document_id: UUID, days: int) -> UUID:
    """created_at을 오늘 기준 days일 옮긴 분석 작업을 넣는다. 최신 판정 순서를 정하려고 쓴다."""
    analysis_id = uuid4()
    session.execute(
        text(
            "INSERT INTO document_analyses (id, document_id, parser_version, created_at)"
            " VALUES (:id, :document_id, 'v1', CURRENT_TIMESTAMP + make_interval(days => :days))"
        ),
        {"id": analysis_id, "document_id": document_id, "days": days},
    )
    return analysis_id


def insert_embedding(
    session: Session, analysis_id: UUID, source_id: UUID, source_type: str
) -> None:
    """분석한 문서의 환자·방문으로 임베딩 하나를 넣는다."""
    session.execute(
        text(
            "INSERT INTO embeddings"
            " (id, source_type, source_id, patient_id, visit_id, embedding, model_name)"
            " SELECT :id, :source_type, :source_id, d.patient_id, d.visit_id,"
            " CAST(:embedding AS vector), 'old-model'"
            " FROM documents d JOIN document_analyses a ON a.document_id = d.id"
            " WHERE a.id = :analysis_id"
        ),
        {
            "id": uuid4(),
            "source_type": source_type,
            "source_id": source_id,
            "embedding": "[" + ",".join(str(value) for value in make_vector(9.0)) + "]",
            "analysis_id": analysis_id,
        },
    )


def insert_old_chunk(
    session: Session, analysis_id: UUID, page_id: UUID, chunk_index: int
) -> UUID:
    """청크 하나와 그 청크의 DOCUMENT_CHUNK 임베딩을 raw SQL로 넣고 청크 id를 돌려준다."""
    chunk_id = uuid4()
    session.execute(
        text(
            "INSERT INTO document_chunks (id, analysis_id, page_id, chunk_index, content)"
            " VALUES (:id, :analysis_id, :page_id, :chunk_index, :content)"
        ),
        {
            "id": chunk_id,
            "analysis_id": analysis_id,
            "page_id": page_id,
            "chunk_index": chunk_index,
            "content": "이전 합성 청크",
        },
    )
    insert_embedding(session, analysis_id, chunk_id, "DOCUMENT_CHUNK")
    return chunk_id


def cite_chunk(session: Session, document_id: UUID, chunk_id: UUID) -> None:
    """설명 인용(explanation_citations)이 청크를 참조하게 만든다."""
    explanation_id = uuid4()
    session.execute(
        text("INSERT INTO ai_explanations (id, document_id) VALUES (:id, :document_id)"),
        {"id": explanation_id, "document_id": document_id},
    )
    session.execute(
        text(
            "INSERT INTO explanation_citations (id, explanation_id, chunk_id)"
            " VALUES (:id, :explanation_id, :chunk_id)"
        ),
        {"id": uuid4(), "explanation_id": explanation_id, "chunk_id": chunk_id},
    )


def chunk_ids_of(session: Session, analysis_id: UUID) -> list[UUID]:
    """분석의 청크 id를 chunk_index 순서로 읽는다."""
    return list(
        session.execute(
            text("SELECT id FROM document_chunks WHERE analysis_id = :id ORDER BY chunk_index"),
            {"id": analysis_id},
        ).scalars()
    )


def embedded_source_ids(session: Session) -> set[UUID]:
    """테스트 트랜잭션에서 보이는 모든 임베딩의 source_id."""
    return set(session.execute(text("SELECT source_id FROM embeddings")).scalars())


def replace_one(
    session_factory: sessionmaker[Session], document_id: UUID, analysis_id: UUID, page_id: UUID
) -> bool:
    """청크 하나와 벡터 하나로 교체를 실행한다."""
    return SqlAlchemyDocumentChunkRepository(session_factory).replace_chunks(
        document_id,
        analysis_id,
        make_chunks(page_id, 0),
        make_embeddings(make_vector(1.0)),
        CREATED_AT,
    )


def test_replace_inserts_chunks_and_matching_embeddings(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    analysis_id: UUID,
    page_id: UUID,
) -> None:
    """청크마다 임베딩 하나를 넣고 source_id, 환자·방문, 모델, 차원, 벡터 값을 맞춘다."""
    replaced = SqlAlchemyDocumentChunkRepository(session_factory).replace_chunks(
        document_id,
        analysis_id,
        make_chunks(page_id, 0, 1),
        make_embeddings(make_vector(0.25), make_vector(-0.5)),
        CREATED_AT,
    )

    assert replaced is True
    assert [tuple(row) for row in fetch_chunks(db_session, analysis_id)] == [
        (analysis_id, page_id, 0, "합성 청크 0", 10, CREATED_AT, True),
        (analysis_id, page_id, 1, "합성 청크 1", 11, CREATED_AT, True),
    ]
    chunk_ids = chunk_ids_of(db_session, analysis_id)
    owner = db_session.execute(
        text("SELECT patient_id, visit_id FROM documents WHERE id = :id"), {"id": document_id}
    ).one()
    rows = {
        row.source_id: row
        for row in db_session.execute(
            select(EmbeddingModel).where(EmbeddingModel.source_id.in_(chunk_ids))
        ).scalars()
    }
    assert set(rows) == set(chunk_ids)
    for chunk_id, seed in zip(chunk_ids, [0.25, -0.5], strict=True):
        row = rows[chunk_id]
        assert row.source_type == "DOCUMENT_CHUNK"
        assert (row.patient_id, row.visit_id) == (owner.patient_id, owner.visit_id)
        assert row.model_name == MODEL_NAME
        assert row.dimensions == DIMENSIONS
        assert row.created_at == CREATED_AT
        assert list(row.embedding) == pytest.approx(make_vector(seed))


def test_replace_removes_previous_analysis_chunks_and_embeddings(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    analysis_id: UUID,
    page_id: UUID,
) -> None:
    """새 analysis를 적재하면 이전 analysis의 청크와 임베딩이 사라진다."""
    old_analysis_id = insert_analysis(db_session, document_id, days=-1)
    old_ids = [insert_old_chunk(db_session, old_analysis_id, page_id, i) for i in (0, 1)]

    assert replace_one(session_factory, document_id, analysis_id, page_id) is True

    new_ids = chunk_ids_of(db_session, analysis_id)
    assert chunk_ids_of(db_session, old_analysis_id) == []
    assert len(new_ids) == 1
    assert embedded_source_ids(db_session) == set(new_ids)
    assert set(new_ids).isdisjoint(old_ids)


def test_cited_previous_chunk_stays_but_loses_embedding(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    analysis_id: UUID,
    page_id: UUID,
) -> None:
    """인용된 이전 청크는 남기고 임베딩만 지워 검색에서 뺀다. 새 청크와 임베딩은 들어간다."""
    old_analysis_id = insert_analysis(db_session, document_id, days=-1)
    old_ids = [insert_old_chunk(db_session, old_analysis_id, page_id, i) for i in (0, 1)]
    cite_chunk(db_session, document_id, old_ids[0])

    assert replace_one(session_factory, document_id, analysis_id, page_id) is True

    new_ids = chunk_ids_of(db_session, analysis_id)
    assert sorted(chunk_ids_of(db_session, old_analysis_id)) == sorted(old_ids)
    assert len(new_ids) == 1
    assert embedded_source_ids(db_session) == set(new_ids)


def test_replace_keeps_health_record_embeddings(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    analysis_id: UUID,
    page_id: UUID,
) -> None:
    """source_type이 HEALTH_RECORD인 임베딩은 source_id가 이전 청크와 같아도 지우지 않는다."""
    old_analysis_id = insert_analysis(db_session, document_id, days=-1)
    old_id = insert_old_chunk(db_session, old_analysis_id, page_id, 0)
    insert_embedding(db_session, old_analysis_id, old_id, "HEALTH_RECORD")

    replace_one(session_factory, document_id, analysis_id, page_id)

    assert db_session.execute(
        text("SELECT source_type FROM embeddings WHERE source_id = :id"), {"id": old_id}
    ).scalars().all() == ["HEALTH_RECORD"]


def test_failure_during_load_keeps_previous_data(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    analysis_id: UUID,
    page_id: UUID,
) -> None:
    """두 번째 벡터가 767차원이라 적재가 실패하면 새 행은 없고 이전 청크·임베딩은 그대로다."""
    old_analysis_id = insert_analysis(db_session, document_id, days=-1)
    old_ids = [insert_old_chunk(db_session, old_analysis_id, page_id, i) for i in (0, 1)]

    with pytest.raises(StatementError):
        SqlAlchemyDocumentChunkRepository(session_factory).replace_chunks(
            document_id,
            analysis_id,
            make_chunks(page_id, 0, 1),
            make_embeddings(make_vector(1.0), make_vector(2.0, DIMENSIONS - 1)),
            CREATED_AT,
        )

    assert chunk_ids_of(db_session, analysis_id) == []
    assert sorted(chunk_ids_of(db_session, old_analysis_id)) == sorted(old_ids)
    assert embedded_source_ids(db_session) == set(old_ids)


def test_mismatched_vector_count_writes_nothing(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    analysis_id: UUID,
    page_id: UUID,
) -> None:
    """청크 수와 벡터 수가 다르면 ValueError를 올리고 아무 행도 쓰지 않는다."""
    with pytest.raises(ValueError):
        SqlAlchemyDocumentChunkRepository(session_factory).replace_chunks(
            document_id,
            analysis_id,
            make_chunks(page_id, 0, 1),
            make_embeddings(make_vector(1.0)),
            CREATED_AT,
        )

    assert chunk_ids_of(db_session, analysis_id) == []
    assert embedded_source_ids(db_session) == set()


def test_not_latest_analysis_changes_nothing(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    analysis_id: UUID,
    page_id: UUID,
) -> None:
    """더 새 분석이 있으면 False를 반환하고 그 분석의 청크·임베딩을 지우지 않는다."""
    newer_analysis_id = insert_analysis(db_session, document_id, days=1)
    newer_id = insert_old_chunk(db_session, newer_analysis_id, page_id, 0)

    assert replace_one(session_factory, document_id, analysis_id, page_id) is False

    assert chunk_ids_of(db_session, analysis_id) == []
    assert chunk_ids_of(db_session, newer_analysis_id) == [newer_id]
    assert embedded_source_ids(db_session) == {newer_id}


def test_deleted_document_changes_nothing(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    analysis_id: UUID,
    page_id: UUID,
) -> None:
    """삭제된 문서는 색인하지 않고 False를 반환한다."""
    db_session.execute(
        text("UPDATE documents SET deleted_at = CURRENT_TIMESTAMP WHERE id = :id"),
        {"id": document_id},
    )

    assert replace_one(session_factory, document_id, analysis_id, page_id) is False

    assert chunk_ids_of(db_session, analysis_id) == []
    assert embedded_source_ids(db_session) == set()


def test_empty_chunks_still_remove_previous_data(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    analysis_id: UUID,
    page_id: UUID,
) -> None:
    """새 분석에 청크가 없어도 이전 청크·임베딩을 지워 검색에 남지 않게 한다."""
    old_analysis_id = insert_analysis(db_session, document_id, days=-1)
    insert_old_chunk(db_session, old_analysis_id, page_id, 0)

    replaced = SqlAlchemyDocumentChunkRepository(session_factory).replace_chunks(
        document_id, analysis_id, [], make_embeddings(), CREATED_AT
    )

    assert replaced is True
    assert chunk_ids_of(db_session, old_analysis_id) == []
    assert chunk_ids_of(db_session, analysis_id) == []
    assert embedded_source_ids(db_session) == set()
