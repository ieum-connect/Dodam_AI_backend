"""document_chunks 저장소 테스트. 실제 PostgreSQL에 쓰고 테스트가 끝나면 롤백한다."""

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Row, select, text
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session, sessionmaker

from app.application.ports.document_chunk_repository import ChunkText, IndexTarget
from app.application.ports.text_embedder import TextEmbeddings
from app.application.use_cases.index_document_chunks import (
    ChunkIndexOutcome,
    ChunkIndexResult,
    ChunkIndexSkipReason,
    index_document_chunks,
)
from app.domain.document.document_analysis import AnalysisStatus
from app.domain.document.document_chunk import DocumentChunk, PageText
from app.domain.document.document_page import PageOcrStatus
from app.infrastructure.db.models.embedding import EmbeddingModel
from app.infrastructure.db.repositories.document_chunk_repository import (
    SqlAlchemyDocumentChunkRepository,
)
from tests.integration.db.conftest import RecordingEmbedder

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


def test_duplicate_chunk_index_rolls_back_whole_replacement(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    analysis_id: UUID,
    page_id: UUID,
) -> None:
    """DB 유일 키 위반으로 실패하면 새 행이 하나도 남지 않고 이전 데이터와 분석 상태는 그대로다."""
    old_analysis_id = insert_analysis(db_session, document_id, days=-1)
    old_id = insert_old_chunk(db_session, old_analysis_id, page_id, 0)
    status_before = fetch_analysis_status(db_session, analysis_id)
    duplicate = make_chunks(page_id, 0) * 2

    with pytest.raises(IntegrityError):
        SqlAlchemyDocumentChunkRepository(session_factory).replace_chunks(
            document_id,
            analysis_id,
            duplicate,
            make_embeddings(make_vector(1.0), make_vector(2.0)),
            CREATED_AT,
        )

    assert chunk_ids_of(db_session, analysis_id) == []
    assert chunk_ids_of(db_session, old_analysis_id) == [old_id]
    assert embedded_source_ids(db_session, document_id) == {old_id}
    assert fetch_analysis_status(db_session, analysis_id) == status_before


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
    session: Session,
    analysis_id: UUID,
    page_id: UUID,
    chunk_index: int,
    source_type: str = "DOCUMENT_CHUNK",
) -> UUID:
    """청크 하나와 그 청크를 가리키는 임베딩을 raw SQL로 넣고 청크 id를 돌려준다."""
    chunk_id = insert_chunk_only(session, analysis_id, page_id, chunk_index, "이전 합성 청크")
    insert_embedding(session, analysis_id, chunk_id, source_type)
    return chunk_id


def insert_chunk_only(
    session: Session, analysis_id: UUID, page_id: UUID, chunk_index: int, content: str
) -> UUID:
    """임베딩 없이 청크 하나만 raw SQL로 넣고 청크 id를 돌려준다."""
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
            "content": content,
        },
    )
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


def embedded_source_ids(session: Session, document_id: UUID) -> set[UUID]:
    """테스트 문서의 환자에 속한 임베딩의 source_id. 공유 DB의 다른 행은 보지 않는다."""
    return set(
        session.execute(
            text(
                "SELECT source_id FROM embeddings WHERE patient_id ="
                " (SELECT patient_id FROM documents WHERE id = :id)"
            ),
            {"id": document_id},
        ).scalars()
    )


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
    assert embedded_source_ids(db_session, document_id) == set(new_ids)
    assert set(new_ids).isdisjoint(old_ids)


def test_cited_previous_chunk_stays_but_loses_embedding(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    analysis_id: UUID,
    page_id: UUID,
) -> None:
    """이전 청크 하나라도 인용되면 이전 청크는 모두 남고, 이전 임베딩은 모두 지워 검색에서 뺀다.

    새 청크와 임베딩은 들어간다.
    """
    old_analysis_id = insert_analysis(db_session, document_id, days=-1)
    old_ids = [insert_old_chunk(db_session, old_analysis_id, page_id, i) for i in (0, 1)]
    cite_chunk(db_session, document_id, old_ids[0])

    assert replace_one(session_factory, document_id, analysis_id, page_id) is True

    new_ids = chunk_ids_of(db_session, analysis_id)
    assert sorted(chunk_ids_of(db_session, old_analysis_id)) == sorted(old_ids)
    assert len(new_ids) == 1
    assert embedded_source_ids(db_session, document_id) == set(new_ids)


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
    assert embedded_source_ids(db_session, document_id) == set(old_ids)


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
    assert embedded_source_ids(db_session, document_id) == set()


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
    assert embedded_source_ids(db_session, document_id) == {newer_id}


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
    assert embedded_source_ids(db_session, document_id) == set()


def test_status_deleted_document_changes_nothing(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    analysis_id: UUID,
    page_id: UUID,
) -> None:
    """deleted_at이 비어 있어도 status가 DELETED면 색인하지 않고 False를 반환한다."""
    db_session.execute(
        text("UPDATE documents SET status = 'DELETED' WHERE id = :id"), {"id": document_id}
    )

    assert replace_one(session_factory, document_id, analysis_id, page_id) is False

    assert chunk_ids_of(db_session, analysis_id) == []
    assert embedded_source_ids(db_session, document_id) == set()


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
    assert embedded_source_ids(db_session, document_id) == set()


def insert_document(session: Session, like: UUID) -> UUID:
    """like 문서와 같은 환자·방문·업로더로 합성 문서를 하나 더 만든다."""
    document_id = uuid4()
    session.execute(
        text(
            "INSERT INTO documents (id, visit_id, uploader_user_id, patient_id,"
            " storage_key, file_name, mime_type, file_size_bytes)"
            " SELECT :id, visit_id, uploader_user_id, patient_id,"
            " 'documents/' || :id || '/original.pdf', 'original.pdf', 'application/pdf', 1"
            " FROM documents WHERE id = :like"
        ),
        {"id": document_id, "like": like},
    )
    return document_id


def set_analysis_status(session: Session, analysis_id: UUID, status: str) -> None:
    """분석 작업의 상태를 바꾼다."""
    session.execute(
        text("UPDATE document_analyses SET status = :status WHERE id = :id"),
        {"id": analysis_id, "status": status},
    )


def make_analyzed_document(
    session: Session,
    like: UUID,
    analysis_status: str = "SUCCEEDED",
    page_status: PageOcrStatus = PageOcrStatus.SUCCEEDED,
) -> tuple[UUID, UUID, UUID]:
    """문서, 분석 하나, 페이지 하나를 만들고 (문서, 분석, 페이지) id를 돌려준다."""
    document_id = insert_document(session, like)
    analysis_id = insert_analysis(session, document_id, days=0)
    set_analysis_status(session, analysis_id, analysis_status)
    text_value = "합성 원문" if page_status == PageOcrStatus.SUCCEEDED else None
    page_id = insert_page(session, document_id, 1, page_status, text_value)
    return document_id, analysis_id, page_id


def test_find_index_target_reads_latest_analysis_and_chunk_counts(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    analysis_id: UUID,
    page_id: UUID,
) -> None:
    """최신 분석의 id·상태와 그 분석의 청크 수, DOCUMENT_CHUNK 임베딩이 없는 청크 수를 읽는다."""
    old_analysis_id = insert_analysis(db_session, document_id, days=-1)
    insert_old_chunk(db_session, old_analysis_id, page_id, 0)
    insert_chunk_only(db_session, old_analysis_id, page_id, 1, "이전 합성 청크")
    repository = SqlAlchemyDocumentChunkRepository(session_factory)

    target = repository.find_index_target(document_id)
    assert target == IndexTarget(analysis_id, AnalysisStatus.PENDING, 0, 0)

    set_analysis_status(db_session, analysis_id, "SUCCEEDED")
    insert_old_chunk(db_session, analysis_id, page_id, 0)
    insert_old_chunk(db_session, analysis_id, page_id, 1, source_type="HEALTH_RECORD")
    insert_chunk_only(db_session, analysis_id, page_id, 2, "합성 청크 2")
    target = repository.find_index_target(document_id)
    assert target == IndexTarget(analysis_id, AnalysisStatus.SUCCEEDED, 3, 2)


def test_find_index_target_is_none_without_live_document_or_analysis(
    db_session: Session, session_factory: sessionmaker[Session], document_id: UUID
) -> None:
    """없는 문서, 분석 행이 없는 문서, 삭제된 문서는 None이다."""
    no_analysis = insert_document(db_session, document_id)
    soft_deleted, _, _ = make_analyzed_document(db_session, document_id)
    status_deleted, _, _ = make_analyzed_document(db_session, document_id)
    db_session.execute(
        text("UPDATE documents SET deleted_at = CURRENT_TIMESTAMP WHERE id = :id"),
        {"id": soft_deleted},
    )
    db_session.execute(
        text("UPDATE documents SET status = 'DELETED' WHERE id = :id"), {"id": status_deleted}
    )
    repository = SqlAlchemyDocumentChunkRepository(session_factory)

    for missing in (uuid4(), no_analysis, soft_deleted, status_deleted):
        assert repository.find_index_target(missing) is None


def test_find_unindexed_document_ids_picks_only_live_succeeded_without_embeddings(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    embedder: RecordingEmbedder,
) -> None:
    """모든 청크에 임베딩 있는 문서, 삭제된 문서, 최신 분석이 SUCCEEDED가 아닌 문서,
    원문 없는 문서는 뺀다. 청크에 HEALTH_RECORD 임베딩만 있는 문서는 목록에 뜨고,
    재색인하면 채워져 빠진다."""
    unindexed, _, _ = make_analyzed_document(db_session, document_id)

    only_old_indexed, _, old_page = make_analyzed_document(db_session, document_id)
    older = insert_analysis(db_session, only_old_indexed, days=-1)
    set_analysis_status(db_session, older, "SUCCEEDED")
    insert_old_chunk(db_session, older, old_page, 0)

    health_only, health_analysis, health_page = make_analyzed_document(db_session, document_id)
    insert_old_chunk(db_session, health_analysis, health_page, 0, source_type="HEALTH_RECORD")

    indexed, indexed_analysis, indexed_page = make_analyzed_document(db_session, document_id)
    assert replace_one(session_factory, indexed, indexed_analysis, indexed_page)

    soft_deleted, _, _ = make_analyzed_document(db_session, document_id)
    db_session.execute(
        text("UPDATE documents SET deleted_at = CURRENT_TIMESTAMP WHERE id = :id"),
        {"id": soft_deleted},
    )
    status_deleted, _, _ = make_analyzed_document(db_session, document_id)
    db_session.execute(
        text("UPDATE documents SET status = 'DELETED' WHERE id = :id"), {"id": status_deleted}
    )

    latest_running, _, _ = make_analyzed_document(db_session, document_id, "RUNNING")
    older_succeeded = insert_analysis(db_session, latest_running, days=-1)
    set_analysis_status(db_session, older_succeeded, "SUCCEEDED")

    no_text, _, _ = make_analyzed_document(
        db_session, document_id, page_status=PageOcrStatus.SKIPPED
    )

    mine = {
        unindexed,
        only_old_indexed,
        health_only,
        indexed,
        soft_deleted,
        status_deleted,
        latest_running,
        no_text,
    }
    repository = SqlAlchemyDocumentChunkRepository(session_factory)
    found = repository.find_unindexed_document_ids()

    # 공유 DB라 다른 문서가 섞일 수 있어 이 테스트가 만든 문서만 비교한다.
    assert set(found) & mine == {unindexed, only_old_indexed, health_only}
    assert found == sorted(found)

    # 예전 기준은 청크가 있다는 이유로 ALREADY_INDEXED로 건너뛰어 영구히 복구되지 않았다.
    result = index_document_chunks(health_only, chunk_repository=repository, embedder=embedder)
    assert (result.outcome, result.chunk_count) == (ChunkIndexOutcome.INDEXED, 1)
    assert health_only not in repository.find_unindexed_document_ids()


@pytest.fixture
def partly_embedded(
    db_session: Session, document_id: UUID, analysis_id: UUID, page_id: UUID
) -> tuple[UUID, list[UUID]]:
    """최신 SUCCEEDED 분석에 청크 3개를 두고 첫 청크에만 임베딩을 넣는다. (분석 id, 청크 id들)."""
    set_analysis_status(db_session, analysis_id, "SUCCEEDED")
    first = insert_old_chunk(db_session, analysis_id, page_id, 0)
    rest = [
        insert_chunk_only(db_session, analysis_id, page_id, index, f"빠진 합성 청크 {index}")
        for index in (1, 2)
    ]
    return analysis_id, [first, *rest]


def test_partly_embedded_document_is_listed_and_only_missing_chunks_are_filled(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    partly_embedded: tuple[UUID, list[UUID]],
    embedder: RecordingEmbedder,
) -> None:
    """일부 청크만 임베딩이 있으면 목록에 뜬다. 재색인은 빠진 청크만 임베딩하고 청크는 그대로 둔다.
    다시 실행하면 ALREADY_INDEXED이고 목록에서 빠진다."""
    analysis_id, chunk_ids = partly_embedded
    repository = SqlAlchemyDocumentChunkRepository(session_factory)
    assert document_id in repository.find_unindexed_document_ids()

    result = index_document_chunks(document_id, chunk_repository=repository, embedder=embedder)

    assert result == ChunkIndexResult(
        ChunkIndexOutcome.INDEXED, analysis_id=analysis_id, chunk_count=2
    )
    assert embedder.calls == [["빠진 합성 청크 1", "빠진 합성 청크 2"]]
    assert chunk_ids_of(db_session, analysis_id) == chunk_ids
    assert embedded_source_ids(db_session, document_id) == set(chunk_ids)
    assert document_id not in repository.find_unindexed_document_ids()

    again = index_document_chunks(document_id, chunk_repository=repository, embedder=embedder)

    assert again.skip_reason == ChunkIndexSkipReason.ALREADY_INDEXED
    assert len(embedder.calls) == 1


def test_fill_works_with_cited_chunks(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    page_id: UUID,
    partly_embedded: tuple[UUID, list[UUID]],
    embedder: RecordingEmbedder,
) -> None:
    """빠진 청크나 이전 분석 청크가 인용되어 있어도 채운다. 어떤 청크도 지우지 않는다."""
    analysis_id, chunk_ids = partly_embedded
    old_analysis_id = insert_analysis(db_session, document_id, days=-1)
    old_id = insert_old_chunk(db_session, old_analysis_id, page_id, 0)
    cite_chunk(db_session, document_id, old_id)
    # ai_explanations는 문서마다 version 1이 하나뿐이라 두 번째 인용은 다른 합성 문서의 설명에 단다.
    cite_chunk(db_session, insert_document(db_session, document_id), chunk_ids[1])
    repository = SqlAlchemyDocumentChunkRepository(session_factory)

    result = index_document_chunks(document_id, chunk_repository=repository, embedder=embedder)

    assert (result.outcome, result.chunk_count) == (ChunkIndexOutcome.INDEXED, 2)
    assert chunk_ids_of(db_session, analysis_id) == chunk_ids
    assert chunk_ids_of(db_session, old_analysis_id) == [old_id]
    assert embedded_source_ids(db_session, document_id) == {*chunk_ids, old_id}


@pytest.mark.parametrize(
    "change_sql",
    [
        "UPDATE documents SET deleted_at = CURRENT_TIMESTAMP WHERE id = :id",
        "UPDATE documents SET status = 'DELETED' WHERE id = :id",
        "INSERT INTO document_analyses (id, document_id, parser_version, created_at)"
        " VALUES (gen_random_uuid(), :id, 'v1', CURRENT_TIMESTAMP + INTERVAL '1 day')",
    ],
    ids=["deleted_at", "status_deleted", "newer_analysis"],
)
def test_add_missing_embeddings_writes_nothing_for_deleted_or_superseded(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    partly_embedded: tuple[UUID, list[UUID]],
    embedder: RecordingEmbedder,
    change_sql: str,
) -> None:
    """문서가 삭제됐거나 더 새 분석이 생겼으면 False를 반환하고 임베딩을 넣지 않는다."""
    analysis_id, chunk_ids = partly_embedded
    repository = SqlAlchemyDocumentChunkRepository(session_factory)
    missing = repository.find_chunks_without_embedding(analysis_id)
    db_session.execute(text(change_sql), {"id": document_id})

    added = repository.add_missing_embeddings(
        document_id,
        analysis_id,
        missing,
        embedder.embed([chunk.content for chunk in missing]),
        CREATED_AT,
    )

    assert added is False
    assert embedded_source_ids(db_session, document_id) == {chunk_ids[0]}


def test_add_missing_embeddings_skips_already_filled_and_checks_vector_count(
    db_session: Session,
    session_factory: sessionmaker[Session],
    document_id: UUID,
    partly_embedded: tuple[UUID, list[UUID]],
) -> None:
    """이미 임베딩이 있는 청크는 다시 넣지 않는다.
    벡터 수가 다르면 ValueError이고 아무것도 쓰지 않는다."""
    analysis_id, chunk_ids = partly_embedded
    repository = SqlAlchemyDocumentChunkRepository(session_factory)
    missing = repository.find_chunks_without_embedding(analysis_id)
    assert [chunk.chunk_id for chunk in missing] == chunk_ids[1:]

    with pytest.raises(ValueError):
        repository.add_missing_embeddings(
            document_id, analysis_id, missing, make_embeddings(make_vector(1.0)), CREATED_AT
        )
    assert embedded_source_ids(db_session, document_id) == {chunk_ids[0]}

    first = ChunkText(chunk_ids[0], "이전 합성 청크")
    added = repository.add_missing_embeddings(
        document_id,
        analysis_id,
        [first, *missing],
        make_embeddings(make_vector(1.0), make_vector(2.0), make_vector(3.0)),
        CREATED_AT,
    )

    assert added is True
    counts = db_session.execute(
        text("SELECT source_id, count(*) FROM embeddings WHERE source_id = ANY(:ids) GROUP BY 1"),
        {"ids": chunk_ids},
    ).all()
    assert dict(counts) == dict.fromkeys(chunk_ids, 1)
