"""document_pages에서 원문을 읽고 document_chunks, embeddings에 쓰는 SQLAlchemy 구현."""

import logging
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Select, delete, exists, func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.application.ports.document_chunk_repository import IndexTarget
from app.application.ports.text_embedder import TextEmbeddings
from app.domain.document.document_analysis import AnalysisStatus
from app.domain.document.document_chunk import DocumentChunk, PageText
from app.domain.document.document_page import PageOcrStatus
from app.infrastructure.db.models.document import DOCUMENT_NOT_DELETED, DocumentModel
from app.infrastructure.db.models.document_analysis import DocumentAnalysisModel
from app.infrastructure.db.models.document_chunk import DocumentChunkModel
from app.infrastructure.db.models.document_page import DocumentPageModel
from app.infrastructure.db.models.embedding import SOURCE_TYPE_DOCUMENT_CHUNK, EmbeddingModel

logger = logging.getLogger(__name__)


class SqlAlchemyDocumentChunkRepository:
    """메서드마다 session을 열고 끝날 때 commit한다. 메서드 하나가 트랜잭션 하나다."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def find_page_texts(self, document_id: UUID) -> list[PageText]:
        """문서의 페이지 원문을 page_no 순서로 읽는다."""
        with self._session_factory() as session:
            rows = session.execute(
                select(
                    DocumentPageModel.id,
                    DocumentPageModel.page_no,
                    DocumentPageModel.ocr_status,
                    DocumentPageModel.ocr_text,
                )
                .where(DocumentPageModel.document_id == document_id)
                .order_by(DocumentPageModel.page_no)
            ).all()
        return [
            PageText(
                page_id=row.id,
                page_no=row.page_no,
                ocr_status=PageOcrStatus(row.ocr_status),
                ocr_text=row.ocr_text,
            )
            for row in rows
        ]

    def find_index_target(self, document_id: UUID) -> IndexTarget | None:
        """삭제되지 않은 문서의 최신 분석 id·상태와 그 분석의 청크 유무를 읽는다."""
        with self._session_factory() as session:
            document = session.execute(
                select(DocumentModel.id).where(
                    DocumentModel.id == document_id, *DOCUMENT_NOT_DELETED
                )
            ).scalar_one_or_none()
            if document is None:
                return None
            latest = session.execute(
                _latest_analysis_query(document_id).add_columns(DocumentAnalysisModel.status)
            ).one_or_none()
            if latest is None:
                return None
            has_chunks = session.execute(
                select(exists().where(DocumentChunkModel.analysis_id == latest.id))
            ).scalar_one()
        return IndexTarget(
            analysis_id=latest.id, status=AnalysisStatus(latest.status), has_chunks=has_chunks
        )

    def find_unindexed_document_ids(self) -> list[UUID]:
        """최신 분석이 SUCCEEDED이고 원문이 있는데 DOCUMENT_CHUNK 임베딩이 없는 문서 id 목록."""
        # 문서마다 최신 분석 하나(rank 1). 정렬 기준은 _latest_analysis_query와 같다.
        ranked = select(
            DocumentAnalysisModel.document_id,
            DocumentAnalysisModel.id,
            DocumentAnalysisModel.status,
            func.row_number()
            .over(
                partition_by=DocumentAnalysisModel.document_id,
                order_by=(DocumentAnalysisModel.created_at.desc(), DocumentAnalysisModel.id.desc()),
            )
            .label("rank"),
        ).subquery()
        latest = select(ranked).where(ranked.c.rank == 1).subquery()
        has_embedding = (
            select(DocumentChunkModel.id)
            .join(
                EmbeddingModel,
                (EmbeddingModel.source_id == DocumentChunkModel.id)
                & (EmbeddingModel.source_type == SOURCE_TYPE_DOCUMENT_CHUNK),
            )
            .where(DocumentChunkModel.analysis_id == latest.c.id)
            .exists()
        )
        # 원문이 없으면 청크가 0개라 임베딩이 생기지 않는다. 목록에 계속 남지 않게 뺀다.
        has_text = (
            select(DocumentPageModel.id)
            .where(
                DocumentPageModel.document_id == latest.c.document_id,
                DocumentPageModel.ocr_status == PageOcrStatus.SUCCEEDED,
            )
            .exists()
        )
        with self._session_factory() as session:
            return list(
                session.execute(
                    select(latest.c.document_id)
                    .join(DocumentModel, DocumentModel.id == latest.c.document_id)
                    .where(
                        *DOCUMENT_NOT_DELETED,
                        latest.c.status == AnalysisStatus.SUCCEEDED,
                        ~has_embedding,
                        has_text,
                    )
                    .order_by(latest.c.document_id)
                ).scalars()
            )

    def replace_chunks(
        self,
        document_id: UUID,
        analysis_id: UUID,
        chunks: Sequence[DocumentChunk],
        embeddings: TextEmbeddings,
        created_at: datetime,
    ) -> bool:
        """새 청크·임베딩 적재와 이전 analysis 청크·임베딩 삭제를 한 트랜잭션으로 묶는다."""
        if len(chunks) != len(embeddings.vectors):
            raise ValueError("청크 수와 벡터 수가 다릅니다.")
        with self._session_factory.begin() as session:
            # 문서 행을 잠가 같은 문서의 교체 두 개가 섞이지 않게 한다.
            document = session.execute(
                select(DocumentModel.patient_id, DocumentModel.visit_id)
                .where(DocumentModel.id == document_id, DocumentModel.deleted_at.is_(None))
                .with_for_update()
            ).one_or_none()
            if document is None or _latest_analysis_id(session, document_id) != analysis_id:
                # 늦게 끝난 이전 분석이 더 새 분석의 벡터를 지우지 않게 한다.
                return False
            old_chunk_ids = _other_analysis_chunk_ids(session, document_id, analysis_id)
            chunk_ids = [uuid4() for _ in chunks]
            if chunks:
                session.execute(
                    insert(DocumentChunkModel),
                    [
                        _chunk_row(chunk_id, analysis_id, chunk, created_at)
                        for chunk_id, chunk in zip(chunk_ids, chunks, strict=True)
                    ],
                )
                session.execute(
                    insert(EmbeddingModel),
                    [
                        {
                            "id": uuid4(),
                            "source_type": SOURCE_TYPE_DOCUMENT_CHUNK,
                            "source_id": chunk_id,
                            "patient_id": document.patient_id,
                            "visit_id": document.visit_id,
                            "embedding": vector,
                            "model_name": embeddings.model_name,
                            "dimensions": len(vector),
                            "created_at": created_at,
                        }
                        for chunk_id, vector in zip(chunk_ids, embeddings.vectors, strict=True)
                    ],
                )
            if old_chunk_ids:
                _delete_old_chunks(session, document_id, analysis_id, old_chunk_ids)
            return True


def _chunk_row(
    chunk_id: UUID, analysis_id: UUID, chunk: DocumentChunk, created_at: datetime
) -> dict[str, object]:
    """document_chunks INSERT 한 행의 값."""
    return {
        "id": chunk_id,
        "analysis_id": analysis_id,
        "page_id": chunk.page_id,
        "chunk_index": chunk.chunk_index,
        "content": chunk.content,
        "bounding_box": chunk.bounding_box,
        "token_count": chunk.token_count,
        "created_at": created_at,
    }


def _latest_analysis_query(document_id: UUID) -> Select[tuple[UUID]]:
    """문서의 최신 분석 id를 고르는 쿼리. 정렬 기준은 find_target, Spring 목록 쿼리와 같다."""
    return (
        select(DocumentAnalysisModel.id)
        .where(DocumentAnalysisModel.document_id == document_id)
        .order_by(DocumentAnalysisModel.created_at.desc(), DocumentAnalysisModel.id.desc())
        .limit(1)
    )


def _latest_analysis_id(session: Session, document_id: UUID) -> UUID | None:
    """문서의 최신 분석 id."""
    return session.execute(_latest_analysis_query(document_id)).scalar_one_or_none()


def _other_analysis_chunk_ids(session: Session, document_id: UUID, analysis_id: UUID) -> list[UUID]:
    """같은 문서의 다른 analysis 청크 id. 이전 교체 때 인용 때문에 남은 청크도 포함한다."""
    return list(
        session.execute(
            select(DocumentChunkModel.id)
            .join(DocumentAnalysisModel, DocumentAnalysisModel.id == DocumentChunkModel.analysis_id)
            .where(
                DocumentAnalysisModel.document_id == document_id,
                DocumentChunkModel.analysis_id != analysis_id,
            )
        ).scalars()
    )


def _delete_old_chunks(
    session: Session, document_id: UUID, analysis_id: UUID, old_chunk_ids: list[UUID]
) -> None:
    """이전 청크의 임베딩을 지우고 청크 삭제를 시도한다. 인용 FK에 걸리면 청크는 남긴다."""
    session.execute(
        delete(EmbeddingModel).where(
            EmbeddingModel.source_type == SOURCE_TYPE_DOCUMENT_CHUNK,
            EmbeddingModel.source_id.in_(old_chunk_ids),
        )
    )
    try:
        # 청크 삭제만 되돌리고 임베딩 삭제와 새 적재는 유지해야 검색에서 이전 청크가 빠진다.
        with session.begin_nested():
            session.execute(delete(DocumentChunkModel).where(DocumentChunkModel.id.in_(old_chunk_ids)))
    except IntegrityError as error:
        logger.warning(
            "인용된 이전 청크를 남겨 둡니다: document_id=%s, analysis_id=%s, error=%s",
            document_id,
            analysis_id,
            type(error).__name__,
        )
