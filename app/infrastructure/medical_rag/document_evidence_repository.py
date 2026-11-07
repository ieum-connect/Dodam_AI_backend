"""pgvector 기반 의료 문서 청크 검색 저장소."""

from collections.abc import Sequence
from uuid import UUID, uuid4

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session, sessionmaker

from app.contracts.document_evidence import DocumentEvidenceItem


class DocumentEvidenceRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def search(
        self,
        *,
        patient_id: UUID,
        allowed_document_ids: Sequence[UUID],
        query_embedding: Sequence[float],
        model_name: str,
        dimensions: int,
        limit: int,
    ) -> list[DocumentEvidenceItem]:
        if not allowed_document_ids:
            return []
        vector_literal = "[" + ",".join(str(value) for value in query_embedding) + "]"
        statement = text(
            """
            SELECT e.source_id AS chunk_id,
                   d.id AS document_id,
                   dp.page_no AS page_no,
                   dc.content AS source_text,
                   d.status AS document_status,
                   1 - (e.embedding <=> CAST(:query_embedding AS vector)) AS similarity
            FROM embeddings e
            JOIN document_chunks dc
              ON dc.id = e.source_id
             AND e.source_type = 'DOCUMENT_CHUNK'
            JOIN document_analyses da ON da.id = dc.analysis_id
            JOIN documents d ON d.id = da.document_id
            LEFT JOIN document_pages dp ON dp.id = dc.page_id
            LEFT JOIN visits v ON v.id = e.visit_id
            WHERE e.patient_id = :patient_id
              AND d.id IN :document_ids
              AND d.deleted_at IS NULL
              AND d.status <> 'DELETED'
              AND (v.id IS NULL OR v.deleted_at IS NULL)
              AND e.model_name = :model_name
              AND e.dimensions = :dimensions
            ORDER BY e.embedding <=> CAST(:query_embedding AS vector)
            LIMIT :limit
            """
        ).bindparams(bindparam("document_ids", expanding=True))
        with self._session_factory() as session:
            rows = session.execute(
                statement,
                {
                    "query_embedding": vector_literal,
                    "patient_id": patient_id,
                    "document_ids": list(allowed_document_ids),
                    "model_name": model_name,
                    "dimensions": dimensions,
                    "limit": limit,
                },
            ).mappings().all()
        return [
            DocumentEvidenceItem(
                citation_id=uuid4(),
                document_id=row["document_id"],
                chunk_id=row["chunk_id"],
                page_no=row["page_no"],
                source_text=row["source_text"],
                similarity=float(row["similarity"]),
                document_status=row["document_status"],
            )
            for row in rows
        ]
