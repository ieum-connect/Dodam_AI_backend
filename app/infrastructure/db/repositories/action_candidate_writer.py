"""Persist candidates inside the analysis result transaction."""

from collections.abc import Mapping
from uuid import UUID

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from app.domain.document.action_candidates import build_candidates
from app.domain.document.page_extraction import PageExtraction


def save_candidates(
    session: Session, document_id: UUID, pages: Mapping[int, PageExtraction]
) -> None:
    candidates = build_candidates(document_id, pages)
    visit_id = session.execute(
        text("SELECT visit_id FROM documents WHERE id=:id"), {"id": document_id}
    ).scalar_one()
    # Keep guardian decisions and their schedules across retries/reanalysis.
    cancel = """UPDATE action_items
        SET approval_status='CANCELED', cancel_reason='SOURCE_REANALYZED'
        WHERE document_id=:document_id AND approval_status='PENDING'"""
    parameters = {"document_id": document_id}
    if candidates:
        statement = text(cancel + " AND id NOT IN :ids").bindparams(
            bindparam("ids", expanding=True)
        )
        session.execute(statement, {**parameters, "ids": [row.id for row in candidates]})
    else:
        session.execute(text(cancel), parameters)
    for candidate in candidates:
        session.execute(
            text("""
            INSERT INTO action_items(id, visit_id, document_id, type, title, description,
                                     due_at, approval_status)
            VALUES (:id,:visit_id,:document_id,:type,:title,:description,:due_at,'PENDING')
            ON CONFLICT (id) DO NOTHING
        """),
            {
                "id": candidate.id,
                "visit_id": visit_id,
                "document_id": document_id,
                "type": candidate.type,
                "title": candidate.title,
                "description": candidate.description,
                "due_at": candidate.due_at,
            },
        )
