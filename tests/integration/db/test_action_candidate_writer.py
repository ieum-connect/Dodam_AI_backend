from sqlalchemy import text

from app.infrastructure.db.repositories.action_candidate_writer import save_candidates
from tests.unit.domain.test_action_candidates import extraction


def test_retry_preserves_guardian_decision_and_does_not_create_schedules(db_session, document_id):
    page = extraction()
    for _ in range(2):
        save_candidates(db_session, document_id, {1: page})
    rows = db_session.execute(
        text("SELECT id,approval_status,due_at FROM action_items WHERE document_id=:id"),
        {"id": document_id},
    ).all()
    assert len(rows) == 1
    assert rows[0].approval_status == "PENDING"
    assert rows[0].due_at is not None
    db_session.execute(
        text("UPDATE action_items SET approval_status='REJECTED' WHERE id=:id"), {"id": rows[0].id}
    )
    save_candidates(db_session, document_id, {1: page})
    assert (
        db_session.execute(
            text("SELECT approval_status FROM action_items WHERE id=:id"), {"id": rows[0].id}
        ).scalar_one()
        == "REJECTED"
    )
    assert (
        db_session.execute(
            text("SELECT count(*) FROM schedules WHERE action_item_id=:id"), {"id": rows[0].id}
        ).scalar_one()
        == 0
    )


def test_successful_reanalysis_cancels_stale_pending_candidates(db_session, document_id):
    save_candidates(db_session, document_id, {1: extraction()})
    save_candidates(db_session, document_id, {})
    assert (
        db_session.execute(
            text("SELECT approval_status FROM action_items WHERE document_id=:id"),
            {"id": document_id},
        ).scalar_one()
        == "CANCELED"
    )


def test_reappearing_candidate_becomes_pending_with_the_same_id(db_session, document_id):
    page = extraction()
    find_candidate = text(
        "SELECT id, approval_status, cancel_reason FROM action_items WHERE document_id=:id"
    )
    params = {"id": document_id}
    save_candidates(db_session, document_id, {1: page})
    original = db_session.execute(find_candidate, params).one()
    save_candidates(db_session, document_id, {})
    assert db_session.execute(find_candidate, params).one() == (
        original.id,
        "CANCELED",
        "SOURCE_REANALYZED",
    )
    save_candidates(db_session, document_id, {2: page})
    assert db_session.execute(find_candidate, params).one() == (original.id, "PENDING", None)
