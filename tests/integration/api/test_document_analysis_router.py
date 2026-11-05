import inspect
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.internal.v1 import document_analysis_router as worker
from app.application.use_cases.analyze_document import analyze_document
from app.core.config import get_settings
from app.main import app
from tests.integration.api.test_error_responses import _settings


def test_internal_auth_and_id_validation_then_background_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr("app.api.internal.v1.document_analysis_router.run_analysis", calls.append)
    app.dependency_overrides[get_settings] = _settings
    document_id = uuid4()
    path = f"/internal/documents/{document_id}/analyze"
    body = {
        "documentId": str(document_id),
        "objectKey": "documents/test.png",
        "mimeType": "image/png",
        "requestId": "test",
    }
    try:
        client = TestClient(app)
        assert client.post(path, json=body).status_code == 401
        headers = {"X-Internal-Token": "test-token"}
        assert (
            client.post(
                path, json={**body, "documentId": str(uuid4())}, headers=headers
            ).status_code
            == 400
        )
        assert calls == []
        assert client.post(path, json=body, headers=headers).status_code == 202
        assert calls == [document_id]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("fails", [False, True])
def test_worker_matches_current_pipeline_and_closes_both_clients(monkeypatch, fails):
    extractor, embedder = Mock(), Mock()
    calls = []
    monkeypatch.setattr(worker, "get_settings", _settings)
    monkeypatch.setattr(worker, "get_session_factory", lambda: object())
    monkeypatch.setattr(worker, "create_page_extractor", lambda settings: extractor)
    monkeypatch.setattr(worker, "create_text_embedder", lambda settings: embedder)

    def run(*args, **kwargs):
        # Catch missing required arguments when another repository changes the pipeline.
        inspect.signature(analyze_document).bind(*args, **kwargs)
        calls.append(kwargs)
        if fails:
            raise RuntimeError("synthetic analysis failure")

    monkeypatch.setattr(worker, "analyze_document", run)
    worker.run_analysis(uuid4())
    assert len(calls) == 1
    assert calls[0]["embedder"] is embedder
    extractor.close.assert_called_once()
    embedder.close.assert_called_once()
