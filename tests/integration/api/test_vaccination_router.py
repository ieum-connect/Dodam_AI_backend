"""예방접종 일정 내부 API의 인증·오류·응답 모양을 확인한다."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import Settings, get_settings
from app.main import app

URL = "/internal/vaccinations/schedule"
BODY = {"birthDate": "2025-05-05", "asOf": "2026-11-08", "completedDoses": ["BCG-1"]}


def _settings() -> Settings:
    return Settings(
        database_url="postgresql+psycopg://localhost/test",
        ai_internal_token=SecretStr("test-token"),
        storage_local_root="./data/object-storage",
        llm_api_key=SecretStr("test-key"),
        llm_base_url="https://example.test/openai/",
        llm_model="test-model",
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_settings] = _settings
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _post(client: TestClient, body: dict, token: str | None = "test-token"):
    headers = {"X-Internal-Token": token} if token else {}
    return client.post(URL, headers=headers, json=body)


@pytest.mark.parametrize("token", [None, "wrong-token"])
def test_missing_or_wrong_token_is_401(client: TestClient, token: str | None) -> None:
    response = _post(client, BODY, token)
    assert response.status_code == 401
    assert response.json()["code"] == "AUTH_001"


def test_valid_request_returns_schedule(client: TestClient) -> None:
    response = _post(client, BODY)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "scheduleVersion", "asOf", "ageMonths", "next", "groups", "counts", "vaccines",
    }
    assert body["scheduleVersion"] == "KDCA 2026-01-02"
    assert body["asOf"] == "2026-11-08"
    assert body["ageMonths"] == 18
    assert set(body["groups"]) == {"now", "soon", "past", "laterCount"}
    assert body["counts"]["done"] == 1
    assert set(body["next"]) == {
        "doseId", "vaccineId", "name", "label", "recFrom", "recTo", "delayFrom", "group",
    }


def test_unknown_dose_id_is_400(client: TestClient) -> None:
    response = _post(client, {**BODY, "completedDoses": ["DTAP-9"]})
    assert response.status_code == 400
    assert response.json()["code"] == "VACCINATION_002"


def test_birth_after_as_of_is_400(client: TestClient) -> None:
    response = _post(client, {**BODY, "birthDate": "2026-11-09"})
    assert response.status_code == 400
    assert response.json()["code"] == "VACCINATION_001"


@pytest.mark.parametrize(
    "body",
    [
        {**BODY, "asOf": "2026/11/08"},
        {**BODY, "asOf": "2026-13-01"},
        {**BODY, "asOf": 1762560000},
        {**BODY, "asOf": "2026-11-08T00:00:00"},
        {**BODY, "completedDoses": "BCG-1"},
        {"asOf": "2026-11-08"},
        {**BODY, "completed_doses": []},
    ],
)
def test_malformed_request_is_400_global(client: TestClient, body: dict) -> None:
    response = _post(client, body)
    assert response.status_code == 400
    assert response.json()["code"] == "GLOBAL_002"
