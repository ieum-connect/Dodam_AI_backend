from app.application.welfare.explain_recommendation import explain
from app.application.welfare.search_policy import search
from app.contracts.welfare import WelfareExplanationRequest, WelfareProgramInput
from app.domain.welfare.policy_chunk import chunk_program
from app.domain.welfare.recommendation_context import WelfareRecommendation
from app.infrastructure.welfare.embedding_store import WelfareEmbeddingStore


def program():
    return {"id": "p1", "name": "아이돌봄", "target": "만 7세 이하 아동", "benefit": "돌봄 지원"}


def test_chunk_program_keeps_source_fields():
    chunks = chunk_program(program())
    assert [(c.chunk_type, c.source_field) for c in chunks] == [("ELIGIBILITY", "target"), ("BENEFIT", "benefit")]


def test_embedding_store_skips_unchanged_content():
    store = WelfareEmbeddingStore()
    chunks = chunk_program(program())
    assert len(store.upsert(chunks)) == 2
    assert store.upsert(chunks) == []


def test_search_returns_matching_evidence():
    store = WelfareEmbeddingStore()
    store.upsert(chunk_program(program()))
    result = search(store, "아동 돌봄")
    assert result and result[0]["program_id"] == "p1"


def test_explanation_preserves_grounded_conditions():
    response = explain(WelfareExplanationRequest(
        program=WelfareProgramInput(program_id="p1", name="아이돌봄"),
        recommendation=WelfareRecommendation(
            program_id="p1", name="아이돌봄", score=90, status="ELIGIBLE",
            matched_conditions=["연령 일치"], missing_conditions=["소득 확인"],
        ), evidence=["만 7세 이하 아동"],
    ))
    assert response.matched_reasons == ["연령 일치"]
    assert response.additional_checks == ["소득 확인"]
    assert response.source_evidence == ["만 7세 이하 아동"]
