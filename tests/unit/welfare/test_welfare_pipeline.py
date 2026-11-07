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


def detailed_request(score=82, llm_reply=None):
    from app.domain.welfare.recommendation_context import WelfareChildContext, WelfareCondition
    return WelfareExplanationRequest(
        program=WelfareProgramInput(
            program_id="p1", name="서울형 아이돌봄비 지원", agency_name="서울특별시 강동구",
            target="서울특별시에 거주하는 만 0세~3세 아동을 양육하는 가정 중 기준중위소득 150% 이하 가구",
            benefit="월 30만원의 아이돌봄비 지원",
        ),
        recommendation=WelfareRecommendation(
            program_id="p1", name="서울형 아이돌봄비 지원", score=score, status="POSSIBLE",
            region_scope="DISTRICT", region_label="서울특별시 강동구",
            conditions=[
                WelfareCondition(key="REGION", label="거주지역", status="MET", weight=30,
                                 message="서울특별시 강동구 거주 조건과 일치해요"),
                WelfareCondition(key="AGE", label="대상 연령", status="MET", weight=25,
                                 message="지원대상 연령(0개월 ~ 만 3세)에 해당해요"),
                WelfareCondition(key="INCOME", label="소득 기준", status="UNKNOWN", weight=18,
                                 message="소득구간 확인이 필요해요"),
            ],
        ),
        child=WelfareChildContext(age_label="생후 30개월", region_label="서울특별시 강동구 길동",
                                  income_band_label="기준중위소득 150% 이하", dual_income=True, profile_saved=True),
        evidence=["만 0세~3세 아동"],
    )


class FakeLlm:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate_text(self, system_prompt, user_prompt):
        self.prompts.append(user_prompt)
        return self.reply


def test_rule_based_summary_quotes_engine_score():
    response = explain(detailed_request())
    assert "82%" in response.summary
    assert response.ai_generated is False
    assert response.score == 82
    assert response.additional_checks == ["소득구간 확인이 필요해요"]


def test_prompt_carries_profile_and_condition_evidence():
    llm = FakeLlm("조건 일치율 82%예요. 월 30만원을 지원해요. 소득 확인이 필요해요.")
    explain(detailed_request(), llm)
    prompt = llm.prompts[0]
    assert "조건 일치율: 82%" in prompt
    assert "[충족] 거주지역" in prompt
    assert "[확인필요] 소득 기준" in prompt
    assert "생후 30개월" in prompt
    assert "지원내용 원문" in prompt


def test_generated_summary_with_invented_percentage_is_discarded():
    response = explain(detailed_request(), FakeLlm("일치율 95%로 적합하고 70% 확률로 선정됩니다."))
    assert response.ai_generated is False
    assert "82%" in response.summary


def test_generated_summary_may_quote_percentages_from_source_text():
    reply = "일치율 82%예요. 기준중위소득 150% 이하 가구가 대상이에요. 확인이 필요해요."
    response = explain(detailed_request(), FakeLlm(reply))
    assert response.ai_generated is True
    assert response.summary == reply


def test_llm_failure_falls_back_to_rules():
    class Broken:
        def generate_text(self, system_prompt, user_prompt):
            raise RuntimeError("provider down")

    response = explain(detailed_request(), Broken())
    assert response.ai_generated is False
    assert "82%" in response.summary
