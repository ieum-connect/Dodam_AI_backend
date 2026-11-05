"""Local E2E contract test only. Model answers are fixed fixtures, NOT actual AI.

Run with python -m tests.manual_contract_server. Never use as a deployment entry point.
"""

import json
import os

import httpx
import uvicorn
from pydantic import SecretStr

from app.api.internal.v1 import document_analysis_router
from app.application.ports.text_embedder import TextEmbeddings
from app.infrastructure.providers.llm.openai_page_extractor import create_page_extractor
from app.main import app

TEXT = (
    "연동 테스트용 가상 진료 안내문\n"
    "2026년 12월 3일 오전 10시 발달 재검진\n"
    "비타민 D 하루 1회 식후 복용\n"
)


def fake_model_response(request):
    items = [
        {
            "item_type": "SCHEDULE",
            "normalized_name": "발달 재검진",
            "raw_value": "2026년 12월 3일 오전 10시",
            "normalized_value": "2026-12-03T10:00",
            "unit": None,
            "reference_range": None,
            "confidence": 0.99,
            "source_text": "2026년 12월 3일 오전 10시 발달 재검진",
            "source_box": None,
        },
        {
            "item_type": "MEDICATION",
            "normalized_name": "비타민 D",
            "raw_value": "하루 1회 식후 복용",
            "normalized_value": None,
            "unit": None,
            "reference_range": None,
            "confidence": 0.99,
            "source_text": "비타민 D 하루 1회 식후 복용",
            "source_box": None,
        },
    ]
    page = {"document_type": "DISCHARGE_GUIDE", "raw_text": TEXT, "items": items}
    return httpx.Response(
        200,
        json={
            "status": "completed",
            "model": "contract-fixture-not-ai",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(page)}]}
            ],
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    )


def extractor(settings):
    return create_page_extractor(
        settings.model_copy(update={"openai_api_key": SecretStr("contract-test-placeholder")}),
        transport=httpx.MockTransport(fake_model_response),
    )


if __name__ == "__main__":

    class ContractEmbedder:
        def embed(self, texts):
            return TextEmbeddings(
                [[1.0] + [0.0] * 767 for _ in texts], "contract-fixture-not-ai", 768
            )

        def close(self):
            pass

    document_analysis_router.create_page_extractor = extractor
    document_analysis_router.create_text_embedder = lambda settings: ContractEmbedder()
    print("CONTRACT TEST ONLY: FIXED MODEL RESPONSES, NO ACTUAL AI CALLS")
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8006")))
