from openai import OpenAI

from app.core.config import Settings


class EmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.embedding_model
        self.client = OpenAI(
            api_key=(settings.llm_api_key or settings.openai_api_key).get_secret_value()
            if (settings.llm_api_key or settings.openai_api_key) else "",
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.client.api_key:
            raise RuntimeError("임베딩 API 키가 설정되지 않았습니다.")
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
