"""의료 문서와 질문에 동일한 모델·차원을 적용하는 임베딩 제공자."""

from openai import OpenAI

from app.core.config import Settings


class MedicalRagEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.medical_rag_embedding_model
        self.dimensions = settings.medical_rag_embedding_dimensions
        api_key = (
            settings.embedding_api_key.get_secret_value()
            if settings.embedding_api_key
            else ""
        )
        self._client = OpenAI(
            api_key=api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    def embed_query(self, question: str) -> list[float]:
        if not self._client.api_key:
            raise RuntimeError("EMBEDDING_API_KEY가 설정되지 않았습니다.")
        response = self._client.embeddings.create(
            model=self.model,
            input=[question],
            dimensions=self.dimensions,
        )
        vector = response.data[0].embedding
        if len(vector) != self.dimensions:
            raise RuntimeError("질문 임베딩 차원이 의료 RAG 설정과 일치하지 않습니다.")
        return vector
