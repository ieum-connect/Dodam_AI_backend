"""문서 색인과 같은 임베더로 의료 채팅 질문을 임베딩한다."""

from app.core.config import Settings
from app.infrastructure.providers.embedding import create_text_embedder


class MedicalRagEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self._embedder = create_text_embedder(settings)

    def embed_query(self, question: str) -> list[float]:
        return self._embed([question])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        result = self._embedder.embed(texts)
        vectors = result.vectors
        if len(vectors) != len(texts) or any(
            len(vector) != self.dimensions for vector in vectors
        ):
            raise RuntimeError("임베딩 결과가 의료 RAG 설정과 일치하지 않습니다.")
        return vectors

    def close(self) -> None:
        self._embedder.close()
