"""텍스트 임베딩 provider. 문서 적재와 질의 검색은 create_text_embedder로 임베더를 만든다."""

from app.infrastructure.providers.embedding.openai_text_embedder import create_text_embedder

__all__ = ["create_text_embedder"]
