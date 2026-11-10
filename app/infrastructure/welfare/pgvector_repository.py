import hashlib
import uuid
from datetime import UTC, datetime

from sqlalchemy import select

from app.infrastructure.db.models.welfare_embedding import WelfareEmbedding
from app.infrastructure.db.session import get_session_factory
from app.infrastructure.welfare.embedding_provider import EmbeddingProvider


class PgVectorEmbeddingRepository:
    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider

    def upsert(self, chunks: list) -> int:
        if not chunks:
            return 0
        now = datetime.now(UTC)
        vectors = self.provider.embed([chunk.text for chunk in chunks])
        changed = 0
        with get_session_factory()() as session:
            for chunk, vector in zip(chunks, vectors, strict=True):
                program_id = uuid.UUID(chunk.program_id)
                digest = hashlib.sha256(chunk.text.encode()).hexdigest()
                row = session.scalar(select(WelfareEmbedding).where(
                    WelfareEmbedding.program_id == program_id,
                    WelfareEmbedding.chunk_type == chunk.chunk_type,
                    WelfareEmbedding.source_field == chunk.source_field,
                ))
                if row and row.content_hash == digest:
                    continue
                if row:
                    row.chunk_text, row.content_hash, row.embedding, row.updated_at = chunk.text, digest, vector, now
                else:
                    session.add(WelfareEmbedding(id=uuid.uuid4(), program_id=program_id, chunk_type=chunk.chunk_type, source_field=chunk.source_field, chunk_text=chunk.text, content_hash=digest, embedding=vector, embedding_model=self.provider.model, created_at=now, updated_at=now))
                changed += 1
            session.commit()
        return changed
