import hashlib
from dataclasses import asdict

from app.domain.welfare.policy_chunk import PolicyChunk


class WelfareEmbeddingStore:
    """외부 벡터 DB 연결 전까지 사용하는 content-hash 기반 저장소."""

    def __init__(self) -> None:
        self._items: dict[str, dict] = {}

    def upsert(self, chunks: list[PolicyChunk]) -> list[dict]:
        changed = []
        for chunk in chunks:
            content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            key = f"{chunk.program_id}:{chunk.chunk_type}:{chunk.source_field}"
            if self._items.get(key, {}).get("content_hash") == content_hash:
                continue
            item = {**asdict(chunk), "content_hash": content_hash}
            self._items[key] = item
            changed.append(item)
        return changed

    def all(self) -> list[dict]:
        return list(self._items.values())
