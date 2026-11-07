import re

from app.infrastructure.welfare.embedding_store import WelfareEmbeddingStore


def search(store: WelfareEmbeddingStore, query: str, limit: int = 10) -> list[dict]:
    tokens = set(re.findall(r"[가-힣A-Za-z0-9]+", query.lower()))
    scored = []
    for item in store.all():
        text_tokens = set(re.findall(r"[가-힣A-Za-z0-9]+", item["text"].lower()))
        score = len(tokens & text_tokens) / max(len(tokens), 1)
        if score > 0:
            scored.append({"program_id": item["program_id"], "name": item["chunk_type"], "score": score, "evidence": [item["text"]]})
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:limit]
