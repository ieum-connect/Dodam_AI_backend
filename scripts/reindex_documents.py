"""저장된 페이지 원문으로 청크·임베딩을 다시 만드는 수동 복구 스크립트.

임베딩이 실패해 최신 분석에 청크·임베딩이 없는 문서를 VLM 호출 없이 다시 색인한다.
문서와 분석의 상태는 바꾸지 않는다. 자동 테스트에 포함되지 않는다.
색인되는 문서마다 OpenAI 임베딩 비용이 들고 DB의 document_chunks, embeddings가 바뀐다.
--missing은 누락 목록 전체를 개수 제한 없이 처리하므로, 목록이 길면 비용도 그만큼 든다.
--missing은 DOCUMENT_CHUNK 임베딩이 없는 문서를, 문서 id 지정은 최신 분석에 청크가 있는지를
기준으로 건너뛴다.
청크와 임베딩은 한 트랜잭션으로 함께 저장되므로 보통 두 기준의 결과가 같다.
EMBEDDING_API_KEY가 없으면 대상을 조회하기 전에 종료한다.
저장소 루트에서 실행한다.

    python -m scripts.reindex_documents <documentId>
    python -m scripts.reindex_documents --missing
"""

import argparse
from collections import Counter
from uuid import UUID

from app.application.use_cases.index_document_chunks import (
    ChunkIndexOutcome,
    ChunkIndexResult,
    index_document_chunks,
)
from app.core.config import get_settings
from app.infrastructure.db.repositories.document_chunk_repository import (
    SqlAlchemyDocumentChunkRepository,
)
from app.infrastructure.db.session import get_session_factory
from app.infrastructure.providers.embedding import create_text_embedder


def parse_args() -> argparse.Namespace:
    """문서 id 하나 또는 --missing 중 하나를 받는다."""
    parser = argparse.ArgumentParser(description="문서 청크·임베딩 재색인")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("document_id", nargs="?", type=UUID, help="재색인할 문서 id")
    group.add_argument("--missing", action="store_true", help="임베딩이 없는 문서를 모두 처리")
    return parser.parse_args()


def describe(result: ChunkIndexResult) -> str:
    """결과를 건너뜀 이유나 오류 코드가 붙은 한 단어로 바꾼다."""
    detail = result.skip_reason or result.error_code
    return f"{result.outcome}({detail})" if detail else str(result.outcome)


def main() -> None:
    """대상 문서를 하나씩 재색인하고, 실패해도 다음 문서로 넘어간 뒤 건수를 요약한다."""
    args = parse_args()
    settings = get_settings()
    key = settings.embedding_api_key
    if key is None or not key.get_secret_value():
        # 키가 없으면 모든 문서가 MISSING_API_KEY로 실패하므로 조회 전에 끝낸다.
        # 키 값은 출력하지 않는다.
        print("EMBEDDING_API_KEY가 설정되지 않아 재색인할 수 없습니다.")
        raise SystemExit(1)
    repository = SqlAlchemyDocumentChunkRepository(get_session_factory())
    embedder = create_text_embedder(settings)
    summary: Counter[str] = Counter()
    try:
        document_ids = (
            repository.find_unindexed_document_ids() if args.missing else [args.document_id]
        )
        print(f"대상 문서: {len(document_ids)}개")
        for document_id in document_ids:
            try:
                result = index_document_chunks(
                    document_id, chunk_repository=repository, embedder=embedder
                )
                label = describe(result)
                chunks = result.chunk_count
            except Exception as error:
                # 유스케이스는 예외를 올리지 않지만, 한 문서 때문에 전체가 멈추지 않게 한다.
                # 메시지에는 원문이 있을 수 있어 클래스 이름만 쓴다.
                label = f"{ChunkIndexOutcome.FAILED}({type(error).__name__})"
                chunks = 0
            summary[label] += 1
            # 원문, 키, 경로는 출력하지 않는다.
            print(f"{document_id}: {label}, chunks={chunks}")
    finally:
        embedder.close()

    failed = sum(n for label, n in summary.items() if label.startswith(ChunkIndexOutcome.FAILED))
    for outcome in ChunkIndexOutcome:
        count = sum(n for label, n in summary.items() if label.startswith(outcome))
        details = ", ".join(
            f"{label} {n}" for label, n in sorted(summary.items()) if label.startswith(outcome)
        )
        print(f"{outcome}: {count}" + (f" ({details})" if details else ""))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
