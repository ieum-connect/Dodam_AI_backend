"""설정 검증 테스트. .env는 읽지 않고 필수값만 합성으로 넣는다."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.infrastructure.db.models.embedding import EMBEDDING_DIMENSIONS


def make_settings(**values: object) -> Settings:
    """.env 없이 필수값과 받은 값으로 설정을 만든다."""
    return Settings(
        _env_file=None,
        database_url="unused",
        ai_internal_token="token",
        storage_local_root="root",
        **values,
    )


def test_embedding_dimensions_accepts_768() -> None:
    """DB 컬럼과 같은 768은 통과한다."""
    assert make_settings(embedding_dimensions=768).embedding_dimensions == 768


@pytest.mark.parametrize("dimensions", [1536, 0])
def test_embedding_dimensions_rejects_other_values(dimensions: int) -> None:
    """768이 아니면 설정 오류이고, 메시지에 vector(768) 고정 이유가 있다."""
    with pytest.raises(ValidationError) as error:
        make_settings(embedding_dimensions=dimensions)

    assert "vector(768)" in str(error.value)


def test_embedding_dimensions_default_matches_db_column() -> None:
    """설정 기본값과 embeddings.embedding 컬럼 차원이 따로 놀지 않게 묶는다."""
    assert Settings.model_fields["embedding_dimensions"].default == EMBEDDING_DIMENSIONS
    assert make_settings().embedding_dimensions == EMBEDDING_DIMENSIONS
