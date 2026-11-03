"""문서 분석 유스케이스 단위 테스트. 저장소, 스토리지, VLM 포트는 가짜로 대신한다."""

import logging
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

import pymupdf
import pytest
from sqlalchemy.exc import IntegrityError

from app.application.ports.document_analysis_repository import AnalysisTarget
from app.application.ports.document_storage import (
    DocumentNotFoundError,
    InvalidObjectKeyError,
    OriginalDocumentReadError,
    RenderedPageWriteError,
)
from app.application.ports.page_extractor import (
    PageExtractionError,
    PageExtractionErrorCode,
    PageExtractionResult,
    VlmUsage,
)
from app.application.ports.text_embedder import (
    TextEmbeddingError,
    TextEmbeddingErrorCode,
    TextEmbeddings,
)
from app.application.use_cases.analyze_document import DocumentAnalysisResult, analyze_document
from app.domain.document.document_analysis import AnalysisFailure, DocumentStatus
from app.domain.document.document_chunk import DocumentChunk, PageText
from app.domain.document.document_page import PageOcrStatus, RenderedPage
from app.domain.document.page_extraction import (
    DocumentType,
    ExtractedItem,
    ExtractedItemType,
    PageExtraction,
)
from app.domain.document.page_split import PageImage, PageSplitError, PageSplitErrorCode
from app.domain.operation.error_code import AnalysisErrorCode, FailedStep

DOCUMENT_ID = UUID("11111111-2222-3333-4444-555555555555")
ANALYSIS_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
PAGE_ID = UUID("99999999-8888-7777-6666-555555555555")
CHUNK_SOURCE_TEXT = "합성 청크 원문"
OBJECT_KEY = f"documents/{DOCUMENT_ID}/original.pdf"
TARGET = AnalysisTarget(
    document_id=DOCUMENT_ID,
    analysis_id=ANALYSIS_ID,
    storage_key=OBJECT_KEY,
    mime_type="application/pdf",
)
ITEM = ExtractedItem(
    item_type=ExtractedItemType.OTHER,
    normalized_name=None,
    raw_value=None,
    normalized_value=None,
    unit=None,
    reference_range=None,
    confidence=None,
    source_text="합성 문장",
    source_box=None,
)
USAGE = VlmUsage(
    model_name="fake-model",
    prompt_version="v1",
    input_tokens=10,
    output_tokens=5,
    reasoning_tokens=2,
)
EXTRACTION = PageExtraction(document_type=DocumentType.UNKNOWN, raw_text="합성 문장", items=[ITEM])
RESULT = PageExtractionResult(extraction=EXTRACTION, usage=USAGE)
BLANK_EXTRACTION = PageExtraction(document_type=DocumentType.UNKNOWN, raw_text="", items=[])
BLANK_RESULT = PageExtractionResult(extraction=BLANK_EXTRACTION, usage=USAGE)
# VLM 오류 코드 → (error_code, failed_step, retryable) 기대값.
EXTRACTION_MAPPING = [
    ("MISSING_API_KEY", "AI_GENERATION_FAILED", "VLM_EXTRACTION", False),
    ("TIMEOUT", "VLM_TIMEOUT", "VLM_EXTRACTION", True),
    ("RATE_LIMITED", "VLM_RATE_LIMIT", "VLM_EXTRACTION", True),
    ("API_ERROR", "AI_GENERATION_FAILED", "VLM_EXTRACTION", True),
    ("REQUEST_REJECTED", "AI_GENERATION_FAILED", "VLM_EXTRACTION", False),
    ("INCOMPLETE", "AI_GENERATION_FAILED", "VLM_EXTRACTION", False),
    ("REFUSED", "AI_GENERATION_FAILED", "VLM_EXTRACTION", False),
    ("SCHEMA_MISMATCH", "STRUCTURE_VALIDATION_FAILED", "STRUCTURE_VALIDATION", True),
]


def make_pdf(page_count: int = 3) -> bytes:
    """빈 페이지 page_count장짜리 PDF를 만든다."""
    document = pymupdf.open()
    for _ in range(page_count):
        document.new_page(width=100, height=200)
    return document.tobytes()


class FakeReader:
    """정해 둔 바이트를 돌려주거나 정해 둔 예외를 던지는 가짜 포트."""

    def __init__(self, data: bytes | None = None, error: Exception | None = None) -> None:
        self.data = make_pdf() if data is None else data
        self.error = error
        self.requested_keys: list[str] = []

    def read_original(self, object_key: str) -> bytes:
        """요청받은 키를 기록한다. 정해 둔 예외가 있으면 던진다."""
        self.requested_keys.append(object_key)
        if self.error is not None:
            raise self.error
        return self.data


class FakeWriter:
    """쓴 키를 기록하는 가짜 포트. 정해 둔 예외가 있으면 던진다."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.written_keys: list[str] = []

    def write_rendered_page(self, object_key: str, png_bytes: bytes) -> None:
        """쓴 키를 기록한다. 정해 둔 예외가 있으면 던진다."""
        if self.error is not None:
            raise self.error
        self.written_keys.append(object_key)


class FakeExtractor:
    """받은 페이지 번호를 기록하는 가짜 포트.

    정해 둔 페이지에서는 정해 둔 예외를 던지거나 글자가 없는 빈 결과를 돌려준다.
    """

    def __init__(
        self,
        errors: Mapping[int, Exception] | None = None,
        blank_page_nos: Collection[int] = (),
    ) -> None:
        self.errors = errors or {}
        self.blank_page_nos = blank_page_nos
        self.requested_page_nos: list[int] = []

    def extract_page(self, page: PageImage) -> PageExtractionResult:
        """호출을 기록한다. 실패하도록 정해 둔 페이지면 그 예외를 던진다."""
        self.requested_page_nos.append(page.page_no)
        if page.page_no in self.errors:
            raise self.errors[page.page_no]
        return BLANK_RESULT if page.page_no in self.blank_page_nos else RESULT


class FakeRepository:
    """호출을 순서대로 기록하는 가짜 저장소. 분석 저장소와 페이지 저장소 포트를 함께 대신한다."""

    def __init__(
        self,
        target: AnalysisTarget | None = TARGET,
        started: bool = True,
        mark_failed_error: Exception | None = None,
    ) -> None:
        self.target = target
        self.started = started
        self.mark_failed_error = mark_failed_error
        self.calls: list[tuple] = []

    def find_target(self, document_id: UUID) -> AnalysisTarget | None:
        """호출을 기록하고 정해 둔 대상을 돌려준다."""
        self.calls.append(("find_target", document_id))
        return self.target

    def mark_started(self, target: AnalysisTarget, started_at: datetime) -> bool:
        """호출을 기록하고 정해 둔 시작 여부를 돌려준다."""
        self.calls.append(("mark_started", target))
        return self.started

    def save_rendered_pages(
        self, document_id: UUID, pages: Sequence[RenderedPage], rendered_at: datetime
    ) -> None:
        """저장한 페이지 번호를 기록한다."""
        self.calls.append(("save_rendered_pages", [page.page_no for page in pages]))

    def save_result(
        self,
        target: AnalysisTarget,
        document_status: DocumentStatus,
        extractions_by_page_no: Mapping[int, PageExtraction],
        model_version: str | None,
        completed_at: datetime,
    ) -> bool:
        """받은 상태, 페이지별 추출 결과, 모델 이름을 기록한다."""
        self.calls.append(
            ("save_result", document_status, dict(extractions_by_page_no), model_version)
        )
        return True

    def mark_failed(
        self, target: AnalysisTarget, failure: AnalysisFailure, completed_at: datetime
    ) -> bool:
        """받은 실패 기록을 기록한다. 정해 둔 예외가 있으면 던진다."""
        if self.mark_failed_error is not None:
            raise self.mark_failed_error
        self.calls.append(("mark_failed", failure))
        return True

    def call_names(self) -> list[str]:
        """기록한 호출의 이름만 순서대로 돌려준다."""
        return [call[0] for call in self.calls]


class FakeChunkRepository:
    """호출을 기록하는 가짜 청크 저장소. 정해 둔 메서드에서는 정해 둔 예외를 던진다."""

    def __init__(
        self,
        calls: list[tuple] | None = None,
        find_error: Exception | None = None,
        save_error: Exception | None = None,
        pages: list[PageText] | None = None,
        replaced: bool = True,
    ) -> None:
        self.calls = [] if calls is None else calls
        self.find_error = find_error
        self.save_error = save_error
        self.pages = (
            [PageText(PAGE_ID, 1, PageOcrStatus.SUCCEEDED, CHUNK_SOURCE_TEXT)]
            if pages is None
            else pages
        )
        self.replaced = replaced

    def find_page_texts(self, document_id: UUID) -> list[PageText]:
        """호출을 기록하고 정해 둔 원문 페이지를 돌려준다."""
        self.calls.append(("find_page_texts", document_id))
        if self.find_error is not None:
            raise self.find_error
        return self.pages

    def replace_chunks(
        self,
        document_id: UUID,
        analysis_id: UUID,
        chunks: Sequence[DocumentChunk],
        embeddings: TextEmbeddings,
        created_at: datetime,
    ) -> bool:
        """받은 문서·분석 id, 청크, 벡터를 기록한다."""
        self.calls.append(
            ("replace_chunks", document_id, analysis_id, list(chunks), embeddings.vectors)
        )
        if self.save_error is not None:
            raise self.save_error
        return self.replaced


class FakeEmbedder:
    """받은 텍스트를 기록하고 입력마다 합성 벡터 하나를 돌려주는 가짜 임베더."""

    def __init__(self, calls: list[tuple] | None = None, error: Exception | None = None) -> None:
        self.calls = [] if calls is None else calls
        self.error = error

    def embed(self, texts: Sequence[str]) -> TextEmbeddings:
        """호출을 기록한다. 정해 둔 예외가 있으면 던진다."""
        self.calls.append(("embed", list(texts)))
        if self.error is not None:
            raise self.error
        return TextEmbeddings(
            vectors=[[float(index)] for index in range(len(texts))],
            model_name="fake-embedding",
            dimensions=1,
        )


def run(
    repository: FakeRepository,
    *,
    reader: FakeReader | None = None,
    writer: FakeWriter | None = None,
    extractor: FakeExtractor | None = None,
    chunk_repository: FakeChunkRepository | None = None,
    embedder: FakeEmbedder | None = None,
) -> DocumentAnalysisResult:
    """가짜 포트로 유스케이스를 실행한다. 넘기지 않은 포트는 성공하는 가짜를 쓴다."""
    return analyze_document(
        DOCUMENT_ID,
        reader=reader or FakeReader(),
        writer=writer or FakeWriter(),
        repository=repository,
        page_repository=repository,
        chunk_repository=chunk_repository or FakeChunkRepository(),
        embedder=embedder or FakeEmbedder(),
        extractor=extractor or FakeExtractor(),
        render_dpi=72,
        max_pages=5,
        max_page_side_px=2048,
    )


def extraction_error(code: PageExtractionErrorCode) -> PageExtractionError:
    """오류 코드로 페이지 추출 예외를 만든다."""
    return PageExtractionError(code)


def test_all_pages_succeed_marks_ready() -> None:
    """모든 페이지가 성공하면 시작 표시 → 페이지 저장 → 결과 저장 순서로 부르고 READY다."""
    repository = FakeRepository()
    reader = FakeReader()

    result = run(repository, reader=reader)

    assert reader.requested_keys == [OBJECT_KEY]
    assert repository.call_names() == [
        "find_target",
        "mark_started",
        "save_rendered_pages",
        "save_result",
    ]
    assert repository.calls[-1] == (
        "save_result",
        DocumentStatus.READY,
        {1: EXTRACTION, 2: EXTRACTION, 3: EXTRACTION},
        "fake-model",
    )
    assert result == DocumentAnalysisResult(
        document_status=DocumentStatus.READY, usages=(USAGE, USAGE, USAGE)
    )


def test_some_pages_fail_marks_needs_review() -> None:
    """일부 페이지가 실패하면 성공한 페이지만 저장하고 NEEDS_REVIEW다."""
    repository = FakeRepository()
    extractor = FakeExtractor({2: extraction_error(PageExtractionErrorCode.TIMEOUT)})

    result = run(repository, extractor=extractor)

    assert extractor.requested_page_nos == [1, 2, 3]
    assert repository.calls[-1] == (
        "save_result",
        DocumentStatus.NEEDS_REVIEW,
        {1: EXTRACTION, 3: EXTRACTION},
        "fake-model",
    )
    assert result == DocumentAnalysisResult(
        document_status=DocumentStatus.NEEDS_REVIEW, usages=(USAGE, USAGE)
    )


def test_blank_page_counts_as_success() -> None:
    """글자가 없는 페이지는 실패로 세지 않아 READY이고, 빈 결과를 그대로 저장소에 넘긴다."""
    repository = FakeRepository()

    result = run(repository, extractor=FakeExtractor(blank_page_nos={2}))

    assert repository.calls[-1] == (
        "save_result",
        DocumentStatus.READY,
        {1: EXTRACTION, 2: BLANK_EXTRACTION, 3: EXTRACTION},
        "fake-model",
    )
    assert result.document_status is DocumentStatus.READY


def test_all_pages_fail_marks_failed_with_first_failure() -> None:
    """모든 페이지가 실패하면 결과를 저장하지 않고 첫 실패 페이지의 이유로 FAILED다."""
    repository = FakeRepository()
    extractor = FakeExtractor(
        {
            1: extraction_error(PageExtractionErrorCode.TIMEOUT),
            2: extraction_error(PageExtractionErrorCode.REFUSED),
            3: extraction_error(PageExtractionErrorCode.REFUSED),
        }
    )

    result = run(repository, extractor=extractor)

    failure = AnalysisFailure(
        error_code=AnalysisErrorCode.VLM_TIMEOUT,
        failed_step=FailedStep.VLM_EXTRACTION,
        retryable=True,
        detail_code="TIMEOUT",
    )
    assert repository.calls[-1] == ("mark_failed", failure)
    assert "save_result" not in repository.call_names()
    assert result == DocumentAnalysisResult(document_status=DocumentStatus.FAILED, failure=failure)


def test_missing_api_key_fails_after_first_page() -> None:
    """API 키가 없으면 첫 페이지만 부르고 문서 전체를 재시도 불가 FAILED로 끝낸다."""
    repository = FakeRepository()
    error = extraction_error(PageExtractionErrorCode.MISSING_API_KEY)
    extractor = FakeExtractor({1: error, 2: error, 3: error})

    result = run(repository, extractor=extractor)

    assert extractor.requested_page_nos == [1]
    assert result.document_status is DocumentStatus.FAILED
    assert result.failure == AnalysisFailure(
        error_code=AnalysisErrorCode.AI_GENERATION_FAILED,
        failed_step=FailedStep.VLM_EXTRACTION,
        retryable=False,
        detail_code="MISSING_API_KEY",
    )


@pytest.mark.parametrize(("code", "error_code", "failed_step", "retryable"), EXTRACTION_MAPPING)
def test_extraction_error_code_mapping(
    code: str, error_code: str, failed_step: str, retryable: bool
) -> None:
    """VLM 오류 코드를 FC-15 error_code, failed_step, retryable로 바꾸고 세부 코드를 남긴다."""
    repository = FakeRepository()
    extractor = FakeExtractor({1: extraction_error(PageExtractionErrorCode(code))})

    result = run(repository, reader=FakeReader(make_pdf(1)), extractor=extractor)

    assert result.failure == AnalysisFailure(
        error_code=AnalysisErrorCode(error_code),
        failed_step=FailedStep(failed_step),
        retryable=retryable,
        detail_code=code,
    )


def test_every_extraction_error_code_is_mapped() -> None:
    """대응표 테스트가 VLM 오류 코드를 빠짐없이 다룬다."""
    mapped = {row[0] for row in EXTRACTION_MAPPING}

    assert mapped == {code.value for code in PageExtractionErrorCode}


@pytest.mark.parametrize(
    ("error", "detail_code", "retryable"),
    [
        (DocumentNotFoundError(OBJECT_KEY), "DOCUMENT_NOT_FOUND", True),
        (InvalidObjectKeyError(OBJECT_KEY), "INVALID_OBJECT_KEY", False),
        (OriginalDocumentReadError(OBJECT_KEY), "ORIGINAL_READ_FAILED", True),
    ],
)
def test_original_read_failure_marks_failed(
    error: Exception, detail_code: str, retryable: bool
) -> None:
    """원본을 읽지 못하면 페이지 저장과 VLM 호출 없이 OBJECT_FETCH 단계 FAILED다."""
    repository = FakeRepository()
    extractor = FakeExtractor()

    result = run(repository, reader=FakeReader(error=error), extractor=extractor)

    failure = AnalysisFailure(
        error_code=AnalysisErrorCode.OBJECT_FETCH_FAILED,
        failed_step=FailedStep.OBJECT_FETCH,
        retryable=retryable,
        detail_code=detail_code,
    )
    assert repository.call_names() == ["find_target", "mark_started", "mark_failed"]
    assert extractor.requested_page_nos == []
    assert result == DocumentAnalysisResult(document_status=DocumentStatus.FAILED, failure=failure)
    # error_message에 들어갈 값에 objectKey가 섞이지 않는다.
    assert OBJECT_KEY not in failure.detail_code


def test_page_split_failure_marks_failed() -> None:
    """페이지로 나누지 못하면 분리 오류 코드를 세부 코드로 남기고 재시도 불가 FAILED다."""
    repository = FakeRepository()
    extractor = FakeExtractor()

    result = run(repository, reader=FakeReader(b"not a pdf"), extractor=extractor)

    assert extractor.requested_page_nos == []
    assert result.document_status is DocumentStatus.FAILED
    assert result.failure == AnalysisFailure(
        error_code=AnalysisErrorCode.DOCUMENT_PAGE_SPLIT_FAILED,
        failed_step=FailedStep.PAGE_SPLIT,
        retryable=False,
        detail_code="CORRUPTED_FILE",
    )


@pytest.mark.parametrize("code", list(PageSplitErrorCode))
def test_page_split_error_code_mapping(code: PageSplitErrorCode) -> None:
    """분리 오류 코드는 모두 DOCUMENT_PAGE_SPLIT_FAILED / PAGE_SPLIT, 재시도 불가로 기록한다."""
    repository = FakeRepository()
    # 입력별 분리 오류는 분리기 테스트가 확인하므로, 여기서는 오류를 바로 던진다.
    reader = FakeReader(error=PageSplitError(code))

    result = run(repository, reader=reader)

    assert repository.calls[-1] == (
        "mark_failed",
        AnalysisFailure(
            error_code=AnalysisErrorCode.DOCUMENT_PAGE_SPLIT_FAILED,
            failed_step=FailedStep.PAGE_SPLIT,
            retryable=False,
            detail_code=code.value,
        ),
    )
    assert result.document_status is DocumentStatus.FAILED


def test_rendered_page_write_failure_marks_failed() -> None:
    """렌더링 이미지를 쓰지 못하면 VLM 호출 없이 재시도 가능 FAILED다."""
    repository = FakeRepository()
    extractor = FakeExtractor()
    writer = FakeWriter(RenderedPageWriteError(f"documents/{DOCUMENT_ID}/pages/1.png"))

    result = run(repository, writer=writer, extractor=extractor)

    assert extractor.requested_page_nos == []
    assert "save_rendered_pages" not in repository.call_names()
    assert result.failure == AnalysisFailure(
        error_code=AnalysisErrorCode.OBJECT_FETCH_FAILED,
        failed_step=FailedStep.OBJECT_FETCH,
        retryable=True,
        detail_code="RENDERED_PAGE_WRITE_FAILED",
    )


@pytest.mark.parametrize(
    ("repository_arguments", "expected_calls"),
    [
        ({"target": None}, ["find_target"]),
        ({"started": False}, ["find_target", "mark_started"]),
    ],
    ids=["no_target", "deleted_or_succeeded"],
)
def test_non_target_changes_nothing(
    repository_arguments: dict[str, Any], expected_calls: list[str]
) -> None:
    """대상 없음, 또는 시작 직전에 삭제됐거나 이미 SUCCEEDED인 분석은 원본도 읽지 않고 끝낸다."""
    repository = FakeRepository(**repository_arguments)
    reader = FakeReader()
    extractor = FakeExtractor()

    result = run(repository, reader=reader, extractor=extractor)

    assert repository.call_names() == expected_calls
    assert reader.requested_keys == []
    assert extractor.requested_page_nos == []
    assert result == DocumentAnalysisResult(document_status=None)


def test_unexpected_error_in_extraction_marks_failed_and_reraises() -> None:
    """VLM 단계의 예상 밖 예외는 FAILED로 기록한 뒤 같은 예외를 다시 올린다."""
    repository = FakeRepository()
    error = RuntimeError("bug")

    with pytest.raises(RuntimeError) as raised:
        run(repository, extractor=FakeExtractor({2: error}))

    assert raised.value is error
    assert repository.calls[-1] == (
        "mark_failed",
        AnalysisFailure(
            error_code=AnalysisErrorCode.AI_GENERATION_FAILED,
            failed_step=FailedStep.VLM_EXTRACTION,
            retryable=True,
            detail_code="UNEXPECTED_ERROR",
        ),
    )


def test_unexpected_error_before_extraction_records_object_fetch_step() -> None:
    """추출 전 단계의 예상 밖 예외는 failed_step을 OBJECT_FETCH로 기록한다."""
    repository = FakeRepository()
    error = OSError("disk")

    with pytest.raises(OSError) as raised:
        run(repository, reader=FakeReader(error=error))

    assert raised.value is error
    assert repository.calls[-1][1].failed_step is FailedStep.OBJECT_FETCH


def test_original_error_is_raised_when_failure_record_also_fails() -> None:
    """FAILED 기록까지 실패해도 기록 예외가 아니라 원래 예외를 올린다."""
    repository = FakeRepository(mark_failed_error=ConnectionError("db down"))
    error = RuntimeError("bug")

    with pytest.raises(RuntimeError) as raised:
        run(repository, extractor=FakeExtractor({1: error}))

    assert raised.value is error


@pytest.mark.parametrize(
    ("errors", "expected_status"),
    [
        ({}, DocumentStatus.READY),
        ({2: extraction_error(PageExtractionErrorCode.TIMEOUT)}, DocumentStatus.NEEDS_REVIEW),
    ],
    ids=["ready", "needs_review"],
)
def test_chunks_are_embedded_and_replaced_after_result(
    errors: dict[int, Exception], expected_status: DocumentStatus
) -> None:
    """결과 저장 뒤 원문을 읽어 청크로 나누고, 그 원문을 임베딩한 벡터와 함께 교체를 부른다."""
    repository = FakeRepository()
    chunk_repository = FakeChunkRepository(calls=repository.calls)
    embedder = FakeEmbedder(calls=repository.calls)

    result = run(
        repository,
        extractor=FakeExtractor(errors),
        chunk_repository=chunk_repository,
        embedder=embedder,
    )

    assert result.document_status is expected_status
    assert repository.call_names()[-4:] == [
        "save_result",
        "find_page_texts",
        "embed",
        "replace_chunks",
    ]
    assert repository.calls[-3] == ("find_page_texts", DOCUMENT_ID)
    assert repository.calls[-2] == ("embed", [CHUNK_SOURCE_TEXT])
    name, document_id, analysis_id, chunks, vectors = repository.calls[-1]
    assert (name, document_id, analysis_id) == ("replace_chunks", DOCUMENT_ID, ANALYSIS_ID)
    assert [(c.chunk_index, c.content, c.page_id, c.bounding_box) for c in chunks] == [
        (0, CHUNK_SOURCE_TEXT, PAGE_ID, None)
    ]
    assert vectors == [[0.0]]


def test_no_text_skips_embedding_but_still_replaces() -> None:
    """청크가 없으면 임베딩을 부르지 않고, 이전 분석 벡터를 지우도록 빈 목록으로 교체한다."""
    chunk_repository = FakeChunkRepository(
        pages=[PageText(PAGE_ID, 1, PageOcrStatus.SKIPPED, None)]
    )
    embedder = FakeEmbedder()

    run(FakeRepository(), chunk_repository=chunk_repository, embedder=embedder)

    assert embedder.calls == []
    assert chunk_repository.calls[-1] == (
        "replace_chunks",
        DOCUMENT_ID,
        ANALYSIS_ID,
        [],
        [],
    )


def test_embedding_failure_keeps_analysis_and_stores_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """임베딩이 끝내 실패하면 교체를 부르지 않아 벡터 없는 청크가 남지 않고, 분석 결과는 같다."""
    repository = FakeRepository()
    chunk_repository = FakeChunkRepository(calls=repository.calls)
    embedder = FakeEmbedder(
        calls=repository.calls, error=TextEmbeddingError(TextEmbeddingErrorCode.RATE_LIMITED)
    )

    with caplog.at_level(logging.ERROR):
        result = run(repository, chunk_repository=chunk_repository, embedder=embedder)

    assert result == DocumentAnalysisResult(
        document_status=DocumentStatus.READY, usages=(USAGE, USAGE, USAGE)
    )
    assert "replace_chunks" not in repository.call_names()
    assert "mark_failed" not in repository.call_names()
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "TextEmbeddingError(RATE_LIMITED)" in errors[0].getMessage()
    assert CHUNK_SOURCE_TEXT not in caplog.text


def test_skipped_replacement_keeps_analysis_result(caplog: pytest.LogCaptureFixture) -> None:
    """문서 삭제나 더 새 분석 때문에 교체를 건너뛰어도 결과는 같고 error 로그는 없다."""
    repository = FakeRepository()

    with caplog.at_level(logging.INFO):
        result = run(repository, chunk_repository=FakeChunkRepository(replaced=False))

    assert result.document_status is DocumentStatus.READY
    assert "mark_failed" not in repository.call_names()
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert "건너뜀" in caplog.text


def test_no_chunking_when_all_pages_fail() -> None:
    """분석이 FAILED로 끝나면 청킹, 임베딩, 교체를 부르지 않는다."""
    chunk_repository = FakeChunkRepository()
    embedder = FakeEmbedder()
    error = extraction_error(PageExtractionErrorCode.REFUSED)

    run(
        FakeRepository(),
        extractor=FakeExtractor({1: error, 2: error, 3: error}),
        chunk_repository=chunk_repository,
        embedder=embedder,
    )

    assert chunk_repository.calls == []
    assert embedder.calls == []


def test_no_chunking_when_result_was_not_saved() -> None:
    """그 사이 문서가 삭제되어 결과를 저장하지 못했으면 청킹을 부르지 않는다."""

    class DeletedRepository(FakeRepository):
        """결과 저장이 언제나 False인 가짜 저장소."""

        def save_result(self, *args: Any) -> bool:
            """문서가 삭제된 경우처럼 False를 돌려준다."""
            return False

    chunk_repository = FakeChunkRepository()

    result = run(DeletedRepository(), chunk_repository=chunk_repository)

    assert result.document_status is None
    assert chunk_repository.calls == []


def test_no_chunking_for_non_target() -> None:
    """분석 대상이 아니면 청킹을 부르지 않는다."""
    chunk_repository = FakeChunkRepository()

    run(FakeRepository(target=None), chunk_repository=chunk_repository)

    assert chunk_repository.calls == []


@pytest.mark.parametrize(
    "chunk_repository_arguments",
    [
        {"find_error": ConnectionError(CHUNK_SOURCE_TEXT)},
        {"save_error": IntegrityError("INSERT", {"content": CHUNK_SOURCE_TEXT}, Exception())},
        {"save_error": ValueError(CHUNK_SOURCE_TEXT)},
    ],
    ids=["find_fails", "replace_conflict", "replace_rejects"],
)
def test_chunking_failure_keeps_analysis_result(
    chunk_repository_arguments: dict[str, Exception], caplog: pytest.LogCaptureFixture
) -> None:
    """청크 저장소가 실패해도 반환값은 같고 mark_failed는 없다. 로그는 error에 원문 없이 남는다."""
    repository = FakeRepository()

    with caplog.at_level(logging.ERROR):
        result = run(
            repository, chunk_repository=FakeChunkRepository(**chunk_repository_arguments)
        )

    assert result == DocumentAnalysisResult(
        document_status=DocumentStatus.READY, usages=(USAGE, USAGE, USAGE)
    )
    assert "mark_failed" not in repository.call_names()
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    expected_class = next(iter(chunk_repository_arguments.values())).__class__.__name__
    assert expected_class in errors[0].getMessage()
    assert CHUNK_SOURCE_TEXT not in caplog.text


def test_chunk_calculation_failure_keeps_analysis_result(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """청크 계산이 실패해도(예: 인코딩 내려받기 실패) 분석 결과는 그대로이고 저장하지 않는다."""

    def fail(pages: Sequence[PageText]) -> list[DocumentChunk]:
        raise OSError(CHUNK_SOURCE_TEXT)

    monkeypatch.setattr("app.application.use_cases.analyze_document.chunk_pages", fail)
    repository = FakeRepository()
    chunk_repository = FakeChunkRepository()

    with caplog.at_level(logging.ERROR):
        result = run(repository, chunk_repository=chunk_repository)

    assert result.document_status is DocumentStatus.READY
    assert "mark_failed" not in repository.call_names()
    assert [call[0] for call in chunk_repository.calls] == ["find_page_texts"]
    assert "OSError" in caplog.text
    assert CHUNK_SOURCE_TEXT not in caplog.text
