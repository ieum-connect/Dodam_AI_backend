import io
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from app.application.ports.document_storage import (
    DocumentNotFoundError, InvalidObjectKeyError, OriginalDocumentReadError,
    RenderedPageWriteError,
)
from app.infrastructure.storage.s3_document_storage import S3DocumentStorage


@pytest.fixture
def storage():
    client = Mock()
    return S3DocumentStorage("private-docs", "ap-southeast-2", client=client), client


def test_reads_existing_key_and_closes_stream(storage):
    adapter, client = storage
    body = io.BytesIO(b"pdf")
    client.get_object.return_value = {"Body": body}
    assert adapter.read_original("documents/patient/original.pdf") == b"pdf"
    client.get_object.assert_called_once_with(Bucket="private-docs", Key="documents/patient/original.pdf")
    assert body.closed


def test_writes_private_encrypted_png(storage):
    adapter, client = storage
    adapter.write_rendered_page("rendered/page.png", b"png")
    client.put_object.assert_called_once_with(Bucket="private-docs", Key="rendered/page.png",
        Body=b"png", ContentType="image/png", ServerSideEncryption="AES256")


@pytest.mark.parametrize("key", ["", "/doc", "../doc", "a/../b", "a//b", "a\\b", "a\0b"])
def test_invalid_keys_never_reach_s3(storage, key):
    adapter, client = storage
    with pytest.raises(InvalidObjectKeyError):
        adapter.read_original(key)
    with pytest.raises(InvalidObjectKeyError):
        adapter.write_rendered_page(key, b"png")
    assert not client.mock_calls


@pytest.mark.parametrize("code,error", [("NoSuchKey", DocumentNotFoundError), ("AccessDenied", OriginalDocumentReadError)])
def test_maps_read_errors(storage, code, error):
    adapter, client = storage
    client.get_object.side_effect = ClientError({"Error": {"Code": code, "Message": "private details"}}, "GetObject")
    with pytest.raises(error) as exc:
        adapter.read_original("docs/a.pdf")
    assert "private details" not in str(exc.value)


def test_maps_write_error(storage):
    adapter, client = storage
    client.put_object.side_effect = ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")
    with pytest.raises(RenderedPageWriteError):
        adapter.write_rendered_page("rendered/a.png", b"png")
