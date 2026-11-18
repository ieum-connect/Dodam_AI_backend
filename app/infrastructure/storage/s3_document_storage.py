"""Private S3 document storage using the SDK's default credential chain."""

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.application.ports.document_storage import (
    DocumentNotFoundError,
    InvalidObjectKeyError,
    OriginalDocumentReadError,
    RenderedPageWriteError,
)


class S3DocumentStorage:
    def __init__(self, bucket: str, region: str, *, client=None) -> None:
        if not bucket.strip():
            raise ValueError("S3 bucket is required")
        self._bucket = bucket
        self._client = client if client is not None else boto3.client("s3", region_name=region)

    @staticmethod
    def _key(key: str) -> str:
        if not key or not key.strip() or "\\" in key or "\0" in key:
            raise InvalidObjectKeyError(key)
        if any(part in ("", ".", "..") for part in key.split("/")):
            raise InvalidObjectKeyError(key)
        return key

    def read_original(self, object_key: str) -> bytes:
        key = self._key(object_key)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            with response["Body"] as body:
                return body.read()
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
                raise DocumentNotFoundError(key) from None
            raise OriginalDocumentReadError(key) from None
        except (BotoCoreError, OSError):
            raise OriginalDocumentReadError(key) from None

    def write_rendered_page(self, object_key: str, png_bytes: bytes) -> None:
        key = self._key(object_key)
        try:
            self._client.put_object(
                Bucket=self._bucket, Key=key, Body=png_bytes,
                ContentType="image/png", ServerSideEncryption="AES256",
            )
        except (BotoCoreError, ClientError, OSError):
            raise RenderedPageWriteError(key) from None
