from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import boto3

from .settings import settings


class StorageBackend(ABC):
    @abstractmethod
    def put_bytes(self, key: str, payload: bytes, content_type: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def put_text(self, key: str, payload: str, content_type: str = "text/plain") -> None:
        raise NotImplementedError

    @abstractmethod
    def get_text(self, key: str) -> str:
        raise NotImplementedError


class LocalStorage(StorageBackend):
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        path = (self.root / key).resolve()
        root = self.root.resolve()
        if root not in path.parents and path != root:
            raise ValueError("invalid storage key")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def put_bytes(self, key: str, payload: bytes, content_type: str) -> None:
        _ = content_type
        self._resolve(key).write_bytes(payload)

    def get_bytes(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def put_text(self, key: str, payload: str, content_type: str = "text/plain") -> None:
        _ = content_type
        self._resolve(key).write_text(payload, encoding="utf-8")

    def get_text(self, key: str) -> str:
        return self._resolve(key).read_text(encoding="utf-8")


class S3Storage(StorageBackend):
    def __init__(self, endpoint_url: str, access_key: str, secret_key: str, bucket: str, region: str):
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        buckets = self.client.list_buckets().get("Buckets", [])
        names = {b["Name"] for b in buckets}
        if self.bucket not in names:
            self.client.create_bucket(Bucket=self.bucket)

    def put_bytes(self, key: str, payload: bytes, content_type: str) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=payload, ContentType=content_type)

    def get_bytes(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def put_text(self, key: str, payload: str, content_type: str = "text/plain") -> None:
        self.put_bytes(key, payload.encode("utf-8"), content_type)

    def get_text(self, key: str) -> str:
        return self.get_bytes(key).decode("utf-8")


_storage: StorageBackend | None = None


def get_storage() -> StorageBackend:
    global _storage
    if _storage is not None:
        return _storage
    if settings.storage_backend == "s3":
        _storage = S3Storage(
            endpoint_url=settings.s3_endpoint_url,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            bucket=settings.s3_bucket,
            region=settings.s3_region,
        )
    else:
        _storage = LocalStorage(settings.local_storage_dir)
    return _storage

