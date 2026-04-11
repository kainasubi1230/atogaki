from dataclasses import dataclass
import os


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")
    jwt_secret: str = os.getenv("JWT_SECRET", "change-me-in-production")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    rq_inline: bool = _bool_env("RQ_INLINE", True)
    storage_backend: str = os.getenv("STORAGE_BACKEND", "local")
    local_storage_dir: str = os.getenv("LOCAL_STORAGE_DIR", "./storage")
    s3_endpoint_url: str = os.getenv("S3_ENDPOINT_URL", "")
    s3_access_key: str = os.getenv("S3_ACCESS_KEY", "")
    s3_secret_key: str = os.getenv("S3_SECRET_KEY", "")
    s3_bucket: str = os.getenv("S3_BUCKET", "handwriting")
    s3_region: str = os.getenv("S3_REGION", "ap-northeast-1")


settings = Settings()

