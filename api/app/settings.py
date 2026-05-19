from dataclasses import dataclass, field
import os


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _list_env(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return default
    items = [v.strip() for v in value.split(",")]
    return [v for v in items if v]


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
    base_model_path: str = os.getenv("BASE_MODEL_PATH", "./storage/models/base_model.pt")
    inference_only: bool = _bool_env("INFERENCE_ONLY", False)
    shared_style_id: int = _int_env("SHARED_STYLE_ID", 0)
    readable_text_svg: bool = _bool_env("READABLE_TEXT_SVG", False)
    cors_allow_origins: list[str] = field(
        default_factory=lambda: _list_env(
            "CORS_ALLOW_ORIGINS",
            ["http://localhost:3000", "http://127.0.0.1:3000"],
        )
    )
    cors_allow_origin_regex: str = os.getenv("CORS_ALLOW_ORIGIN_REGEX", r"https?://.*")


settings = Settings()
