from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int


class DatasetUploadResponse(BaseModel):
    dataset_id: int
    user_id: int
    consent: bool
    preprocess_status: str | None = None
    preprocess_error_code: str | None = None
    segment_count: int | None = None
    labeled_segment_count: int | None = None


class TrajectoryPoint(BaseModel):
    x: float
    y: float
    t: int
    pen_state: str = Field(pattern="^(down|up)$")
    width: float = Field(default=2.0, ge=0.1, le=20.0)


class TrajectoryUploadRequest(BaseModel):
    consent: bool = True
    label: str = Field(min_length=1, max_length=64)
    points: list[TrajectoryPoint] = Field(min_length=2, max_length=20000)


class TrajectoryUploadResponse(BaseModel):
    dataset_id: int
    user_id: int
    consent: bool
    point_count: int
    artifact_key: str


class JobResponse(BaseModel):
    job_id: str
    kind: str
    status: str


class JobDetailResponse(BaseModel):
    job_id: str
    kind: str
    status: str
    error_code: str | None
    payload: dict
    result: dict
    updated_at: datetime


class TrainStyleResponse(BaseModel):
    style_id: int
    job_id: str
    status: str


class StyleCoverageResponse(BaseModel):
    style_id: int
    status: str
    total_target_chars: int
    covered_count: int
    missing_count: int
    covered_chars: str
    missing_hiragana: str
    missing_katakana: str
    missing_kanji_core: str
    missing_latin: str = ""


class GenerateRequest(BaseModel):
    user_id: int
    style_id: int
    text: str = Field(min_length=1, max_length=500)
    purpose: str


class GenerateResponse(BaseModel):
    output_id: int
    svg: str
    trajectory: list[dict]


class OutputResponse(BaseModel):
    output_id: int
    user_id: int
    style_id: int
    text: str
    purpose: str
    watermark_text: str
    svg: str
    trajectory: list[dict]
    created_at: datetime


class AuditLogResponse(BaseModel):
    id: int
    user_id: int | None
    method: str
    path: str
    event_type: str
    status_code: int | None
    detail: dict
    created_at: datetime

