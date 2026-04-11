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

