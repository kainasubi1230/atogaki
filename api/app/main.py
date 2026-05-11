from datetime import datetime
import uuid

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import and_
from sqlalchemy.orm import Session

from trainerlib.model import generate_trajectory
from trainerlib.svg import trajectory_to_svg

from .audit import log_event
from .database import Base, SessionLocal, engine, get_db
from .deps import get_current_user
from .jsonutil import dumps, loads
from .models import AuditLog, Dataset, Job, Output, StyleAdapter, User
from .queueing import enqueue
from .schemas import (
    AuditLogResponse,
    DatasetUploadResponse,
    GenerateRequest,
    GenerateResponse,
    JobDetailResponse,
    JobResponse,
    LoginRequest,
    OutputResponse,
    SignupRequest,
    TokenResponse,
    TrainStyleResponse,
)
from .security import create_access_token, hash_password, parse_access_token, verify_password
from .storage import get_storage
from .tasks import run_preprocess_job, run_train_lora_job

WATERMARK_TEXT = "AI生成（アクセシビリティ支援）"

app = FastAPI(title="Accessibility Handwriting MVP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        db = SessionLocal()
        try:
            user_id = None
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header.replace("Bearer ", "", 1).strip()
                user_id = parse_access_token(token)
            log_event(
                db,
                path=request.url.path,
                method=request.method,
                event_type="http_request",
                detail={"query": str(request.url.query)},
                user_id=user_id,
                status_code=response.status_code if response else 500,
            )
        finally:
            db.close()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/auth/signup", response_model=TokenResponse)
def signup(payload: SignupRequest, db: Session = Depends(get_db)) -> TokenResponse:
    exists = db.query(User).filter(User.email == payload.email).first()
    if exists:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="email_exists")
    user = User(email=payload.email, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user.id)
    log_event(db, path="/auth/signup", method="POST", event_type="auth_signup", detail={"user_id": user.id}, user_id=user.id)
    return TokenResponse(access_token=token, user_id=user.id)


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.query(User).filter(User.email == payload.email).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_credentials")
    token = create_access_token(user.id)
    log_event(db, path="/auth/login", method="POST", event_type="auth_login", detail={"user_id": user.id}, user_id=user.id)
    return TokenResponse(access_token=token, user_id=user.id)


@app.post("/datasets/upload-scan", response_model=DatasetUploadResponse)
async def upload_scan(
    consent: bool = Form(...),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DatasetUploadResponse:
    key = f"scans/u{user.id}/{uuid.uuid4().hex}-{file.filename}"
    payload = await file.read()
    storage = get_storage()
    storage.put_bytes(key, payload, file.content_type or "application/octet-stream")

    dataset = Dataset(user_id=user.id, object_key=key, consent=consent, active=True)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    log_event(
        db,
        path="/datasets/upload-scan",
        method="POST",
        event_type="dataset_upload",
        detail={"dataset_id": dataset.id, "consent": consent},
        user_id=user.id,
        status_code=200,
    )
    return DatasetUploadResponse(dataset_id=dataset.id, user_id=user.id, consent=consent)


@app.post("/datasets/{user_id}/preprocess", response_model=JobResponse)
def preprocess_datasets(user_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> JobResponse:
    if user.id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="user_mismatch")

    job_id = uuid.uuid4().hex
    job = Job(id=job_id, user_id=user_id, kind="preprocess", status="queued", payload_json=dumps({"user_id": user_id}))
    db.add(job)
    db.commit()

    enqueue("default", run_preprocess_job, job_id)
    log_event(
        db,
        path=f"/datasets/{user_id}/preprocess",
        method="POST",
        event_type="preprocess_requested",
        detail={"job_id": job_id},
        user_id=user_id,
        status_code=202,
    )
    refreshed = db.query(Job).filter(Job.id == job_id).first()
    return JobResponse(job_id=job_id, kind="preprocess", status=refreshed.status if refreshed else "queued")


@app.post("/styles/{user_id}/train-lora", response_model=TrainStyleResponse)
def train_lora(user_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> TrainStyleResponse:
    if user.id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="user_mismatch")

    style = StyleAdapter(user_id=user_id, status="training")
    db.add(style)
    db.commit()
    db.refresh(style)

    job_id = uuid.uuid4().hex
    job = Job(
        id=job_id,
        user_id=user_id,
        kind="train_lora",
        status="queued",
        payload_json=dumps({"user_id": user_id, "style_id": style.id}),
    )
    db.add(job)
    db.commit()

    enqueue("trainer", run_train_lora_job, job_id)
    log_event(
        db,
        path=f"/styles/{user_id}/train-lora",
        method="POST",
        event_type="train_lora_requested",
        detail={"job_id": job_id, "style_id": style.id},
        user_id=user_id,
        status_code=202,
    )
    refreshed = db.query(Job).filter(Job.id == job_id).first()
    return TrainStyleResponse(style_id=style.id, job_id=job_id, status=refreshed.status if refreshed else "queued")


@app.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> GenerateResponse:
    if user.id != payload.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="user_mismatch")
    if payload.purpose != "accessibility":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="purpose_must_be_accessibility")

    style = db.query(StyleAdapter).filter(StyleAdapter.id == payload.style_id).first()
    if style is None or style.user_id != payload.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="style_owner_mismatch")
    if style.disabled or style.status != "ready" or not style.adapter_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="style_not_ready")

    storage = get_storage()
    adapter_seed = storage.get_text(style.adapter_key)
    trajectory = generate_trajectory(payload.text, adapter_seed)
    svg = trajectory_to_svg(trajectory, WATERMARK_TEXT)

    output_svg_key = f"outputs/u{user.id}/{uuid.uuid4().hex}.svg"
    output_trajectory_key = f"outputs/u{user.id}/{uuid.uuid4().hex}.json"
    storage.put_text(output_svg_key, svg, content_type="image/svg+xml")
    storage.put_text(output_trajectory_key, dumps(trajectory), content_type="application/json")

    output = Output(
        user_id=user.id,
        style_id=style.id,
        text=payload.text,
        purpose=payload.purpose,
        svg_key=output_svg_key,
        trajectory_key=output_trajectory_key,
        watermark_text=WATERMARK_TEXT,
    )
    db.add(output)
    db.commit()
    db.refresh(output)

    log_event(
        db,
        path="/generate",
        method="POST",
        event_type="generation_created",
        detail={"output_id": output.id, "purpose": payload.purpose, "char_count": len(payload.text)},
        user_id=user.id,
        status_code=200,
    )
    return GenerateResponse(output_id=output.id, svg=svg, trajectory=trajectory)


@app.get("/jobs/{job_id}", response_model=JobDetailResponse)
def get_job(job_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> JobDetailResponse:
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job_not_found")
    if job.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")
    return JobDetailResponse(
        job_id=job.id,
        kind=job.kind,
        status=job.status,
        error_code=job.error_code,
        payload=loads(job.payload_json),
        result=loads(job.result_json),
        updated_at=job.updated_at,
    )


@app.get("/outputs/{output_id}", response_model=OutputResponse)
def get_output(output_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> OutputResponse:
    output = db.query(Output).filter(Output.id == output_id).first()
    if output is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="output_not_found")
    if output.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")
    storage = get_storage()
    return OutputResponse(
        output_id=output.id,
        user_id=output.user_id,
        style_id=output.style_id,
        text=output.text,
        purpose=output.purpose,
        watermark_text=output.watermark_text,
        svg=storage.get_text(output.svg_key),
        trajectory=loads(storage.get_text(output.trajectory_key)),
        created_at=output.created_at,
    )


@app.get("/audit/logs", response_model=list[AuditLogResponse])
def get_audit_logs(
    user_id: int | None = Query(default=None),
    from_ts: datetime | None = Query(default=None, alias="from"),
    to_ts: datetime | None = Query(default=None, alias="to"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[AuditLogResponse]:
    target_user_id = user_id if user_id is not None else current_user.id
    if target_user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")

    conditions = [AuditLog.user_id == target_user_id]
    if from_ts is not None:
        conditions.append(AuditLog.created_at >= from_ts)
    if to_ts is not None:
        conditions.append(AuditLog.created_at <= to_ts)
    logs = db.query(AuditLog).filter(and_(*conditions)).order_by(AuditLog.created_at.desc()).limit(500).all()
    return [
        AuditLogResponse(
            id=log.id,
            user_id=log.user_id,
            method=log.method,
            path=log.path,
            event_type=log.event_type,
            status_code=log.status_code,
            detail=loads(log.detail_json),
            created_at=log.created_at,
        )
        for log in logs
    ]


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
