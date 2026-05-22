from datetime import datetime, timezone
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from trainerlib.model import train_lora_adapter
from trainerlib.preprocess import preprocess_scan

from .database import SessionLocal
from .jsonutil import dumps, loads
from .models import Dataset, Job, StyleAdapter
from .storage import get_storage


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _set_job_status(db: Session, job: Job, status: str, result: dict | None = None, error_code: str | None = None) -> None:
    job.status = status
    job.updated_at = _utcnow()
    if result is not None:
        job.result_json = dumps(result)
    job.error_code = error_code
    db.add(job)
    db.commit()


def run_preprocess_job(job_id: str) -> None:
    db = SessionLocal()
    storage = get_storage()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job is None:
            return
        _set_job_status(db, job, "running")

        payload = loads(job.payload_json)
        user_id = payload["user_id"]

        consented_datasets = (
            db.query(Dataset)
            .filter(Dataset.user_id == user_id, Dataset.active.is_(True), Dataset.consent.is_(True))
            .all()
        )
        if not consented_datasets:
            _set_job_status(db, job, "failed", {"message": "no consented datasets"}, "NO_CONSENT_DATA")
            return
        datasets = [d for d in consented_datasets if d.preprocess_status != "done"]
        if not datasets:
            _set_job_status(
                db,
                job,
                "completed",
                {
                    "success_count": 0,
                    "failure_count": 0,
                    "failure_codes": {},
                    "skipped_done_count": len(consented_datasets),
                },
            )
            return

        successes = 0
        failures = 0
        failure_codes: dict[str, int] = {}
        for dataset in datasets:
            result = preprocess_scan(storage.get_bytes(dataset.object_key))
            artifact_key = f"preprocessed/u{dataset.user_id}/d{dataset.id}.json"
            storage.put_text(artifact_key, dumps(result), content_type="application/json")
            dataset.preprocess_artifact_key = artifact_key
            if result["success"]:
                dataset.preprocess_status = "done"
                dataset.preprocess_error_code = None
                successes += 1
            else:
                dataset.preprocess_status = "failed"
                dataset.preprocess_error_code = result["reason_code"]
                failures += 1
                failure_codes[result["reason_code"]] = failure_codes.get(result["reason_code"], 0) + 1
            db.add(dataset)
        db.commit()

        _set_job_status(
            db,
            job,
            "completed",
            {
                "success_count": successes,
                "failure_count": failures,
                "failure_codes": failure_codes,
            },
        )
    except Exception:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job is not None:
            _set_job_status(db, job, "failed", {"message": "unexpected error"}, "PREPROCESS_EXCEPTION")
    finally:
        db.close()


def run_train_lora_job(job_id: str) -> None:
    db = SessionLocal()
    storage = get_storage()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job is None:
            return
        _set_job_status(db, job, "running")
        payload = loads(job.payload_json)
        user_id = payload["user_id"]
        style_id = payload["style_id"]

        style = db.query(StyleAdapter).filter(StyleAdapter.id == style_id, StyleAdapter.user_id == user_id).first()
        if style is None:
            _set_job_status(db, job, "failed", {"message": "style not found"}, "STYLE_NOT_FOUND")
            return

        datasets = (
            db.query(Dataset)
            .filter(
                Dataset.user_id == user_id,
                Dataset.active.is_(True),
                Dataset.consent.is_(True),
                Dataset.preprocess_status == "done",
            )
            .all()
        )
        if not datasets:
            style.status = "failed"
            db.add(style)
            db.commit()
            _set_job_status(db, job, "failed", {"message": "no preprocessed data"}, "NO_PREPROCESSED_DATA")
            return

        adapter_key = f"adapters/u{user_id}/style-{style_id}-{uuid.uuid4().hex}.json"
        adapter_path = f"tmp_{adapter_key.replace('/', '_')}"
        train_result = train_lora_adapter(user_id, style_id, len(datasets), adapter_path)
        adapter_content = Path(train_result.adapter_path).read_text(encoding="utf-8")
        storage.put_text(adapter_key, adapter_content, "application/json")
        Path(train_result.adapter_path).unlink(missing_ok=True)

        style.adapter_key = adapter_key
        style.status = "ready"
        db.add(style)
        db.commit()

        _set_job_status(
            db,
            job,
            "completed",
            {
                "style_id": style_id,
                "adapter_key": adapter_key,
                "similarity_score": train_result.similarity_score,
                "cer": train_result.cer,
            },
        )
    except Exception:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job is not None:
            _set_job_status(db, job, "failed", {"message": "unexpected error"}, "TRAIN_EXCEPTION")
    finally:
        db.close()
