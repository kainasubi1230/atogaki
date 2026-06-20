from datetime import datetime, timezone
import uuid
from pathlib import Path
import json
from typing import Any

from sqlalchemy.orm import Session

from trainerlib.model import train_lora_adapter, _trajectory_quality_score
from trainerlib.preprocess import preprocess_scan
from trainerlib.char_token import char_to_model_id

from .database import SessionLocal
from .jsonutil import dumps, loads
from .models import Dataset, Job, StyleAdapter
from .storage import get_storage
from .settings import settings

HIRAGANA_TARGET = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
KATAKANA_TARGET = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
KANJI_CORE_TARGET = "日月火水木金土山川田天気学年人大小中上下左右先生今来行見話書読食飲休車電駅校友名本語文字漢"
LATIN_TARGET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
HIRAGANA_TARGET_SET = set(HIRAGANA_TARGET)
KATAKANA_TARGET_SET = set(KATAKANA_TARGET)
KANJI_CORE_TARGET_SET = set(KANJI_CORE_TARGET)
LATIN_TARGET_SET = set(LATIN_TARGET)

# Hiragana is the main focus; keep katakana/kanji covered but lighter.
BOOTSTRAP_CAP_HIRAGANA = 18
BOOTSTRAP_CAP_KATAKANA = 10
BOOTSTRAP_CAP_KANJI = 6
BOOTSTRAP_CAP_LATIN = 8

BOOTSTRAP_DATASETS = [
    "storage/base/base_dataset_latin_v1.jsonl",
    "storage/base/base_dataset_handwritten_mix_v1.jsonl",
    "storage/base/base_dataset_hiragana_k49_mix.jsonl",
    "storage/base/base_dataset_katakana_handwritten_like_v8best.jsonl",
    "storage/base/base_dataset_joyo_full_plus_user_v1.jsonl",
]


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


def _trajectory_to_sequence(points: list[dict]) -> list[list[float]]:
    if not isinstance(points, list) or len(points) < 2:
        return []
    ordered = sorted(points, key=lambda p: int(p.get("t", 0)))
    prev_x = float(ordered[0].get("x", 0.0))
    prev_y = float(ordered[0].get("y", 0.0))
    seq: list[list[float]] = []
    for p in ordered[1:]:
        x = float(p.get("x", prev_x))
        y = float(p.get("y", prev_y))
        dx = x - prev_x
        dy = y - prev_y
        pen = 1.0 if p.get("pen_state") == "down" else 0.0
        width = float(p.get("width", 1.0)) / 4.0
        seq.append([dx / 20.0, dy / 20.0, pen, max(0.1, min(1.6, width))])
        prev_x, prev_y = x, y
    return seq


def _label_from_segment(segment: dict[str, Any]) -> str | None:
    raw = segment.get("label")
    if not isinstance(raw, str):
        return None
    label = raw.strip()
    if len(label) != 1:
        return None
    return label


def _collect_user_samples_from_artifact(
    artifact: dict[str, Any],
    *,
    dataset_id: int,
    user_id: int,
    style_id: int,
    source: str = "user_trajectory",
) -> list[dict]:
    if not artifact.get("success"):
        return []
    segments = artifact.get("segments")
    if not isinstance(segments, list):
        return []
    out: list[dict] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        label = _label_from_segment(segment)
        if not label:
            continue
        quality = segment.get("quality")
        if not isinstance(quality, (int, float)):
            quality = _trajectory_quality_score(segment.get("trajectory", []), label)
        if quality < -120.0:
            continue
        seq = _trajectory_to_sequence(segment.get("trajectory", []))
        if len(seq) < 8:
            continue
        out.append(
            {
                "char_id": char_to_model_id(label),
                "style_id": style_id,
                "dataset_id": dataset_id,
                "user_id": user_id,
                "sequence": seq,
                "meta": {"source": source, "char": label, "quality": round(float(quality), 4)},
            }
        )
    return out


def _sample_char_from_row(row: dict[str, Any]) -> str | None:
    meta = row.get("meta")
    if isinstance(meta, dict):
        ch = meta.get("char")
        if isinstance(ch, str) and len(ch) == 1:
            return ch
    return None


def _bootstrap_cap_for_char(ch: str) -> int:
    if ch in HIRAGANA_TARGET_SET:
        return BOOTSTRAP_CAP_HIRAGANA
    if ch in KATAKANA_TARGET_SET:
        return BOOTSTRAP_CAP_KATAKANA
    if ch in KANJI_CORE_TARGET_SET:
        return BOOTSTRAP_CAP_KANJI
    if ch in LATIN_TARGET_SET:
        return BOOTSTRAP_CAP_LATIN
    return BOOTSTRAP_CAP_KANJI


def _collect_bootstrap_samples(
    *,
    style_id: int,
    user_id: int,
    existing_chars: set[str],
) -> list[dict]:
    # Prioritize hiragana, then katakana, practical core kanji, and Latin.
    target_chars = list(HIRAGANA_TARGET + KATAKANA_TARGET + KANJI_CORE_TARGET + LATIN_TARGET)
    needed = [ch for ch in target_chars if ch not in existing_chars]
    if not needed:
        return []
    need_set = set(needed)
    buckets: dict[str, list[dict]] = {ch: [] for ch in needed}
    per_char_caps = {ch: _bootstrap_cap_for_char(ch) for ch in needed}
    out: list[dict] = []

    for path_str in BOOTSTRAP_DATASETS:
        path = Path(path_str)
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    ch = _sample_char_from_row(row)
                    if ch is None or ch not in need_set:
                        continue
                    seq = row.get("sequence")
                    if not isinstance(seq, list) or len(seq) < 8:
                        continue
                    if len(buckets[ch]) >= per_char_caps[ch]:
                        continue
                    sample = {
                        "char_id": char_to_model_id(ch),
                        "style_id": style_id,
                        "dataset_id": 0,
                        "user_id": user_id,
                        "sequence": seq,
                        "meta": {"source": f"bootstrap:{path.name}", "char": ch},
                    }
                    buckets[ch].append(sample)
        except Exception:
            continue

        # Stop early only when each needed char reached its target cap.
        if all(len(buckets[ch]) >= per_char_caps[ch] for ch in needed):
            break

    for ch in needed:
        out.extend(buckets[ch][: per_char_caps[ch]])
    return out


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

        user_samples: list[dict] = []
        unlabeled_segments = 0
        for ds in datasets:
            if not ds.preprocess_artifact_key:
                continue
            try:
                artifact = loads(storage.get_text(ds.preprocess_artifact_key))
            except Exception:
                continue
            segs = artifact.get("segments")
            if isinstance(segs, list):
                for segment in segs:
                    if isinstance(segment, dict) and _label_from_segment(segment) is None:
                        unlabeled_segments += 1
            source = "user_scan" if ds.object_key.startswith("scans/") else "user_trajectory"
            user_samples.extend(
                _collect_user_samples_from_artifact(
                    artifact,
                    dataset_id=ds.id,
                    user_id=user_id,
                    style_id=style_id,
                    source=source,
                )
            )

        existing_chars = {
            str((s.get("meta") or {}).get("char", ""))
            for s in user_samples
            if isinstance((s.get("meta") or {}).get("char"), str) and len(str((s.get("meta") or {}).get("char", ""))) == 1
        }
        bootstrap_samples = _collect_bootstrap_samples(
            style_id=style_id,
            user_id=user_id,
            existing_chars=existing_chars,
        )
        user_samples.extend(bootstrap_samples)

        adapter_key = f"adapters/u{user_id}/style-{style_id}-{uuid.uuid4().hex}.json"
        adapter_path = f"tmp_{adapter_key.replace('/', '_')}"
        train_result = train_lora_adapter(
            user_id,
            style_id,
            len(datasets),
            adapter_path,
            user_samples=user_samples,
            base_model_path=settings.base_model_path,
        )
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
                "sample_count": len(user_samples),
                "char_coverage": len({int(s["char_id"]) for s in user_samples}),
                "char_coverage_text": len({str((s.get("meta") or {}).get("char", "")) for s in user_samples if (s.get("meta") or {}).get("char")}),
                "bootstrap_sample_count": len(bootstrap_samples),
                "unlabeled_segments": unlabeled_segments,
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
