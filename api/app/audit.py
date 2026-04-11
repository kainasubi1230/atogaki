from datetime import datetime, timezone

from sqlalchemy.orm import Session

from .jsonutil import dumps
from .models import AuditLog


def log_event(
    db: Session,
    *,
    path: str,
    method: str,
    event_type: str,
    detail: dict | None = None,
    user_id: int | None = None,
    status_code: int | None = None,
) -> AuditLog:
    log = AuditLog(
        user_id=user_id,
        path=path,
        method=method,
        event_type=event_type,
        status_code=status_code,
        detail_json=dumps(detail or {}),
        created_at=datetime.now(timezone.utc),
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log
