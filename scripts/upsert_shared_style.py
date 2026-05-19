from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app.database import Base, SessionLocal, engine
from api.app.models import StyleAdapter


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Insert or update a shared style row for inference-only mode.")
    parser.add_argument("--style-id", type=int, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--adapter-key", required=True, help="Storage key like adapters/u1/style-1-xxxx.json")
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        style = db.query(StyleAdapter).filter(StyleAdapter.id == args.style_id).first()
        if style is None:
            style = StyleAdapter(
                id=args.style_id,
                user_id=args.user_id,
                status="ready",
                adapter_key=args.adapter_key,
                disabled=False,
                created_at=_utcnow(),
            )
            db.add(style)
        else:
            style.user_id = args.user_id
            style.status = "ready"
            style.adapter_key = args.adapter_key
            style.disabled = False
            db.add(style)
        db.commit()
        print(
            {
                "style_id": style.id,
                "user_id": style.user_id,
                "status": style.status,
                "disabled": style.disabled,
                "adapter_key": style.adapter_key,
            }
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()

