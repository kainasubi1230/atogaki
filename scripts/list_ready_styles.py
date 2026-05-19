from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app.database import SessionLocal
from api.app.models import StyleAdapter


def main() -> None:
    db = SessionLocal()
    try:
        rows = (
            db.query(StyleAdapter)
            .filter(StyleAdapter.disabled.is_(False), StyleAdapter.status == "ready", StyleAdapter.adapter_key.is_not(None))
            .order_by(StyleAdapter.id.asc())
            .all()
        )
    finally:
        db.close()

    if not rows:
        print("No ready styles found.")
        return

    print("style_id\tuser_id\tstatus\tdisabled\tadapter_key")
    for row in rows:
        print(f"{row.id}\t{row.user_id}\t{row.status}\t{row.disabled}\t{row.adapter_key}")


if __name__ == "__main__":
    main()

