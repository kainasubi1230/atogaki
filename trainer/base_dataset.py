from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from sqlalchemy import inspect
from sqlalchemy.orm import Session

from api.app.database import SessionLocal, engine
from api.app.models import Dataset
from api.app.storage import get_storage
from trainerlib.char_token import char_to_model_id


def _sorted_trajectory(points: list[dict]) -> list[dict]:
    return sorted(points, key=lambda p: int(p.get("t", 0)))


def _to_feature_sequence(points: list[dict]) -> list[list[float]]:
    points = _sorted_trajectory(points)
    if len(points) < 2:
        return []

    sequence: list[list[float]] = []
    prev_x = float(points[0].get("x", 0))
    prev_y = float(points[0].get("y", 0))
    for point in points[1:]:
        x = float(point.get("x", prev_x))
        y = float(point.get("y", prev_y))
        dx = x - prev_x
        dy = y - prev_y
        pen = 1.0 if point.get("pen_state") == "down" else 0.0
        width = float(point.get("width", 1.0)) / 4.0
        sequence.append([dx / 20.0, dy / 20.0, pen, width])
        prev_x, prev_y = x, y
    return sequence


def _pseudo_char_id(dataset_id: int, bbox: dict, sequence_len: int) -> int:
    seed = f"{dataset_id}:{bbox.get('x0', 0)}:{bbox.get('y0', 0)}:{bbox.get('x1', 0)}:{bbox.get('y1', 0)}:{sequence_len}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 4096


def _char_id_from_segment(source: Dataset, segment: dict, sequence_len: int) -> int:
    label = segment.get("label")
    if isinstance(label, str):
        normalized = label.strip()
        if normalized:
            return char_to_model_id(normalized)
    return _pseudo_char_id(source.id, segment.get("bbox", {}), sequence_len)


def _normalized_label_from_segment(segment: dict) -> str | None:
    label = segment.get("label")
    if isinstance(label, str):
        normalized = label.strip()
        if normalized:
            return normalized
    return None


@dataclass
class BuildStats:
    dataset_rows: int
    sample_count: int
    skipped_count: int
    output_path: str


def _fetch_source_datasets(db: Session) -> list[Dataset]:
    return (
        db.query(Dataset)
        .filter(
            Dataset.active.is_(True),
            Dataset.consent.is_(True),
            Dataset.preprocess_status == "done",
            Dataset.preprocess_artifact_key.is_not(None),
        )
        .all()
    )


def _fetch_source_datasets_with_prefix(db: Session, object_key_prefix: str | None) -> list[Dataset]:
    query = db.query(Dataset).filter(
        Dataset.active.is_(True),
        Dataset.consent.is_(True),
        Dataset.preprocess_status == "done",
        Dataset.preprocess_artifact_key.is_not(None),
    )
    if object_key_prefix:
        query = query.filter(Dataset.object_key.like(f"{object_key_prefix}%"))
    return query.all()


def build_base_dataset(
    output_path: str,
    *,
    min_points: int = 8,
    max_segments_per_dataset: int = 500,
    object_key_prefix: str | None = None,
    require_label: bool = False,
) -> BuildStats:
    if not inspect(engine).has_table("datasets"):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")
        return BuildStats(dataset_rows=0, sample_count=0, skipped_count=0, output_path=str(output))

    storage = get_storage()
    db = SessionLocal()
    try:
        source_datasets = _fetch_source_datasets_with_prefix(db, object_key_prefix)
    finally:
        db.close()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    skipped_count = 0
    with output.open("w", encoding="utf-8") as writer:
        for source in source_datasets:
            artifact_key = source.preprocess_artifact_key
            if artifact_key is None:
                skipped_count += 1
                continue
            artifact = json.loads(storage.get_text(artifact_key))
            if not artifact.get("success", False):
                skipped_count += 1
                continue
            segments = artifact.get("segments", [])[: max(1, max_segments_per_dataset)]
            for segment in segments:
                if require_label and _normalized_label_from_segment(segment) is None:
                    skipped_count += 1
                    continue
                sequence = _to_feature_sequence(segment.get("trajectory", []))
                if len(sequence) < min_points:
                    skipped_count += 1
                    continue
                char_id = _char_id_from_segment(source, segment, len(sequence))
                sample = {
                    "char_id": char_id,
                    "style_id": 0,
                    "dataset_id": source.id,
                    "user_id": source.user_id,
                    "sequence": sequence,
                }
                writer.write(json.dumps(sample, ensure_ascii=False) + "\n")
                sample_count += 1

    return BuildStats(
        dataset_rows=len(source_datasets),
        sample_count=sample_count,
        skipped_count=skipped_count,
        output_path=str(output),
    )
