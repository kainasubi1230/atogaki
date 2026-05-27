from io import BytesIO
import json

import numpy as np
from PIL import Image
from trainer.public_dataset import _image_to_sequence


LOW_CONTRAST = "LOW_CONTRAST"
NO_TEXT_DETECTED = "NO_TEXT_DETECTED"
TOO_FEW_SEGMENTS = "TOO_FEW_SEGMENTS"


def _binary_projection_groups(mask: np.ndarray, axis: int) -> list[tuple[int, int]]:
    projection = (mask.sum(axis=axis) > 0).astype(np.int32)
    groups: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(projection):
        if value == 1 and start is None:
            start = idx
        if value == 0 and start is not None:
            groups.append((start, idx))
            start = None
    if start is not None:
        groups.append((start, len(projection)))
    return groups


def _stroke_from_mask(mask: np.ndarray) -> list[dict]:
    if mask.ndim != 2:
        return []
    # Convert foreground mask into grayscale tile for robust path extraction.
    tile = np.where(mask, 0, 255).astype(np.uint8)
    seq = _image_to_sequence(tile, max_points=220, threshold=128, smooth_profile="default")
    if not seq:
        return []

    points: list[dict] = []
    x = 0.0
    y = 0.0
    t = 0
    for row in seq:
        if not isinstance(row, list) or len(row) < 4:
            continue
        x += float(row[0]) * 20.0
        y += float(row[1]) * 20.0
        pen_down = float(row[2]) > 0.5
        w = max(1, min(4, int(round(float(row[3]) * 4.0))))
        points.append(
            {
                "x": int(round(x)),
                "y": int(round(y)),
                "t": t,
                "pen_state": "down" if pen_down else "up",
                "width": w,
            }
        )
        t += 1
    if len(points) < 2:
        return []
    # Ensure explicit pen-up at stroke end.
    points[-1]["pen_state"] = "up"
    return points


def preprocess_scan(image_bytes: bytes) -> dict:
    image = Image.open(BytesIO(image_bytes)).convert("L")
    arr = np.array(image)
    contrast = float(arr.std())
    if contrast < 9.0:
        return {"success": False, "reason_code": LOW_CONTRAST}

    binary = arr < 200
    if float(binary.mean()) < 0.003:
        return {"success": False, "reason_code": NO_TEXT_DETECTED}

    lines = _binary_projection_groups(binary, axis=1)
    segments: list[dict] = []
    for y0, y1 in lines:
        line_mask = binary[y0:y1, :]
        chars = _binary_projection_groups(line_mask, axis=0)
        for x0, x1 in chars:
            char_mask = line_mask[:, x0:x1]
            if char_mask.shape[0] < 4 or char_mask.shape[1] < 4:
                continue
            stroke = _stroke_from_mask(char_mask)
            if not stroke:
                continue
            # Move local stroke coordinates back to full image coordinates.
            for p in stroke:
                p["x"] = int(p["x"]) + int(x0)
                p["y"] = int(p["y"]) + int(y0)
            segments.append(
                {
                    "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                    "trajectory": stroke,
                }
            )

    if len(segments) < 2:
        return {"success": False, "reason_code": TOO_FEW_SEGMENTS}

    return {
        "success": True,
        "reason_code": None,
        "segment_count": len(segments),
        "segments": segments,
    }


def serialize_preprocess(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False)

