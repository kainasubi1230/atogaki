from io import BytesIO
import json

import numpy as np
from PIL import Image


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
    y_coords, x_coords = np.where(mask)
    if len(x_coords) == 0:
        return []
    order = np.argsort(y_coords * mask.shape[1] + x_coords)
    points: list[dict] = []
    stride = max(1, len(order) // 120)
    t = 0
    for idx in order[::stride]:
        points.append(
            {
                "x": int(x_coords[idx]),
                "y": int(y_coords[idx]),
                "t": t,
                "pen_state": "down",
                "width": 2,
            }
        )
        t += 1
    if points:
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

