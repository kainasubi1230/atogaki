from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from threading import Lock
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import and_
from sqlalchemy.orm import Session

from trainer.public_dataset import _image_to_sequence, _image_to_sequence_contours, _sequence_shape_iou
from trainerlib.kana_image import generate_kana_image_best
from trainerlib.model import generate_trajectory
from trainerlib.preprocess import preprocess_scan
from trainerlib.svg import text_to_svg, text_to_svg_readable, trajectory_to_svg

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
    TrajectoryUploadRequest,
    TrajectoryUploadResponse,
    StyleCoverageResponse,
    TrainStyleResponse,
)
from .security import create_access_token, hash_password, parse_access_token, verify_password
from .settings import settings
from .storage import get_storage
from .tasks import run_preprocess_job, run_train_lora_job

WATERMARK_TEXT = "AI生成（アクセシビリティ支援）"
HIRAGANA_TARGET = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
KATAKANA_TARGET = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
KANJI_CORE_TARGET = "日月火水木金土山川田天気学年人大小中上下左右先生今来行見話書読食飲休車電駅校友名本語文字漢"
TARGET_CHARS = HIRAGANA_TARGET + KATAKANA_TARGET + KANJI_CORE_TARGET
RUNTIME_DATASET_FILES = [
    Path("storage/base/base_dataset_runtime_master.jsonl"),
    Path("storage/base/base_dataset_hiragana_images_v2.jsonl"),
    Path("storage/base/base_dataset_katakana_handwritten_like_v8best.jsonl"),
    Path("storage/base/base_dataset_kanjivg_joyo_full_v5.jsonl"),
]
_RUNTIME_CHAR_BANK: dict[str, list[list[list[float]]]] | None = None
_RUNTIME_CHAR_BANK_LOCK = Lock()
_HIRAGANA_STROKE_RANGE: dict[str, tuple[int, int]] = {
    "い": (2, 3),
    "う": (2, 3),
    "え": (2, 4),
    "お": (3, 5),
    # Stabilize na-row by excluding over-fragmented trajectories.
    "な": (3, 5),
    "に": (2, 4),
    "ぬ": (1, 3),
    "ね": (1, 3),
    "の": (1, 2),
}
_HIRAGANA_NA_ROW = {"な", "に", "ぬ", "ね", "の"}

app = FastAPI(title="Accessibility Handwriting MVP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_origin_regex=settings.cors_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


def _get_shared_style(db: Session) -> StyleAdapter:
    if settings.shared_style_id <= 0:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="shared_style_not_configured")
    style = db.query(StyleAdapter).filter(StyleAdapter.id == settings.shared_style_id).first()
    if style is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="shared_style_not_found")
    if style.disabled or style.status != "ready" or not style.adapter_key:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="shared_style_not_ready")
    return style


def _split_label_chars(raw: str) -> list[str]:
    chars: list[str] = []
    for ch in raw:
        if ch.isspace():
            continue
        chars.append(ch)
    return chars


def _is_hiragana_char(ch: str) -> bool:
    if len(ch) != 1:
        return False
    code = ord(ch)
    return 0x3041 <= code <= 0x3096


def _is_katakana_char(ch: str) -> bool:
    if len(ch) != 1:
        return False
    code = ord(ch)
    return (
        0x30A1 <= code <= 0x30FA
        or 0x30FD <= code <= 0x30FF
        or 0x31F0 <= code <= 0x31FF
        or 0xFF66 <= code <= 0xFF9D
    )


def _is_kana_char(ch: str) -> bool:
    return _is_hiragana_char(ch) or _is_katakana_char(ch)


def _is_kanji_char(ch: str) -> bool:
    if len(ch) != 1:
        return False
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def _is_japanese_char(ch: str) -> bool:
    if _is_kana_char(ch) or _is_kanji_char(ch):
        return True
    # Common JP punctuation marks often appear in phrases.
    return ch in {"ー", "々", "〆", "ヶ", "、", "。", "・", "「", "」", "『", "』", "（", "）", "！", "？", "〜", "…"}


def _is_kana_text(text: str) -> bool:
    has_kana = False
    for ch in text:
        if ch.isspace():
            continue
        if _is_kana_char(ch):
            has_kana = True
            continue
        return False
    return has_kana


def _is_hiragana_text(text: str) -> bool:
    has_hiragana = False
    for ch in text:
        if ch.isspace():
            continue
        if _is_hiragana_char(ch):
            has_hiragana = True
            continue
        return False
    return has_hiragana


def _is_katakana_text(text: str) -> bool:
    has_katakana = False
    for ch in text:
        if ch.isspace():
            continue
        if _is_katakana_char(ch):
            has_katakana = True
            continue
        return False
    return has_katakana


def _is_japanese_text(text: str) -> bool:
    has_japanese = False
    for ch in text:
        if ch.isspace():
            continue
        if _is_japanese_char(ch):
            has_japanese = True
            continue
        return False
    return has_japanese


def _has_kana_coverage(text: str, exemplars_text: dict[str, object] | None) -> bool:
    if not isinstance(exemplars_text, dict):
        return False
    for ch in text:
        if ch.isspace():
            continue
        if not _is_kana_char(ch):
            return False
        bucket = exemplars_text.get(ch)
        if not isinstance(bucket, list) or len(bucket) == 0:
            return False
    return True


def _is_path_artifact(points: list[dict], text: str) -> bool:
    if not points:
        return True
    down = [p for p in points if p.get("pen_state") == "down"]
    if len(down) < max(6, len(text) * 4):
        return True
    xs = [int(p.get("x", 0)) for p in down]
    ys = [int(p.get("y", 0)) for p in down]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 0 or height <= 0:
        return True
    if width > max(1100, len(text) * 260) or height > 1300:
        return True
    # Guard for almost-horizontal "just a line" outputs.
    if height < 6 and width > 24:
        return True
    y_bins = {int(round(y / 3.0)) for y in ys}
    if len(y_bins) <= 2 and width > 36:
        return True
    # Detect rapid left-right / up-down jitter (zigzag artifacts).
    headings: list[float] = []
    for idx in range(1, len(points)):
        prev = points[idx - 1]
        cur = points[idx]
        if prev.get("pen_state") != "down" or cur.get("pen_state") != "down":
            continue
        dx = float(int(cur.get("x", 0)) - int(prev.get("x", 0)))
        dy = float(int(cur.get("y", 0)) - int(prev.get("y", 0)))
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        headings.append(math.atan2(dy, dx))
    if len(headings) >= 6:
        sharp = 0
        for idx in range(1, len(headings)):
            delta = headings[idx] - headings[idx - 1]
            while delta > math.pi:
                delta -= 2 * math.pi
            while delta < -math.pi:
                delta += 2 * math.pi
            if abs(delta) > 1.20:
                sharp += 1
        sharp_ratio = sharp / max(1, len(headings) - 1)
        # Hiragana should be smoother than katakana.
        zigzag_threshold = 0.46
        if _is_hiragana_text(text):
            zigzag_threshold = 0.30
        elif _is_katakana_text(text):
            zigzag_threshold = 0.36
        if sharp_ratio > zigzag_threshold:
            return True
    # Large discontinuities are a common failure mode for zigzag artifacts.
    # Compare only continuous pen-down segments; stroke boundaries (pen-up -> pen-down)
    # are expected to "jump".
    bad_jumps = 0
    for idx in range(1, len(points)):
        prev = points[idx - 1]
        cur = points[idx]
        if prev.get("pen_state") != "down" or cur.get("pen_state") != "down":
            continue
        dx = abs(int(cur.get("x", 0)) - int(prev.get("x", 0)))
        dy = abs(int(cur.get("y", 0)) - int(prev.get("y", 0)))
        if dx > 42 or dy > 42:
            bad_jumps += 1
            if bad_jumps >= 2:
                return True
    return False


def _is_degenerate_trajectory(points: list[dict], text: str) -> bool:
    if not points:
        return True
    down = [p for p in points if p.get("pen_state") == "down"]
    if len(down) < max(10, len(text) * 6):
        return True
    xs = [int(p.get("x", 0)) for p in down]
    ys = [int(p.get("y", 0)) for p in down]
    if any(abs(x) > 5000 for x in xs) or any(abs(y) > 5000 for y in ys):
        return True
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 0 or height <= 0:
        return True
    if width > max(900, len(text) * 280) or height > 1400:
        return True
    if width < max(18, len(text) * 8):
        return True
    # Very flat outputs appear as "just a line".
    if height < max(10, int(width * 0.14)):
        return True
    aspect = height / max(1, width)
    if aspect < 0.12 or aspect > 6.5:
        return True

    # Line-like outputs often have little vertical diversity.
    y_bins = {int(round(y / 4.0)) for y in ys}
    if len(y_bins) <= 3 and width > max(36, len(text) * 12):
        return True
    return False


def _split_down_strokes(points: list[dict]) -> list[list[tuple[int, int]]]:
    strokes: list[list[tuple[int, int]]] = []
    cur: list[tuple[int, int]] = []
    for p in points:
        if p.get("pen_state") == "down":
            cur.append((int(p.get("x", 0)), int(p.get("y", 0))))
            continue
        if cur:
            strokes.append(cur)
            cur = []
    if cur:
        strokes.append(cur)
    return strokes


def _trajectory_candidate_score(points: list[dict], text: str) -> float:
    if not points:
        return float("-inf")
    if _is_path_artifact(points, text) or _is_degenerate_trajectory(points, text):
        return float("-inf")
    down = [p for p in points if p.get("pen_state") == "down"]
    if len(down) < max(8, len(text) * 4):
        return float("-inf")
    xs = [int(p.get("x", 0)) for p in down]
    ys = [int(p.get("y", 0)) for p in down]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 0 or height <= 0:
        return float("-inf")

    strokes = _split_down_strokes(points)
    if not strokes:
        return float("-inf")
    tiny_strokes = sum(1 for s in strokes if len(s) <= 2)
    short_strokes = sum(1 for s in strokes if len(s) <= 3)
    stroke_count = len(strokes)
    y_bins = len({int(round(y / 2.0)) for y in ys})

    aspect = height / max(1.0, width)
    expected_aspect = 0.9 if _is_hiragana_text(text) else 0.7
    aspect_penalty = abs(aspect - expected_aspect) * 6.0

    return (
        min(24.0, float(len(down)) / 4.0)
        + min(12.0, float(width) / max(6.0, len(text) * 2.5))
        + min(12.0, float(height) / 2.2)
        + min(8.0, float(y_bins) / 2.0)
        - tiny_strokes * 5.0
        - short_strokes * 2.0
        - max(0, stroke_count - (len(text) * 6)) * 2.4
        - aspect_penalty
    )


def _is_usable_trajectory(points: list[dict], text: str) -> bool:
    score = _trajectory_candidate_score(points, text)
    if score == float("-inf"):
        return False
    strokes = _split_down_strokes(points)
    if not strokes:
        return False
    tiny_strokes = sum(1 for s in strokes if len(s) <= 2)
    if tiny_strokes > max(0, len(text) // 3):
        return False
    if len(strokes) < max(1, len(text)):
        return False
    return True


def _chaikin_smooth(points: list[tuple[float, float]], iterations: int = 2) -> list[tuple[float, float]]:
    out = list(points)
    if len(out) < 3:
        return out
    for _ in range(max(0, iterations)):
        if len(out) < 3:
            break
        nxt: list[tuple[float, float]] = [out[0]]
        for i in range(len(out) - 1):
            x0, y0 = out[i]
            x1, y1 = out[i + 1]
            q = (0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1)
            r = (0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1)
            nxt.append(q)
            nxt.append(r)
        nxt.append(out[-1])
        out = nxt
    return out


def _resample_polyline(points: list[tuple[float, float]], spacing: float = 1.2) -> list[tuple[float, float]]:
    if len(points) < 2:
        return list(points)
    out: list[tuple[float, float]] = [points[0]]
    carry = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        dx = x1 - x0
        dy = y1 - y0
        seg = math.hypot(dx, dy)
        if seg < 1e-6:
            continue
        ux = dx / seg
        uy = dy / seg
        dist = spacing - carry
        while dist <= seg:
            px = x0 + ux * dist
            py = y0 + uy * dist
            out.append((px, py))
            dist += spacing
        carry = max(0.0, seg - (dist - spacing))
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


def _strokes_to_bitmap(
    strokes: list[list[tuple[float, float, int]]],
    *,
    size: int = 64,
    pad: int = 6,
) -> list[int]:
    all_pts = [(x, y, w) for stroke in strokes for (x, y, w) in stroke]
    if len(all_pts) < 4:
        return [0] * (size * size)
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    scale = min((size - 2 * pad) / span_x, (size - 2 * pad) / span_y)
    off_x = (size - span_x * scale) * 0.5
    off_y = (size - span_y * scale) * 0.5

    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        mapped: list[tuple[float, float, int]] = []
        for x, y, w in stroke:
            px = off_x + (x - min_x) * scale
            py = off_y + (y - min_y) * scale
            mapped.append((px, py, w))
        for i in range(1, len(mapped)):
            x0, y0, w0 = mapped[i - 1]
            x1, y1, w1 = mapped[i]
            width = max(1, min(3, int(round((w0 + w1) * 0.5))))
            draw.line((x0, y0, x1, y1), fill=255, width=width)
    data = list(img.getdata())
    return [1 if v >= 32 else 0 for v in data]


def _bitmap_iou(a: list[int], b: list[int]) -> float:
    inter = 0
    union = 0
    for x, y in zip(a, b):
        if x or y:
            union += 1
            if x and y:
                inter += 1
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _strokes_shape_iou(
    a: list[list[tuple[float, float, int]]],
    b: list[list[tuple[float, float, int]]],
) -> float:
    return _bitmap_iou(_strokes_to_bitmap(a), _strokes_to_bitmap(b))


def _apply_runtime_variation(
    strokes: list[list[tuple[float, float, int]]],
    *,
    seed: str,
    ch: str,
) -> list[list[tuple[float, float, int]]]:
    if not strokes:
        return []
    all_pts = [(x, y) for stroke in strokes for (x, y, _w) in stroke]
    if len(all_pts) < 4:
        return strokes
    cx = sum(p[0] for p in all_pts) / float(len(all_pts))
    cy = sum(p[1] for p in all_pts) / float(len(all_pts))
    rng = np.random.default_rng(_stable_int_seed(seed) & 0xFFFFFFFF)

    # Keep shape stable; only add subtle handwritten variation.
    rot_deg = float(rng.uniform(-2.1, 2.1))
    scale_x = float(1.0 + rng.uniform(-0.028, 0.028))
    scale_y = float(1.0 + rng.uniform(-0.028, 0.028))
    shear = float(rng.uniform(-0.022, 0.022))
    shift_x = float(rng.uniform(-0.45, 0.45))
    shift_y = float(rng.uniform(-0.45, 0.45))
    if _is_kana_char(ch):
        rot_deg = float(rng.uniform(-0.80, 0.80))
        scale_x = float(1.0 + rng.uniform(-0.010, 0.010))
        scale_y = float(1.0 + rng.uniform(-0.010, 0.010))
        shear = float(rng.uniform(-0.006, 0.006))
        shift_x = float(rng.uniform(-0.16, 0.16))
        shift_y = float(rng.uniform(-0.16, 0.16))
    if ch in _HIRAGANA_NA_ROW:
        rot_deg = float(rng.uniform(-1.05, 1.05))
        scale_x = float(1.0 + rng.uniform(-0.014, 0.014))
        scale_y = float(1.0 + rng.uniform(-0.014, 0.014))
        shear = float(rng.uniform(-0.010, 0.010))
        shift_x = float(rng.uniform(-0.24, 0.24))
        shift_y = float(rng.uniform(-0.24, 0.24))
    if _is_kanji_char(ch):
        rot_deg *= 0.7
        scale_x = 1.0 + (scale_x - 1.0) * 0.6
        scale_y = 1.0 + (scale_y - 1.0) * 0.6
        shear *= 0.6
        shift_x *= 0.8
        shift_y *= 0.8

    rot = math.radians(rot_deg)
    cr = math.cos(rot)
    sr = math.sin(rot)

    varied: list[list[tuple[float, float, int]]] = []
    for stroke_idx, stroke in enumerate(strokes):
        if len(stroke) < 2:
            varied.append(stroke)
            continue
        amp = float(rng.uniform(0.06, 0.17))
        if _is_kanji_char(ch):
            amp *= 0.7
        if _is_kana_char(ch):
            amp = 0.0
        if ch in _HIRAGANA_NA_ROW:
            amp *= 0.42
        s_tx = float(rng.uniform(-0.18, 0.18))
        s_ty = float(rng.uniform(-0.18, 0.18))
        if ch in _HIRAGANA_NA_ROW:
            s_tx = float(rng.uniform(-0.09, 0.09))
            s_ty = float(rng.uniform(-0.09, 0.09))
        # Low-frequency bend: avoids digital-looking micro-wiggles.
        bend_center = float(rng.uniform(0.32, 0.68))
        bend_width = float(rng.uniform(0.42, 0.78))
        if ch in _HIRAGANA_NA_ROW:
            bend_center = float(rng.uniform(0.40, 0.60))
            bend_width = float(rng.uniform(0.62, 0.88))
        bend_sign = -1.0 if (stroke_idx % 2 == 0 and rng.uniform() < 0.5) else 1.0

        out_stroke: list[tuple[float, float, int]] = []
        n = len(stroke)
        for i, (x, y, w) in enumerate(stroke):
            dx = x - cx
            dy = y - cy
            sx = dx * scale_x + dy * shear
            sy = dy * scale_y
            rx = sx * cr - sy * sr
            ry = sx * sr + sy * cr
            px = rx + cx + shift_x + s_tx
            py = ry + cy + shift_y + s_ty

            if i == 0:
                txv = stroke[1][0] - stroke[0][0]
                tyv = stroke[1][1] - stroke[0][1]
            elif i == n - 1:
                txv = stroke[-1][0] - stroke[-2][0]
                tyv = stroke[-1][1] - stroke[-2][1]
            else:
                txv = stroke[i + 1][0] - stroke[i - 1][0]
                tyv = stroke[i + 1][1] - stroke[i - 1][1]
            tn = math.hypot(txv, tyv)
            if tn > 1e-6 and amp > 1e-6:
                nx = -tyv / tn
                ny = txv / tn
                t = float(i) / float(max(1, n - 1))
                envelope = math.sin(math.pi * t) ** 1.20
                d = abs(t - bend_center) / max(1e-6, bend_width)
                local = max(0.0, 1.0 - d * d)
                bend = bend_sign * amp * envelope * local
                px += nx * bend
                py += ny * bend

            # Slight endpoint anchor keeps shapes readable.
            t2 = float(i) / float(max(1, n - 1))
            endpoint_pull = max(0.0, 0.24 - abs(t2 - 0.5)) * (0.58 if ch in _HIRAGANA_NA_ROW else 0.42)
            px = px * (1.0 - endpoint_pull) + x * endpoint_pull
            py = py * (1.0 - endpoint_pull) + y * endpoint_pull

            out_stroke.append((px, py, w))
        varied.append(out_stroke)
    return varied


def _extract_strokes_from_local(local: list[tuple[float, float, str, int]]) -> list[list[tuple[float, float, int]]]:
    strokes: list[list[tuple[float, float, int]]] = []
    cur: list[tuple[float, float, int]] = []
    for lx, ly, pen, width in local:
        if pen == "down":
            cur.append((lx, ly, width))
            continue
        if len(cur) >= 2:
            strokes.append(cur)
        cur = []
    if len(cur) >= 2:
        strokes.append(cur)
    return strokes


def _sequence_to_strokes(seq: list[list[float]]) -> list[list[tuple[float, float, int]]]:
    if not seq:
        return []
    local: list[tuple[float, float, str, int]] = []
    lx = 0.0
    ly = 0.0
    for rowv in seq:
        if not isinstance(rowv, list) or len(rowv) < 4:
            continue
        lx += float(rowv[0]) * 20.0
        ly += float(rowv[1]) * 20.0
        pen = "down" if float(rowv[2]) > 0.5 else "up"
        widthv = max(1, min(4, int(round(float(rowv[3]) * 4.0))))
        local.append((lx, ly, pen, widthv))
    return _extract_strokes_from_local(local)


def _stable_int_seed(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _sequence_step_jump_penalty(seq: list[list[float]]) -> tuple[int, int]:
    big_any = 0
    big_down = 0
    for row in seq:
        if not isinstance(row, list) or len(row) < 3:
            continue
        dx = abs(float(row[0]))
        dy = abs(float(row[1]))
        if dx > 1.8 or dy > 1.8:
            big_any += 1
            if float(row[2]) > 0.5:
                big_down += 1
    return big_any, big_down


def _char_sequence_quality(seq: list[list[float]], ch: str, source: str = "") -> float:
    if not isinstance(seq, list) or len(seq) < 8:
        return float("-inf")
    big_any, big_down = _sequence_step_jump_penalty(seq)
    # Exclude broken trajectories with huge jumps.
    if big_any > 6 or big_down > 2:
        return float("-inf")
    strokes = _sequence_to_strokes(seq)
    if not strokes:
        return float("-inf")
    all_pts = [(x, y) for stroke in strokes for (x, y, _w) in stroke]
    if len(all_pts) < 10:
        return float("-inf")
    stroke_count = len(strokes)
    point_count = len(all_pts)
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    if span_x <= 1e-6 or span_y <= 1e-6:
        return float("-inf")
    if _is_kana_char(ch):
        if stroke_count > 8:
            return float("-inf")
        if point_count > 170:
            return float("-inf")
    if _is_hiragana_char(ch):
        stroke_range = _HIRAGANA_STROKE_RANGE.get(ch)
        if stroke_range is not None:
            lo, hi = stroke_range
            if stroke_count < lo or stroke_count > hi:
                return float("-inf")
    redraw_count = sum(_stroke_redraw_count(s) for s in strokes)
    turn_penalty = sum(_stroke_turn_penalty(s) for s in strokes) / max(1, len(strokes))
    total_len = sum(_stroke_path_len(s) for s in strokes)
    length_ratio = total_len / max(1.0, span_x + span_y)
    short_frag = sum(1 for s in strokes if len(s) <= 3)
    tiny_frag = sum(1 for s in strokes if _stroke_path_len(s) < 1.8)
    aspect = span_y / max(1.0, span_x)
    target_aspect = 0.92 if _is_hiragana_char(ch) else 0.80 if _is_katakana_char(ch) else 1.0
    aspect_penalty = abs(aspect - target_aspect) * 18.0
    source_bonus = 0.0
    if source == "hiragana-dataset-github":
        source_bonus = 18.0
    elif source == "k49":
        source_bonus = 8.0
    elif source.startswith("katakana-generated-handlike"):
        source_bonus = 14.0
    elif source.startswith("kanjivg"):
        source_bonus = 12.0
    # For these characters, k49 often yields over-simplified or broken forms.
    if ch in {"い", "う", "え", "お"} and source == "k49":
        source_bonus -= 18.0
    preferred_strokes = 2.8 if _is_kana_char(ch) else 6.0
    stroke_penalty = abs(stroke_count - preferred_strokes) * (5.5 if _is_kana_char(ch) else 2.5)
    return (
        min(180.0, total_len * 0.55)
        + min(120.0, span_x + span_y)
        + source_bonus
        - redraw_count * 6.5
        - turn_penalty * 64.0
        - short_frag * 8.0
        - tiny_frag * 10.0
        - stroke_penalty
        - max(0.0, length_ratio - 4.2) * 42.0
        - max(0, point_count - 110) * 1.4
        - aspect_penalty
        - big_any * 8.0
        - big_down * 14.0
    )


def _load_runtime_char_bank() -> dict[str, list[list[list[float]]]]:
    ranked: dict[str, list[tuple[float, list[list[float]]]]] = {}
    for path in RUNTIME_DATASET_FILES:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = line.strip()
                    if not row:
                        continue
                    try:
                        payload = json.loads(row)
                    except Exception:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    meta = payload.get("meta")
                    ch = meta.get("char") if isinstance(meta, dict) else None
                    if not isinstance(ch, str) or len(ch) != 1:
                        continue
                    seq = payload.get("sequence")
                    if not isinstance(seq, list):
                        continue
                    source = str(meta.get("source", "")) if isinstance(meta, dict) else ""
                    score = _char_sequence_quality(seq, ch, source)
                    if score == float("-inf"):
                        continue
                    bucket = ranked.setdefault(ch, [])
                    bucket.append((score, seq))
        except Exception:
            continue

    bank: dict[str, list[list[list[float]]]] = {}
    for ch, rows in ranked.items():
        rows.sort(key=lambda it: it[0], reverse=True)
        # Keep only top stable candidates per character to avoid child-like/noisy samples.
        bank[ch] = [seq for (_score, seq) in rows[:20]]
    return bank


def _get_runtime_char_bank() -> dict[str, list[list[list[float]]]]:
    global _RUNTIME_CHAR_BANK
    if _RUNTIME_CHAR_BANK is not None:
        return _RUNTIME_CHAR_BANK
    with _RUNTIME_CHAR_BANK_LOCK:
        if _RUNTIME_CHAR_BANK is None:
            _RUNTIME_CHAR_BANK = _load_runtime_char_bank()
    return _RUNTIME_CHAR_BANK or {}


def _select_runtime_char_sequence(ch: str, *, seed: str) -> list[list[float]] | None:
    bank = _get_runtime_char_bank()
    candidates = bank.get(ch)
    if not candidates:
        return None
    if ch in {"い", "う", "え", "お"}:
        top_n = min(3, len(candidates))
    else:
        top_n = min(6, len(candidates))
    idx = _stable_int_seed(seed) % top_n
    return candidates[idx]


def _score_runtime_char_strokes(strokes: list[list[tuple[float, float, int]]], ch: str) -> float:
    if not strokes:
        return float("-inf")
    all_pts = [(x, y) for stroke in strokes for (x, y, _w) in stroke]
    if len(all_pts) < 8:
        return float("-inf")
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    if span_x <= 1e-6 or span_y <= 1e-6:
        return float("-inf")
    total_len = sum(_stroke_path_len(s) for s in strokes)
    redraw_count = sum(_stroke_redraw_count(s) for s in strokes)
    turn_penalty = sum(_stroke_turn_penalty(s) for s in strokes) / max(1, len(strokes))
    stroke_count = len(strokes)
    point_count = len(all_pts)
    if _is_hiragana_char(ch):
        stroke_range = _HIRAGANA_STROKE_RANGE.get(ch)
        if stroke_range is not None:
            lo, hi = stroke_range
            if stroke_count < lo or stroke_count > hi:
                return float("-inf")
    length_ratio = total_len / max(1.0, span_x + span_y)
    expected_strokes = 2.8 if _is_kana_char(ch) else 6.0
    return (
        min(120.0, total_len * 0.42)
        + min(90.0, span_x + span_y)
        - abs(stroke_count - expected_strokes) * (6.0 if _is_kana_char(ch) else 3.0)
        - redraw_count * 8.0
        - turn_penalty * 72.0
        - max(0.0, length_ratio - 4.0) * 50.0
        - max(0, point_count - 96) * 1.8
    )


def _generate_hiragana_runtime_trajectory(text: str) -> list[dict]:
    points: list[dict] = []
    cursor_x = 24.0
    baseline_y = 52.0
    t = 0

    for idx, ch in enumerate(text):
        if ch == "\n":
            cursor_x = 24.0
            baseline_y += 42.0
            continue
        if ch.isspace():
            cursor_x += 12.0
            continue
        if not _is_hiragana_char(ch):
            return []

        img = generate_kana_image_best(
            ch,
            style_seed=f"runtime-hira:{text}:{idx}",
            size=192,
            trials=8,
        )
        arr = np.array(img.convert("L"), dtype=np.uint8)
        seq = _image_to_sequence(arr, max_points=180, threshold=128, smooth_profile="default")
        if not seq:
            return []

        local: list[tuple[float, float, str, int]] = []
        lx = 0.0
        ly = 0.0
        for row in seq:
            if not isinstance(row, list) or len(row) < 4:
                continue
            lx += float(row[0]) * 20.0
            ly += float(row[1]) * 20.0
            pen = "down" if float(row[2]) > 0.5 else "up"
            width = max(1, min(4, int(round(float(row[3]) * 4.0))))
            local.append((lx, ly, pen, width))
        if len(local) < 8:
            return []

        strokes = _extract_strokes_from_local(local)
        if not strokes:
            return []

        all_pts = [(x, y) for stroke in strokes for (x, y, _w) in stroke]
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        span_x = max(1e-6, max_x - min_x)
        span_y = max(1e-6, max_y - min_y)
        target_w = 19.0
        target_h = 30.0
        sx = max(0.08, min(3.0, target_w / span_x))
        sy = max(0.08, min(3.0, target_h / span_y))
        char_w = span_x * sx
        base_y = baseline_y + (target_h - span_y * sy) * 0.5

        for stroke in strokes:
            raw_xy = [(x, y) for (x, y, _w) in stroke]
            smoothed_xy = _chaikin_smooth(raw_xy, iterations=2)
            sampled_xy = _resample_polyline(smoothed_xy, spacing=1.15)
            if len(sampled_xy) < 2:
                continue
            width_values = [w for (_x, _y, w) in stroke]
            stroke_width = max(1, min(4, int(round(sum(width_values) / max(1, len(width_values))))))
            for x_raw, y_raw in sampled_xy:
                x = cursor_x + (x_raw - min_x) * sx
                y = base_y + (y_raw - min_y) * sy
                points.append(
                    {
                        "x": round(x, 3),
                        "y": round(y, 3),
                        "t": t,
                        "pen_state": "down",
                        "width": stroke_width,
                    }
                )
                t += 1
            # explicit stroke end
            points.append(
                {
                    "x": round(cursor_x + (sampled_xy[-1][0] - min_x) * sx, 3),
                    "y": round(base_y + (sampled_xy[-1][1] - min_y) * sy, 3),
                    "t": t,
                    "pen_state": "up",
                    "width": stroke_width,
                }
            )
            t += 1
        cursor_x += char_w + 6.0
    return points


def _is_pure_kana_text(text: str) -> bool:
    has_kana = False
    for ch in text:
        if ch == "\n" or ch.isspace():
            continue
        if not _is_kana_char(ch):
            return False
        has_kana = True
    return has_kana


def _render_runtime_char_image(ch: str, *, seed: str, size: int = 136) -> Image.Image | None:
    if _is_kana_char(ch):
        # Use dataset-driven kana image as-is (no post normalization).
        return generate_kana_image_best(
            ch,
            style_seed=seed,
            size=size,
            trials=10,
        ).convert("L")
    elif not _is_japanese_char(ch):
        return None
    else:
        canvas = Image.new("L", (size, size), 255)
        draw = ImageDraw.Draw(canvas)
        font_candidates = [
            "storage/fonts/KleeOne-Regular.ttf",
            "storage/fonts/NotoSansJP-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        font: ImageFont.FreeTypeFont | None = None
        for font_path in font_candidates:
            p = Path(font_path)
            if not p.exists():
                continue
            try:
                font = ImageFont.truetype(str(p), int(size * 0.72))
                break
            except Exception:
                continue
        if font is None:
            return None
        bbox = draw.textbbox((0, 0), ch, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (size - tw) / 2 - bbox[0]
        y = (size - th) / 2 - bbox[1]
        draw.text((x, y), ch, fill=0, font=font)
        return canvas


def _strokes_from_runtime_image(img: Image.Image) -> list[list[tuple[float, float, int]]]:
    arr = np.array(img.convert("L"), dtype=np.uint8)
    dark_mask = arr < 128
    bright_mask = arr > 128
    ref_mask = dark_mask if int(dark_mask.sum()) <= int(bright_mask.sum()) else bright_mask
    candidates: list[tuple[str, list[list[float]]]] = []
    for thr in (128, 142, 156):
        candidates.append(
            (
                "centerline",
                _image_to_sequence(arr, max_points=320, threshold=thr, smooth_profile="default"),
            )
        )
    for thr in (128, 150):
        candidates.append(
            (
                "contour",
                _image_to_sequence_contours(arr, max_points=360, threshold=thr, smooth_profile="default"),
            )
        )

    best_strokes: list[list[tuple[float, float, int]]] = []
    best_score = float("-inf")
    for mode, seq in candidates:
        if not seq:
            continue
        strokes = _sequence_to_strokes(seq)
        if not strokes:
            continue
        quality = _score_strokes_quality(strokes)
        if quality == float("-inf"):
            continue
        iou = _sequence_shape_iou(seq, ref_mask)
        # Prioritize shape fidelity to source image, then stroke continuity.
        score = (iou * 160.0) + quality
        # Contour mode can double-trace strokes; prefer centerline when similar quality.
        if mode == "contour":
            score -= 26.0
        if score > best_score:
            best_score = score
            best_strokes = strokes
    if best_strokes:
        return best_strokes

    # Last-resort extraction: prefer "something readable" over dropping the char.
    for thr in (120, 136, 152, 168):
        seq = _image_to_sequence(arr, max_points=520, threshold=thr, smooth_profile="default")
        if not seq:
            continue
        strokes = _merge_fragmented_strokes(_sequence_to_strokes(seq))
        if strokes:
            return strokes
    for thr in (128, 150, 172):
        seq = _image_to_sequence_contours(arr, max_points=560, threshold=thr, smooth_profile="default")
        if not seq:
            continue
        strokes = _merge_fragmented_strokes(_sequence_to_strokes(seq))
        if strokes:
            return strokes
    return []


def _merge_fragmented_strokes(strokes: list[list[tuple[float, float, int]]]) -> list[list[tuple[float, float, int]]]:
    if not strokes:
        return []
    merged: list[list[tuple[float, float, int]]] = []
    for stroke in strokes:
        if not stroke:
            continue
        if not merged:
            merged.append(list(stroke))
            continue
        prev = merged[-1]
        dx = float(stroke[0][0] - prev[-1][0])
        dy = float(stroke[0][1] - prev[-1][1])
        dist = math.hypot(dx, dy)
        can_merge = dist <= 2.8
        if can_merge and len(prev) >= 2 and len(stroke) >= 2:
            pvx = float(prev[-1][0] - prev[-2][0])
            pvy = float(prev[-1][1] - prev[-2][1])
            nvx = float(stroke[1][0] - stroke[0][0])
            nvy = float(stroke[1][1] - stroke[0][1])
            plen = math.hypot(pvx, pvy)
            nlen = math.hypot(nvx, nvy)
            if plen > 1e-6 and nlen > 1e-6:
                dot = (pvx * nvx + pvy * nvy) / (plen * nlen)
                # Avoid connecting strokes that face opposite directions.
                if dot < 0.10:
                    can_merge = False
        if can_merge:
            bridge_w = int(round((prev[-1][2] + stroke[0][2]) * 0.5))
            prev.append((stroke[0][0], stroke[0][1], max(1, min(4, bridge_w))))
            prev.extend(stroke[1:])
        else:
            merged.append(list(stroke))
    cleaned: list[list[tuple[float, float, int]]] = []
    for stroke in merged:
        if len(stroke) < 3:
            continue
        if _stroke_path_len(stroke) < 2.0:
            continue
        cleaned.append(stroke)
    return cleaned


def _stroke_path_len(stroke: list[tuple[float, float, int]]) -> float:
    if len(stroke) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(stroke)):
        total += math.hypot(stroke[i][0] - stroke[i - 1][0], stroke[i][1] - stroke[i - 1][1])
    return total


def _stroke_redraw_count(stroke: list[tuple[float, float, int]]) -> int:
    # Count revisits to already drawn cells (ignoring immediate neighbors).
    if len(stroke) < 4:
        return 0
    seen: dict[tuple[int, int], int] = {}
    redraw = 0
    for idx, (x, y, _w) in enumerate(stroke):
        cell = (int(round(x)), int(round(y)))
        prev_idx = seen.get(cell)
        if prev_idx is not None and (idx - prev_idx) > 2:
            redraw += 1
        seen[cell] = idx
    return redraw


def _stroke_turn_penalty(stroke: list[tuple[float, float, int]]) -> float:
    if len(stroke) < 5:
        return 0.0
    sharp = 0
    total = 0
    for i in range(2, len(stroke)):
        x0, y0, _ = stroke[i - 2]
        x1, y1, _ = stroke[i - 1]
        x2, y2, _ = stroke[i]
        v1x = x1 - x0
        v1y = y1 - y0
        v2x = x2 - x1
        v2y = y2 - y1
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        dot = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
        ang = math.acos(dot)
        total += 1
        if ang > 1.18:
            sharp += 1
    if total == 0:
        return 0.0
    return float(sharp) / float(total)


def _score_strokes_quality(strokes: list[list[tuple[float, float, int]]]) -> float:
    if not strokes:
        return float("-inf")
    all_pts = [(x, y) for stroke in strokes for (x, y, _w) in stroke]
    if len(all_pts) < 10:
        return float("-inf")
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    if span_x <= 1e-6 or span_y <= 1e-6:
        return float("-inf")
    short_frag = sum(1 for s in strokes if len(s) <= 3)
    tiny_frag = sum(1 for s in strokes if _stroke_path_len(s) < 1.8)
    total_len = sum(_stroke_path_len(s) for s in strokes)
    redraw_count = sum(_stroke_redraw_count(s) for s in strokes)
    turn_penalty = sum(_stroke_turn_penalty(s) for s in strokes) / max(1, len(strokes))
    density = len(all_pts) / max(1.0, span_x + span_y)
    return (
        min(260.0, total_len)
        + min(80.0, float(span_x + span_y))
        - short_frag * 9.0
        - tiny_frag * 12.0
        - redraw_count * 5.4
        - turn_penalty * 48.0
        - max(0.0, density - 2.8) * 22.0
    )


def _runtime_kana_image_svg(text: str, watermark_text: str) -> str:
    lines = text.splitlines() or [text]
    if not lines:
        lines = [""]

    pad_x = 10
    pad_y = 10
    char_w = 40
    char_h = 50
    line_gap = 8
    max_len = max((len(line) for line in lines), default=1)
    width = max(180, pad_x * 2 + max_len * char_w + 10)
    height = max(120, pad_y * 2 + len(lines) * (char_h + line_gap) + 20)
    # Per-request nonce for random handwriting variation.
    render_nonce = uuid.uuid4().hex

    stroke_paths: list[tuple[str, float]] = []
    for row, line in enumerate(lines):
        x = pad_x
        y = pad_y + row * (char_h + line_gap)
        for col, ch in enumerate(line):
            if ch.isspace():
                x += int(char_w * 0.55)
                continue
            attempt_rows: list[tuple[float, list[list[tuple[float, float, int]]]]] = []
            for attempt in range(8):
                # Keep base character shape deterministic for stability.
                seed = f"runtime-ds-base:v1:{text}:{row}:{col}:{ch}:{attempt}"
                seq = _select_runtime_char_sequence(ch, seed=seed)
                if seq is not None:
                    strokes = _merge_fragmented_strokes(_sequence_to_strokes(seq))
                else:
                    # Fallback only when dataset trajectory for char is unavailable.
                    img = _render_runtime_char_image(ch, seed=seed, size=160)
                    if img is None:
                        continue
                    strokes = _merge_fragmented_strokes(_strokes_from_runtime_image(img))
                score = _score_runtime_char_strokes(strokes, ch)
                if score == float("-inf"):
                    continue
                attempt_rows.append((score, strokes))

            if not attempt_rows:
                # Character-level rescue: do not silently drop unsupported/bad chars.
                rescue_seed = f"runtime-ds-rescue:v1:{text}:{row}:{col}:{ch}"
                rescue_img = _render_runtime_char_image(ch, seed=rescue_seed, size=172)
                if rescue_img is not None:
                    rescue_strokes = _strokes_from_runtime_image(rescue_img)
                    if rescue_strokes:
                        rescue_score = _score_runtime_char_strokes(rescue_strokes, ch)
                        if rescue_score == float("-inf"):
                            rescue_score = 0.0
                        attempt_rows.append((rescue_score, rescue_strokes))
                if not attempt_rows:
                    x += char_w
                    continue

            attempt_rows.sort(key=lambda it: it[0], reverse=True)
            best_score = attempt_rows[0][0]
            # Always keep best base shape; randomness is applied only as a light variation.
            best_strokes = attempt_rows[0][1]

            # Generate several random variants, then keep only high-quality ones.
            variant_pool: list[list[list[tuple[float, float, int]]]] = [best_strokes]
            for v_idx in range(6):
                varied = _apply_runtime_variation(
                    best_strokes,
                    seed=f"runtime-var:v2:{render_nonce}:{text}:{row}:{col}:{ch}:{v_idx}",
                    ch=ch,
                )
                v_score = _score_runtime_char_strokes(varied, ch) if varied else float("-inf")
                if v_score < (best_score - (8.0 if _is_kana_char(ch) else 20.0)):
                    continue
                shape_iou = _strokes_shape_iou(best_strokes, varied)
                min_iou = 0.93 if _is_kana_char(ch) else 0.72
                if shape_iou < min_iou:
                    continue
                variant_pool.append(varied)
            pick_seed = _stable_int_seed(f"runtime-variant-pick:v1:{render_nonce}:{text}:{row}:{col}:{ch}")
            draw_strokes = variant_pool[pick_seed % len(variant_pool)]
            if not draw_strokes:
                x += char_w
                continue

            all_pts = [(sx, sy) for stroke in draw_strokes for (sx, sy, _w) in stroke]
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)
            span_x = max(1e-6, max_x - min_x)
            span_y = max(1e-6, max_y - min_y)
            scale = min((char_w - 4) / span_x, (char_h - 4) / span_y)
            scale = max(0.02, min(3.2, scale))
            off_x = x + (char_w - span_x * scale) * 0.5
            off_y = y + (char_h - span_y * scale) * 0.5

            for stroke_idx, stroke in enumerate(draw_strokes):
                raw_xy = [(px, py) for (px, py, _w) in stroke]
                if _is_kana_char(ch):
                    sampled_xy = _resample_polyline(raw_xy, spacing=1.32)
                else:
                    # Avoid overly uniform digital roundness: keep some natural angularity.
                    smooth_iter = 2 if len(raw_xy) >= 16 else 1
                    smoothed_xy = _chaikin_smooth(raw_xy, iterations=smooth_iter)
                    sampled_xy = _resample_polyline(smoothed_xy, spacing=1.18)
                if len(sampled_xy) < 2:
                    continue
                seg_parts: list[str] = []
                first_x = off_x + (sampled_xy[0][0] - min_x) * scale
                first_y = off_y + (sampled_xy[0][1] - min_y) * scale
                seg_parts.append(f"M{first_x:.2f},{first_y:.2f}")
                if _is_kana_char(ch):
                    for idx2 in range(1, len(sampled_xy)):
                        lx = off_x + (sampled_xy[idx2][0] - min_x) * scale
                        ly = off_y + (sampled_xy[idx2][1] - min_y) * scale
                        seg_parts.append(f"L{lx:.2f},{ly:.2f}")
                else:
                    for idx2 in range(1, len(sampled_xy) - 1):
                        cx, cy = sampled_xy[idx2]
                        nx, ny = sampled_xy[idx2 + 1]
                        ccx = off_x + (cx - min_x) * scale
                        ccy = off_y + (cy - min_y) * scale
                        mx = off_x + ((cx + nx) * 0.5 - min_x) * scale
                        my = off_y + ((cy + ny) * 0.5 - min_y) * scale
                        seg_parts.append(f"Q{ccx:.2f},{ccy:.2f} {mx:.2f},{my:.2f}")
                    last_x = off_x + (sampled_xy[-1][0] - min_x) * scale
                    last_y = off_y + (sampled_xy[-1][1] - min_y) * scale
                    seg_parts.append(f"L{last_x:.2f},{last_y:.2f}")
                width_values = [w for (_x, _y, w) in stroke]
                mean_w = sum(width_values) / max(1, len(width_values))
                width_seed = _stable_int_seed(f"runtime-sw:v1:{render_nonce}:{text}:{row}:{col}:{ch}:{stroke_idx}")
                width_jitter = ((width_seed % 1000) / 1000.0 - 0.5) * 0.14
                stroke_w = max(0.78, min(1.35, 0.78 + mean_w * 0.17 + width_jitter))
                stroke_paths.append((" ".join(seg_parts), stroke_w))
            x += char_w

    watermark_y = height - 14
    path_svg = "".join(
        f"<path d='{d}' stroke='#1f1f1f' fill='none' stroke-width='{sw:.2f}' "
        "stroke-linecap='round' stroke-linejoin='round'/>"
        for (d, sw) in stroke_paths
    )
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
        "<rect width='100%' height='100%' fill='white'/>"
        f"{path_svg}"
        f"<text x='12' y='{watermark_y}' fill='#c62828' font-size='14'>{WATERMARK_TEXT}</text>"
        "</svg>"
    )


def _chars_from_adapter_payload(adapter_payload: dict[str, object] | None) -> set[str]:
    if not isinstance(adapter_payload, dict):
        return set()
    coverage_chars = adapter_payload.get("coverage_chars")
    if isinstance(coverage_chars, list):
        out_cov: set[str] = set()
        for ch in coverage_chars:
            if isinstance(ch, str) and len(ch) == 1:
                out_cov.add(ch)
        if out_cov:
            return out_cov
    exemplars_text = adapter_payload.get("user_char_exemplars_text")
    if not isinstance(exemplars_text, dict):
        return set()
    out: set[str] = set()
    for key, value in exemplars_text.items():
        if not isinstance(key, str) or len(key) != 1:
            continue
        if isinstance(value, list) and value:
            out.add(key)
    return out


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
    label: str | None = Form(default=None),
    labels: str | None = Form(default=None),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DatasetUploadResponse:
    key = f"scans/u{user.id}/{uuid.uuid4().hex}-{file.filename}"
    payload = await file.read()
    storage = get_storage()
    storage.put_bytes(key, payload, file.content_type or "application/octet-stream")

    normalized_label = (label or "").strip()
    normalized_labels = _split_label_chars((labels or "").strip())
    dataset = Dataset(user_id=user.id, object_key=key, consent=consent, active=True)
    segment_count: int | None = None
    labeled_segment_count: int | None = None
    if normalized_label or normalized_labels:
        result = preprocess_scan(payload)
        if result.get("success"):
            segments = result.get("segments", [])
            segment_count = len(segments)
            labeled_segment_count = 0
            if normalized_labels:
                for idx, segment in enumerate(segments):
                    if not isinstance(segment, dict):
                        continue
                    if idx >= len(normalized_labels):
                        continue
                    segment["label"] = normalized_labels[idx]
                    labeled_segment_count += 1
            elif normalized_label:
                for segment in segments:
                    if isinstance(segment, dict):
                        segment["label"] = normalized_label
                        labeled_segment_count += 1
            result["segment_count"] = len(segments)
            result["labeled_segment_count"] = labeled_segment_count
            result["label_mode"] = "multi" if normalized_labels else "single"
        artifact_key = f"preprocessed/u{user.id}/{uuid.uuid4().hex}.json"
        storage.put_text(artifact_key, dumps(result), content_type="application/json")
        dataset.preprocess_artifact_key = artifact_key
        if result.get("success"):
            dataset.preprocess_status = "done"
            dataset.preprocess_error_code = None
        else:
            dataset.preprocess_status = "failed"
            dataset.preprocess_error_code = str(result.get("reason_code") or "PREPROCESS_FAILED")

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    log_event(
        db,
        path="/datasets/upload-scan",
        method="POST",
        event_type="dataset_upload",
        detail={
            "dataset_id": dataset.id,
            "consent": consent,
            "label": normalized_label or None,
            "labels_count": len(normalized_labels) if normalized_labels else None,
            "segment_count": segment_count,
            "labeled_segment_count": labeled_segment_count,
            "preprocess_status": dataset.preprocess_status,
        },
        user_id=user.id,
        status_code=200,
    )
    return DatasetUploadResponse(
        dataset_id=dataset.id,
        user_id=user.id,
        consent=consent,
        preprocess_status=dataset.preprocess_status,
        preprocess_error_code=dataset.preprocess_error_code,
        segment_count=segment_count,
        labeled_segment_count=labeled_segment_count,
    )


@app.post("/datasets/upload-trajectory", response_model=TrajectoryUploadResponse)
def upload_trajectory(
    payload: TrajectoryUploadRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TrajectoryUploadResponse:
    label = payload.label.strip()
    if not label:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="label_required")

    normalized_points: list[dict] = []
    for idx, p in enumerate(payload.points):
        x = float(p.x)
        y = float(p.y)
        t = int(p.t) if p.t >= 0 else idx
        width = float(p.width)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(width)):
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="invalid_point_values")
        normalized_points.append(
            {
                "x": int(round(x)),
                "y": int(round(y)),
                "t": t,
                "pen_state": p.pen_state,
                "width": width,
            }
        )

    normalized_points.sort(key=lambda p: int(p["t"]))
    if normalized_points[-1]["pen_state"] != "up":
        normalized_points[-1]["pen_state"] = "up"
    down_points = [p for p in normalized_points if p["pen_state"] == "down"]
    if not down_points:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="no_pen_down_points")
    if len(normalized_points) < 16 or len(down_points) < 12:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="trajectory_too_short")

    xs = [int(p["x"]) for p in normalized_points]
    ys = [int(p["y"]) for p in normalized_points]
    bbox = {"x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)}
    if (bbox["x1"] - bbox["x0"]) < 10 and (bbox["y1"] - bbox["y0"]) < 10:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="trajectory_too_small")

    storage = get_storage()
    raw_key = f"trajectories/u{user.id}/{uuid.uuid4().hex}.json"
    artifact_key = f"preprocessed/u{user.id}/{uuid.uuid4().hex}.json"

    raw_payload = {
        "user_id": user.id,
        "label": label,
        "point_count": len(normalized_points),
        "points": normalized_points,
    }
    storage.put_text(raw_key, dumps(raw_payload), content_type="application/json")

    artifact_payload = {
        "success": True,
        "reason_code": None,
        "segment_count": 1,
        "segments": [
            {
                "bbox": bbox,
                "trajectory": normalized_points,
                "label": label,
            }
        ],
    }
    storage.put_text(artifact_key, dumps(artifact_payload), content_type="application/json")

    dataset = Dataset(
        user_id=user.id,
        object_key=raw_key,
        consent=payload.consent,
        active=True,
        preprocess_status="done",
        preprocess_artifact_key=artifact_key,
        preprocess_error_code=None,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    log_event(
        db,
        path="/datasets/upload-trajectory",
        method="POST",
        event_type="trajectory_upload",
        detail={"dataset_id": dataset.id, "point_count": len(normalized_points), "consent": payload.consent},
        user_id=user.id,
        status_code=200,
    )
    return TrajectoryUploadResponse(
        dataset_id=dataset.id,
        user_id=user.id,
        consent=payload.consent,
        point_count=len(normalized_points),
        artifact_key=artifact_key,
    )


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
    if settings.inference_only:
        shared_style = _get_shared_style(db)
        return TrainStyleResponse(style_id=shared_style.id, job_id="inference-only", status="ready")

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


@app.get("/styles/{style_id}/coverage", response_model=StyleCoverageResponse)
def get_style_coverage(style_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> StyleCoverageResponse:
    style = (
        db.query(StyleAdapter)
        .filter(StyleAdapter.id == style_id, StyleAdapter.user_id == user.id)
        .first()
    )
    if style is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="style_not_found")

    covered_chars: set[str] = set()
    if style.adapter_key:
        storage = get_storage()
        try:
            adapter_payload = loads(storage.get_text(style.adapter_key))
            covered_chars = _chars_from_adapter_payload(adapter_payload if isinstance(adapter_payload, dict) else None)
        except Exception:
            covered_chars = set()

    missing_hira = "".join(ch for ch in HIRAGANA_TARGET if ch not in covered_chars)
    missing_kata = "".join(ch for ch in KATAKANA_TARGET if ch not in covered_chars)
    missing_kanji_core = "".join(ch for ch in KANJI_CORE_TARGET if ch not in covered_chars)
    covered_in_target = "".join(ch for ch in TARGET_CHARS if ch in covered_chars)
    total_target = len(TARGET_CHARS)
    covered_count = len(covered_in_target)
    missing_count = total_target - covered_count

    return StyleCoverageResponse(
        style_id=style.id,
        status=style.status,
        total_target_chars=total_target,
        covered_count=covered_count,
        missing_count=missing_count,
        covered_chars=covered_in_target,
        missing_hiragana=missing_hira,
        missing_katakana=missing_kata,
        missing_kanji_core=missing_kanji_core,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> GenerateResponse:
    if user.id != payload.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="user_mismatch")
    if payload.purpose != "accessibility":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="purpose_must_be_accessibility")

    use_shared_style = settings.inference_only and settings.shared_style_id > 0
    # Use learned handwriting path for Japanese text except pure hiragana.
    # Pure hiragana uses dedicated runtime image->trajectory generation.
    use_runtime_hiragana = (not settings.text_only_mode) and _is_hiragana_text(payload.text)
    use_model_for_text = (
        (not settings.text_only_mode)
        and _is_japanese_text(payload.text)
        and (not _is_hiragana_text(payload.text))
    )
    style: StyleAdapter | None = None
    if use_shared_style:
        style = _get_shared_style(db)
    else:
        style = (
            db.query(StyleAdapter)
            .filter(StyleAdapter.id == payload.style_id, StyleAdapter.user_id == payload.user_id)
            .first()
        )
        if style is None and (not use_model_for_text):
            style = (
                db.query(StyleAdapter)
                .filter(StyleAdapter.user_id == payload.user_id)
                .order_by(StyleAdapter.id.desc())
                .first()
            )
        if style is None and (not use_model_for_text):
            style = StyleAdapter(user_id=payload.user_id, status="ready", adapter_key=None, disabled=False)
            db.add(style)
            db.commit()
            db.refresh(style)
        if style is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="style_owner_mismatch")

    def _fallback_svg(text: str) -> str:
        if _is_japanese_text(text):
            return text_to_svg_readable(text, WATERMARK_TEXT)
        return text_to_svg(text, WATERMARK_TEXT)

    if settings.text_only_mode or (not use_model_for_text and not use_runtime_hiragana):
        trajectory = []
        svg = _fallback_svg(payload.text)
    elif _is_japanese_text(payload.text):
        trajectory = []
        svg = _runtime_kana_image_svg(payload.text, WATERMARK_TEXT)
    elif use_runtime_hiragana:
        trajectory = []
        # 1) Prefer learned style (known-good route before recent regression).
        if style is not None and style.adapter_key and style.status == "ready" and (not style.disabled):
            storage = get_storage()
            try:
                adapter_seed = storage.get_text(style.adapter_key)
            except Exception:
                adapter_seed = ""
            if adapter_seed:
                style_candidate = generate_trajectory(payload.text, adapter_seed, settings.base_model_path)
                if _is_usable_trajectory(style_candidate, payload.text):
                    trajectory = style_candidate
        # 2) Runtime synthesis for hiragana if style path failed.
        if not trajectory:
            runtime_candidate = _generate_hiragana_runtime_trajectory(payload.text)
            if _is_usable_trajectory(runtime_candidate, payload.text):
                trajectory = runtime_candidate
        # 3) Last-resort base model only if still no usable path.
        if not trajectory:
            base_candidate = generate_trajectory(payload.text, "{}", settings.base_model_path)
            if _is_usable_trajectory(base_candidate, payload.text):
                trajectory = base_candidate

        if not trajectory:
            trajectory = []
            svg = _fallback_svg(payload.text)
        else:
            svg = trajectory_to_svg(trajectory, WATERMARK_TEXT)
    else:
        if style.disabled or style.status != "ready" or not style.adapter_key:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="style_not_ready")

        storage = get_storage()
        adapter_seed = storage.get_text(style.adapter_key)
        adapter_payload = loads(adapter_seed)
        exemplars_text = (
            adapter_payload.get("user_char_exemplars_text")
            if isinstance(adapter_payload, dict)
            else None
        )
        # Do not require user-only coverage here: generate_trajectory() merges
        # user exemplars with base-model exemplars, so kana not present in the
        # user sample can still be rendered from the learned base dataset.
        has_user_coverage = _has_kana_coverage(payload.text, exemplars_text if isinstance(exemplars_text, dict) else None)
        trajectory = []
        style_candidate = generate_trajectory(payload.text, adapter_seed, settings.base_model_path)
        if _is_usable_trajectory(style_candidate, payload.text):
            trajectory = style_candidate
        if not trajectory:
            base_candidate = generate_trajectory(payload.text, "{}", settings.base_model_path)
            if _is_usable_trajectory(base_candidate, payload.text):
                trajectory = base_candidate
        use_text_fallback = settings.readable_text_svg or not trajectory
        # If user coverage is missing and generated path still looks bad, prefer readability.
        if not has_user_coverage and not trajectory:
            use_text_fallback = True
        if use_text_fallback:
            trajectory = []
            svg = _fallback_svg(payload.text)
        else:
            svg = trajectory_to_svg(trajectory, WATERMARK_TEXT)

    storage = get_storage()
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
