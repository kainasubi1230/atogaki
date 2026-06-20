"""
api/app/analyze_scan.py

OpenCV (なければ scipy/numpy フォールバック) でスキャン画像を解析し、
文字ごとの bbox・切り抜き画像(base64)・writer_style を返す APIRouter。
main.py に `app.include_router(analyze_scan.router)` で組み込む。
"""

from __future__ import annotations

import base64
import io as _io
import os
import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

router = APIRouter(prefix="/analyze", tags=["analyze"])

_DEFAULT_IMG_PATH = Path("storage/S__14434308.jpg")

_ROW_TEMPLATES: list[list[str]] = [
    list("あいうえおかきくけこさしすせそ"),
    list("たちつてとなにぬねのはひふへほ"),
    list("まみむめもや ゆ よらりるれろ"),
    list("アイウエオカキクケコサシスセソ"),
    list("タチツテトナニヌネノハヒフヘホ"),
    list("マミムメモヤ ユ ヨラリルレロ"),
]


def _extract_segments(img_path: Path) -> dict:
    """
    画像を読み込み、OpenCV(なければ scipy) でバイナリ化→輪郭抽出→
    行列ソート→ラベル付け→切り抜き base64 埋め込みを行い、
    segments と writer_style を含む dict を返す。
    """
    if not img_path.is_file():
        return {"success": False, "reason": f"file_not_found: {img_path}"}

    pil_img = Image.open(img_path).convert("RGB")
    gray_arr = np.array(pil_img.convert("L"))
    img_h, img_w = gray_arr.shape

    # ── OpenCV があれば大津の二値化 ──────────────────────────────
    try:
        import cv2  # type: ignore

        _, binary = cv2.threshold(
            gray_arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes_raw = [cv2.boundingRect(c) for c in contours]

    # ── フォールバック: scipy ─────────────────────────────────────
    except ImportError:
        from scipy.ndimage import binary_opening as _bop  # type: ignore
        from scipy.ndimage import label as _label  # type: ignore

        thr = int(np.mean(gray_arr))
        opened = _bop(gray_arr < thr, structure=np.ones((3, 3), dtype=bool))
        labeled, n = _label(opened)
        boxes_raw = []
        for i in range(1, n + 1):
            ys, xs = np.where(labeled == i)
            if not len(xs):
                continue
            boxes_raw.append(
                (
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1),
                )
            )

    # 小さすぎる候補を除外
    boxes_raw = [
        (x, y, w, h)
        for (x, y, w, h) in boxes_raw
        if w * h >= 80 and w >= 5 and h >= 5
    ]

    # ── 行クラスタリング ──────────────────────────────────────────
    row_tol = max(20, img_h // 20)
    row_clusters: list[list[tuple]] = []
    for box in sorted(boxes_raw, key=lambda b: b[1]):
        _, y, _, h = box
        cy = y + h / 2
        placed = False
        for cluster in row_clusters:
            cy_c = sum(b[1] + b[3] / 2 for b in cluster) / len(cluster)
            if abs(cy - cy_c) < row_tol:
                cluster.append(box)
                placed = True
                break
        if not placed:
            row_clusters.append([box])

    # 各行を左→右でソート、短い行は除外、最大 6 行
    row_clusters = [
        sorted(c, key=lambda b: b[0])
        for c in row_clusters
        if len(c) >= 3
    ]
    row_clusters.sort(key=lambda c: sum(b[1] for b in c) / len(c))
    row_clusters = row_clusters[:6]

    # ── セグメント構築 ────────────────────────────────────────────
    segments: list[dict] = []
    for row_idx, cluster in enumerate(row_clusters):
        template = _ROW_TEMPLATES[row_idx] if row_idx < len(_ROW_TEMPLATES) else []
        for col_idx, (x, y, w, h) in enumerate(cluster):
            label_char: str | None = None
            if col_idx < len(template) and template[col_idx].strip():
                label_char = template[col_idx]

            # 切り抜き画像を PNG base64 で埋め込む
            crop = pil_img.crop((x, y, x + w, y + h))
            bio = _io.BytesIO()
            crop.save(bio, format="PNG")
            char_b64 = base64.b64encode(bio.getvalue()).decode()

            segments.append(
                {
                    "index": len(segments),
                    "row": row_idx,
                    "col": col_idx,
                    "label": label_char,
                    "bbox": {
                        "x0": int(x),
                        "y0": int(y),
                        "x1": int(x + w),
                        "y1": int(y + h),
                    },
                    "image_b64": char_b64,
                    "style_features": {
                        "width": int(w),
                        "height": int(h),
                        "aspect_ratio": round(float(h) / float(max(w, 1)), 4),
                        "area": int(w * h),
                    },
                }
            )

    # 92 個に補完（欠損セルは null エントリ）
    while len(segments) < 92:
        segments.append(
            {
                "index": len(segments),
                "row": None,
                "col": None,
                "label": None,
                "bbox": None,
                "image_b64": None,
                "style_features": None,
            }
        )

    # ── writer_style: 形状統計 ────────────────────────────────────
    valid = [s for s in segments if s["style_features"] is not None]

    def _stats(vals: list) -> tuple:
        if not vals:
            return None, None
        a = np.array(vals, dtype=float)
        return float(a.mean()), float(a.std())

    m_ar, s_ar = _stats([s["style_features"]["aspect_ratio"] for s in valid])
    m_w, s_w = _stats([s["style_features"]["width"] for s in valid])
    m_h, s_h = _stats([s["style_features"]["height"] for s in valid])
    m_a, s_a = _stats([s["style_features"]["area"] for s in valid])

    # 元画像 base64（フロントでオーバーレイ表示用）
    bio_orig = _io.BytesIO()
    pil_img.save(bio_orig, format="JPEG", quality=85)
    orig_b64 = base64.b64encode(bio_orig.getvalue()).decode()

    return {
        "success": True,
        "image_b64": orig_b64,
        "image_width": img_w,
        "image_height": img_h,
        "segment_count": len(segments),
        "segments": segments,
        "writer_style": {
            "mean_aspect_ratio": m_ar,
            "std_aspect_ratio": s_ar,
            "mean_char_width": m_w,
            "std_char_width": s_w,
            "mean_char_height": m_h,
            "std_char_height": s_h,
            "mean_area": m_a,
            "std_area": s_a,
            "detected_count": len(valid),
        },
    }


# ── エンドポイント ────────────────────────────────────────────────


@router.get("/kana-scan")
async def analyze_kana_scan() -> JSONResponse:
    """
    storage/S__14434308.jpg を解析して文字bbox・切り抜き・writer_styleを返す。
    認証不要（デバッグ・開発用エンドポイント）。
    """
    result = _extract_segments(_DEFAULT_IMG_PATH)
    return JSONResponse(content=result)


@router.post("/kana-scan-upload")
async def analyze_kana_scan_upload(file: UploadFile = File(...)) -> JSONResponse:
    """任意の画像をアップロードして解析する（認証不要）。"""
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        result = _extract_segments(tmp_path)
    finally:
        os.unlink(tmp_path)
    return JSONResponse(content=result)
