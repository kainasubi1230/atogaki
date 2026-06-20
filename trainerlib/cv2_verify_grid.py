"""
cv2_verify_grid.py

OpenCV-based post-processing to verify and correct bounding boxes:
- Renders a font-based template for each expected kana character
- Computes normalized cross-correlation between the scanned crop and the template
- Uses the score to identify mismatches and adjusts via a fallback search within column boundaries
"""
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_opening
import cv2

FONT_PATH = "storage/fonts/KleeOne-Regular.ttf"
TEMPLATE_SIZE = 64

_template_cache = {}

def get_kana_template(char: str, font_path: str = FONT_PATH, size: int = TEMPLATE_SIZE) -> np.ndarray:
    """Render a reference (template) image for `char` using the given font."""
    key = (char, font_path, size)
    if key in _template_cache:
        return _template_cache[key]
    try:
        font = ImageFont.truetype(font_path, int(size * 1.25))
    except Exception:
        _template_cache[key] = None
        return None
    
    canvas_w = size * 2
    canvas_h = size * 2
    canvas = Image.new("L", (canvas_w, canvas_h), 255)
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas_w - tw) / 2 - bbox[0]
    y = (canvas_h - th) / 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)
    
    raw_tmpl = (np.array(canvas, dtype=np.uint8) < 128)
    ys, xs = np.where(raw_tmpl)
    if len(ys) == 0 or len(xs) == 0:
        tmpl = np.zeros((size, size), dtype=np.uint8)
        _template_cache[key] = tmpl
        return tmpl
    cropped = raw_tmpl[int(ys.min()): int(ys.max()) + 1, int(xs.min()): int(xs.max()) + 1]
    h, w = cropped.shape
    target = int(size * 0.78)
    scale = min(target / float(w), target / float(h))
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    resized = cv2.resize(
        cropped.astype(np.uint8) * 255,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )
    tmpl = np.zeros((size, size), dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    tmpl[y0:y0 + new_h, x0:x0 + new_w] = resized
    _template_cache[key] = tmpl
    return tmpl


def _normalize_binary_image(ink: np.ndarray, size: int) -> np.ndarray:
    ys, xs = np.where(ink)
    if len(ys) == 0 or len(xs) == 0:
        return np.zeros((size, size), dtype=np.uint8)
    cropped = ink[int(ys.min()): int(ys.max()) + 1, int(xs.min()): int(xs.max()) + 1]
    h, w = cropped.shape
    if h < 2 or w < 2:
        return np.zeros((size, size), dtype=np.uint8)

    target = int(size * 0.78)
    scale = min(target / float(w), target / float(h))
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    resized = cv2.resize(
        cropped.astype(np.uint8) * 255,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((size, size), dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def compute_match_score(
    binary_crop: np.ndarray,
    char: str,
    template_size: int = TEMPLATE_SIZE,
) -> float:
    """
    Compute a normalized match score (0.0–1.0) between a binary crop
    (True=ink) and the font template for `char`.
    """
    tmpl = get_kana_template(char, size=template_size)
    if tmpl is None or binary_crop.shape[0] < 4 or binary_crop.shape[1] < 4:
        return 0.0

    candidate = _normalize_binary_image(binary_crop, template_size)

    c_f = candidate.astype(np.float32)
    t_f = tmpl.astype(np.float32)

    dot = float((c_f * t_f).sum())
    norm_c = float(np.sqrt((c_f * c_f).sum()))
    norm_t = float(np.sqrt((t_f * t_f).sum()))

    if norm_c < 1.0 or norm_t < 1.0:
        return 0.0

    corr = dot / (norm_c * norm_t)
    c_bin = candidate > 32
    t_bin = tmpl > 32
    intersection = float(np.logical_and(c_bin, t_bin).sum())
    union_like = float(c_bin.sum() + t_bin.sum())
    dice = (2.0 * intersection / union_like) if union_like > 0 else 0.0
    return float((corr * 0.7) + (dice * 0.3))