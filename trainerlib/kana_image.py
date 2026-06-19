from __future__ import annotations

import hashlib
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


CHAR_TO_ROMAJI = {
    "あ": "A",
    "い": "I",
    "う": "U",
    "え": "E",
    "お": "O",
    "か": "KA",
    "き": "KI",
    "く": "KU",
    "け": "KE",
    "こ": "KO",
    "さ": "SA",
    "し": "SHI",
    "す": "SU",
    "せ": "SE",
    "そ": "SO",
    "た": "TA",
    "ち": "CHI",
    "つ": "TSU",
    "て": "TE",
    "と": "TO",
    "な": "NA",
    "に": "NI",
    "ぬ": "NU",
    "ね": "NE",
    "の": "NO",
    "は": "HA",
    "ひ": "HI",
    "ふ": "FU",
    "へ": "HE",
    "ほ": "HO",
    "ま": "MA",
    "み": "MI",
    "む": "MU",
    "め": "ME",
    "も": "MO",
    "や": "YA",
    "ゆ": "YU",
    "よ": "YO",
    "ら": "RA",
    "り": "RI",
    "る": "RU",
    "れ": "RE",
    "ろ": "RO",
    "わ": "WA",
    "を": "WO",
    "ん": "N",
    "ば": "BA",
    "だ": "DA",
    "じ": "JI",
    "ぴ": "PI",
}

_CHAR_TEMPLATE_CACHE: dict[tuple[str, str, int], np.ndarray] = {}
HIRAGANA_RANGE_START = ord("ぁ")
HIRAGANA_RANGE_END = ord("ゖ")
KATAKANA_RANGE_START = ord("ァ")
KATAKANA_RANGE_END = ord("ヶ")


def _rng_from_seed(seed: str) -> random.Random:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    token = int(digest[:16], 16)
    return random.Random(token)


def _interior_only(mask: np.ndarray) -> np.ndarray:
    # Remove connected components touching image border.
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    stack: list[tuple[int, int]] = []
    for x in range(w):
        if mask[0, x]:
            stack.append((0, x))
        if mask[h - 1, x]:
            stack.append((h - 1, x))
    for y in range(h):
        if mask[y, 0]:
            stack.append((y, 0))
        if mask[y, w - 1]:
            stack.append((y, w - 1))

    while stack:
        y, x = stack.pop()
        if not (0 <= y < h and 0 <= x < w):
            continue
        if visited[y, x] or not mask[y, x]:
            continue
        visited[y, x] = True
        stack.append((y - 1, x))
        stack.append((y + 1, x))
        stack.append((y, x - 1))
        stack.append((y, x + 1))
    return mask & (~visited)


def _mask_shape_score(mask: np.ndarray) -> float:
    count = int(mask.sum())
    if count <= 0:
        return 1e12
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    box_area = max(1, (y1 - y0 + 1) * (x1 - x0 + 1))
    fill = count / box_area
    # Prefer sparse stroke-like masks over dense filled blocks.
    fill_penalty = abs(fill - 0.18) * 6000.0
    area_penalty = abs(count - 1600) * 0.12
    return area_penalty + fill_penalty


def _pick_source_image(input_dir: Path, char: str, rng: random.Random) -> Path:
    romaji = CHAR_TO_ROMAJI.get(char)
    if romaji is None:
        raise ValueError(f"unsupported char for image generation: {char}")
    files = sorted(input_dir.glob(f"kana{romaji}*.jpg"))
    if not files:
        raise FileNotFoundError(f"source images not found for {char} (kana{romaji}*.jpg)")
    return files[int(rng.random() * len(files)) % len(files)]


def _is_katakana_char(char: str) -> bool:
    if len(char) != 1:
        return False
    code = ord(char)
    return KATAKANA_RANGE_START <= code <= KATAKANA_RANGE_END


def _is_hiragana_char(char: str) -> bool:
    if len(char) != 1:
        return False
    code = ord(char)
    return HIRAGANA_RANGE_START <= code <= HIRAGANA_RANGE_END


def _render_kana_with_font(
    char: str,
    *,
    style_seed: str,
    font_path: str,
    size: int,
) -> Image.Image:
    rng = _rng_from_seed(f"font:{style_seed}:{char}:{size}")
    canvas = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(canvas)
    is_katakana = _is_katakana_char(char)
    # Narrower randomness for katakana so line thickness is more consistent.
    # Wider randomness for a more natural, casual look.
    if is_katakana:
        font_size = int(size * rng.uniform(0.48, 0.64))
    else:
        font_size = int(size * rng.uniform(0.50, 0.70))
    font = ImageFont.truetype(font_path, font_size)
    bbox = draw.textbbox((0, 0), char, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    if is_katakana:
        jitter = size * 0.05
    else:
        jitter = size * 0.07
    x = (size - tw) / 2 - bbox[0] + rng.uniform(-jitter, jitter)
    y = (size - th) / 2 - bbox[1] + rng.uniform(-jitter, jitter)
    draw.text((x, y), char, fill=0, font=font)

    if is_katakana:
        angle = rng.uniform(-10.0, 10.0)
        scale = rng.uniform(0.92, 1.08)
    else:
        angle = rng.uniform(-14.0, 14.0)
        scale = rng.uniform(0.88, 1.12)
    transformed = canvas.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=255,
    )
    nw = max(32, int(round(transformed.width * scale)))
    nh = max(32, int(round(transformed.height * scale)))
    transformed = transformed.resize((nw, nh), Image.Resampling.BICUBIC)
    out = Image.new("L", (size, size), 255)
    ox = int((size - transformed.width) / 2 + rng.uniform(-size * 0.05, size * 0.05))
    oy = int((size - transformed.height) / 2 + rng.uniform(-size * 0.05, size * 0.05))
    out.paste(transformed, (ox, oy))
    out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.08, 0.28)))
    return out


def _binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(max(0, iterations)):
        h, w = out.shape
        p = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        nxt = np.zeros_like(out, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                nxt |= p[dy : dy + h, dx : dx + w]
        out = nxt
    return out


def _binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(max(0, iterations)):
        h, w = out.shape
        p = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        nxt = np.ones_like(out, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                nxt &= p[dy : dy + h, dx : dx + w]
        out = nxt
    return out


def _regularize_katakana_mask(mask: np.ndarray, *, style_seed: str) -> np.ndarray:
    rng = _rng_from_seed(f"katakana-mask:{style_seed}")
    out = mask.astype(bool)
    # Opening removes burr noise and helps consistent pen-like width.
    out = _binary_dilate(_binary_erode(out, 1), 1)
    target_low = rng.uniform(0.035, 0.048)
    target_high = rng.uniform(0.055, 0.070)
    for _ in range(3):
        fill = float(out.mean())
        if fill > target_high:
            out = _binary_erode(out, 1)
            continue
        if fill < target_low:
            out = _binary_dilate(out, 1)
            continue
        break
    # Closing to smooth stair-steps after width normalization.
    out = _binary_erode(_binary_dilate(out, 1), 1)
    return out


def generate_kana_image(
    char: str,
    *,
    style_seed: str,
    input_dir: str = "storage/public_cache/net_datasets/hiragana-dataset/hiragana_images",
    fallback_font_path: str = "storage/fonts/KleeOne-Regular.ttf",
    size: int = 512,
) -> Image.Image:
    rng = _rng_from_seed(f"{style_seed}:{char}:{size}")
    if char in CHAR_TO_ROMAJI:
        src_dir = Path(input_dir)
        src = _pick_source_image(src_dir, char, rng)
        base = Image.open(src).convert("L")
        base = ImageOps.autocontrast(base)
        base = ImageEnhance.Contrast(base).enhance(rng.uniform(1.2, 1.7))
    elif _is_katakana_char(char) or _is_hiragana_char(char):
        font_path = Path(fallback_font_path)
        if not font_path.exists():
            raise FileNotFoundError(f"kana fallback font not found: {font_path}")
        base = _render_kana_with_font(
            char,
            style_seed=style_seed,
            font_path=str(font_path),
            size=max(128, size),
        )
    else:
        raise ValueError(f"unsupported char for image generation: {char}")

    # Keep augmentation wide for natural handwriting variety.
    angle = rng.uniform(-12.0, 12.0)
    scale = rng.uniform(0.88, 1.15)
    tx = rng.uniform(-5.0, 5.0)
    ty = rng.uniform(-5.0, 5.0)

    transformed = base.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=255,
    )
    nw = max(32, int(round(transformed.width * scale)))
    nh = max(32, int(round(transformed.height * scale)))
    transformed = transformed.resize((nw, nh), Image.Resampling.BICUBIC)

    canvas = Image.new("L", (size, size), 255)
    ox = int((size - transformed.width) / 2 + tx * (size / 96.0))
    oy = int((size - transformed.height) / 2 + ty * (size / 96.0))
    canvas.paste(transformed, (ox, oy))
    canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.55)))

    # Convert to clean black-on-white.
    arr_img = ImageOps.autocontrast(canvas)
    arr = np.array(arr_img, dtype=np.uint8)
    dark_mask = _interior_only(arr < 128)
    bright_mask = _interior_only(arr > 128)

    dark_count = int(dark_mask.sum())
    bright_count = int(bright_mask.sum())
    if dark_count == 0 and bright_count == 0:
        # Fallback to simple polarity when both got stripped out.
        raw_dark = arr < 128
        raw_bright = arr > 128
        fg = raw_dark if int(raw_dark.sum()) <= int(raw_bright.sum()) else raw_bright
    else:
        candidates: list[np.ndarray] = []
        if dark_count > 0:
            candidates.append(dark_mask)
        if bright_count > 0:
            candidates.append(bright_mask)
        fg = min(candidates, key=_mask_shape_score)
    if _is_katakana_char(char):
        fg = _regularize_katakana_mask(fg, style_seed=style_seed)
    out = np.full_like(arr, 255, dtype=np.uint8)
    out[fg] = 0
    return Image.fromarray(out, mode="L")


def _image_to_mask(image: Image.Image, size: int = 96) -> np.ndarray:
    arr = np.array(image.convert("L").resize((size, size), Image.Resampling.BICUBIC), dtype=np.uint8)
    return arr < 128


def _char_template(char: str, input_dir: str, size: int = 96) -> np.ndarray:
    key = (char, input_dir, size)
    cached = _CHAR_TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached
    masks = []
    if char in CHAR_TO_ROMAJI:
        src_dir = Path(input_dir)
        romaji = CHAR_TO_ROMAJI.get(char)
        files = sorted(src_dir.glob(f"kana{romaji}*.jpg"))
        if not files:
            raise FileNotFoundError(f"source images not found for {char} (kana{romaji}*.jpg)")
        for path in files:
            norm = generate_kana_image(char, style_seed=f"template:{path.name}", input_dir=input_dir, size=size)
            masks.append(_image_to_mask(norm, size=size))
    elif _is_katakana_char(char) or _is_hiragana_char(char):
        for idx in range(12):
            norm = generate_kana_image(char, style_seed=f"template_font:{char}:{idx}", input_dir=input_dir, size=size)
            masks.append(_image_to_mask(norm, size=size))
    else:
        raise ValueError(f"unsupported char for template: {char}")
    avg = np.mean(np.stack(masks, axis=0).astype(np.float32), axis=0)
    tmpl = avg > 0.35
    _CHAR_TEMPLATE_CACHE[key] = tmpl
    return tmpl


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(max(1, union))


def generate_kana_image_best(
    char: str,
    *,
    style_seed: str,
    input_dir: str = "storage/public_cache/net_datasets/hiragana-dataset/hiragana_images",
    size: int = 512,
    trials: int = 12,
) -> Image.Image:
    template = _char_template(char, input_dir=input_dir, size=96)
    rng = _rng_from_seed(f"best:{style_seed}:{char}")
    best_img: Image.Image | None = None
    best_score = -1.0
    for idx in range(max(1, trials)):
        img = generate_kana_image(
            char,
            style_seed=f"{style_seed}:trial:{idx}",
            input_dir=input_dir,
            size=size,
        )
        score = _iou(template, _image_to_mask(img, size=96))
        # Add small random noise to allow organic variation
        score_perturbed = score + rng.uniform(-0.06, 0.06)
        if score_perturbed > best_score:
            best_score = score_perturbed
            best_img = img
    if best_img is None:
        return generate_kana_image(char, style_seed=style_seed, input_dir=input_dir, size=size)
    return best_img


def kana_image_quality_score(image: Image.Image) -> float:
    # Lower is better.
    mask = _image_to_mask(image, size=96)
    return _mask_shape_score(mask)
