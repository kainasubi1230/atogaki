from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET
import math

from svgpathtools import parse_path

from trainerlib.char_token import char_to_model_id


@dataclass
class KanjiVGImportStats:
    source: str
    requested_chars: int
    imported_count: int
    skipped_count: int
    output_path: str
    unique_chars: int


def _load_joyo_chars(limit: int | None = None) -> list[str]:
    import kanji_lists

    chars = list(kanji_lists.JOYO)
    if limit is not None and limit > 0:
        chars = chars[: int(limit)]
    return chars


def _load_kana_chars(script: str = "both") -> list[str]:
    key = script.strip().lower()
    hiragana = [chr(cp) for cp in range(0x3041, 0x3097)]
    katakana = [chr(cp) for cp in range(0x30A1, 0x30FB)]
    if key == "hiragana":
        return hiragana
    if key == "katakana":
        return katakana
    if key == "both":
        return hiragana + katakana
    raise ValueError(f"unsupported kana script: {script}")


def _svg_paths(svg_path: Path) -> list[list[tuple[float, float]]]:
    root = ET.parse(svg_path).getroot()
    strokes: list[list[tuple[float, float]]] = []
    for el in root.iter():
        if not str(el.tag).endswith("path"):
            continue
        d = el.attrib.get("d")
        if not d:
            continue
        try:
            p = parse_path(d)
        except Exception:
            continue
        length = float(p.length(error=1e-3))
        n = max(10, min(96, int(length / 1.8)))
        pts: list[tuple[float, float]] = []
        if n <= 1:
            continue
        for i in range(n):
            t = float(i) / float(n - 1)
            z = p.point(t)
            pts.append((float(z.real), float(z.imag)))
        if len(pts) >= 2:
            strokes.append(pts)
    return strokes


def _normalize_strokes(
    strokes: list[list[tuple[float, float]]],
    *,
    canvas: float = 96.0,
    margin: float = 8.0,
) -> list[list[tuple[float, float]]]:
    all_pts = [pt for s in strokes for pt in s]
    if not all_pts:
        return []
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max(1e-6, max_x - min_x)
    h = max(1e-6, max_y - min_y)
    scale = min((canvas - 2 * margin) / w, (canvas - 2 * margin) / h)
    ox = (canvas - w * scale) * 0.5 - min_x * scale
    oy = (canvas - h * scale) * 0.5 - min_y * scale
    out: list[list[tuple[float, float]]] = []
    for s in strokes:
        out.append([(x * scale + ox, y * scale + oy) for x, y in s])
    return out


def _affine_variant(
    strokes: list[list[tuple[float, float]]],
    *,
    rng: random.Random,
    center: tuple[float, float] = (48.0, 48.0),
) -> list[list[tuple[float, float]]]:
    import math

    cx, cy = center
    rot = math.radians(rng.uniform(-4.0, 4.0))
    sc = rng.uniform(0.95, 1.05)
    tx = rng.uniform(-2.0, 2.0)
    ty = rng.uniform(-2.0, 2.0)
    cr = math.cos(rot)
    sr = math.sin(rot)
    out: list[list[tuple[float, float]]] = []
    for s in strokes:
        ns: list[tuple[float, float]] = []
        for x, y in s:
            dx = (x - cx) * sc
            dy = (y - cy) * sc
            rx = dx * cr - dy * sr
            ry = dx * sr + dy * cr
            ns.append((rx + cx + tx, ry + cy + ty))
        out.append(ns)
    return out


def _stroke_length(stroke: list[tuple[float, float]]) -> float:
    if len(stroke) < 2:
        return 0.0
    total = 0.0
    px, py = stroke[0]
    for x, y in stroke[1:]:
        total += math.hypot(x - px, y - py)
        px, py = x, y
    return total


def _smooth_stroke_points(
    stroke: list[tuple[float, float]],
    *,
    passes: int = 2,
    blend: float = 0.30,
) -> list[tuple[float, float]]:
    if len(stroke) < 4 or passes <= 0 or blend <= 0.0:
        return stroke
    out = list(stroke)
    for _ in range(passes):
        nxt = [out[0]]
        for i in range(1, len(out) - 1):
            px, py = out[i - 1]
            cx, cy = out[i]
            nx, ny = out[i + 1]
            ax = (px + nx) * 0.5
            ay = (py + ny) * 0.5
            sx = cx * (1.0 - blend) + ax * blend
            sy = cy * (1.0 - blend) + ay * blend
            nxt.append((sx, sy))
        nxt.append(out[-1])
        out = nxt
    return out


def _local_handwritten_variant(
    strokes: list[list[tuple[float, float]]],
    *,
    rng: random.Random,
    jitter_strength: float = 1.0,
    center: tuple[float, float] = (48.0, 48.0),
) -> list[list[tuple[float, float]]]:
    cx, cy = center
    out: list[list[tuple[float, float]]] = []
    for s in strokes:
        if len(s) < 2:
            out.append(s)
            continue

        length = _stroke_length(s)
        mx = sum(p[0] for p in s) / float(len(s))
        my = sum(p[1] for p in s) / float(len(s))
        short_dot = length < 8.0

        # Per-stroke transform: small but noticeable handwritten variability.
        js = max(0.4, min(2.2, float(jitter_strength)))
        local_rot_deg = rng.uniform(-4.6 * js, 4.6 * js)
        local_scale = rng.uniform(max(0.88, 1.0 - 0.07 * js), min(1.12, 1.0 + 0.07 * js))
        local_tx = rng.uniform(-1.2 * js, 1.2 * js)
        local_ty = rng.uniform(-1.2 * js, 1.2 * js)

        # For short "dot-like" strokes, randomize orientation a bit stronger.
        if short_dot:
            local_rot_deg += rng.uniform(-32.0 * js, 32.0 * js)
            local_scale *= rng.uniform(max(0.78, 1.0 - 0.22 * js), min(1.25, 1.0 + 0.22 * js))
            local_tx += rng.uniform(-0.7 * js, 0.7 * js)
            local_ty += rng.uniform(-0.7 * js, 0.7 * js)

        local_rot = math.radians(local_rot_deg)
        cr = math.cos(local_rot)
        sr = math.sin(local_rot)

        # Smooth normal-direction wobble (avoid jaggy line noise).
        phase = rng.uniform(0.0, math.pi * 2.0)
        # Lower frequency wobble avoids jagged, over-oscillated lines.
        period = rng.uniform(0.45, 0.95)
        wobble_amp = rng.uniform(0.22 * js, 0.72 * js)
        if short_dot:
            wobble_amp *= rng.uniform(0.5, 0.85)

        ns: list[tuple[float, float]] = []
        npts = len(s)
        for i, (x, y) in enumerate(s):
            dx = (x - mx) * local_scale
            dy = (y - my) * local_scale
            rx = dx * cr - dy * sr
            ry = dx * sr + dy * cr
            bx = rx + mx + local_tx
            by = ry + my + local_ty

            if npts >= 3:
                if i == 0:
                    txv = s[1][0] - s[0][0]
                    tyv = s[1][1] - s[0][1]
                elif i == npts - 1:
                    txv = s[-1][0] - s[-2][0]
                    tyv = s[-1][1] - s[-2][1]
                else:
                    txv = s[i + 1][0] - s[i - 1][0]
                    tyv = s[i + 1][1] - s[i - 1][1]
                tn = math.hypot(txv, tyv)
                if tn > 1e-6:
                    nx = -tyv / tn
                    ny = txv / tn
                    t = float(i) / float(max(1, npts - 1))
                    envelope = math.sin(math.pi * t)
                    wobble = wobble_amp * envelope * math.sin((t * math.pi * 2.0 * period) + phase)
                    bx += nx * wobble
                    by += ny * wobble

            # Very light anchor pull so very short strokes stay near canonical position.
            if short_dot:
                bx = bx * 0.86 + mx * 0.14
                by = by * 0.86 + my * 0.14

            # Keep within drawing frame.
            bx = min(95.0, max(1.0, bx))
            by = min(95.0, max(1.0, by))
            ns.append((bx, by))

        # Keep handwritten variation while reducing jagged polyline artifacts.
        smooth_passes = 1 if short_dot else (2 if js <= 1.2 else 3)
        smooth_blend = 0.24 if short_dot else min(0.36, 0.28 + (js - 1.0) * 0.06)
        ns = _smooth_stroke_points(ns, passes=smooth_passes, blend=smooth_blend)

        clamped: list[tuple[float, float]] = []
        for x, y in ns:
            clamped.append((min(95.0, max(1.0, x)), min(95.0, max(1.0, y))))
        ns = clamped
        out.append(ns)
    return out


def _scale_strokes_about_center(
    strokes: list[list[tuple[float, float]]],
    *,
    scale: float,
    center: tuple[float, float] = (48.0, 48.0),
) -> list[list[tuple[float, float]]]:
    if abs(scale - 1.0) < 1e-6:
        return strokes
    cx, cy = center
    out: list[list[tuple[float, float]]] = []
    for s in strokes:
        ns: list[tuple[float, float]] = []
        for x, y in s:
            nx = (x - cx) * scale + cx
            ny = (y - cy) * scale + cy
            nx = min(95.0, max(1.0, nx))
            ny = min(95.0, max(1.0, ny))
            ns.append((nx, ny))
        out.append(ns)
    return out


def _strokes_to_sequence(
    strokes: list[list[tuple[float, float]]],
    *,
    stroke_width: float = 0.78,
) -> list[list[float]]:
    seq: list[list[float]] = []
    cursor_x = 0.0
    cursor_y = 0.0
    for s in strokes:
        if len(s) < 2:
            continue
        sx, sy = s[0]
        if abs(sx - cursor_x) > 1e-6 or abs(sy - cursor_y) > 1e-6:
            seq.append([(sx - cursor_x) / 20.0, (sy - cursor_y) / 20.0, 0.0, stroke_width])
            cursor_x, cursor_y = sx, sy
        prev_x, prev_y = sx, sy
        for x, y in s[1:]:
            seq.append([(x - prev_x) / 20.0, (y - prev_y) / 20.0, 1.0, stroke_width])
            prev_x, prev_y = x, y
        if seq:
            seq[-1][2] = 0.0
        cursor_x, cursor_y = prev_x, prev_y
    return seq


def import_kanjivg_joyo_to_base_dataset(
    output_path: str,
    *,
    repo_dir: str = "storage/public_cache/net_datasets/kanjivg/kanji",
    chars_limit: int = 0,
    variants_per_char: int = 3,
    compact_scale: float = 0.90,
    hand_jitter: float = 1.15,
    seed: int = 42,
    append: bool = True,
    source_name: str = "kanjivg-joyo",
) -> KanjiVGImportStats:
    src = Path(repo_dir)
    if not src.exists():
        raise FileNotFoundError(f"KanjiVG directory not found: {src}")

    joyo = _load_joyo_chars(chars_limit if chars_limit > 0 else None)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    rng = random.Random(seed)

    imported = 0
    skipped = 0
    seen_chars: set[str] = set()
    rows: list[dict] = []

    for ch in joyo:
        code = f"{ord(ch):05x}.svg"
        svg_path = src / code
        if not svg_path.exists():
            skipped += 1
            continue
        try:
            base = _normalize_strokes(_svg_paths(svg_path))
        except Exception:
            skipped += 1
            continue
        if not base:
            skipped += 1
            continue
        nvar = max(1, int(variants_per_char))
        for i in range(nvar):
            var_rng = random.Random(rng.randint(0, 2**31 - 1) ^ (ord(ch) << 8) ^ i)
            strokes = _affine_variant(base, rng=var_rng)
            strokes = _local_handwritten_variant(strokes, rng=var_rng, jitter_strength=float(hand_jitter))
            strokes = _scale_strokes_about_center(strokes, scale=float(compact_scale))
            width = min(0.95, max(0.70, 0.80 + var_rng.uniform(-0.05, 0.06)))
            seq = _strokes_to_sequence(strokes, stroke_width=width)
            if len(seq) < 8:
                continue
            rows.append(
                {
                    "char_id": char_to_model_id(ch),
                    "style_id": 0,
                    "dataset_id": -900,
                    "user_id": 0,
                    "sequence": seq,
                    "meta": {
                        "source": source_name,
                        "char": ch,
                        "svg_file": code,
                        "variant": i,
                        "profile": "kanjivg_local_jitter_v5_smooth",
                        "compact_scale": float(compact_scale),
                        "hand_jitter": float(hand_jitter),
                    },
                }
            )
            imported += 1
            seen_chars.add(ch)

    with out_path.open(mode, encoding="utf-8") as w:
        for row in rows:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")

    requested = len(joyo)
    return KanjiVGImportStats(
        source=source_name,
        requested_chars=requested,
        imported_count=imported,
        skipped_count=max(0, requested - len(seen_chars)),
        output_path=str(out_path),
        unique_chars=len(seen_chars),
    )


def import_kanjivg_kana_to_base_dataset(
    output_path: str,
    *,
    repo_dir: str = "storage/public_cache/net_datasets/kanjivg/kanji",
    script: str = "both",
    variants_per_char: int = 3,
    compact_scale: float = 0.90,
    hand_jitter: float = 1.05,
    seed: int = 42,
    append: bool = True,
    source_name: str = "kanjivg-kana",
) -> KanjiVGImportStats:
    src = Path(repo_dir)
    if not src.exists():
        raise FileNotFoundError(f"KanjiVG directory not found: {src}")

    chars = _load_kana_chars(script)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    rng = random.Random(seed)

    imported = 0
    skipped = 0
    seen_chars: set[str] = set()
    rows: list[dict] = []

    for ch in chars:
        code = f"{ord(ch):05x}.svg"
        svg_path = src / code
        if not svg_path.exists():
            skipped += 1
            continue
        try:
            base = _normalize_strokes(_svg_paths(svg_path))
        except Exception:
            skipped += 1
            continue
        if not base:
            skipped += 1
            continue
        nvar = max(1, int(variants_per_char))
        for i in range(nvar):
            var_rng = random.Random(rng.randint(0, 2**31 - 1) ^ (ord(ch) << 8) ^ i)
            strokes = _affine_variant(base, rng=var_rng)
            strokes = _local_handwritten_variant(strokes, rng=var_rng, jitter_strength=float(hand_jitter))
            strokes = _scale_strokes_about_center(strokes, scale=float(compact_scale))
            width = min(0.90, max(0.68, 0.76 + var_rng.uniform(-0.05, 0.05)))
            seq = _strokes_to_sequence(strokes, stroke_width=width)
            if len(seq) < 8:
                continue
            rows.append(
                {
                    "char_id": char_to_model_id(ch),
                    "style_id": 0,
                    "dataset_id": -901,
                    "user_id": 0,
                    "sequence": seq,
                    "meta": {
                        "source": source_name,
                        "char": ch,
                        "svg_file": code,
                        "variant": i,
                        "profile": "kanjivg_kana_v1",
                        "compact_scale": float(compact_scale),
                        "hand_jitter": float(hand_jitter),
                    },
                }
            )
            imported += 1
            seen_chars.add(ch)

    with out_path.open(mode, encoding="utf-8") as w:
        for row in rows:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")

    requested = len(chars)
    return KanjiVGImportStats(
        source=source_name,
        requested_chars=requested,
        imported_count=imported,
        skipped_count=max(0, requested - len(seen_chars)),
        output_path=str(out_path),
        unique_chars=len(seen_chars),
    )
