from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

from trainer.public_dataset import _preprocess_labeled_char_image, _image_to_sequence
from trainerlib.char_token import char_to_model_id


LATIN_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

Point = tuple[float, float, float]
Stroke = list[Point]
Template = list[list[tuple[float, float]]]


@dataclass
class LatinBuildResult:
    output_path: str
    char_count: int
    rows_written: int
    variants_per_char: int


def _soften_polyline(points: list[tuple[float, float]], *, iterations: int = 1) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    softened = list(points)
    for _ in range(max(0, iterations)):
        nxt: list[tuple[float, float]] = [softened[0]]
        for p0, p1 in zip(softened, softened[1:]):
            q = (p0[0] * 0.72 + p1[0] * 0.28, p0[1] * 0.72 + p1[1] * 0.28)
            r = (p0[0] * 0.28 + p1[0] * 0.72, p0[1] * 0.28 + p1[1] * 0.72)
            nxt.extend([q, r])
        nxt.append(softened[-1])
        softened = nxt
    return softened


def _smooth_wobble(
    pts: list[tuple[float, float]],
    *,
    rng: random.Random,
    amp: float = 0.6,
) -> list[tuple[float, float]]:
    """Apply a smooth perpendicular sine-wave wobble to a polyline.

    Uses an envelope (sin(π·t)) so that endpoints stay anchored.
    This mirrors the approach in kanjivg_dataset._local_handwritten_variant.
    """
    n = len(pts)
    if n < 3 or amp < 1e-6:
        return pts
    phase = rng.uniform(0.0, math.tau)
    period = rng.uniform(0.5, 1.0)
    out: list[tuple[float, float]] = []
    for i, (x, y) in enumerate(pts):
        t = float(i) / float(max(1, n - 1))
        # Tangent direction -> normal direction.
        if i == 0:
            txv, tyv = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
        elif i == n - 1:
            txv, tyv = pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]
        else:
            txv, tyv = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
        tn = math.hypot(txv, tyv)
        if tn > 1e-6:
            nx, ny = -tyv / tn, txv / tn
            envelope = math.sin(math.pi * t)
            wobble = amp * envelope * math.sin(t * math.pi * 2.0 * period + phase)
            x += nx * wobble
            y += ny * wobble
        out.append((x, y))
    return out


def _line(points: list[tuple[float, float]], *, width: float, rng: random.Random, jitter: float = 0.0) -> Stroke:
    """Render a polyline stroke with organic smooth wobble instead of random noise."""
    out: Stroke = []
    # First soften the control points, then apply smooth wobble.
    softened_points = _soften_polyline(points, iterations=2)

    # Bow / arc per segment (large-scale curve feel)
    num_segs = len(softened_points) - 1
    seg_bows = [rng.uniform(-0.012, 0.012) for _ in range(num_segs)]

    # Sample the full polyline at a fixed resolution.
    sampled: list[tuple[float, float]] = []
    for idx in range(num_segs):
        x0, y0 = softened_points[idx]
        x1, y1 = softened_points[idx + 1]
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(8, int(dist / 4.0))
        if dist > 1e-6:
            nx = -(y1 - y0) / dist
            ny = (x1 - x0) / dist
        else:
            nx, ny = 0.0, 0.0
        bow_dist = seg_bows[idx] * dist
        for step in range(steps + 1):
            if idx > 0 and step == 0:
                continue
            t = step / max(1, steps)
            ease = t * t * (3.0 - 2.0 * t)
            arc = math.sin(math.pi * t) * bow_dist
            x = x0 + (x1 - x0) * ease + nx * arc
            y = y0 + (y1 - y0) * ease + ny * arc
            sampled.append((x, y))

    # Apply smooth wobble over the whole stroke (envelope keeps endpoints anchored).
    wobble_amp = jitter * 14.0  # jitter is ~0.01-0.06; amp in canvas units
    wobbled = _smooth_wobble(sampled, rng=rng, amp=wobble_amp)

    for x, y in wobbled:
        out.append((x, y, width * rng.uniform(0.92, 1.10)))
    return out


def _quad(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    width: float,
    rng: random.Random,
    points: int = 18,
    jitter: float = 0.0,
) -> Stroke:
    sampled: list[tuple[float, float]] = []
    for i in range(points):
        t = i / max(1, points - 1)
        mt = 1.0 - t
        x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
        y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
        sampled.append((x, y))
    wobbled = _smooth_wobble(sampled, rng=rng, amp=jitter * 14.0)
    return [(x, y, width * rng.uniform(0.92, 1.10)) for x, y in wobbled]


def _cubic(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    *,
    width: float,
    rng: random.Random,
    points: int = 22,
    jitter: float = 0.0,
) -> Stroke:
    sampled: list[tuple[float, float]] = []
    for i in range(points):
        t = i / max(1, points - 1)
        mt = 1.0 - t
        x = mt**3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t**3 * p3[1]
        sampled.append((x, y))
    wobbled = _smooth_wobble(sampled, rng=rng, amp=jitter * 14.0)
    return [(x, y, width * rng.uniform(0.92, 1.10)) for x, y in wobbled]


def _oval(cx: float, cy: float, rx: float, ry: float, *, width: float, rng: random.Random, points: int = 30) -> Stroke:
    out: Stroke = []
    phase = rng.uniform(-0.22, 0.22)
    for i in range(points + 1):
        t = math.tau * i / points + phase
        wob = 1.0 + rng.uniform(-0.055, 0.055)
        out.append((cx + math.cos(t) * rx * wob, cy + math.sin(t) * ry * wob, width * rng.uniform(0.92, 1.10)))
    return out


def _dot(x: float, y: float, *, width: float, rng: random.Random) -> Stroke:
    return _oval(x, y, 3.0, 3.0, width=width, rng=rng, points=10)


UPPER: dict[str, Template] = {
    "A": [[(18, 88), (50, 12), (82, 88)], [(31, 56), (69, 56)]],
    "B": [[(22, 88), (22, 12), (55, 12), (72, 27), (56, 45), (22, 45)], [(22, 45), (60, 45), (78, 63), (58, 88), (22, 88)]],
    "C": [[(76, 24), (57, 9), (28, 20), (18, 50), (29, 81), (61, 91), (78, 76)]],
    "D": [[(22, 88), (22, 12), (56, 14), (79, 39), (77, 64), (56, 86), (22, 88)]],
    "E": [[(76, 14), (22, 14), (22, 88), (78, 88)], [(25, 50), (65, 50)]],
    "F": [[(22, 88), (22, 14), (78, 14)], [(25, 50), (64, 50)]],
    "G": [[(78, 27), (58, 10), (28, 20), (18, 51), (31, 82), (64, 88), (79, 67), (60, 62)]],
    "H": [[(22, 14), (22, 88)], [(78, 14), (78, 88)], [(24, 51), (76, 51)]],
    "I": [[(32, 14), (68, 14)], [(50, 14), (50, 88)], [(32, 88), (68, 88)]],
    "J": [[(72, 14), (72, 67), (63, 86), (41, 90), (27, 77)]],
    "K": [[(22, 14), (22, 88)], [(76, 15), (24, 52), (78, 88)]],
    "L": [[(22, 14), (22, 88), (78, 88)]],
    "M": [[(18, 88), (20, 14), (50, 60), (80, 14), (82, 88)]],
    "N": [[(20, 88), (20, 14), (80, 88), (80, 14)]],
    "O": [[(50, 12), (76, 24), (83, 52), (72, 80), (47, 90), (22, 78), (15, 49), (27, 20), (50, 12)]],
    "P": [[(22, 88), (22, 14), (58, 14), (78, 31), (62, 51), (22, 51)]],
    "Q": [[(50, 12), (76, 24), (83, 52), (72, 80), (47, 90), (22, 78), (15, 49), (27, 20), (50, 12)], [(57, 70), (80, 93)]],
    "R": [[(22, 88), (22, 14), (58, 14), (78, 31), (62, 51), (24, 51), (78, 88)]],
    "S": [[(76, 25), (58, 10), (29, 19), (27, 40), (68, 54), (76, 77), (51, 91), (24, 78)]],
    "T": [[(18, 14), (82, 14)], [(50, 14), (50, 88)]],
    "U": [[(20, 14), (21, 66), (32, 86), (50, 91), (69, 86), (80, 65), (80, 14)]],
    "V": [[(18, 14), (50, 89), (82, 14)]],
    "W": [[(14, 14), (30, 88), (50, 48), (70, 88), (86, 14)]],
    "X": [[(20, 14), (80, 88)], [(80, 14), (20, 88)]],
    "Y": [[(18, 14), (50, 48), (82, 14)], [(50, 48), (50, 88)]],
    "Z": [[(20, 14), (80, 14), (22, 88), (82, 88)]],
}

DIGITS: dict[str, Template] = {
    "0": [[(50, 12), (75, 24), (82, 52), (70, 82), (48, 90), (25, 78), (18, 48), (30, 20), (50, 12)]],
    "1": [[(38, 30), (51, 14), (51, 88)], [(35, 88), (68, 88)]],
    "2": [[(25, 31), (41, 12), (68, 17), (75, 38), (22, 88), (78, 88)]],
    "3": [[(27, 22), (53, 11), (75, 27), (57, 49), (76, 66), (61, 88), (30, 78)]],
    "4": [[(70, 88), (70, 13), (20, 63), (82, 63)]],
    "5": [[(74, 15), (28, 15), (24, 46), (55, 43), (78, 61), (66, 86), (33, 82)]],
    "6": [[(72, 22), (52, 12), (28, 30), (20, 61), (36, 87), (67, 83), (76, 60), (58, 45), (28, 52)]],
    "7": [[(22, 15), (80, 15), (45, 88)]],
    "8": [[(50, 50), (73, 36), (64, 14), (37, 14), (27, 36), (50, 50), (77, 66), (62, 90), (36, 87), (24, 66), (50, 50)]],
    "9": [[(72, 49), (45, 54), (24, 39), (34, 16), (65, 16), (80, 43), (70, 74), (48, 90), (28, 80)]],
}

LOWER: dict[str, Template] = {
    "a": [[(70, 43), (54, 33), (31, 42), (25, 65), (42, 82), (66, 71), (70, 43), (72, 82)]],
    "b": [[(27, 14), (27, 88)], [(29, 52), (52, 35), (75, 52), (70, 78), (45, 86), (28, 72)]],
    "c": [[(72, 46), (55, 34), (32, 42), (24, 63), (36, 81), (62, 81), (75, 68)]],
    "d": [[(73, 14), (73, 88)], [(71, 52), (48, 35), (25, 52), (30, 78), (55, 86), (72, 72)]],
    "e": [[(25, 61), (72, 57), (65, 39), (40, 35), (24, 55), (31, 77), (58, 84), (75, 70)]],
    "f": [[(66, 18), (51, 12), (43, 30), (43, 88)], [(28, 45), (62, 45)]],
    "g": [[(70, 43), (52, 33), (29, 43), (25, 66), (43, 80), (68, 70), (70, 43), (68, 96), (45, 108), (25, 94)]],
    "h": [[(27, 14), (27, 88)], [(29, 56), (49, 35), (70, 47), (72, 88)]],
    "i": [[(50, 42), (50, 88)], [(50, 24), (50, 25)]],
    "j": [[(55, 42), (55, 94), (40, 108), (26, 98)], [(55, 24), (55, 25)]],
    "k": [[(27, 14), (27, 88)], [(70, 40), (29, 64), (72, 88)]],
    "l": [[(50, 14), (50, 88)]],
    "m": [[(20, 88), (20, 42), (38, 36), (50, 53), (50, 88)], [(50, 53), (68, 36), (82, 52), (82, 88)]],
    "n": [[(26, 88), (26, 42), (50, 35), (72, 50), (72, 88)]],
    "o": [[(50, 35), (72, 48), (70, 72), (50, 85), (28, 73), (26, 49), (50, 35)]],
    "p": [[(27, 42), (27, 108)], [(29, 52), (52, 35), (75, 52), (70, 78), (45, 86), (28, 72)]],
    "q": [[(70, 43), (52, 33), (29, 43), (25, 66), (43, 80), (68, 70), (70, 43), (70, 108)]],
    "r": [[(28, 88), (28, 43), (48, 36), (67, 43)]],
    "s": [[(70, 45), (49, 35), (28, 45), (35, 61), (67, 66), (71, 80), (48, 87), (27, 76)]],
    "t": [[(47, 20), (47, 82), (60, 88)], [(30, 43), (66, 43)]],
    "u": [[(26, 42), (27, 71), (42, 86), (66, 78), (72, 42), (72, 88)]],
    "v": [[(22, 42), (49, 88), (78, 42)]],
    "w": [[(16, 42), (32, 88), (50, 60), (68, 88), (84, 42)]],
    "x": [[(28, 42), (72, 88)], [(72, 42), (28, 88)]],
    "y": [[(22, 42), (50, 88), (77, 42)], [(50, 88), (38, 108), (24, 101)]],
    "z": [[(28, 42), (72, 42), (29, 88), (74, 88)]],
}


def _template_for_char(ch: str) -> Template:
    if ch in UPPER:
        return UPPER[ch]
    if ch in LOWER:
        return LOWER[ch]
    if ch in DIGITS:
        return DIGITS[ch]
    return []


def _strokes_for_char(ch: str, rng: random.Random) -> list[Stroke]:
    template = _template_for_char(ch)
    if not template:
        return []
    width = rng.uniform(1.30, 1.70)

    is_upper = ch.isupper()
    is_lower = ch.islower()

    if is_upper:
        # Uppercase: clean and balanced.
        # Tight, symmetric parameters so letters look neat and well-proportioned.
        slant = rng.uniform(-0.03, 0.03)          # symmetric – no italic bias
        rot = math.radians(rng.uniform(-1.5, 1.5)) # gentle tilt only
        sx = rng.uniform(0.94, 1.04)               # tight x-scale
        sy = rng.uniform(0.94, 1.04)               # tight y-scale
        tx = rng.uniform(-0.8, 0.8)
        ty = rng.uniform(-0.8, 0.8)
        base_jitter = rng.uniform(0.008, 0.018)    # subtle wobble – stays clean
        compact = 1.0                               # full template size
    elif is_lower:
        # Lowercase: slightly smaller than uppercase + organic handwritten wobble.
        slant = rng.uniform(-0.04, 0.08)           # mild italic feel
        rot = math.radians(rng.uniform(-3.0, 3.0))
        sx = rng.uniform(0.92, 1.06)
        sy = rng.uniform(0.92, 1.06)
        tx = rng.uniform(-1.0, 1.0)
        ty = rng.uniform(-1.0, 1.0)
        base_jitter = rng.uniform(0.030, 0.065)    # stronger wobble for handwritten feel
        compact = 0.88                             # shrink to ~88 % of template size
    else:
        # Digits: moderate parameters.
        slant = rng.uniform(-0.03, 0.05)
        rot = math.radians(rng.uniform(-2.0, 2.0))
        sx = rng.uniform(0.93, 1.05)
        sy = rng.uniform(0.93, 1.05)
        tx = rng.uniform(-1.0, 1.0)
        ty = rng.uniform(-1.0, 1.0)
        base_jitter = rng.uniform(0.018, 0.038)
        compact = 1.0

    cr = math.cos(rot)
    sr = math.sin(rot)

    def transform(pt: tuple[float, float]) -> tuple[float, float]:
        x, y = pt
        # Apply compact scale first (shrinks toward canvas centre 50, 50).
        x = (x - 50.0) * compact
        y = (y - 50.0) * compact
        # Then apply per-variant slant, scale, rotation, shift.
        xp = x * sx + slant * y
        yp = y * sy
        return (50.0 + xp * cr - yp * sr + tx, 50.0 + xp * sr + yp * cr + ty)

    strokes: list[Stroke] = []
    for raw in template:
        pts = [transform(p) for p in raw]
        if len(pts) == 1:
            strokes.append(_dot(pts[0][0], pts[0][1], width=width, rng=rng))
        else:
            # Each stroke gets its own slightly varied wobble strength.
            stroke_jitter = base_jitter * rng.uniform(0.7, 1.3)
            strokes.append(_line(pts, width=width, rng=rng, jitter=stroke_jitter))
    return strokes


def _sequence_from_strokes(strokes: list[Stroke]) -> list[list[float]]:
    seq: list[list[float]] = []
    prev_x = 0.0
    prev_y = 0.0
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        sx, sy, sw = stroke[0]
        seq.append([(sx - prev_x) / 20.0, (sy - prev_y) / 20.0, 0.0, sw / 4.0])
        prev_x, prev_y = sx, sy
        for x, y, w in stroke:
            seq.append([(x - prev_x) / 20.0, (y - prev_y) / 20.0, 1.0, w / 4.0])
            prev_x, prev_y = x, y
        seq.append([0.0, 0.0, 0.0, sw / 4.0])
    return seq


def _generate_latin_sequence(ch: str, rng: random.Random) -> list[list[float]]:
    font_path = "storage/fonts/KleeOne-Regular.ttf"
    size = 128
    
    is_upper = ch.isupper()
    is_lower = ch.islower()
    
    if is_upper:
        # Uppercase: large, clean, very minimal rotation to keep left/right balance
        font_size = int(size * rng.uniform(0.68, 0.72))
        angle = rng.uniform(-0.8, 0.8)  # minimal tilt
        scale = rng.uniform(0.97, 1.03)  # tight scale
        tx = rng.uniform(-0.8, 0.8)
        ty = rng.uniform(-0.8, 0.8)
    elif is_lower:
        # Lowercase: smaller, slightly more slant/rotation for natural hand-written feel
        # Scaled down according to user request: "小文字はちょっとサイズ落として"
        font_size = int(size * rng.uniform(0.52, 0.58))
        angle = rng.uniform(-3.5, 3.5)  # handwriting-like tilt
        scale = rng.uniform(0.93, 1.05)
        tx = rng.uniform(-1.2, 1.2)
        ty = rng.uniform(0.5, 2.0)  # slightly lower baseline adjustment
    else:
        # Digits: clean, balanced
        font_size = int(size * rng.uniform(0.64, 0.68))
        angle = rng.uniform(-1.5, 1.5)
        scale = rng.uniform(0.95, 1.05)
        tx = rng.uniform(-0.8, 0.8)
        ty = rng.uniform(-0.8, 0.8)
        
    canvas = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(canvas)
    
    font = ImageFont.truetype(font_path, font_size)
    bbox = draw.textbbox((0, 0), ch, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    x = (size - tw) / 2.0 - bbox[0] + tx
    y = (size - th) / 2.0 - bbox[1] + ty
    
    draw.text((x, y), ch, fill=0, font=font)
    
    # Apply rotation and scaling
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
    ox = int((size - transformed.width) / 2)
    oy = int((size - transformed.height) / 2)
    out.paste(transformed, (ox, oy))
    
    # Slight blur to smooth edges
    out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.12, 0.28)))
    
    # Thinning & sequence extraction
    arr = _preprocess_labeled_char_image(out, normalize_size=96)
    seq = _image_to_sequence(arr, max_points=120, threshold=128, smooth_profile="latin_klee")
    return seq


def build_latin_dataset(
    *,
    output_path: str | Path,
    variants_per_char: int = 48,
    seed: int = 20260612,
    append: bool = False,
) -> LatinBuildResult:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    rows_written = 0
    with output.open(mode, encoding="utf-8", newline="\n") as f:
        for char_index, ch in enumerate(LATIN_CHARS):
            for variant in range(max(1, variants_per_char)):
                rng = random.Random(f"latin-image-v1:{seed}:{ch}:{variant}")
                seq = _generate_latin_sequence(ch, rng)
                if len(seq) < 8:
                    continue
                payload = {
                    "char_id": char_to_model_id(ch),
                    "style_id": 0,
                    "dataset_id": -970,
                    "user_id": 0,
                    "sequence": seq,
                    "meta": {
                        "source": "latin-generated-v4",
                        "char": ch,
                        "variant": variant,
                        "char_index": char_index,
                    },
                }
                f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows_written += 1
    return LatinBuildResult(
        output_path=str(output),
        char_count=len(LATIN_CHARS),
        rows_written=rows_written,
        variants_per_char=max(1, variants_per_char),
    )
