from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path

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


def _line(points: list[tuple[float, float]], *, width: float, rng: random.Random, jitter: float = 0.0) -> Stroke:
    out: Stroke = []
    softened_points = _soften_polyline(points, iterations=1)
    for idx in range(len(softened_points) - 1):
        x0, y0 = softened_points[idx]
        x1, y1 = softened_points[idx + 1]
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(8, int(dist / 4.0))
        if dist > 1e-6:
            nx = -(y1 - y0) / dist
            ny = (x1 - x0) / dist
        else:
            nx = 0.0
            ny = 0.0
        bow = rng.uniform(-0.050, 0.050) * dist
        wave = rng.uniform(-0.020, 0.020) * dist
        for step in range(steps + 1):
            if idx > 0 and step == 0:
                continue
            t = step / max(1, steps)
            ease = t * t * (3.0 - 2.0 * t)
            curve = math.sin(math.pi * t) * bow + math.sin(math.tau * t + idx * 0.71) * wave
            x = x0 + (x1 - x0) * ease + nx * curve + rng.uniform(-jitter, jitter)
            y = y0 + (y1 - y0) * ease + ny * curve + rng.uniform(-jitter, jitter)
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
    out: Stroke = []
    for i in range(points):
        t = i / max(1, points - 1)
        mt = 1.0 - t
        x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
        y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
        out.append((x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter), width * rng.uniform(0.92, 1.10)))
    return out


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
    out: Stroke = []
    for i in range(points):
        t = i / max(1, points - 1)
        mt = 1.0 - t
        x = mt**3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t**3 * p3[1]
        out.append((x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter), width * rng.uniform(0.92, 1.10)))
    return out


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
    width = rng.uniform(1.25, 1.95)
    slant = rng.uniform(-0.10, 0.16)
    rot = math.radians(rng.uniform(-3.2, 3.2))
    sx = rng.uniform(0.88, 1.08)
    sy = rng.uniform(0.90, 1.08)
    tx = rng.uniform(-2.4, 2.4)
    ty = rng.uniform(-2.4, 2.4)
    cr = math.cos(rot)
    sr = math.sin(rot)

    def transform(pt: tuple[float, float]) -> tuple[float, float]:
        x, y = pt
        x = (x - 50.0) * sx + slant * (y - 50.0)
        y = (y - 50.0) * sy
        return (50.0 + x * cr - y * sr + tx, 50.0 + x * sr + y * cr + ty)

    strokes: list[Stroke] = []
    for raw in template:
        pts = [transform(p) for p in raw]
        if len(pts) == 1:
            strokes.append(_dot(pts[0][0], pts[0][1], width=width, rng=rng))
        else:
            strokes.append(_line(pts, width=width, rng=rng, jitter=rng.uniform(0.10, 0.32)))
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
                rng = random.Random(f"latin-v2:{seed}:{ch}:{variant}")
                strokes = _strokes_for_char(ch, rng)
                seq = _sequence_from_strokes(strokes)
                if len(seq) < 8:
                    continue
                payload = {
                    "char_id": char_to_model_id(ch),
                    "style_id": 0,
                    "dataset_id": -970,
                    "user_id": 0,
                    "sequence": seq,
                    "meta": {
                        "source": "latin-generated-v2",
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
