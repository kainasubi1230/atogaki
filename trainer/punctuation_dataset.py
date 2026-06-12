from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


PUNCTUATION_CHARS = (
    "。、，．！？!?・…ー〜—-－"
    "「」『』（）()【】[]［］{}｛｝〈〉《》〔〕"
    "：；:;,.／/＼\\｜|＋+=＝＊*＆&％%＃#＠@￥¥$"
    "“”‘’\"'"
)


@dataclass
class PunctuationBuildResult:
    output_path: str
    char_count: int
    rows_written: int
    variants_per_char: int


Point = tuple[float, float, float]
Stroke = list[Point]


def _jitter(rng: random.Random, value: float, amount: float) -> float:
    return value + rng.uniform(-amount, amount)


def _bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    points: int,
    width: float,
    rng: random.Random,
    jitter: float = 0.45,
) -> Stroke:
    out: Stroke = []
    for i in range(points):
        t = i / max(1, points - 1)
        mt = 1.0 - t
        x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
        y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
        taper = 1.0 - 0.22 * t
        out.append((_jitter(rng, x, jitter), _jitter(rng, y, jitter), max(1.1, width * taper)))
    return out


def _cubic(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    *,
    points: int,
    width: float,
    rng: random.Random,
    jitter: float = 0.45,
) -> Stroke:
    out: Stroke = []
    for i in range(points):
        t = i / max(1, points - 1)
        mt = 1.0 - t
        x = mt**3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t**3 * p3[1]
        taper = 1.0 - 0.14 * t
        out.append((_jitter(rng, x, jitter), _jitter(rng, y, jitter), max(1.0, width * taper)))
    return out


def _oval(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    *,
    points: int,
    width: float,
    rng: random.Random,
    wobble: float = 0.08,
) -> Stroke:
    out: Stroke = []
    phase = rng.uniform(-0.15, 0.15)
    for i in range(points + 1):
        t = math.tau * i / points + phase
        wob = 1.0 + rng.uniform(-wobble, wobble)
        x = cx + math.cos(t) * rx * wob
        y = cy + math.sin(t) * ry * wob
        out.append((x, y, width * (0.94 + rng.random() * 0.12)))
    return out


def _line(points: list[tuple[float, float]], *, width: float, rng: random.Random, jitter: float = 0.55) -> Stroke:
    out: Stroke = []
    for idx in range(len(points) - 1):
        x0, y0 = points[idx]
        x1, y1 = points[idx + 1]
        for step in range(12):
            if idx > 0 and step == 0:
                continue
            t = step / 11.0
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            out.append((_jitter(rng, x, jitter), _jitter(rng, y, jitter), width * (0.92 + rng.random() * 0.14)))
    return out


def _wave(*, rng: random.Random, width: float) -> Stroke:
    out: Stroke = []
    for i in range(18):
        t = i / 17.0
        x = 12 + t * 76
        y = 50 + math.sin(t * math.tau - 0.3) * 8
        out.append((_jitter(rng, x, 0.45), _jitter(rng, y, 0.45), width))
    return out


def _strokes_for_char(ch: str, rng: random.Random) -> list[Stroke]:
    width = rng.uniform(1.35, 1.85)
    if ch in {"。", "．"}:
        return [_oval(50, 50, rng.uniform(7.0, 9.0), rng.uniform(6.3, 8.2), points=18, width=width, rng=rng)]
    if ch in {".", "・"}:
        return [_oval(50, 50, rng.uniform(3.0, 4.3), rng.uniform(3.0, 4.2), points=14, width=width, rng=rng)]
    if ch in {"、", "，", ","}:
        # Keep it closer to a lower teardrop than a long diagonal slash.
        return [
            _bezier(
                (_jitter(rng, 44, 0.9), _jitter(rng, 45, 1.1)),
                (_jitter(rng, 47, 1.0), _jitter(rng, 58, 1.2)),
                (_jitter(rng, 55, 1.2), _jitter(rng, 70, 1.4)),
                points=11,
                width=rng.uniform(1.65, 2.18),
                rng=rng,
                jitter=0.28,
            )
        ]
    if ch in {"！", "!"}:
        main = _bezier((50, 10), (48, 34), (47, 61), points=15, width=rng.uniform(1.55, 2.1), rng=rng)
        dot = _oval(47.5, 80, 5.0, 5.4, points=12, width=rng.uniform(1.5, 2.0), rng=rng)
        return [main, dot]
    if ch in {"？", "?"}:
        top = _cubic((31, 24), (42, 6), (68, 10), (66, 31), points=16, width=rng.uniform(1.55, 2.05), rng=rng)
        tail = _bezier((66, 31), (58, 47), (50, 60), points=11, width=rng.uniform(1.35, 1.85), rng=rng)
        dot = _oval(49, 81, 4.8, 5.2, points=12, width=rng.uniform(1.45, 1.95), rng=rng)
        return [top, tail, dot]
    if ch == "…":
        return [
            _oval(25, 50, 3.4, 3.7, points=12, width=width, rng=rng),
            _oval(50, 50, 3.4, 3.7, points=12, width=width, rng=rng),
            _oval(75, 50, 3.4, 3.7, points=12, width=width, rng=rng),
        ]
    if ch == "ー":
        return [_bezier((12, 51), (45, 48), (88, 51), points=18, width=rng.uniform(1.45, 1.95), rng=rng, jitter=0.35)]
    if ch == "〜":
        return [_wave(rng=rng, width=rng.uniform(1.35, 1.85))]
    if ch in {"：", ":"}:
        return [
            _oval(50, 32, 4.0, 4.4, points=12, width=width, rng=rng),
            _oval(50, 69, 4.0, 4.4, points=12, width=width, rng=rng),
        ]
    if ch in {"；", ";"}:
        return [
            _oval(50, 31, 4.0, 4.4, points=12, width=width, rng=rng),
            _bezier((52, 61), (48, 75), (40, 88), points=10, width=width, rng=rng),
        ]
    if ch in {"「", "『", "【", "[", "［", "{", "｛", "〈", "《", "〔"}:
        inset = 22 if ch in {"「", "【", "[", "［", "{", "｛", "〈", "〔"} else 27
        if ch in {"〈", "《"}:
            return [_line([(70, 12), (34, 50), (70, 88)], width=rng.uniform(1.45, 1.95), rng=rng)]
        if ch in {"{", "｛"}:
            return [_cubic((66, 10), (38, 20), (56, 39), (39, 50), points=13, width=rng.uniform(1.35, 1.85), rng=rng), _cubic((39, 50), (56, 61), (38, 80), (66, 90), points=13, width=rng.uniform(1.35, 1.85), rng=rng)]
        return [_line([(78, 12), (inset, 12), (inset, 88)], width=rng.uniform(1.55, 2.05), rng=rng)]
    if ch in {"」", "』", "】", "]", "］", "}", "｝", "〉", "》", "〕"}:
        inset = 78 if ch in {"」", "】", "]", "］", "}", "｝", "〉", "〕"} else 73
        if ch in {"〉", "》"}:
            return [_line([(30, 12), (66, 50), (30, 88)], width=rng.uniform(1.45, 1.95), rng=rng)]
        if ch in {"}", "｝"}:
            return [_cubic((34, 10), (62, 20), (44, 39), (61, 50), points=13, width=rng.uniform(1.35, 1.85), rng=rng), _cubic((61, 50), (44, 61), (62, 80), (34, 90), points=13, width=rng.uniform(1.35, 1.85), rng=rng)]
        return [_line([(22, 88), (inset, 88), (inset, 12)], width=rng.uniform(1.55, 2.05), rng=rng)]
    if ch in {"（", "("}:
        return [_cubic((67, 8), (42, 24), (36, 74), (67, 94), points=24, width=rng.uniform(1.45, 1.95), rng=rng)]
    if ch in {"）", ")"}:
        return [_cubic((33, 8), (58, 24), (64, 74), (33, 94), points=24, width=rng.uniform(1.45, 1.95), rng=rng)]
    if ch in {"／", "/"}:
        return [_line([(72, 12), (28, 88)], width=rng.uniform(1.35, 1.9), rng=rng)]
    if ch in {"＼", "\\"}:
        return [_line([(28, 12), (72, 88)], width=rng.uniform(1.35, 1.9), rng=rng)]
    if ch in {"｜", "|"}:
        return [_line([(50, 10), (49, 90)], width=rng.uniform(1.35, 1.85), rng=rng)]
    if ch in {"—", "-", "－"}:
        return [_bezier((15, 52), (48, 49), (85, 52), points=16, width=rng.uniform(1.35, 1.85), rng=rng, jitter=0.32)]
    if ch in {"＋", "+"}:
        return [
            _line([(18, 52), (82, 51)], width=rng.uniform(1.35, 1.85), rng=rng),
            _line([(50, 18), (50, 84)], width=rng.uniform(1.35, 1.85), rng=rng),
        ]
    if ch in {"=", "＝"}:
        return [
            _bezier((18, 40), (48, 37), (82, 40), points=12, width=width, rng=rng, jitter=0.28),
            _bezier((18, 61), (48, 58), (82, 61), points=12, width=width, rng=rng, jitter=0.28),
        ]
    if ch in {"＊", "*"}:
        return [
            _line([(50, 20), (50, 80)], width=width, rng=rng),
            _line([(24, 34), (76, 66)], width=width, rng=rng),
            _line([(76, 34), (24, 66)], width=width, rng=rng),
        ]
    if ch in {"＆", "&"}:
        return [
            _cubic((60, 29), (44, 10), (28, 27), (43, 43), points=16, width=width, rng=rng),
            _cubic((43, 43), (75, 68), (50, 93), (29, 72), points=20, width=width, rng=rng),
            _line([(32, 55), (78, 90)], width=width, rng=rng),
        ]
    if ch in {"％", "%"}:
        return [
            _oval(31, 31, 7, 8, points=12, width=width, rng=rng),
            _line([(72, 16), (28, 86)], width=width, rng=rng),
            _oval(69, 70, 7, 8, points=12, width=width, rng=rng),
        ]
    if ch in {"＃", "#"}:
        return [
            _line([(37, 18), (31, 84)], width=width, rng=rng),
            _line([(63, 18), (57, 84)], width=width, rng=rng),
            _line([(18, 40), (82, 40)], width=width, rng=rng),
            _line([(18, 62), (82, 62)], width=width, rng=rng),
        ]
    if ch in {"＠", "@"}:
        return [
            _oval(50, 52, 28, 30, points=28, width=width, rng=rng),
            _oval(51, 52, 9, 11, points=16, width=width, rng=rng),
            _bezier((60, 52), (68, 73), (80, 57), points=12, width=width, rng=rng),
        ]
    if ch in {"￥", "¥"}:
        return [
            _line([(28, 16), (50, 47), (72, 16)], width=width, rng=rng),
            _line([(50, 47), (50, 88)], width=width, rng=rng),
            _line([(30, 55), (70, 55)], width=width, rng=rng),
            _line([(31, 68), (69, 68)], width=width, rng=rng),
        ]
    if ch == "$":
        return [
            _line([(50, 14), (50, 88)], width=width, rng=rng),
            _cubic((68, 29), (47, 12), (25, 29), (39, 47), points=15, width=width, rng=rng),
            _cubic((39, 47), (76, 57), (65, 88), (31, 74), points=18, width=width, rng=rng),
        ]
    if ch in {"“", "”", "\""}:
        return [
            _bezier((41, 26), (36, 39), (39, 50), points=9, width=width, rng=rng),
            _bezier((58, 26), (53, 39), (56, 50), points=9, width=width, rng=rng),
        ]
    if ch in {"‘", "’", "'"}:
        return [_bezier((50, 25), (45, 39), (49, 51), points=10, width=width, rng=rng)]
    return []


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


def build_punctuation_dataset(
    *,
    output_path: str | Path,
    variants_per_char: int = 48,
    seed: int = 20260529,
    append: bool = False,
) -> PunctuationBuildResult:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    rows_written = 0
    with output.open(mode, encoding="utf-8", newline="\n") as f:
        for char_index, ch in enumerate(PUNCTUATION_CHARS):
            for variant in range(max(1, variants_per_char)):
                rng = random.Random(f"punctuation-v1:{seed}:{ch}:{variant}")
                strokes = _strokes_for_char(ch, rng)
                seq = _sequence_from_strokes(strokes)
                if len(seq) < 8:
                    continue
                payload = {
                    "char_id": ord(ch),
                    "style_id": 0,
                    "dataset_id": -960,
                    "user_id": 0,
                    "sequence": seq,
                    "meta": {
                        "source": "punctuation-generated-v1",
                        "char": ch,
                        "variant": variant,
                        "char_index": char_index,
                    },
                }
                f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows_written += 1
    return PunctuationBuildResult(
        output_path=str(output),
        char_count=len(PUNCTUATION_CHARS),
        rows_written=rows_written,
        variants_per_char=max(1, variants_per_char),
    )
