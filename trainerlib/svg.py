from __future__ import annotations

import hashlib
from html import escape
import random


def text_to_svg(text: str, watermark_text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        lines = [text if text else ""]

    font_size = 46
    line_height = 62
    start_x = 20.0
    start_y = 56.0

    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    tspans: list[str] = []
    max_x = start_x
    for line_idx, line in enumerate(lines):
        x = start_x
        baseline = start_y + line_idx * line_height + rng.uniform(-1.0, 1.0)
        for ch in line:
            dx = 26.5 + rng.uniform(-4.0, 2.4)
            rotate = rng.uniform(-9.0, 9.0)
            y = baseline + rng.uniform(-2.6, 2.6)
            char_size = font_size + rng.uniform(-2.2, 1.6)
            tspans.append(
                f"<tspan x='{x:.1f}' y='{y:.1f}' rotate='{rotate:.1f}' font-size='{char_size:.1f}'>{escape(ch)}</tspan>"
            )
            x += dx
        max_x = max(max_x, x)

    width = max(220, int(max_x + 28))
    height = max(120, int(start_y + len(lines) * line_height + 26))
    watermark_y = height - 14
    glyphs = "".join(tspans)
    # Prefer common Japanese system fonts; fallback keeps text readable.
    font_family = (
        "'Klee One', 'Yusei Magic', 'Shippori Mincho', "
        "'Yuji Syuku', 'Hiragino Mincho ProN', 'Yu Mincho', serif"
    )
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
        "<rect width='100%' height='100%' fill='white'/>"
        f"<text fill='black' font-size='{font_size}' font-family=\"{font_family}\" "
        "style='font-weight:500;letter-spacing:0.45px'>"
        f"{glyphs}</text>"
        f"<text x='12' y='{watermark_y}' fill='#c62828' font-size='14'>{escape(watermark_text)}</text>"
        "</svg>"
    )


def text_to_svg_readable(text: str, watermark_text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        lines = [text if text else ""]

    font_size = 46
    line_height = 64
    start_x = 20
    start_y = 58

    max_chars = max((len(line) for line in lines), default=1)
    width = max(220, int(start_x + max_chars * 30 + 28))
    height = max(120, int(start_y + len(lines) * line_height + 24))
    watermark_y = height - 14

    font_family = (
        "'Noto Sans JP', 'Hiragino Kaku Gothic ProN', 'Yu Gothic', "
        "'Meiryo', sans-serif"
    )

    tspans: list[str] = []
    for line_idx, line in enumerate(lines):
        y = start_y + line_idx * line_height
        escaped = escape(line)
        tspans.append(f"<tspan x='{start_x}' y='{y}'>{escaped}</tspan>")
    glyphs = "".join(tspans)

    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
        "<rect width='100%' height='100%' fill='white'/>"
        f"<text fill='black' font-size='{font_size}' font-family=\"{font_family}\" "
        "style='font-weight:500;letter-spacing:0.2px'>"
        f"{glyphs}</text>"
        f"<text x='12' y='{watermark_y}' fill='#c62828' font-size='14'>{escape(watermark_text)}</text>"
        "</svg>"
    )


def trajectory_to_svg(trajectory: list[dict], watermark_text: str) -> str:
    if not trajectory:
        return (
            "<svg xmlns='http://www.w3.org/2000/svg' width='800' height='160'>"
            f"<text x='10' y='140' fill='#d11'>{escape(watermark_text)}</text></svg>"
        )

    xs = [float(p.get("x", 0.0)) for p in trajectory]
    ys = [float(p.get("y", 0.0)) for p in trajectory]
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)
    pad_x = 20
    pad_y = 18
    width = max(120.0, (max_x - min_x) + pad_x * 2)
    height = max(96.0, (max_y - min_y) + pad_y * 2 + 20)

    strokes: list[list[tuple[float, float]]] = []
    cur_stroke: list[tuple[float, float]] = []
    for p in trajectory:
        state = str(p.get("pen_state", "up"))
        if state != "down":
            if cur_stroke:
                strokes.append(cur_stroke)
                cur_stroke = []
            continue
        x = float(p.get("x", 0.0)) - min_x + pad_x
        y = float(p.get("y", 0.0)) - min_y + pad_y
        if not cur_stroke or cur_stroke[-1] != (x, y):
            cur_stroke.append((x, y))
    if cur_stroke:
        strokes.append(cur_stroke)

    path_parts: list[str] = []
    for stroke in strokes:
        if len(stroke) == 1:
            continue
        if len(stroke) == 2:
            x0, y0 = stroke[0]
            x1, y1 = stroke[1]
            if ((x1 - x0) ** 2 + (y1 - y0) ** 2) < 1.0:
                continue
            path_parts.append(f"M{x0:.2f},{y0:.2f} L{x1:.2f},{y1:.2f}")
            continue
        x0, y0 = stroke[0]
        path_parts.append(f"M{x0:.2f},{y0:.2f}")
        for idx in range(1, len(stroke) - 1):
            cx, cy = stroke[idx]
            nx, ny = stroke[idx + 1]
            mx = (cx + nx) / 2.0
            my = (cy + ny) / 2.0
            path_parts.append(f"Q{cx:.2f},{cy:.2f} {mx:.2f},{my:.2f}")
        lx, ly = stroke[-1]
        path_parts.append(f"L{lx:.2f},{ly:.2f}")

    path_data = " ".join(path_parts)
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{int(round(width))}' height='{int(round(height))}' viewBox='0 0 {width:.2f} {height:.2f}'>"
        "<rect width='100%' height='100%' fill='white'/>"
        f"<path d='{path_data}' stroke='black' fill='none' stroke-width='1.25' vector-effect='non-scaling-stroke' stroke-linecap='round' stroke-linejoin='round'/>"
        f"<text x='12' y='{height - 14:.2f}' fill='#c62828' font-size='14'>{escape(watermark_text)}</text>"
        "</svg>"
    )
