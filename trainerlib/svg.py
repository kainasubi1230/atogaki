from __future__ import annotations

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

    rng = random.Random(hash(text) & 0xFFFFFFFF)
    tspans: list[str] = []
    max_x = start_x
    for line_idx, line in enumerate(lines):
        x = start_x
        baseline = start_y + line_idx * line_height + rng.uniform(-1.0, 1.0)
        for ch in line:
            dx = 26.0 + rng.uniform(-3.0, 2.0)
            rotate = rng.uniform(-7.0, 7.0)
            y = baseline + rng.uniform(-2.2, 2.2)
            tspans.append(
                f"<tspan x='{x:.1f}' y='{y:.1f}' rotate='{rotate:.1f}'>{escape(ch)}</tspan>"
            )
            x += dx
        max_x = max(max_x, x)

    width = max(220, int(max_x + 28))
    height = max(120, int(start_y + len(lines) * line_height + 26))
    watermark_y = height - 14
    glyphs = "".join(tspans)
    # Prefer common Japanese system fonts; fallback keeps text readable.
    font_family = (
        "'Hiragino Kaku Gothic ProN', 'Yu Gothic', 'Meiryo', "
        "'Noto Sans JP', sans-serif"
    )
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
        "<rect width='100%' height='100%' fill='white'/>"
        f"<text fill='black' font-size='{font_size}' font-family=\"{font_family}\" "
        "style='font-weight:500;letter-spacing:0.5px'>"
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

    width = max(p["x"] for p in trajectory) + 40
    height = max(p["y"] for p in trajectory) + 60

    path_parts: list[str] = []
    pen_down = False
    for p in trajectory:
        cmd = "M" if not pen_down or p["pen_state"] != "down" else "L"
        path_parts.append(f"{cmd}{p['x']},{p['y']}")
        pen_down = p["pen_state"] == "down"

    path_data = " ".join(path_parts)
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
        "<rect width='100%' height='100%' fill='white'/>"
        f"<path d='{path_data}' stroke='black' fill='none' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/>"
        f"<text x='12' y='{height - 14}' fill='#c62828' font-size='14'>{escape(watermark_text)}</text>"
        "</svg>"
    )
