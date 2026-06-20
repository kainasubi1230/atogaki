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


def text_to_svg_english(text: str, watermark_text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        lines = [text if text else ""]

    font_size = 44
    line_height = 58
    start_x = 22.0
    start_y = 54.0

    seed = int(hashlib.sha256(f"en:{text}".encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)

    def advance(ch: str) -> float:
        if ch == " ":
            return 14.0 + rng.uniform(-1.2, 2.2)
        if ch in "ilI.,'!|":
            return 12.0 + rng.uniform(-1.0, 1.4)
        if ch in "mwMW@#":
            return 34.0 + rng.uniform(-2.4, 2.6)
        if ch.isupper():
            return 27.0 + rng.uniform(-2.0, 2.5)
        return 23.0 + rng.uniform(-2.2, 2.2)

    tspans: list[str] = []
    max_x = start_x
    for line_idx, line in enumerate(lines):
        x = start_x + rng.uniform(-1.5, 1.5)
        baseline = start_y + line_idx * line_height + rng.uniform(-1.0, 1.8)
        for ch in line:
            if ch == " ":
                x += advance(ch)
                continue
            rotate = rng.uniform(-6.0, 6.0)
            y = baseline + rng.uniform(-2.2, 2.8)
            char_size = font_size + rng.uniform(-2.8, 2.0)
            tspans.append(
                f"<tspan x='{x:.1f}' y='{y:.1f}' rotate='{rotate:.1f}' font-size='{char_size:.1f}'>{escape(ch)}</tspan>"
            )
            x += advance(ch)
        max_x = max(max_x, x)

    width = max(240, int(max_x + 30))
    height = max(118, int(start_y + len(lines) * line_height + 28))
    watermark_y = height - 14
    glyphs = "".join(tspans)
    font_family = (
        "'Segoe Print', 'Bradley Hand', 'Comic Sans MS', "
        "'Marker Felt', 'Chalkboard SE', cursive"
    )
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
        "<rect width='100%' height='100%' fill='white'/>"
        f"<text fill='black' font-size='{font_size}' font-family=\"{font_family}\" "
        "style='font-weight:500;letter-spacing:0'>"
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

    # 1. 軌跡からストロークを抽出（座標と太さ width を取得）
    strokes: list[list[tuple[float, float, float]]] = []
    cur_stroke: list[tuple[float, float, float]] = []
    for p in trajectory:
        state = str(p.get("pen_state", "up"))
        if state != "down":
            if cur_stroke:
                strokes.append(cur_stroke)
                cur_stroke = []
            continue
        x = float(p.get("x", 0.0)) - min_x + pad_x
        y = float(p.get("y", 0.0)) - min_y + pad_y
        w = float(p.get("width", 1.5))
        if not cur_stroke or cur_stroke[-1][:2] != (x, y):
            cur_stroke.append((x, y, w))
    if cur_stroke:
        strokes.append(cur_stroke)

    # 2. 各ストロークを描画する SVG 要素を作成
    svg_paths: list[str] = []
    for stroke in strokes:
        n = len(stroke)
        if n < 2:
            continue

        # 隣り合う点同士の距離と累積距離を計算
        dists = [0.0]
        cum_dists = [0.0]
        for i in range(1, n):
            dx = stroke[i][0] - stroke[i-1][0]
            dy = stroke[i][1] - stroke[i-1][1]
            d = (dx*dx + dy*dy) ** 0.5
            dists.append(d)
            cum_dists.append(cum_dists[-1] + d)

        total_len = cum_dists[-1]

        # 物理的な進捗距離に基づくフェード（テーパリング）の長さ設定
        # 短い画（点など）は極端に細くなったり消えたりしないように制限
        if total_len < 10.0:
            taper_start = total_len * 0.15
            taper_end = total_len * 0.25
        else:
            taper_start = 3.5
            taper_end = 5.0

        def get_fade_at(dist_from_start: float) -> float:
            if total_len <= 0.01:
                return 1.0
            d_from_start = dist_from_start
            d_from_end = total_len - dist_from_start
            
            f_start = 1.0 if taper_start <= 0 else min(1.0, d_from_start / taper_start)
            f_end = 1.0 if taper_end <= 0 else min(1.0, d_from_end / taper_end)
            
            # Smoothstep による滑らかな非線形フェード (3t^2 - 2t^3)
            f_start = 3 * (f_start ** 2) - 2 * (f_start ** 3)
            f_end = 3 * (f_end ** 2) - 2 * (f_end ** 3)
            return f_start * f_end

        stroke_paths: list[str] = []

        # n == 2 の場合 (単純な直線)
        if n == 2:
            x0, y0, w0 = stroke[0]
            x1, y1, w1 = stroke[1]
            if dists[1] < 0.5:
                continue
            w_base = (w0 + w1) / 2.0
            vel_factor = max(0.65, min(1.2, 1.2 - 0.12 * dists[1]))
            fade = get_fade_at(dists[1] / 2.0)
            w_eff = w_base * vel_factor * (0.45 + 0.55 * fade)
            w_eff = max(0.55, min(2.2, w_eff))
            
            d_path = f"M{x0:.2f},{y0:.2f} L{x1:.2f},{y1:.2f}"
            stroke_paths.append(
                f"<path d='{d_path}' stroke='#1a1a1a' fill='none' stroke-width='{w_eff:.2f}' "
                f"stroke-linecap='round' stroke-linejoin='round'/>"
            )
        else:
            # --- 最初の区間 (制御点: stroke[1], 終点: m1) ---
            x0, y0, w0 = stroke[0]
            x1, y1, w1 = stroke[1]
            x2, y2, _ = stroke[2]
            m1_x = (x1 + x2) / 2.0
            m1_y = (y1 + y2) / 2.0
            
            w_base = w1
            d = dists[1]
            vel_factor = max(0.65, min(1.2, 1.2 - 0.12 * d))
            fade = get_fade_at(cum_dists[1] / 2.0)
            w_eff = w_base * vel_factor * (0.45 + 0.55 * fade)
            w_eff = max(0.55, min(2.2, w_eff))
            
            d_path = f"M{x0:.2f},{y0:.2f} Q{x1:.2f},{y1:.2f} {m1_x:.2f},{m1_y:.2f}"
            stroke_paths.append(
                f"<path d='{d_path}' stroke='#1a1a1a' fill='none' stroke-width='{w_eff:.2f}' "
                f"stroke-linecap='round' stroke-linejoin='round'/>"
            )

            # --- 中間区間 (制御点: stroke[idx]) ---
            for idx in range(2, n - 1):
                prev_x, prev_y, _ = stroke[idx-1]
                cx, cy, cw = stroke[idx]
                nx, ny, _ = stroke[idx+1]
                m_prev_x = (prev_x + cx) / 2.0
                m_prev_y = (prev_y + cy) / 2.0
                m_next_x = (cx + nx) / 2.0
                m_next_y = (cy + ny) / 2.0

                w_base = cw
                d = dists[idx]
                vel_factor = max(0.65, min(1.2, 1.2 - 0.12 * d))
                fade = get_fade_at(cum_dists[idx])
                w_eff = w_base * vel_factor * (0.45 + 0.55 * fade)
                w_eff = max(0.55, min(2.2, w_eff))

                d_path = f"M{m_prev_x:.2f},{m_prev_y:.2f} Q{cx:.2f},{cy:.2f} {m_next_x:.2f},{m_next_y:.2f}"
                stroke_paths.append(
                    f"<path d='{d_path}' stroke='#1a1a1a' fill='none' stroke-width='{w_eff:.2f}' "
                    f"stroke-linecap='round' stroke-linejoin='round'/>"
                )

            # --- 最後の区間 (m_{n-2} -> p_{n-1} 直線) ---
            penult_x, penult_y, _ = stroke[-2]
            last_x, last_y, last_w = stroke[-1]
            m_last_x = (penult_x + last_x) / 2.0
            m_last_y = (penult_y + last_y) / 2.0

            w_base = last_w
            d = dists[-1]
            vel_factor = max(0.65, min(1.2, 1.2 - 0.12 * d))
            fade = get_fade_at(total_len - dists[-1] / 2.0)
            w_eff = w_base * vel_factor * (0.45 + 0.55 * fade)
            w_eff = max(0.55, min(2.2, w_eff))

            d_path = f"M{m_last_x:.2f},{m_last_y:.2f} L{last_x:.2f},{last_y:.2f}"
            stroke_paths.append(
                f"<path d='{d_path}' stroke='#1a1a1a' fill='none' stroke-width='{w_eff:.2f}' "
                f"stroke-linecap='round' stroke-linejoin='round'/>"
            )

        # 全ての path を `<g opacity='0.92'>` で囲み、重複部分の二重半透明ブレンド（数珠状の斑点）を防ぎつつ、紙染みインク感を表現
        paths_joined_str = "\n  ".join(stroke_paths)
        svg_paths.append(
            f"<g opacity='0.92'>\n  {paths_joined_str}\n</g>"
        )

    paths_joined = "\n".join(svg_paths)
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{int(round(width))}' height='{int(round(height))}' viewBox='0 0 {width:.2f} {height:.2f}'>"
        "<rect width='100%' height='100%' fill='white'/>"
        f"{paths_joined}"
        f"<text x='12' y='{height - 14:.2f}' fill='#c62828' font-size='14'>{escape(watermark_text)}</text>"
        "</svg>"
    )
