from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import statistics

from PIL import Image, ImageDraw


DEFAULT_RUNTIME_SOURCE_PATHS = [
    "storage/base/base_dataset_hiragana_images_v2.jsonl",
    "storage/base/base_dataset_hiragana_k49_mix.jsonl",
    "storage/base/base_dataset_hiragana_thin.jsonl",
    "storage/base/base_dataset_katakana_handwritten_like_v8best.jsonl",
    "storage/base/base_dataset_katakana_handwritten_like_v8raw.jsonl",
    "storage/base/base_dataset_kanjivg_joyo_full_v5.jsonl",
    "storage/base/base_dataset_kkanji_joyo_v3_plus.jsonl",
]

_HIRAGANA_STROKE_RANGE: dict[str, tuple[int, int]] = {
    "い": (2, 3),
    "う": (2, 3),
    "え": (2, 4),
    "お": (3, 5),
    "く": (1, 2),
    "ほ": (3, 5),
    "れ": (1, 3),
    "を": (2, 4),
    "す": (1, 3),
    "へ": (1, 2),
    "も": (2, 4),
    "ろ": (1, 2),
}
_KATAKANA_STROKE_RANGE: dict[str, tuple[int, int]] = {
    "ア": (2, 2), "イ": (2, 2), "ウ": (3, 3), "エ": (3, 3), "オ": (3, 3),
    "カ": (2, 2), "キ": (3, 3), "ク": (2, 2), "ケ": (3, 3), "コ": (2, 2),
    "サ": (3, 3), "シ": (3, 3), "ス": (2, 2), "セ": (2, 2), "ソ": (2, 2),
    "タ": (3, 3), "チ": (3, 3), "ツ": (3, 3), "テ": (3, 3), "ト": (2, 2),
    "ナ": (2, 2), "ニ": (2, 2), "ヌ": (2, 2), "ネ": (4, 4), "ノ": (1, 1),
    "ハ": (2, 2), "ヒ": (2, 2), "フ": (1, 1), "ヘ": (1, 1), "ホ": (4, 4),
    "マ": (2, 2), "ミ": (3, 3), "ム": (2, 2), "メ": (2, 2), "モ": (3, 3),
    "ヤ": (2, 2), "ユ": (2, 2), "ヨ": (3, 3), "ラ": (2, 2), "リ": (2, 2),
    "ル": (2, 2), "レ": (1, 1), "ロ": (3, 3), "ワ": (2, 2), "ヲ": (3, 3), "ン": (2, 2),
}


@dataclass
class RuntimeMasterStats:
    output_path: str
    input_files: list[str]
    considered_rows: int
    imported_rows: int
    skipped_rows: int
    unique_chars: int
    per_char_limit: int


def _is_hiragana_char(ch: str) -> bool:
    if len(ch) != 1:
        return False
    code = ord(ch)
    return 0x3041 <= code <= 0x3096


def _is_katakana_char(ch: str) -> bool:
    if len(ch) != 1:
        return False
    code = ord(ch)
    return (
        0x30A1 <= code <= 0x30FA
        or 0x30FD <= code <= 0x30FF
        or 0x31F0 <= code <= 0x31FF
        or 0xFF66 <= code <= 0xFF9D
    )


def _is_kanji_char(ch: str) -> bool:
    if len(ch) != 1:
        return False
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def _is_target_char(ch: str) -> bool:
    return _is_hiragana_char(ch) or _is_katakana_char(ch) or _is_kanji_char(ch)


def _source_allowed(ch: str, source: str, source_profile: str) -> bool:
    key = source_profile.strip().lower()
    if key in {"", "balanced", "default"}:
        return True
    if key != "hq_handwriting":
        return True
    if _is_hiragana_char(ch):
        return source in {"hira-synth-hq", "hiragana-thin", "k49"} or source.startswith("kanjivg-kana")
    if _is_katakana_char(ch):
        return source.startswith("katakana-generated-handlike-v8") or source.startswith("kanjivg-kana")
    if _is_kanji_char(ch):
        return source.startswith("kanjivg-joyo")
    return False


def _sequence_to_strokes(seq: list[list[float]]) -> list[list[tuple[float, float, int]]]:
    if not seq:
        return []
    strokes: list[list[tuple[float, float, int]]] = []
    cur: list[tuple[float, float, int]] = []
    x = 0.0
    y = 0.0
    for row in seq:
        if not isinstance(row, list) or len(row) < 4:
            continue
        x += float(row[0]) * 20.0
        y += float(row[1]) * 20.0
        pen_down = float(row[2]) > 0.5
        w = max(1, min(4, int(round(float(row[3]) * 4.0))))
        if pen_down:
            cur.append((x, y, w))
        elif len(cur) >= 2:
            strokes.append(cur)
            cur = []
        else:
            cur = []
    if len(cur) >= 2:
        strokes.append(cur)
    return strokes


def _stroke_path_len(stroke: list[tuple[float, float, int]]) -> float:
    if len(stroke) < 2:
        return 0.0
    total = 0.0
    px, py, _ = stroke[0]
    for x, y, _w in stroke[1:]:
        total += math.hypot(x - px, y - py)
        px, py = x, y
    return total


def _stroke_turn_penalty(stroke: list[tuple[float, float, int]]) -> float:
    if len(stroke) < 5:
        return 0.0
    sharp = 0
    total = 0
    for i in range(2, len(stroke)):
        x0, y0, _ = stroke[i - 2]
        x1, y1, _ = stroke[i - 1]
        x2, y2, _ = stroke[i]
        v1x = x1 - x0
        v1y = y1 - y0
        v2x = x2 - x1
        v2y = y2 - y1
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        dot = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
        ang = math.acos(dot)
        total += 1
        if ang > 1.18:
            sharp += 1
    if total == 0:
        return 0.0
    return float(sharp) / float(total)


def _step_jump_penalty(seq: list[list[float]]) -> tuple[int, int]:
    big_any = 0
    big_down = 0
    for row in seq:
        if not isinstance(row, list) or len(row) < 3:
            continue
        dx = abs(float(row[0]))
        dy = abs(float(row[1]))
        if dx > 1.8 or dy > 1.8:
            big_any += 1
            if float(row[2]) > 0.5:
                big_down += 1
    return big_any, big_down


def _sequence_quality(seq: list[list[float]], ch: str, source: str) -> float:
    if not isinstance(seq, list) or len(seq) < 8:
        return float("-inf")

    big_any, big_down = _step_jump_penalty(seq)
    if big_any > 6 or big_down > 2:
        return float("-inf")

    strokes = _sequence_to_strokes(seq)
    if not strokes:
        return float("-inf")

    pts = [(x, y) for stroke in strokes for (x, y, _w) in stroke]
    if len(pts) < 10:
        return float("-inf")

    stroke_count = len(strokes)
    point_count = len(pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    if span_x <= 1e-6 or span_y <= 1e-6:
        return float("-inf")

    if _is_hiragana_char(ch):
        stroke_range = _HIRAGANA_STROKE_RANGE.get(ch)
        if stroke_range is not None:
            lo, hi = stroke_range
            if stroke_count < lo or stroke_count > hi:
                return float("-inf")
    if _is_katakana_char(ch):
        stroke_range = _KATAKANA_STROKE_RANGE.get(ch)
        if stroke_range is not None:
            lo, hi = stroke_range
            if stroke_count < lo or stroke_count > hi:
                return float("-inf")

    if _is_hiragana_char(ch) or _is_katakana_char(ch):
        if stroke_count > 8 or point_count > 170:
            return float("-inf")

    total_len = sum(_stroke_path_len(s) for s in strokes)
    turn_pen = sum(_stroke_turn_penalty(s) for s in strokes) / max(1, len(strokes))
    short_frag = sum(1 for s in strokes if len(s) <= 3)
    tiny_frag = sum(1 for s in strokes if _stroke_path_len(s) < 1.8)
    length_ratio = total_len / max(1.0, span_x + span_y)

    if _is_hiragana_char(ch) or _is_katakana_char(ch):
        # Hard rejects for jagged/fragmented kana that become scribbles.
        if turn_pen > 0.34:
            return float("-inf")
        if short_frag > 4:
            return float("-inf")
        if tiny_frag > 3:
            return float("-inf")
        if length_ratio > 3.9:
            return float("-inf")

    aspect = span_y / max(1.0, span_x)
    target_aspect = 0.92 if _is_hiragana_char(ch) else 0.80 if _is_katakana_char(ch) else 1.0
    aspect_penalty = abs(aspect - target_aspect) * 18.0

    source_bonus = 0.0
    if source == "hiragana-dataset-github":
        source_bonus += 18.0
    elif source.startswith("katakana-generated-handlike"):
        source_bonus += 14.0
    elif source.startswith("kanjivg"):
        source_bonus += 12.0
    elif source.startswith("kkanji"):
        source_bonus += 10.0
    elif source == "k49":
        source_bonus += 8.0
    if ch in {"い", "う", "え", "お"} and source == "k49":
        source_bonus -= 18.0

    preferred_strokes = 2.8 if (_is_hiragana_char(ch) or _is_katakana_char(ch)) else 6.0
    stroke_penalty = abs(stroke_count - preferred_strokes) * (5.5 if preferred_strokes < 4 else 2.5)

    return (
        min(180.0, total_len * 0.55)
        + min(120.0, span_x + span_y)
        + source_bonus
        - turn_pen * 64.0
        - short_frag * 8.0
        - tiny_frag * 10.0
        - stroke_penalty
        - max(0.0, length_ratio - 4.2) * 42.0
        - max(0, point_count - 110) * 1.4
        - aspect_penalty
        - big_any * 8.0
        - big_down * 14.0
    )


def _sequence_bitmap(
    seq: list[list[float]],
    *,
    size: int = 64,
    pad: int = 6,
) -> list[int]:
    strokes = _sequence_to_strokes(seq)
    pts = [(x, y, w) for stroke in strokes for (x, y, w) in stroke]
    if len(pts) < 4:
        return [0] * (size * size)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    scale = min((size - 2 * pad) / span_x, (size - 2 * pad) / span_y)
    off_x = (size - span_x * scale) * 0.5
    off_y = (size - span_y * scale) * 0.5

    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        mapped: list[tuple[float, float, int]] = []
        for x, y, w in stroke:
            px = off_x + (x - min_x) * scale
            py = off_y + (y - min_y) * scale
            mapped.append((px, py, w))
        for i in range(1, len(mapped)):
            x0, y0, w0 = mapped[i - 1]
            x1, y1, w1 = mapped[i]
            width = max(1, min(3, int(round((w0 + w1) * 0.5))))
            draw.line((x0, y0, x1, y1), fill=255, width=width)
    data = list(img.getdata())
    return [1 if v >= 32 else 0 for v in data]


def _binary_iou(a: list[int], b: list[int]) -> float:
    inter = 0
    union = 0
    for x, y in zip(a, b):
        if x or y:
            union += 1
            if x and y:
                inter += 1
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _consensus_bonus(bitmaps: list[list[int]], idx: int) -> float:
    target = bitmaps[idx]
    if not bitmaps or not target:
        return 0.0
    n = len(bitmaps)
    if n <= 1:
        return 0.0
    sims: list[float] = []
    for j, other in enumerate(bitmaps):
        if j == idx:
            continue
        sims.append(_binary_iou(target, other))
    if not sims:
        return 0.0
    # Median similarity is robust against a few bad outliers.
    med = statistics.median(sims)
    return med * 120.0


def build_runtime_master_dataset(
    *,
    output_path: str,
    input_paths: list[str] | None = None,
    per_char_limit: int = 24,
    min_score: float = 20.0,
    seed: int = 42,
    source_profile: str = "balanced",
) -> RuntimeMasterStats:
    paths = [Path(p) for p in (input_paths or DEFAULT_RUNTIME_SOURCE_PATHS)]
    rng = random.Random(seed)

    ranked: dict[str, list[tuple[float, str, dict, list[int]]]] = {}
    considered = 0
    skipped = 0

    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = line.strip()
                if not row:
                    continue
                try:
                    payload = json.loads(row)
                except Exception:
                    skipped += 1
                    continue
                if not isinstance(payload, dict):
                    skipped += 1
                    continue

                meta = payload.get("meta")
                seq = payload.get("sequence")
                if not isinstance(meta, dict) or not isinstance(seq, list):
                    skipped += 1
                    continue

                ch = meta.get("char")
                if not isinstance(ch, str) or len(ch) != 1 or not _is_target_char(ch):
                    skipped += 1
                    continue

                source = str(meta.get("source", ""))
                if not _source_allowed(ch, source, source_profile):
                    skipped += 1
                    continue
                score = _sequence_quality(seq, ch, source)
                considered += 1
                if score == float("-inf"):
                    skipped += 1
                    continue
                bitmap = _sequence_bitmap(seq, size=64, pad=6)
                if sum(bitmap) <= 8:
                    skipped += 1
                    continue

                meta2 = dict(meta)
                meta2["runtime_master_score"] = round(float(score), 4)
                meta2["runtime_master_source_file"] = str(path)
                meta2["runtime_master_profile"] = source_profile
                payload2 = dict(payload)
                payload2["meta"] = meta2
                ranked.setdefault(ch, []).append((score, source, payload2, bitmap))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    imported = 0
    with output.open("w", encoding="utf-8") as out:
        for ch in sorted(ranked.keys()):
            rows = ranked[ch]
            rows.sort(key=lambda it: it[0], reverse=True)
            # Evaluate consensus inside top quality bucket only.
            pre_n = min(max(per_char_limit * 4, 24), len(rows))
            pre = rows[:pre_n]
            bitmaps = [it[3] for it in pre]
            rescored: list[tuple[float, str, dict]] = []
            for i, (score, source, payload, _bmp) in enumerate(pre):
                bonus = _consensus_bonus(bitmaps, i)
                # Small source prior for handwritten feel.
                src_bonus = 0.0
                if _is_hiragana_char(ch) and source == "hiragana-thin":
                    src_bonus += 6.0
                if _is_hiragana_char(ch) and source == "hira-synth-hq":
                    src_bonus += 9.0
                if (_is_hiragana_char(ch) or _is_katakana_char(ch)) and source.startswith("kanjivg-kana"):
                    src_bonus += 8.0
                if _is_katakana_char(ch) and source.startswith("katakana-generated-handlike-v8"):
                    src_bonus += 4.0
                if _is_kanji_char(ch) and source.startswith("kanjivg-joyo"):
                    src_bonus += 3.0
                final_score = score + bonus + src_bonus
                rescored.append((final_score, source, payload))
            # Tiny jitter avoids deterministic tie-locking.
            rescored.sort(key=lambda it: (it[0] + rng.uniform(-1e-5, 1e-5)), reverse=True)

            selected: list[tuple[float, str, dict]] = []
            per_source: dict[str, int] = {}
            source_cap = max(2, int(math.ceil(per_char_limit * 0.65)))
            qualified = [item for item in rescored if float(item[0]) >= float(min_score)]
            if not qualified:
                qualified = rescored
            for item in qualified:
                if len(selected) >= per_char_limit:
                    break
                source = item[1]
                if per_source.get(source, 0) >= source_cap:
                    continue
                selected.append(item)
                per_source[source] = per_source.get(source, 0) + 1

            # Fill the remaining slots without source cap.
            if len(selected) < per_char_limit:
                used_ids = {id(x[2]) for x in selected}
                for item in qualified:
                    if len(selected) >= per_char_limit:
                        break
                    if id(item[2]) in used_ids:
                        continue
                    selected.append(item)
                    used_ids.add(id(item[2]))

            for rank, (_score, _source, payload) in enumerate(selected, start=1):
                meta = payload.get("meta")
                if isinstance(meta, dict):
                    meta["runtime_master_rank"] = rank
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                imported += 1

    return RuntimeMasterStats(
        output_path=str(output),
        input_files=[str(p) for p in paths],
        considered_rows=considered,
        imported_rows=imported,
        skipped_rows=skipped,
        unique_chars=len(ranked),
        per_char_limit=per_char_limit,
    )
