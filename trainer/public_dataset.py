from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import random
import struct
import tarfile
from urllib.request import urlretrieve

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from trainerlib.char_token import char_to_model_id


K49_FILES = {
    "train_images": "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz",
    "train_labels": "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz",
    "test_images": "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz",
    "test_labels": "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz",
    "classmap": "http://codh.rois.ac.jp/kmnist/dataset/k49/k49_classmap.csv",
}

MNIST_FILES = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

KKANJI_FILES = {
    "archive": "http://codh.rois.ac.jp/kmnist/dataset/kkanji/kkanji.tar",
}

HIRAGANA_ROMAJI_TO_CHAR = {
    "A": "あ",
    "I": "い",
    "U": "う",
    "E": "え",
    "O": "お",
    "KA": "か",
    "KI": "き",
    "KU": "く",
    "KE": "け",
    "KO": "こ",
    "SA": "さ",
    "SHI": "し",
    "SU": "す",
    "SE": "せ",
    "SO": "そ",
    "TA": "た",
    "CHI": "ち",
    "TSU": "つ",
    "TE": "て",
    "TO": "と",
    "NA": "な",
    "NI": "に",
    "NU": "ぬ",
    "NE": "ね",
    "NO": "の",
    "HA": "は",
    "HI": "ひ",
    "FU": "ふ",
    "HE": "へ",
    "HO": "ほ",
    "MA": "ま",
    "MI": "み",
    "MU": "む",
    "ME": "め",
    "MO": "も",
    "YA": "や",
    "YU": "ゆ",
    "YO": "よ",
    "RA": "ら",
    "RI": "り",
    "RU": "る",
    "RE": "れ",
    "RO": "ろ",
    "WA": "わ",
    "WO": "を",
    "N": "ん",
    "BA": "ば",
    "DA": "だ",
    "JI": "じ",
    "PI": "ぴ",
}

SUPPORTED_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


@dataclass
class ImportStats:
    source: str
    requested_count: int
    imported_count: int
    skipped_count: int
    output_path: str
    image_shape: list[int]


def _is_hiragana_char(char: str) -> bool:
    if len(char) != 1:
        return False
    code = ord(char)
    return 0x3041 <= code <= 0x3096


def _is_katakana_char(char: str) -> bool:
    if len(char) != 1:
        return False
    code = ord(char)
    return 0x30A1 <= code <= 0x30FA


def _is_kanji_char(char: str) -> bool:
    if len(char) != 1:
        return False
    code = ord(char)
    # CJK Unified Ideographs + Extension A.
    return (0x3400 <= code <= 0x4DBF) or (0x4E00 <= code <= 0x9FFF)


def _char_matches_script(char: str, script_filter: str) -> bool:
    key = script_filter.strip().lower()
    if key == "all":
        return True
    if key == "hiragana":
        return _is_hiragana_char(char)
    if key == "katakana":
        return _is_katakana_char(char)
    if key == "kana":
        return _is_hiragana_char(char) or _is_katakana_char(char)
    if key == "kanji":
        return _is_kanji_char(char)
    raise ValueError(f"unsupported script_filter: {script_filter}")


def _extract_char_from_path(path: Path) -> str | None:
    # Preferred: directory label `.../<char>/img_001.png`.
    parent = path.parent.name.strip()
    if parent.upper().startswith("U+"):
        token = parent[2:]
        try:
            cp = int(token, 16)
            ch = chr(cp)
            if len(ch) == 1:
                return ch
        except Exception:
            pass
    if len(parent) == 1 and not parent.isascii():
        return parent

    # Fallback: filename label `<char>_xx.png` or `<char>-xx.png`.
    stem = path.stem.strip()
    for sep in ("_", "-", " "):
        token = stem.split(sep, 1)[0].strip()
        if len(token) == 1 and not token.isascii():
            return token
    if stem and len(stem[0]) == 1 and not stem[0].isascii():
        return stem[0]
    return None


def _remove_border_components(mask: np.ndarray, *, min_size: int = 8) -> np.ndarray:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    ys, xs = np.where(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        stack = [(sy, sx)]
        comp: list[tuple[int, int]] = []
        touch_border = False
        visited[sy, sx] = True
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            if y == 0 or x == 0 or y == h - 1 or x == w - 1:
                touch_border = True
            for ny, nx in _iter_neighbors(y, x, h, w):
                if mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if touch_border or len(comp) < min_size:
            continue
        for y, x in comp:
            out[y, x] = True
    return out


def _foreground_mask_score(mask: np.ndarray) -> float:
    h, w = mask.shape
    total = float(h * w)
    count = float(mask.sum())
    if count <= 0.0:
        return 1e12
    ratio = count / total
    if ratio > 0.70:
        return 1e11 + ratio * 1e6

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    box_area = float(max(1, bw * bh))
    fill = count / box_area

    # Prefer medium occupancy and non-trivial coverage.
    ratio_penalty = abs(ratio - 0.10) * 9000.0
    fill_penalty = abs(fill - 0.22) * 7000.0
    tiny_penalty = 0.0
    if bw < 10 or bh < 10 or count < 40:
        tiny_penalty += 9000.0
    return ratio_penalty + fill_penalty + tiny_penalty


def _otsu_threshold(image: np.ndarray) -> int:
    # image: uint8 grayscale.
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    total = float(image.size)
    sum_total = float(np.dot(np.arange(256, dtype=np.float64), hist))
    sum_bg = 0.0
    weight_bg = 0.0
    best_threshold = 127
    max_between = -1.0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg <= 0.0:
            continue
        weight_fg = total - weight_bg
        if weight_fg <= 0.0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * ((mean_bg - mean_fg) ** 2)
        if between > max_between:
            max_between = between
            best_threshold = t
    return int(best_threshold)


def _preprocess_labeled_char_image(image: Image.Image, *, normalize_size: int = 96) -> np.ndarray:
    base = image.convert("L")
    size = max(48, int(normalize_size))
    inner = max(24, size - 8)
    contained = ImageOps.contain(base, (inner, inner), method=Image.Resampling.BICUBIC)
    canvas = Image.new("L", (size, size), 255)
    ox = (size - contained.width) // 2
    oy = (size - contained.height) // 2
    canvas.paste(contained, (ox, oy))
    base = canvas
    base = ImageOps.autocontrast(base)
    base = base.filter(ImageFilter.MedianFilter(size=3))
    arr = np.array(base, dtype=np.uint8)
    thr = _otsu_threshold(arr)
    dark_raw = arr < thr
    bright_raw = arr > thr
    dark = _remove_border_components(dark_raw, min_size=10)
    bright = _remove_border_components(bright_raw, min_size=10)

    # If border-cleaning removed too much, keep raw mask.
    if int(dark.sum()) < max(32, int(dark_raw.sum() * 0.35)):
        dark = dark_raw
    if int(bright.sum()) < max(32, int(bright_raw.sum() * 0.35)):
        bright = bright_raw

    candidates = [dark, bright]
    mask = min(candidates, key=_foreground_mask_score)
    out = np.full_like(arr, 255, dtype=np.uint8)
    out[mask] = 0
    return out


def _download_if_missing(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    urlretrieve(url, str(path))


def _extract_tar_if_missing(archive_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted.ok"
    if marker.exists():
        return
    # Fast path: top-level U+ directories already present.
    try:
        if any(p.is_dir() and p.name.upper().startswith("U+") for p in out_dir.iterdir()):
            marker.write_text("ok\n", encoding="utf-8")
            return
    except Exception:
        pass
    with tarfile.open(archive_path, "r") as tf:
        tf.extractall(path=out_dir)
    marker.write_text("ok\n", encoding="utf-8")


def _load_npz(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if "arr_0" in arr:
        return arr["arr_0"]
    # Fallback for custom npz naming.
    first_key = next(iter(arr.keys()))
    return arr[first_key]


def _sample_indices(total_count: int, target_count: int, seed: int) -> list[int]:
    capped = min(total_count, max(0, target_count))
    rng = random.Random(seed)
    return rng.sample(range(total_count), capped)


def _load_k49_classmap(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(str(row.get("index", "")).strip())
            except ValueError:
                continue
            char = str(row.get("char", "")).strip()
            if not char:
                continue
            mapping[idx] = char
    return mapping


_NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 1),   # right
    (1, 1),   # down-right
    (1, 0),   # down
    (1, -1),  # down-left
    (0, -1),  # left
    (-1, -1), # up-left
    (-1, 0),  # up
    (-1, 1),  # up-right
)


def _iter_neighbors(y: int, x: int, h: int, w: int):
    for dy, dx in _NEIGHBOR_OFFSETS:
        ny = y + dy
        nx = x + dx
        if 0 <= ny < h and 0 <= nx < w:
            yield ny, nx


def _split_connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    ys, xs = np.where(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if visited[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            comp.append((cy, cx))
            for ny, nx in _iter_neighbors(cy, cx, h, w):
                if mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if comp:
            components.append(comp)
    return components


def _turn_cost(
    prev: tuple[int, int] | None,
    cur: tuple[int, int],
    nxt: tuple[int, int],
) -> float:
    if prev is None:
        return 0.0
    v1x = cur[1] - prev[1]
    v1y = cur[0] - prev[0]
    v2x = nxt[1] - cur[1]
    v2y = nxt[0] - cur[0]
    dot = float(v1x * v2x + v1y * v2y)
    n1 = max(1e-6, float((v1x * v1x + v1y * v1y) ** 0.5))
    n2 = max(1e-6, float((v2x * v2x + v2y * v2y) ** 0.5))
    cos = dot / (n1 * n2)
    # lower is better. straight (cos=1) -> 0, reverse (cos=-1) -> 2.
    return 1.0 - max(-1.0, min(1.0, cos))


def _component_to_nonoverlap_paths(component: list[tuple[int, int]], mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Traverse a connected skeleton component as a single continuous path using a
    DFS-with-backtrack strategy (approximating an Euler path).

    When greedy traversal reaches a dead end while unvisited pixels remain, the
    algorithm backtracks along the already-visited path until it finds a pixel that
    has unvisited neighbors, then continues forward from there.  This produces ONE
    connected path per component rather than many fragments, eliminating mid-stroke
    pen-up jumps (e.g. the 'H' right leg splitting into two pieces).
    """
    h, w = mask.shape
    pixels = set(component)
    if not pixels:
        return []

    # Build adjacency (8-connected).
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for py, px in pixels:
        nbr = [n for n in (
            (py + dy, px + dx) for dy, dx in _NEIGHBOR_OFFSETS
        ) if (n[0], n[1]) in pixels and 0 <= n[0] < h and 0 <= n[1] < w]
        neighbors[(py, px)] = nbr

    # Start from an endpoint (degree ≤ 1) if available; prefer top-left.
    endpoints = [p for p in pixels if len(neighbors[p]) <= 1]
    if endpoints:
        start = min(endpoints, key=lambda p: (p[1], p[0]))
    else:
        start = min(pixels, key=lambda p: (p[1], p[0]))

    visited: set[tuple[int, int]] = set()
    path: list[tuple[int, int]] = [start]
    visited.add(start)
    prev: tuple[int, int] | None = None
    cur = start

    while len(visited) < len(pixels):
        # Prefer unvisited neighbors; sort by turn cost for smoothness.
        unvisited = [n for n in neighbors[cur] if n not in visited]
        if unvisited:
            nxt = min(unvisited, key=lambda n: _turn_cost(prev, cur, n))
            visited.add(nxt)
            path.append(nxt)
            prev, cur = cur, nxt
        else:
            # Dead end: backtrack along already-visited path until we find a
            # pixel adjacent to an unvisited one.
            backtrack_idx = len(path) - 2  # go one step back
            found = False
            while backtrack_idx >= 0:
                candidate = path[backtrack_idx]
                if any(n not in visited for n in neighbors[candidate]):
                    # Walk back to this pixel (append the return path).
                    path.extend(reversed(path[backtrack_idx + 1:]))
                    prev = path[-2] if len(path) >= 2 else None
                    cur = candidate
                    found = True
                    break
                backtrack_idx -= 1
            if not found:
                break  # All reachable pixels visited.

    # Trim tiny duplicate-point runs at the end of backtracking segments.
    return [path] if len(path) >= 2 else []


def _resample_path(points: list[tuple[int, int]], keep_points: int) -> list[tuple[int, int]]:
    """Arc-length based resampling: places keep_points evenly along the actual
    curve length to avoid the bunching/gapping produced by index-uniform sampling."""
    n = len(points)
    if n <= keep_points:
        return points
    if keep_points <= 1:
        return [points[0]]
    # Build cumulative arc-length table.
    dists = [0.0]
    for i in range(1, n):
        dy = points[i][0] - points[i - 1][0]
        dx = points[i][1] - points[i - 1][1]
        dists.append(dists[-1] + float((dy * dy + dx * dx) ** 0.5))
    total = dists[-1]
    if total < 1e-9:
        idx = np.linspace(0, n - 1, num=keep_points, dtype=np.int32)
        return [points[int(i)] for i in idx.tolist()]
    targets = np.linspace(0.0, total, num=keep_points)
    result: list[tuple[int, int]] = []
    j = 0
    for t in targets:
        while j < n - 1 and dists[j + 1] < t:
            j += 1
        result.append(points[j])
    return result


def _catmull_rom_segment(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    steps: int,
) -> list[tuple[float, float]]:
    """Interpolate one Catmull-Rom segment from p1 to p2 using steps sub-points."""
    out: list[tuple[float, float]] = []
    for k in range(steps):
        t = float(k) / float(steps)
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        # Catmull-Rom tangents (alpha=0.5 centripetal).
        m1y = 0.5 * (p2[0] - p0[0])
        m1x = 0.5 * (p2[1] - p0[1])
        m2y = 0.5 * (p3[0] - p1[0])
        m2x = 0.5 * (p3[1] - p1[1])
        y = h00 * p1[0] + h10 * m1y + h01 * p2[0] + h11 * m2y
        x = h00 * p1[1] + h10 * m1x + h01 * p2[1] + h11 * m2x
        out.append((y, x))
    return out


def _smooth_polyline(
    points: list[tuple[int, int]],
    *,
    passes: int = 2,
    blend: float = 0.58,
) -> list[tuple[float, float]]:
    """Smooth a pixel-skeleton path using Laplacian pre-smoothing followed by
    Catmull-Rom spline interpolation to eliminate staircase artifacts.
    The resulting list has roughly (len(points) * steps_per_seg) points;
    callers should re-sample afterwards if a fixed count is required.
    `passes` and `blend` still control the initial Laplacian pre-smooth."""
    if len(points) <= 2:
        return [(float(y), float(x)) for y, x in points]

    # --- 1. Mild Laplacian pass to remove coarse pixel-grid noise ---
    out: list[tuple[float, float]] = [(float(y), float(x)) for y, x in points]
    alpha = max(0.0, min(1.0, float(blend)))
    lap_passes = max(0, int(passes))
    for _ in range(lap_passes):
        nxt = [out[0]]
        for i in range(1, len(out) - 1):
            py, px = out[i - 1]
            cy, cx = out[i]
            ny, nx = out[i + 1]
            sy = cy * (1.0 - alpha) + (py + ny) * 0.5 * alpha
            sx = cx * (1.0 - alpha) + (px + nx) * 0.5 * alpha
            nxt.append((sy, sx))
        nxt.append(out[-1])
        out = nxt

    # --- 2. Catmull-Rom spline through Laplacian-smoothed control points ---
    # Fixed 3 sub-steps per segment gives smooth curves without over-densifying.
    seg_steps = 3
    spline: list[tuple[float, float]] = []
    # Phantom endpoints mirror first/last segments for natural curve termination.
    ctrl = [
        (2.0 * out[0][0] - out[1][0], 2.0 * out[0][1] - out[1][1]),
        *out,
        (2.0 * out[-1][0] - out[-2][0], 2.0 * out[-1][1] - out[-2][1]),
    ]
    for i in range(1, len(ctrl) - 2):
        seg = _catmull_rom_segment(ctrl[i - 1], ctrl[i], ctrl[i + 1], ctrl[i + 2], steps=seg_steps)
        spline.extend(seg)
    spline.append(out[-1])  # include the final endpoint
    return spline


def _path_to_sequence(
    sampled: list[tuple[float, float]],
    *,
    stroke_width: float = 0.5,
) -> list[list[float]]:
    if len(sampled) < 2:
        return []
    seq: list[list[float]] = []
    prev_y, prev_x = sampled[0]
    for y, x in sampled[1:]:
        dx = float(x - prev_x) / 20.0
        dy = float(y - prev_y) / 20.0
        seq.append([dx, dy, 1.0, float(stroke_width)])
        prev_y, prev_x = y, x
    if seq:
        seq[-1][2] = 0.0
    return seq


def _path_to_sequence_with_cursor(
    sampled: list[tuple[float, float]],
    cursor: tuple[float, float],
    *,
    stroke_width: float = 0.5,
) -> tuple[list[list[float]], tuple[float, float]]:
    if len(sampled) < 2:
        return [], cursor
    seq: list[list[float]] = []

    cur_y, cur_x = cursor
    start_y, start_x = sampled[0]
    if abs(start_x - cur_x) > 1e-6 or abs(start_y - cur_y) > 1e-6:
        # Pen-up move to the next stroke start to preserve relative component
        # placement (e.g. dakuten/handakuten circles).
        seq.append([(start_x - cur_x) / 20.0, (start_y - cur_y) / 20.0, 0.0, float(stroke_width)])

    prev_y, prev_x = start_y, start_x
    for y, x in sampled[1:]:
        dx = float(x - prev_x) / 20.0
        dy = float(y - prev_y) / 20.0
        seq.append([dx, dy, 1.0, float(stroke_width)])
        prev_y, prev_x = y, x
    if seq:
        seq[-1][2] = 0.0
    return seq, (prev_y, prev_x)


def _normalize_binary_mask(mask: np.ndarray, size: int = 64) -> np.ndarray:
    if mask.ndim != 2 or int(mask.sum()) < 2:
        return np.zeros((size, size), dtype=bool)
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    crop = mask[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8) * 255
    tile = Image.fromarray(crop, mode="L")
    tile = ImageOps.contain(tile, (size - 4, size - 4), method=Image.Resampling.NEAREST)
    canvas = Image.new("L", (size, size), 0)
    ox = (size - tile.width) // 2
    oy = (size - tile.height) // 2
    canvas.paste(tile, (ox, oy))
    arr = np.array(canvas, dtype=np.uint8)
    return arr > 127


def _sequence_shape_iou(seq: list[list[float]], reference_mask: np.ndarray) -> float:
    if not seq:
        return 0.0
    x = 0.0
    y = 0.0
    pts: list[tuple[float, float, float]] = [(x, y, 0.0)]
    for step in seq:
        if not isinstance(step, list) or len(step) < 3:
            continue
        x += float(step[0]) * 20.0
        y += float(step[1]) * 20.0
        pts.append((x, y, float(step[2])))
    if len(pts) < 2:
        return 0.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x = int(np.floor(min(xs))) - 2
    max_x = int(np.ceil(max(xs))) + 2
    min_y = int(np.floor(min(ys))) - 2
    max_y = int(np.ceil(max(ys))) + 2
    w = max(8, max_x - min_x + 1)
    h = max(8, max_y - min_y + 1)
    pred = np.zeros((h, w), dtype=bool)
    px, py, _ = pts[0]
    for x1, y1, pen in pts[1:]:
        if pen > 0.0:
            dx = x1 - px
            dy = y1 - py
            steps = max(1, int(max(abs(dx), abs(dy)) * 2.0))
            for k in range(steps + 1):
                t = float(k) / float(steps)
                xx = px + dx * t
                yy = py + dy * t
                ix = int(round(xx - min_x))
                iy = int(round(yy - min_y))
                if 0 <= iy < h and 0 <= ix < w:
                    pred[iy, ix] = True
        px, py = x1, y1

    ref_norm = _normalize_binary_mask(reference_mask, size=64)
    pred_norm = _normalize_binary_mask(pred, size=64)
    inter = int(np.logical_and(ref_norm, pred_norm).sum())
    union = int(np.logical_or(ref_norm, pred_norm).sum())
    return float(inter) / float(max(1, union))


def _adaptive_min_shape_iou(reference_mask: np.ndarray, base_iou: float) -> float:
    base = max(0.0, float(base_iou))
    if base <= 0.0:
        return 0.0
    if reference_mask.ndim != 2 or int(reference_mask.sum()) < 2:
        return base * 0.5

    pix = int(reference_mask.sum())
    components = _split_connected_components(reference_mask)
    comp_n = max(1, len(components))
    ys, xs = np.where(reference_mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    box_area = max(1, (y1 - y0 + 1) * (x1 - x0 + 1))
    fill = float(pix) / float(box_area)

    # Complex kanji (many parts, sparse fill, many pixels) should get a lower
    # threshold; simple compact shapes should stay stricter.
    comp_term = max(0.0, float(comp_n - 1)) * 0.11
    sparse_term = max(0.0, 0.24 - fill) * 2.1
    size_term = max(0.0, float(pix - 520)) / 2600.0
    complexity = min(1.0, comp_term + sparse_term + size_term)

    adjusted = base * (1.0 - 0.58 * complexity)
    if comp_n <= 2 and fill > 0.18 and pix < 760:
        adjusted = min(0.65, adjusted * 1.08)
    return max(0.02, float(adjusted))


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    ys, xs = np.where(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        for ny, nx in _iter_neighbors(y, x, h, w):
            if not mask[ny, nx]:
                out[y, x] = True
                break
    return out


def _trace_boundary_paths(boundary: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = boundary.shape
    remaining = set(zip(*np.where(boundary)))
    paths: list[list[tuple[int, int]]] = []
    neighbor_dirs = list(_NEIGHBOR_OFFSETS)

    while remaining:
        start = min(remaining, key=lambda p: (p[1], p[0]))
        remaining.remove(start)
        path = [start]
        cur = start
        prev_dir_idx = 0

        while True:
            candidates: list[tuple[int, int, int]] = []
            for dir_idx, (dy, dx) in enumerate(neighbor_dirs):
                ny = cur[0] + dy
                nx = cur[1] + dx
                if 0 <= ny < h and 0 <= nx < w and (ny, nx) in remaining:
                    # Prefer smoother continuation to avoid jagged jumps.
                    turn_cost = abs(dir_idx - prev_dir_idx)
                    turn_cost = min(turn_cost, 8 - turn_cost)
                    candidates.append((turn_cost, dir_idx, ny * w + nx))
            if not candidates:
                break
            candidates.sort()
            _, dir_idx, key = candidates[0]
            ny = key // w
            nx = key % w
            nxt = (ny, nx)
            remaining.remove(nxt)
            path.append(nxt)
            cur = nxt
            prev_dir_idx = dir_idx

        if len(path) >= 2:
            paths.append(path)
    return paths


def _zhang_suen_thinning(mask: np.ndarray, max_iter: int = 48) -> np.ndarray:
    # Morphological thinning to approximate 1-pixel centerlines.
    img = mask.astype(np.uint8).copy()
    if img.ndim != 2 or img.size == 0:
        return mask
    h, w = img.shape
    if h < 3 or w < 3:
        return mask

    for _ in range(max_iter):
        changed = False

        # step 1
        to_remove = np.zeros_like(img, dtype=bool)
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if img[y, x] != 1:
                    continue
                p2 = img[y - 1, x]
                p3 = img[y - 1, x + 1]
                p4 = img[y, x + 1]
                p5 = img[y + 1, x + 1]
                p6 = img[y + 1, x]
                p7 = img[y + 1, x - 1]
                p8 = img[y, x - 1]
                p9 = img[y - 1, x - 1]
                b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                if b < 2 or b > 6:
                    continue
                seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
                a = 0
                for i in range(8):
                    if seq[i] == 0 and seq[i + 1] == 1:
                        a += 1
                if a != 1:
                    continue
                if p2 * p4 * p6 != 0:
                    continue
                if p4 * p6 * p8 != 0:
                    continue
                to_remove[y, x] = True
        if np.any(to_remove):
            img[to_remove] = 0
            changed = True

        # step 2
        to_remove = np.zeros_like(img, dtype=bool)
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if img[y, x] != 1:
                    continue
                p2 = img[y - 1, x]
                p3 = img[y - 1, x + 1]
                p4 = img[y, x + 1]
                p5 = img[y + 1, x + 1]
                p6 = img[y + 1, x]
                p7 = img[y + 1, x - 1]
                p8 = img[y, x - 1]
                p9 = img[y - 1, x - 1]
                b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                if b < 2 or b > 6:
                    continue
                seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
                a = 0
                for i in range(8):
                    if seq[i] == 0 and seq[i + 1] == 1:
                        a += 1
                if a != 1:
                    continue
                if p2 * p4 * p8 != 0:
                    continue
                if p2 * p6 * p8 != 0:
                    continue
                to_remove[y, x] = True
        if np.any(to_remove):
            img[to_remove] = 0
            changed = True

        if not changed:
            break

    return img.astype(bool)


def _remove_singletons(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    ys, xs = np.where(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        deg = 0
        for ny, nx in _iter_neighbors(y, x, h, w):
            if mask[ny, nx]:
                deg += 1
        if deg == 0:
            out[y, x] = False
    return out


def _image_to_sequence_contours(
    image: np.ndarray,
    max_points: int,
    threshold: int,
    *,
    smooth_profile: str = "default",
) -> list[list[float]]:
    dark_mask = image < threshold
    bright_mask = image > threshold
    dark_count = int(dark_mask.sum())
    bright_count = int(bright_mask.sum())
    mask = dark_mask if dark_count <= bright_count else bright_mask
    if mask.ndim != 2 or int(mask.sum()) < 2:
        return []

    boundary = _boundary_mask(mask)
    if int(boundary.sum()) < 2:
        return []

    paths = _trace_boundary_paths(boundary)
    if not paths:
        return []
    paths.sort(key=lambda p: (min(pt[1] for pt in p), min(pt[0] for pt in p)))
    raw_len = sum(len(p) for p in paths)
    if raw_len <= 0:
        return []

    keep_total = max(2, int(max_points))
    seq: list[list[float]] = []
    cursor = (0.0, 0.0)
    profile = str(smooth_profile).strip().lower()
    for path in paths:
        keep = max(2, int(round((len(path) / raw_len) * keep_total)))
        sampled_i = _resample_path(path, keep)
        if profile == "kanji_fine":
            plen = len(sampled_i)
            if plen < 18:
                passes, blend, stroke_w = 1, 0.34, 0.70
            elif plen < 40:
                passes, blend, stroke_w = 2, 0.45, 0.72
            else:
                passes, blend, stroke_w = 2, 0.52, 0.74
        elif profile == "latin_klee":
            passes, blend, stroke_w = 1, 0.32, 0.70
        else:
            passes, blend, stroke_w = 3, 0.60, 0.64
        spline_raw_c = _smooth_polyline(sampled_i, passes=passes, blend=blend)
        # Arc-length re-sample on spline output.
        sampled_fc: list[tuple[float, float]] = spline_raw_c
        target_pts_c = keep
        if len(sampled_fc) > target_pts_c:
            dists_fc = [0.0]
            for _si in range(1, len(sampled_fc)):
                dy_fc = sampled_fc[_si][0] - sampled_fc[_si - 1][0]
                dx_fc = sampled_fc[_si][1] - sampled_fc[_si - 1][1]
                dists_fc.append(dists_fc[-1] + float((dy_fc * dy_fc + dx_fc * dx_fc) ** 0.5))
            total_fc = dists_fc[-1]
            if total_fc > 1e-9:
                targets_fc = np.linspace(0.0, total_fc, num=target_pts_c)
                resampled_fc: list[tuple[float, float]] = []
                jj_c = 0
                for _t_c in targets_fc:
                    while jj_c < len(sampled_fc) - 1 and dists_fc[jj_c + 1] < _t_c:
                        jj_c += 1
                    resampled_fc.append(sampled_fc[jj_c])
                sampled_fc = resampled_fc
        sampled = sampled_fc
        part, cursor = _path_to_sequence_with_cursor(sampled, cursor, stroke_width=stroke_w)
        if part:
            seq.extend(part)

    if len(seq) > keep_total:
        stride = len(seq) / float(keep_total)
        reduced: list[list[float]] = []
        i = 0.0
        while int(i) < len(seq) and len(reduced) < keep_total:
            reduced.append(seq[int(i)])
            i += stride
        seq = reduced
        if seq:
            seq[-1][2] = 0.0
    return seq


def _image_to_sequence(
    image: np.ndarray,
    max_points: int,
    threshold: int,
    *,
    smooth_profile: str = "default",
) -> list[list[float]]:
    # image is expected to be uint8 grayscale.
    # Some datasets store ink as dark pixels, others as bright pixels.
    # Pick the sparser side as foreground to avoid tracing the background.
    dark_mask = image < threshold
    bright_mask = image > threshold
    dark_count = int(dark_mask.sum())
    bright_count = int(bright_mask.sum())
    mask = dark_mask if dark_count <= bright_count else bright_mask
    if mask.ndim != 2:
        return []

    # Remove isolated single-pixel noise while preserving thin strokes.
    h, w = mask.shape
    neighbor_count = np.zeros_like(image, dtype=np.uint8)
    for dy, dx in _NEIGHBOR_OFFSETS:
        src_y0 = max(0, -dy)
        src_y1 = min(h, h - dy)
        src_x0 = max(0, -dx)
        src_x1 = min(w, w - dx)
        dst_y0 = max(0, dy)
        dst_y1 = min(h, h + dy)
        dst_x0 = max(0, dx)
        dst_x1 = min(w, w + dx)
        neighbor_count[dst_y0:dst_y1, dst_x0:dst_x1] += mask[src_y0:src_y1, src_x0:src_x1].astype(np.uint8)
    mask = mask & (neighbor_count > 0)

    if int(mask.sum()) < 2:
        return []

    # Convert filled strokes into 1-pixel centerlines first to preserve
    # glyph shape while avoiding double outline trajectories.
    mask = _zhang_suen_thinning(mask)
    mask = _remove_singletons(mask)
    if int(mask.sum()) < 2:
        return []

    components = _split_connected_components(mask)
    if not components:
        return []
    components.sort(key=lambda comp: (min(p[1] for p in comp), min(p[0] for p in comp)))

    paths: list[list[tuple[int, int]]] = []
    raw_len = 0
    for comp in components:
        comp_paths = _component_to_nonoverlap_paths(comp, mask)
        for path in comp_paths:
            if len(path) < 2:
                continue
            paths.append(path)
            raw_len += len(path)

    if not paths:
        return []

    keep_total = max(2, int(max_points))
    seq: list[list[float]] = []
    cursor = (0.0, 0.0)
    profile = str(smooth_profile).strip().lower()
    for path in paths:
        # Keep points proportional to each component size.
        comp_keep = max(2, int(round((len(path) / max(1, raw_len)) * keep_total)))
        sampled_i = _resample_path(path, comp_keep)
        if profile == "kanji_fine":
            plen = len(sampled_i)
            if plen < 16:
                passes, blend, stroke_w = 1, 0.30, 0.70
            elif plen < 36:
                passes, blend, stroke_w = 2, 0.42, 0.72
            elif plen < 80:
                passes, blend, stroke_w = 2, 0.52, 0.74
            else:
                passes, blend, stroke_w = 3, 0.58, 0.76
        elif profile == "latin_klee":
            passes, blend, stroke_w = 1, 0.32, 0.70
        else:
            # For Latin / default characters: stronger smoothing to tame
            # the noisy Zhang-Suen skeleton from EMNIST bitmaps.
            passes, blend, stroke_w = 3, 0.65, 0.72
        # After Catmull-Rom expansion, re-sample back to comp_keep points so
        # the total budget is respected while keeping the smooth curve shape.
        spline_raw = _smooth_polyline(sampled_i, passes=passes, blend=blend)
        # Arc-length re-sample on float points (reuse the integer version logic).
        sampled_f: list[tuple[float, float]] = spline_raw
        target_pts = comp_keep
        if len(sampled_f) > target_pts:
            dists_f = [0.0]
            for _si in range(1, len(sampled_f)):
                dy_f = sampled_f[_si][0] - sampled_f[_si - 1][0]
                dx_f = sampled_f[_si][1] - sampled_f[_si - 1][1]
                dists_f.append(dists_f[-1] + float((dy_f * dy_f + dx_f * dx_f) ** 0.5))
            total_f = dists_f[-1]
            if total_f > 1e-9:
                targets_f = np.linspace(0.0, total_f, num=target_pts)
                resampled_f: list[tuple[float, float]] = []
                jj = 0
                for _t in targets_f:
                    while jj < len(sampled_f) - 1 and dists_f[jj + 1] < _t:
                        jj += 1
                    resampled_f.append(sampled_f[jj])
                sampled_f = resampled_f
        sampled = sampled_f
        part, cursor = _path_to_sequence_with_cursor(sampled, cursor, stroke_width=stroke_w)
        if part:
            seq.extend(part)

    if len(seq) > keep_total:
        stride = len(seq) / float(keep_total)
        reduced: list[list[float]] = []
        i = 0.0
        while int(i) < len(seq) and len(reduced) < keep_total:
            reduced.append(seq[int(i)])
            i += stride
        seq = reduced
        if seq:
            seq[-1][2] = 0.0
    return seq


def _resolve_split(images: np.ndarray, labels: np.ndarray, split: str) -> tuple[np.ndarray, np.ndarray]:
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images and labels length mismatch")
    if split not in {"train", "test", "all"}:
        raise ValueError("split must be one of train/test/all")
    return images, labels


def _iter_pairs(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    if split == "train":
        return _resolve_split(train_images, train_labels, split)
    if split == "test":
        return _resolve_split(test_images, test_labels, split)
    # all
    all_images = np.concatenate([train_images, test_images], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    return _resolve_split(all_images, all_labels, split)


def _read_idx_images_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(16)
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"invalid image magic: {magic}")
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size != count * rows * cols:
        raise ValueError("image payload length mismatch")
    return arr.reshape(count, rows, cols)


def _read_idx_labels_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(8)
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"invalid label magic: {magic}")
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size != count:
        raise ValueError("label payload length mismatch")
    return arr


def import_k49_to_base_dataset(
    output_path: str,
    *,
    target_count: int = 40000,
    split: str = "train",
    seed: int = 42,
    cache_dir: str = "storage/public_cache/k49",
    append: bool = True,
    threshold: int = 200,
    max_points: int = 120,
    min_points: int = 8,
) -> ImportStats:
    cache = Path(cache_dir)
    train_imgs_path = cache / "k49-train-imgs.npz"
    train_labels_path = cache / "k49-train-labels.npz"
    test_imgs_path = cache / "k49-test-imgs.npz"
    test_labels_path = cache / "k49-test-labels.npz"
    classmap_path = cache / "k49_classmap.csv"

    _download_if_missing(K49_FILES["train_images"], train_imgs_path)
    _download_if_missing(K49_FILES["train_labels"], train_labels_path)
    _download_if_missing(K49_FILES["test_images"], test_imgs_path)
    _download_if_missing(K49_FILES["test_labels"], test_labels_path)
    _download_if_missing(K49_FILES["classmap"], classmap_path)

    train_images = _load_npz(train_imgs_path)
    train_labels = _load_npz(train_labels_path)
    test_images = _load_npz(test_imgs_path)
    test_labels = _load_npz(test_labels_path)
    classmap = _load_k49_classmap(classmap_path)

    images, labels = _iter_pairs(train_images, train_labels, test_images, test_labels, split)
    indices = _sample_indices(images.shape[0], target_count, seed)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    imported = 0
    skipped = 0
    with out.open(mode, encoding="utf-8") as w:
        for idx in indices:
            image = images[idx]
            label = int(labels[idx])
            char = classmap.get(label)
            if char is None:
                skipped += 1
                continue
            seq = _image_to_sequence(image, max_points=max_points, threshold=threshold)
            if len(seq) < min_points:
                skipped += 1
                continue
            sample = {
                "char_id": char_to_model_id(char),
                "style_id": 0,
                "dataset_id": -49,
                "user_id": 0,
                "sequence": seq,
                "meta": {
                    "source": "k49",
                    "split": split,
                    "source_index": int(idx),
                    "label": label,
                    "char": char,
                },
            }
            w.write(json.dumps(sample, ensure_ascii=False) + "\n")
            imported += 1

    shape = list(images.shape[1:]) if images.ndim >= 2 else []
    return ImportStats(
        source="k49",
        requested_count=target_count,
        imported_count=imported,
        skipped_count=skipped,
        output_path=str(out),
        image_shape=shape,
    )


def import_mnist_to_base_dataset(
    output_path: str,
    *,
    target_count: int = 40000,
    split: str = "train",
    seed: int = 42,
    cache_dir: str = "storage/public_cache/mnist",
    append: bool = True,
    threshold: int = 200,
    max_points: int = 120,
    min_points: int = 8,
) -> ImportStats:
    cache = Path(cache_dir)
    train_imgs_path = cache / "train-images-idx3-ubyte.gz"
    train_labels_path = cache / "train-labels-idx1-ubyte.gz"
    test_imgs_path = cache / "t10k-images-idx3-ubyte.gz"
    test_labels_path = cache / "t10k-labels-idx1-ubyte.gz"

    _download_if_missing(MNIST_FILES["train_images"], train_imgs_path)
    _download_if_missing(MNIST_FILES["train_labels"], train_labels_path)
    _download_if_missing(MNIST_FILES["test_images"], test_imgs_path)
    _download_if_missing(MNIST_FILES["test_labels"], test_labels_path)

    train_images = _read_idx_images_gz(train_imgs_path)
    train_labels = _read_idx_labels_gz(train_labels_path)
    test_images = _read_idx_images_gz(test_imgs_path)
    test_labels = _read_idx_labels_gz(test_labels_path)

    images, labels = _iter_pairs(train_images, train_labels, test_images, test_labels, split)
    indices = _sample_indices(images.shape[0], target_count, seed)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    imported = 0
    skipped = 0
    with out.open(mode, encoding="utf-8") as w:
        for idx in indices:
            image = images[idx]
            label = int(labels[idx])
            label_char = str(label)
            seq = _image_to_sequence(image, max_points=max_points, threshold=threshold)
            if len(seq) < min_points:
                skipped += 1
                continue
            sample = {
                "char_id": char_to_model_id(label_char),
                "style_id": 0,
                "dataset_id": -1,
                "user_id": 0,
                "sequence": seq,
                "meta": {
                    "source": "mnist",
                    "split": split,
                    "source_index": int(idx),
                    "label": label,
                    "char": label_char,
                },
            }
            w.write(json.dumps(sample, ensure_ascii=False) + "\n")
            imported += 1

    shape = list(images.shape[1:]) if images.ndim >= 2 else []
    return ImportStats(
        source="mnist",
        requested_count=target_count,
        imported_count=imported,
        skipped_count=skipped,
        output_path=str(out),
        image_shape=shape,
    )


def import_hiragana_images_to_base_dataset(
    output_path: str,
    *,
    input_dir: str = "storage/public_cache/net_datasets/hiragana-dataset/hiragana_images",
    target_count: int = 1000,
    seed: int = 42,
    append: bool = True,
    threshold: int = 200,
    max_points: int = 120,
    min_points: int = 8,
) -> ImportStats:
    src_dir = Path(input_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {src_dir}")

    files = sorted(src_dir.glob("*.jpg"))
    if not files:
        raise ValueError(f"no jpg files found in: {src_dir}")

    indices = _sample_indices(len(files), target_count, seed)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    imported = 0
    skipped = 0
    for_write: list[dict] = []

    for idx in indices:
        path = files[idx]
        name = path.stem
        if not name.startswith("kana"):
            skipped += 1
            continue
        romaji = "".join(ch for ch in name[4:] if ch.isalpha()).upper()
        char = HIRAGANA_ROMAJI_TO_CHAR.get(romaji)
        if not char:
            skipped += 1
            continue

        arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        seq = _image_to_sequence_contours(arr, max_points=max_points, threshold=threshold)
        if len(seq) < min_points:
            skipped += 1
            continue
        for_write.append(
            {
                "char_id": char_to_model_id(char),
                "style_id": 0,
                "dataset_id": -500,
                "user_id": 0,
                "sequence": seq,
                "meta": {
                    "source": "hiragana-dataset-github",
                    "file_name": path.name,
                    "romaji": romaji,
                    "char": char,
                },
            }
        )
        imported += 1

    with out.open(mode, encoding="utf-8") as w:
        for sample in for_write:
            w.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return ImportStats(
        source="hiragana-dataset-github",
        requested_count=target_count,
        imported_count=imported,
        skipped_count=skipped,
        output_path=str(out),
        image_shape=[],
    )


def import_kkanji_to_base_dataset(
    output_path: str,
    *,
    target_count: int = 20000,
    seed: int = 42,
    cache_dir: str = "storage/public_cache/kkanji",
    append: bool = True,
    sequence_mode: str = "centerline",
    quality_profile: str = "kanji_fine",
    normalize_size: int = 128,
    max_points: int = 240,
    min_points: int = 24,
    min_shape_iou: float = 0.12,
    source_name: str = "kkanji",
) -> ImportStats:
    cache = Path(cache_dir)
    archive_path = cache / "kkanji.tar"
    extract_dir = cache / "kkanji_extracted"

    _download_if_missing(KKANJI_FILES["archive"], archive_path)
    _extract_tar_if_missing(archive_path, extract_dir)

    # Find directory that contains U+XXXX class folders.
    candidate_roots = [extract_dir]
    candidate_roots.extend([p for p in extract_dir.iterdir() if p.is_dir()])
    src_dir: Path | None = None
    for cand in candidate_roots:
        if not cand.is_dir():
            continue
        has_uplus = any(p.is_dir() and p.name.upper().startswith("U+") for p in cand.iterdir())
        if has_uplus:
            src_dir = cand
            break
    if src_dir is None:
        raise FileNotFoundError(f"failed to find U+ class directories in: {extract_dir}")

    return import_character_images_to_base_dataset(
        output_path=output_path,
        input_dir=str(src_dir),
        target_count=target_count,
        seed=seed,
        append=append,
        script_filter="kanji",
        normalize_size=normalize_size,
        sequence_mode=sequence_mode,
        quality_profile=quality_profile,
        min_shape_iou=min_shape_iou,
        max_points=max_points,
        min_points=min_points,
        source_name=source_name,
    )


def import_character_images_to_base_dataset(
    output_path: str,
    *,
    input_dir: str,
    target_count: int = 2000,
    seed: int = 42,
    append: bool = True,
    script_filter: str = "all",
    normalize_size: int = 96,
    sequence_mode: str = "centerline",
    quality_profile: str = "default",
    min_shape_iou: float = 0.26,
    max_points: int = 120,
    min_points: int = 8,
    source_name: str = "labeled-images",
) -> ImportStats:
    src_dir = Path(input_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {src_dir}")

    files = sorted(
        p
        for p in src_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )
    if not files:
        raise ValueError(f"no image files found in: {src_dir}")

    indices = _sample_indices(len(files), target_count, seed)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    imported = 0
    skipped = 0
    for_write: list[dict] = []

    for idx in indices:
        path = files[idx]
        char = _extract_char_from_path(path)
        if not char:
            skipped += 1
            continue
        if not _char_matches_script(char, script_filter):
            skipped += 1
            continue
        try:
            arr = _preprocess_labeled_char_image(Image.open(path), normalize_size=normalize_size)
        except Exception:
            skipped += 1
            continue
        seq_mode = str(sequence_mode).strip().lower()
        profile = str(quality_profile).strip().lower()
        if seq_mode == "centerline":
            seq = _image_to_sequence(arr, max_points=max_points, threshold=128, smooth_profile=profile)
        elif seq_mode == "contour":
            seq = _image_to_sequence_contours(arr, max_points=max_points, threshold=128, smooth_profile=profile)
        else:
            raise ValueError(f"unsupported sequence_mode: {sequence_mode}")
        if len(seq) < min_points:
            skipped += 1
            continue
        required_iou = float(min_shape_iou)
        if profile == "kanji_fine":
            required_iou = _adaptive_min_shape_iou(arr < 128, required_iou)
        shape_iou = _sequence_shape_iou(seq, arr < 128)
        if shape_iou < required_iou:
            skipped += 1
            continue
        rel_path = str(path.relative_to(src_dir)) if path.is_relative_to(src_dir) else path.name
        for_write.append(
            {
                "char_id": char_to_model_id(char),
                "style_id": 0,
                "dataset_id": -700,
                "user_id": 0,
                "sequence": seq,
                "meta": {
                    "source": source_name,
                    "file_name": path.name,
                    "relative_path": rel_path,
                    "char": char,
                    "script_filter": script_filter,
                    "sequence_mode": seq_mode,
                    "quality_profile": profile,
                    "required_shape_iou": required_iou,
                    "shape_iou": shape_iou,
                },
            }
        )
        imported += 1

    with out.open(mode, encoding="utf-8") as w:
        for sample in for_write:
            w.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return ImportStats(
        source=source_name,
        requested_count=target_count,
        imported_count=imported,
        skipped_count=skipped,
        output_path=str(out),
        image_shape=[],
    )


def import_emnist_to_base_dataset(
    output_path: str,
    *,
    target_count: int = 40000,
    split: str = "train",
    seed: int = 42,
    cache_dir: str = "storage/public_cache/emnist",
    append: bool = True,
    threshold: int = 200,
    max_points: int = 120,
    min_points: int = 8,
) -> ImportStats:
    from torchvision.datasets import EMNIST

    is_train = True
    if split == "test":
        is_train = False

    if split == "all":
        train_ds = EMNIST(cache_dir, split="byclass", train=True, download=True)
        test_ds = EMNIST(cache_dir, split="byclass", train=False, download=True)
        images = np.concatenate([train_ds.data.numpy(), test_ds.data.numpy()], axis=0)
        labels = np.concatenate([train_ds.targets.numpy(), test_ds.targets.numpy()], axis=0)
    else:
        ds = EMNIST(cache_dir, split="byclass", train=is_train, download=True)
        images = ds.data.numpy()
        labels = ds.targets.numpy()

    indices = _sample_indices(images.shape[0], target_count, seed)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    imported = 0
    skipped = 0
    with out.open(mode, encoding="utf-8") as w:
        for idx in indices:
            # EMNIST raw images are transposed by default; transpose to correct orientation
            image = images[idx].T
            label = int(labels[idx])
            
            if 0 <= label <= 9:
                char = str(label)
            elif 10 <= label <= 35:
                char = chr(ord('A') + label - 10)
            elif 36 <= label <= 61:
                char = chr(ord('a') + label - 36)
            else:
                skipped += 1
                continue

            seq = _image_to_sequence(image, max_points=max_points, threshold=threshold)
            if len(seq) < min_points:
                skipped += 1
                continue

            sample = {
                "char_id": char_to_model_id(char),
                "style_id": 0,
                "dataset_id": -900,
                "user_id": 0,
                "sequence": seq,
                "meta": {
                    "source": "emnist",
                    "split": split,
                    "source_index": int(idx),
                    "label": label,
                    "char": char,
                },
            }
            w.write(json.dumps(sample, ensure_ascii=False) + "\n")
            imported += 1

    shape = list(images.shape[1:]) if images.ndim >= 2 else []
    return ImportStats(
        source="emnist",
        requested_count=target_count,
        imported_count=imported,
        skipped_count=skipped,
        output_path=str(out),
        image_shape=shape,
    )

