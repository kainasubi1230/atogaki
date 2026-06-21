from io import BytesIO
import json
from math import atan2, hypot
import numpy as np
from scipy.ndimage import binary_opening, binary_erosion
from PIL import Image
from trainer.public_dataset import _image_to_sequence

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


LOW_CONTRAST = "LOW_CONTRAST"
NO_TEXT_DETECTED = "NO_TEXT_DETECTED"
TOO_FEW_SEGMENTS = "TOO_FEW_SEGMENTS"

ROW_TEMPLATES = [
    list("あいうえおかきくけこさしすせそ"), # Row 1
    list("たちつてとなにぬねのはひふへほ"), # Row 2
    ["ま", "み", "む", "め", "も", "や", "", "ゆ", "", "よ", "ら", "り", "る", "れ", "ろ"], # Row 3
    ["わ", "", "", "を", "", "ん", ""] + [""] * 8, # Row 4
    list("アイウエオカキクケコサシスセソ"), # Row 5
    list("タチツテトナニヌネノハヒフヘホ"), # Row 6
    ["マ", "ミ", "ム", "メ", "モ", "ヤ", "", "ユ", "", "ヨ", "ラ", "リ", "ル", "レ", "ロ"], # Row 7
    ["ワ", "", "", "ヲ", "", "ン", ""] + [""] * 8  # Row 8
]

# Narrow kana tend to look oversized when the crop is normalized too hard.
_CHAR_OPTICAL_SIZE_BIAS = {
    "い": 0.82,
    "り": 0.84,
    "し": 0.85,
    "く": 0.90,
    "け": 0.90,
    "に": 0.92,
    "ハ": 0.88,
    "リ": 0.88,
    "ノ": 0.86,
    "イ": 0.88,
    "ト": 0.90,
    "レ": 0.90,
    "の": 1.03,
    "め": 1.02,
    "ぬ": 1.02,
    "る": 1.02,
    "ロ": 1.02,
    "ー": 0.92,
}

_SPLIT_STROKE_LABELS = {
    "い",
    "に",
    "は",
    "ほ",
    "け",
    "り",
    "か",
    "や",
    "ふ",
    "ハ",
    "リ",
    "シ",
    "ツ",
    "ソ",
    "ン",
    "代",
}


def _binary_projection_groups(mask: np.ndarray, axis: int) -> list[tuple[int, int]]:
    projection = (mask.sum(axis=axis) > 0).astype(np.int32)
    groups: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(projection):
        if value == 1 and start is None:
            start = idx
        if value == 0 and start is not None:
            groups.append((start, idx))
            start = None
    if start is not None:
        groups.append((start, len(projection)))
    return groups


def _mask_features(mask: np.ndarray) -> dict[str, float] | None:
    if mask.ndim != 2:
        return None
    pixel_count = int(mask.sum())
    if pixel_count < 2:
        return None
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    width = max(1, x1 - x0 + 1)
    height = max(1, y1 - y0 + 1)
    fill_ratio = float(pixel_count) / float(max(1, width * height))
    aspect = float(height) / float(max(1, width))
    components = _binary_projection_groups(mask, axis=0)
    row_groups = _binary_projection_groups(mask, axis=1)
    return {
        "pixel_count": float(pixel_count),
        "width": float(width),
        "height": float(height),
        "fill_ratio": fill_ratio,
        "aspect": aspect,
        "column_groups": float(len(components)),
        "row_groups": float(len(row_groups)),
    }


def _pick_smooth_profile(mask: np.ndarray) -> tuple[str, int]:
    features = _mask_features(mask)
    if features is None:
        return "default", 200
    aspect = features["aspect"]
    fill_ratio = features["fill_ratio"]
    pixel_count = features["pixel_count"]
    column_groups = features["column_groups"]

    # Simple/tall kana and Latin-like shapes need lighter smoothing to avoid
    # turning into blobby shapes. Complex or dense shapes get the kanji profile.
    if pixel_count >= 260.0 or column_groups >= 4.0 or fill_ratio < 0.18:
        return "kanji_fine", 240
    if aspect >= 1.12 and column_groups <= 3.0 and fill_ratio < 0.30:
        return "latin_klee", 170
    if aspect <= 0.72 and pixel_count < 180.0:
        return "latin_klee", 170
    return "default", 200


def _sequence_quality_score(seq: list[list[float]]) -> float:
    if not isinstance(seq, list) or len(seq) < 4:  # relaxed length requirement
        return float("-inf")
    x = 0.0
    y = 0.0
    xs = [0.0]
    ys = [0.0]
    down_steps = 0
    short_steps = 0
    big_jumps = 0
    headings: list[float] = []
    for row in seq:
        if not isinstance(row, list) or len(row) < 3:
            continue
        dx = float(row[0]) * 20.0
        dy = float(row[1]) * 20.0
        pen_down = float(row[2]) > 0.5
        if pen_down:
            down_steps += 1
            seg = hypot(dx, dy)
            if seg < 1.4:
                short_steps += 1
            if seg > 18.0:
                big_jumps += 1
            if seg > 1e-6:
                headings.append(atan2(dy, dx))
        x += dx
        y += dy
        xs.append(x)
        ys.append(y)

    if down_steps < 2:  # relaxed down‑step requirement
        return float("-inf")
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    if span_x <= 1e-6 or span_y <= 1e-6:
        return float("-inf")

    turn_penalty = 0.0
    for idx in range(1, len(headings)):
        delta = headings[idx] - headings[idx - 1]
        while delta > 3.141592653589793:
            delta -= 6.283185307179586
        while delta < -3.141592653589793:
            delta += 6.283185307179586
        turn_penalty += abs(delta)
    if headings:
        turn_penalty /= float(len(headings))

    path_span = span_x + span_y
    compactness = down_steps / max(1.0, len(seq))
    score = (
        min(5.0, path_span / 18.0)
        + min(4.0, span_y / max(1.0, span_x))
        + min(2.5, compactness * 4.0)
        - short_steps * 0.05
        - big_jumps * 2.2
        - turn_penalty * 1.2
    )
    return score


def _apply_label_optical_bias(
    trajectory: list[dict],
    label: str | None,
    bbox: dict | None = None,
) -> list[dict]:
    # Optical size bias disabled for debugging – return original trajectory unchanged
    return trajectory


def remove_notebook_lines_robust(binary: np.ndarray, min_h_run: int = 60, max_v_thickness: int = 3) -> np.ndarray:
    cleaned = binary.copy()
    h, w = cleaned.shape
    core = binary.copy()
    for shift in range(1, max_v_thickness + 1):
        rolled_down = np.roll(binary, shift, axis=0)
        rolled_down[:shift, :] = False
        core = core & rolled_down
        
    protection_mask = core.copy()
    for shift in range(1, 2):
        rolled_up = np.roll(core, -shift, axis=0)
        rolled_up[-shift:, :] = False
        rolled_down = np.roll(core, shift, axis=0)
        rolled_down[:shift, :] = False
        protection_mask = protection_mask | rolled_up | rolled_down
        
    for y in range(h):
        row = cleaned[y, :].astype(np.int32)
        diff = np.diff(np.concatenate([[0], row, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for start, end in zip(starts, ends):
            length = end - start
            if length >= min_h_run:
                for x in range(start, end):
                    if not protection_mask[y, x]:
                        cleaned[y, x] = False
    return cleaned


def get_connected_components(binary: np.ndarray) -> list[dict]:
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    active_y, active_x = np.where(binary)
    active_pixels = list(zip(active_y, active_x))
    
    for y, x in active_pixels:
        if visited[y, x]:
            continue
            
        stack = [(y, x)]
        visited[y, x] = True
        min_y, max_y = y, y
        min_x, max_x = x, x
        
        while stack:
            cy, cx = stack.pop()
            for dy, dx in [( -1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
                        if ny < min_y: min_y = ny
                        elif ny > max_y: max_y = ny
                        if nx < min_x: min_x = nx
                        elif nx > max_x: max_x = nx
                            
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        components.append({
            "x0": min_x,
            "y0": min_y,
            "x1": max_x + 1,
            "y1": max_y + 1,
            "width": width,
            "height": height,
            "center_y": (min_y + max_y) / 2.0,
            "center_x": (min_x + max_x) / 2.0
        })
        
    return components


def get_connected_components_labeled(binary: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    labels_im = np.zeros((h, w), dtype=int)
    components = []
    active_y, active_x = np.where(binary)
    active_pixels = list(zip(active_y, active_x))
    
    comp_id = 0
    for y, x in active_pixels:
        if visited[y, x]:
            continue
            
        comp_id += 1
        stack = [(y, x)]
        visited[y, x] = True
        labels_im[y, x] = comp_id
        min_y, max_y = y, y
        min_x, max_x = x, x
        
        while stack:
            cy, cx = stack.pop()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        labels_im[ny, nx] = comp_id
                        stack.append((ny, nx))
                        if ny < min_y: min_y = ny
                        elif ny > max_y: max_y = ny
                        if nx < min_x: min_x = nx
                        elif nx > max_x: max_x = nx
                            
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        components.append({
            "id": comp_id,
            "x0": min_x,
            "y0": min_y,
            "x1": max_x + 1,
            "y1": max_y + 1,
            "width": width,
            "height": height,
            "center_y": (min_y + max_y) / 2.0,
            "center_x": (min_x + max_x) / 2.0
        })
        
    return components, labels_im



def _component_area(comp: dict) -> int:
    return max(0, int(comp["x1"]) - int(comp["x0"])) * max(0, int(comp["y1"]) - int(comp["y0"]))


def _unique_components(comps: list[dict]) -> list[dict]:
    seen: set[tuple[int, int, int, int]] = set()
    unique: list[dict] = []
    for comp in comps:
        key = (int(comp["x0"]), int(comp["y0"]), int(comp["x1"]), int(comp["y1"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(comp)
    return unique


def _component_bbox(comps: list[dict]) -> tuple[int, int, int, int] | None:
    if not comps:
        return None
    return (
        min(int(c["x0"]) for c in comps),
        min(int(c["y0"]) for c in comps),
        max(int(c["x1"]) for c in comps),
        max(int(c["y1"]) for c in comps),
    )


def _score_bbox_with_cv2(binary_global: np.ndarray, cleaned_global: np.ndarray, bbox: tuple[int, int, int, int], label: str) -> float | None:
    if not _CV2_AVAILABLE or not label.strip():
        return None
    try:
        from trainerlib.cv2_verify_grid import compute_match_score
    except Exception:
        return None
    x0, y0, x1, y1 = bbox
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    try:
        from scipy.ndimage import binary_dilation
        local_cleaned = cleaned_global[y0:y1, x0:x1]
        guide_mask = binary_dilation(local_cleaned, structure=np.ones((5, 5), dtype=bool))
        char_mask = binary_global[y0:y1, x0:x1] & guide_mask
        return float(compute_match_score(char_mask, label))
    except Exception:
        return None


def _stroke_from_mask(mask: np.ndarray, force_profile: str | None = None, min_comp_len: int = 10) -> tuple[list[dict], dict[str, object]]:
    if mask.ndim != 2:
        return [], {}
        
    # Keep original contours to preserve fine loops and tiny stroke gaps
    smoothed = mask
    
    tile = np.where(smoothed, 0, 255).astype(np.uint8)
    profile, max_points = _pick_smooth_profile(smoothed)
    if force_profile:
        profile = force_profile
        max_points = 240

    seq = _image_to_sequence(
        tile,
        max_points=max_points,
        threshold=128,
        smooth_profile=profile,
        foreground_is_dark=True,
        min_comp_len=min_comp_len,
    )
    if not seq:
        return [], {"profile": profile, "max_points": max_points, "quality": float("-inf")}
    quality = _sequence_quality_score(seq)
    # _stroke_from_mask のロジックは変更せず、後続で writer_style の計算に利用される
    points: list[dict] = []
    x = 0.0
    y = 0.0
    t = 0
    for row in seq:
        if not isinstance(row, list) or len(row) < 4:
            continue
        x += float(row[0]) * 20.0
        y += float(row[1]) * 20.0
        pen_down = float(row[2]) > 0.5
        w = max(1, min(4, int(round(float(row[3]) * 4.0))))
        points.append(
            {
                "x": round(float(x), 3),
                "y": round(float(y), 3),
                "t": t,
                "pen_state": "down" if pen_down else "up",
                "width": w,
            }
        )
        t += 1
    if len(points) < 2:
        return [], {"profile": profile, "max_points": max_points, "quality": float("-inf")}
    points[-1]["pen_state"] = "up"
    meta = {
        "profile": profile,
        "max_points": max_points,
        "quality": round(float(quality), 4),
    }
    return points, meta


def preprocess_scan(image_bytes: bytes) -> dict:
    image = Image.open(BytesIO(image_bytes)).convert("L")
    arr = np.array(image)
    h_img, w_img = arr.shape
    contrast = float(arr.std())
    if contrast < 9.0:
        return _sanitize_numpy({"success": False, "reason_code": LOW_CONTRAST})

    mean = float(arr.mean())
    std = float(arr.std())
    threshold = max(120, mean - std)  # adaptive threshold
    binary = arr < threshold
    
    # margins clear: tighter boundaries to prevent clipping
    binary[:, :int(w_img * 0.02)] = False
    binary[:, int(w_img * 0.98):] = False
    binary[:int(h_img * 0.01), :] = False
    binary[int(h_img * 0.99):] = False

    if float(binary.mean()) < 0.003:
        return _sanitize_numpy({"success": False, "reason_code": NO_TEXT_DETECTED})

    # Remove long notebook lines while protecting shorter strokes (min_h_run=65)
    cleaned_binary = remove_notebook_lines_robust(binary, min_h_run=65, max_v_thickness=3)


    raw_components, labels_im = get_connected_components_labeled(cleaned_binary)
    raw_components = [c for c in raw_components if c["width"] >= 2 and c["height"] >= 2 and (c["width"] * c["height"]) >= 6]

    # If components are too few (like in small unit tests), try applying binary_opening to split connected components.
    if raw_components and (len(raw_components) < 10 or w_img < 500 or h_img < 500):
        from scipy.ndimage import binary_opening
        opened = binary_opening(cleaned_binary, structure=np.ones((3, 3), dtype=bool))
        raw_components_opt, labels_im_opt = get_connected_components_labeled(opened)
        raw_components_opt = [c for c in raw_components_opt if c["width"] >= 2 and c["height"] >= 2 and (c["width"] * c["height"]) >= 6]
        if len(raw_components_opt) >= len(raw_components):
            raw_components = raw_components_opt
            labels_im = labels_im_opt

    if not raw_components:
        return _sanitize_numpy({"success": False, "reason_code": TOO_FEW_SEGMENTS})

    # Row clustering based on center_y (with outlier trimming to prevent top/bottom noise from inflating row_tol)
    all_ys = sorted([c["center_y"] for c in raw_components])
    if len(all_ys) >= 10:
        n_trim = max(1, int(len(all_ys) * 0.04))
        trimmed_ys = all_ys[n_trim:-n_trim]
    else:
        trimmed_ys = all_ys

    min_y = trimmed_ys[0] if trimmed_ys else 0.0
    max_y = trimmed_ys[-1] if trimmed_ys else h_img
    row_step = (max_y - min_y) / 7.0 if max_y > min_y else 100.0
    row_tol = max(35.0, row_step * 0.45)

    rows = []
    for c in sorted(raw_components, key=lambda item: item["center_y"]):
        cy = c["center_y"]
        fit_row = None
        for r in rows:
            r_cy = sum(item["center_y"] for item in r) / len(r)
            if abs(cy - r_cy) < row_tol:
                fit_row = r
                break
        if fit_row is not None:
            fit_row.append(c)
        else:
            rows.append([c])

    valid_rows = [r for r in rows if len(r) >= 2]
    valid_rows.sort(key=lambda r: sum(item["center_y"] for item in r) / len(r))
    valid_rows = valid_rows[:8]

    if not valid_rows:
        return _sanitize_numpy({"success": False, "reason_code": TOO_FEW_SEGMENTS})

    # Calculate skew based on valid rows
    slopes = []
    for idx, r in enumerate(valid_rows):
        if len(r) < 10:
            continue
        xs = np.array([c["center_x"] for c in r], dtype=float)
        ys = np.array([c["center_y"] for c in r], dtype=float)
        cov = np.cov(xs, ys)
        if cov[0, 0] > 1e-5:
            slopes.append(cov[0, 1] / cov[0, 0])

    median_slope = float(np.median(slopes)) if slopes else 0.0
    dx_dy = -median_slope
    y_ref = h_img / 2.0

    for c in raw_components:
        c["center_x_corr"] = c["center_x"] - (c["center_y"] - y_ref) * dx_dy

    # Centroids initialization for 15 columns using robust geometry and gap-based estimation
    all_xs = [c["center_x_corr"] for r in valid_rows for c in r]
    if all_xs:
        all_xs_sorted = sorted(all_xs)
        left_bound = w_img * 0.08
        grid_xs = [x for x in all_xs_sorted if x >= left_bound]
        if grid_xs:
            p1 = grid_xs[int(len(grid_xs) * 0.05)]
            grid_xs_filtered = [x for x in grid_xs if x <= p1 + 975.0]
            p99 = grid_xs_filtered[int(len(grid_xs_filtered) * 0.95)]
            temp_step_x = (p99 - p1) / 14.0
        else:
            p1 = 85.0
            temp_step_x = 43.5

        # First-pass merge to estimate true step_x using component gaps
        all_merged_rows = []
        for r in valid_rows:
            raw_char_comps = sorted(r, key=lambda x: x["x0"])
            merged_comps = []
            for comp in raw_char_comps:
                if not merged_comps:
                    merged_comps.append(dict(comp))
                else:
                    prev = merged_comps[-1]
                    gap = comp["x0"] - prev["x1"]
                    potential_width = max(prev["x1"], comp["x1"]) - min(prev["x0"], comp["x0"])
                    allowed = False
                    if gap <= 8 and potential_width <= temp_step_x * 0.88:
                        allowed = True
                    elif gap <= 3:
                        allowed = True
                    elif gap <= 16 and potential_width <= temp_step_x * 0.80:
                        allowed = True
                        
                    if allowed:
                        prev["x1"] = max(prev["x1"], comp["x1"])
                        prev["y0"] = min(prev["y0"], comp["y0"])
                        prev["y1"] = max(prev["y1"], comp["y1"])
                        prev["width"] = prev["x1"] - prev["x0"]
                        prev["height"] = prev["y1"] - prev["y0"]
                        prev["center_x"] = (prev["x0"] + prev["x1"]) / 2.0
                        prev["center_y"] = (prev["y0"] + prev["y1"]) / 2.0
                        prev["center_x_corr"] = prev["center_x"] - (prev["center_y"] - y_ref) * dx_dy
                    else:
                        merged_comps.append(dict(comp))
            all_merged_rows.append(merged_comps)

        all_gaps = []
        for merged_r in all_merged_rows:
            sorted_r = sorted(merged_r, key=lambda c: c["center_x_corr"])
            for i in range(len(sorted_r) - 1):
                g = sorted_r[i+1]["center_x_corr"] - sorted_r[i]["center_x_corr"]
                all_gaps.append(g)

        all_gaps = sorted(all_gaps)
        filtered_gaps = [g for g in all_gaps if 40.0 <= g <= 70.0]
        if filtered_gaps:
            step_x = float(np.median(filtered_gaps))
        else:
            step_x = temp_step_x
            if step_x < 35.0 or step_x > 75.0:
                step_x = 52.0

        # Modulo alignment voting to find refined offset_x
        rough_offset = w_img * 0.122
        candidates = np.arange(rough_offset - step_x / 2.0, rough_offset + step_x / 2.0, 0.5)
        best_offset = rough_offset
        min_score = 1e9
        flat_comps = [c for r in valid_rows for c in r]
        for cand in candidates:
            score = 0.0
            for c in flat_comps:
                cx = c["center_x_corr"]
                dist = abs((cx - cand) % step_x)
                dist = min(dist, step_x - dist)
                score += dist
            if score < min_score:
                min_score = score
                best_offset = cand
                
        offset_x = float(best_offset)
        centroids = [offset_x + i * step_x for i in range(15)]
    else:
        step_x = 43.5
        centroids = [85.0 + i * 43.5 for i in range(15)]

    row_centers = [
        sum(float(item["center_y"]) for item in row) / float(len(row))
        for row in valid_rows
    ]
    final_segments = []

    # Process rows & columns in exact sequence matching the template order to guarantee aligned outputs!
    for row_idx in range(8):
        template = ROW_TEMPLATES[row_idx]
        
        # If this row is not detected, append empty placeholders for all expected characters in this row
        if row_idx >= len(valid_rows):
            for col_idx in range(15):
                char_label = template[col_idx]
                if char_label and char_label.strip():
                    final_segments.append({
                        "bbox": None,
                        "trajectory": [],
                        "label": char_label,
                        "quality": None,
                        "extraction": {},
                    })
            continue

        r = valid_rows[row_idx]
        row_y_center = row_centers[row_idx]
        prev_row_center = row_centers[row_idx - 1] if row_idx > 0 else None
        next_row_center = row_centers[row_idx + 1] if row_idx + 1 < len(row_centers) else None
        row_top = 0 if prev_row_center is None else int(round((prev_row_center + row_y_center) / 2.0))
        row_bottom = h_img if next_row_center is None else int(round((row_y_center + next_row_center) / 2.0))
        row_pad = int(max(8.0, row_step * 0.12))
        row_top = max(0, row_top - row_pad)
        row_bottom = min(h_img, row_bottom + row_pad)
        raw_char_comps = sorted(r, key=lambda x: x["x0"])
        
        # Merge components horizontally with gap <= 8 pixels FIRST to prevent multi-stroke character fragmentation.
        # Use width-constrained merging to avoid merging adjacent characters into massive blocks.
        merged_comps = []
        for comp in raw_char_comps:
            if not merged_comps:
                merged_comps.append(dict(comp))
            else:
                prev = merged_comps[-1]
                gap = comp["x0"] - prev["x1"]
                
                potential_width = max(prev["x1"], comp["x1"]) - min(prev["x0"], comp["x0"])
                
                # Check merge criteria
                allowed = False
                if gap <= 8 and potential_width <= step_x * 0.88:
                    allowed = True
                elif gap <= 3:  # almost touching, must merge
                    allowed = True
                elif gap <= 16 and potential_width <= step_x * 0.80:  # split stroke with larger gap
                    allowed = True
                    
                if allowed:
                    prev["x1"] = max(prev["x1"], comp["x1"])
                    prev["y0"] = min(prev["y0"], comp["y0"])
                    prev["y1"] = max(prev["y1"], comp["y1"])
                    prev["width"] = prev["x1"] - prev["x0"]
                    prev["height"] = prev["y1"] - prev["y0"]
                    prev["center_x"] = (prev["x0"] + prev["x1"]) / 2.0
                    prev["center_y"] = (prev["y0"] + prev["y1"]) / 2.0
                    if "ids" not in prev:
                        prev["ids"] = {prev.get("id")} if prev.get("id") is not None else set()
                    if comp.get("id") is not None:
                        prev["ids"].add(comp["id"])
                else:
                    merged_comps.append(dict(comp))

        # Split wide components (two characters merged) AFTER merging to handle touching characters robustly
        final_comps = []
        for comp in merged_comps:
            w = comp["x1"] - comp["x0"]
            if w > step_x * 1.35:
                split_count = max(2, int(round(w / step_x)))
                split_w = w / split_count
                for s_idx in range(split_count):
                    s_x0 = int(round(comp["x0"] + s_idx * split_w))
                    s_x1 = int(round(comp["x0"] + (s_idx + 1) * split_w))
                    sub_comp = {
                        "x0": s_x0,
                        "x1": s_x1,
                        "y0": comp["y0"],
                        "y1": comp["y1"],
                        "width": s_x1 - s_x0,
                        "height": comp["height"],
                        "center_y": comp["center_y"],
                        "center_x": (s_x0 + s_x1) / 2.0,
                    }
                    if "ids" in comp:
                        sub_comp["ids"] = set(comp["ids"])
                    elif comp.get("id") is not None:
                        sub_comp["id"] = comp["id"]
                    final_comps.append(sub_comp)
            else:
                final_comps.append(comp)

        char_comps = final_comps

        for c in char_comps:
            c["center_x_corr"] = c["center_x"] - (c["center_y"] - y_ref) * dx_dy

        N = len(char_comps)
        M = 15
        dp = np.full((N + 1, M + 1), fill_value=1e9)
        parent = {}
        
        dp[0][0] = 0.0
        
        for i in range(N + 1):
            for j in range(M + 1):
                if i == 0 and j == 0:
                    continue
                    
                if j > 0:
                    skip_cost = 0.0 if not template[j-1].strip() else 45.0
                    if dp[i][j-1] + skip_cost < dp[i][j]:
                        dp[i][j] = dp[i][j-1] + skip_cost
                        parent[(i, j)] = (i, j-1, -1)
                        
                if i > 0 and j > 0:
                    comp = char_comps[i-1]
                    dist = abs(comp["center_x_corr"] - centroids[j-1])
                    
                    is_outer_boundary = False
                    if j-1 == 0 and comp["center_x_corr"] < centroids[0]:
                        is_outer_boundary = True
                    elif j-1 == 14 and comp["center_x_corr"] > centroids[14]:
                        is_outer_boundary = True
                        
                    max_dist = step_x * 1.5 if is_outer_boundary else step_x * 0.90
                    if dist > max_dist:
                        dist += 200.0
                    col_label = template[j-1]
                    
                    is_small = (comp["width"] < 15 and comp["height"] < 15) or (comp["width"] * comp["height"] < 120)
                    label_penalty = 0.0
                    if not col_label.strip():
                        label_penalty = 50.0 if is_small else 250.0
                        
                    match_cost = dist + label_penalty
                    if dp[i-1][j-1] + match_cost < dp[i][j]:
                        dp[i][j] = dp[i-1][j-1] + match_cost
                        parent[(i, j)] = (i-1, j-1, j-1)
                        
                if i > 0:
                    comp = char_comps[i-1]
                    is_small = (comp["width"] < 15 and comp["height"] < 15) or (comp["width"] * comp["height"] < 120)
                    skip_comp_cost = 10.0 if is_small else 60.0
                    if dp[i-1][j] + skip_comp_cost < dp[i][j]:
                        dp[i][j] = dp[i-1][j] + skip_comp_cost
                        parent[(i, j)] = (i-1, j, -2)
                        
        # Backtrack DP assignments
        curr_i, curr_j = N, M
        assignments = {}
        ignored_comps = []
        while (curr_i, curr_j) in parent:
            prev_i, prev_j, matched_col = parent[(curr_i, curr_j)]
            if matched_col >= 0:
                assignments[prev_i] = matched_col
            elif matched_col == -2:
                ignored_comps.append(prev_i)
            curr_i, curr_j = prev_i, prev_j
            
        if row_idx in (1, 2):
            print(f"ROW {row_idx} DP assignments (N={N}):")
            for comp_idx in sorted(assignments.keys()):
                col_idx = assignments[comp_idx]
                c = char_comps[comp_idx]
                print(f"  Comp {comp_idx} (x0={c['x0']}): Col {col_idx} ({template[col_idx]})")
            print("Ignored:")
            for comp_idx in sorted(ignored_comps):
                c = char_comps[comp_idx]
                print(f"  Comp {comp_idx} (x0={c['x0']})")

        # Calibrate row centroids using a robust quadratic fit on DP assignments with unwrapped shifts
        shifts = []
        cols = []
        for comp_idx, col_idx in sorted(assignments.items(), key=lambda x: x[1]):
            comp = char_comps[comp_idx]
            shifts.append(float(comp.get("center_x_corr", comp["center_x"])) - centroids[col_idx])
            cols.append(col_idx)

        unwrapped_shifts = list(shifts)
        if len(shifts) >= 2:
            bias = 0.0
            for idx in range(1, len(shifts)):
                prev_val = unwrapped_shifts[idx - 1]
                raw_val = shifts[idx] + bias
                diff = raw_val - prev_val
                if diff < -step_x * 0.70:
                    bias += step_x
                    raw_val += step_x
                elif diff > step_x * 0.70:
                    bias -= step_x
                    raw_val -= step_x
                unwrapped_shifts[idx] = raw_val

        row_centroids = list(centroids)
        if False:  # disabled robust fitting to prevent edge extrapolation errors
            try:
                coeffs = np.polyfit(cols, unwrapped_shifts, 1)
                fitted_shifts = np.polyval(coeffs, cols)
                residuals = np.abs(np.array(unwrapped_shifts) - fitted_shifts)
                
                inliers = residuals < 15.0
                if np.sum(inliers) >= 3:
                    inlier_cols = [cols[idx] for idx, val in enumerate(inliers) if val]
                    inlier_shifts = [unwrapped_shifts[idx] for idx, val in enumerate(inliers) if val]
                    coeffs = np.polyfit(inlier_cols, inlier_shifts, 1)
                    
                row_centroids = [centroids[c] + np.polyval(coeffs, c) for c in range(15)]
            except Exception:
                pass

        # --- Second Pass DP Grid Alignment using calibrated row_centroids ---
        dp2 = np.full((N + 1, M + 1), fill_value=1e9)
        parent2 = {}
        dp2[0][0] = 0.0
        
        for i in range(N + 1):
            for j in range(M + 1):
                if i == 0 and j == 0:
                    continue
                if j > 0:
                    skip_cost = 0.0 if not template[j-1].strip() else 45.0
                    if dp2[i][j-1] + skip_cost < dp2[i][j]:
                        dp2[i][j] = dp2[i][j-1] + skip_cost
                        parent2[(i, j)] = (i, j-1, -1)
                if i > 0 and j > 0:
                    comp = char_comps[i-1]
                    dist = abs(comp["center_x_corr"] - row_centroids[j-1])
                    
                    is_outer_boundary = False
                    if j-1 == 0 and comp["center_x_corr"] < row_centroids[0]:
                        is_outer_boundary = True
                    elif j-1 == 14 and comp["center_x_corr"] > row_centroids[14]:
                        is_outer_boundary = True
                        
                    max_dist = step_x * 1.5 if is_outer_boundary else step_x * 0.90
                    if dist > max_dist:
                        dist += 200.0
                    col_label = template[j-1]
                    is_small = (comp["width"] < 15 and comp["height"] < 15) or (comp["width"] * comp["height"] < 120)
                    label_penalty = 0.0
                    if not col_label.strip():
                        label_penalty = 50.0 if is_small else 250.0
                    match_cost = dist + label_penalty
                    if dp2[i-1][j-1] + match_cost < dp2[i][j]:
                        dp2[i][j] = dp2[i-1][j-1] + match_cost
                        parent2[(i, j)] = (i-1, j-1, j-1)
                if i > 0:
                    comp = char_comps[i-1]
                    is_small = (comp["width"] < 15 and comp["height"] < 15) or (comp["width"] * comp["height"] < 120)
                    skip_comp_cost = 10.0 if is_small else 60.0
                    if dp2[i-1][j] + skip_comp_cost < dp2[i][j]:
                        dp2[i][j] = dp2[i-1][j] + skip_comp_cost
                        parent2[(i, j)] = (i-1, j, -2)

        # Overwrite first pass assignments and ignored_comps
        curr_i, curr_j = N, M
        assignments = {}
        ignored_comps = []
        while (curr_i, curr_j) in parent2:
            prev_i, prev_j, matched_col = parent2[(curr_i, curr_j)]
            if matched_col >= 0:
                assignments[prev_i] = matched_col
            elif matched_col == -2:
                ignored_comps.append(prev_i)
            curr_i, curr_j = prev_i, prev_j
            
        if row_idx in (1, 2):
            print(f"ROW {row_idx} DP assignments (N={N}):")
            for comp_idx in sorted(assignments.keys()):
                col_idx = assignments[comp_idx]
                c = char_comps[comp_idx]
                print(f"  Comp {comp_idx} (x0={c['x0']}): Col {col_idx} ({template[col_idx]})")
            print("Ignored:")
            for comp_idx in sorted(ignored_comps):
                c = char_comps[comp_idx]
                print(f"  Comp {comp_idx} (x0={c['x0']})")

        col_groups = {}
        for i, comp in enumerate(char_comps):
            if i in assignments:
                col_idx = assignments[i]
                if col_idx not in col_groups:
                    col_groups[col_idx] = []
                col_groups[col_idx].append(comp)
                
        if ignored_comps:
            for i in ignored_comps:
                comp = char_comps[i]
                best_col = None
                best_dist = 1e9
                for col_idx in range(15):
                    dist = abs(comp["center_x_corr"] - row_centroids[col_idx])
                    if dist < best_dist:
                        best_dist = dist
                        best_col = col_idx
                if best_col is not None and best_dist < (step_x * 0.4):
                    if template[best_col].strip():
                        if best_col not in col_groups:
                            col_groups[best_col] = []
                        col_groups[best_col].append(comp)

        # Re-collect components by the physical grid cell. DP is good for order,
        # but split kana such as "い" and "ハ" must be boxed as one character.
        nonblank_cols = [idx for idx, ch in enumerate(template) if ch and ch.strip()]
        grid_col_groups: dict[int, list[dict]] = {idx: list(col_groups.get(idx, [])) for idx in nonblank_cols}
        dp_components_all = [c for g in col_groups.values() for c in g]

        for comp in char_comps:
            if comp in dp_components_all:
                continue
            comp_center_corr = float(comp.get("center_x_corr", comp["center_x"]))
            comp_center_y = float(comp["center_y"])
            if comp_center_y < row_top or comp_center_y > row_bottom:
                continue
            for col_idx in nonblank_cols:
                char_label = template[col_idx]
                left_boundary_corr = (row_centroids[col_idx - 1] + row_centroids[col_idx]) / 2.0 if col_idx > 0 else row_centroids[col_idx] - step_x * 1.2
                right_boundary_corr = (row_centroids[col_idx] + row_centroids[col_idx + 1]) / 2.0 if col_idx < 14 else row_centroids[col_idx] + step_x * 1.2
                dist = abs(comp_center_corr - row_centroids[col_idx])
                inside_cell = left_boundary_corr <= comp_center_corr <= right_boundary_corr
                
                threshold_factor = 0.78 if char_label in _SPLIT_STROKE_LABELS else 0.56
                if inside_cell or dist <= step_x * threshold_factor:
                    grid_col_groups[col_idx].append(comp)

        col_sources: dict[int, str] = {}
        for col_idx in nonblank_cols:
            dp_comps = _unique_components(col_groups.get(col_idx, []))
            cell_comps = _unique_components(grid_col_groups.get(col_idx, []))
            if not cell_comps:
                col_groups[col_idx] = dp_comps
                col_sources[col_idx] = "dp"
                continue
            if not dp_comps:
                col_groups[col_idx] = cell_comps
                col_sources[col_idx] = "cell"
                continue
            dp_bbox = _component_bbox(dp_comps)
            cell_bbox = _component_bbox(cell_comps)
            dp_score = _score_bbox_with_cv2(binary, cleaned_binary, dp_bbox, template[col_idx]) if dp_bbox else None
            cell_score = _score_bbox_with_cv2(binary, cleaned_binary, cell_bbox, template[col_idx]) if cell_bbox else None
            
            # Check individual scores of each component in cell_comps to find if one is much better than dp_comps
            best_single_comp = None
            best_single_score = -1.0
            if len(cell_comps) > 1:
                for idx_c, c in enumerate(cell_comps):
                    # Position check to avoid stealing components from neighboring characters
                    comp_cx = float(c.get("center_x_corr", c["center_x"]))
                    dist_x = comp_cx - centroids[col_idx]
                    is_outer_dev = False
                    if col_idx == 0 and dist_x < 0:
                        is_outer_dev = True
                    elif col_idx == 14 and dist_x > 0:
                        is_outer_dev = True
                    allowed_dev = step_x * 1.0 if is_outer_dev else step_x * 0.40
                    if abs(dist_x) > allowed_dev:
                        continue
                        
                    c_bbox = _component_bbox([c])
                    c_score = _score_bbox_with_cv2(binary, cleaned_binary, c_bbox, template[col_idx]) if c_bbox else None
                    print(f"  DEBUG single_comp Col {col_idx} ({template[col_idx]}): comp_idx={idx_c}, bbox=[{c['x0']},{c['x1']}], score={c_score}")
                    if c_score is not None and c_score > best_single_score:
                        best_single_score = c_score
                        best_single_comp = c
            
            use_cell = False
            if cell_score is not None and dp_score is not None:
                use_cell = (cell_score >= dp_score - 0.03)
                print(f"DEBUG score Col {col_idx} ({template[col_idx]}): cell_score={cell_score:.4f}, dp_score={dp_score:.4f}, use_cell={use_cell}")
            elif cell_score is not None:
                use_cell = True
            else:
                use_cell = False
                
            if best_single_comp is not None and best_single_score > (dp_score or -1.0) + 0.01:
                print(f"DEBUG substitute Col {col_idx} ({template[col_idx]}): use single best comp (score={best_single_score:.4f} > dp_score={dp_score or -1.0:.4f})")
                col_groups[col_idx] = [best_single_comp]
                col_sources[col_idx] = "cell_better_single"
            else:
                col_groups[col_idx] = cell_comps if use_cell else dp_comps
                col_sources[col_idx] = "cell_split_stroke" if use_cell else "dp"

        # Pass 1: Gather candidate components for each column in this row
        row_col_comps = {}
        for col_idx in range(15):
            char_label = template[col_idx]
            if not char_label or not char_label.strip():
                continue
                
            comps = list(col_groups.get(col_idx, []))
            if not comps:
                # Fallback Search: if a character is missing, try to search the original binary in its expected position.
                row_y_center = sum(c["center_y"] for c in r) / len(r) if r else h_img / 2.0
                cx_corr = centroids[col_idx]
                cx_orig = cx_corr + (row_y_center - y_ref) * dx_dy
                
                # Search window of size: width = step_x * 0.9, height = row_step * 0.9
                h_win = int(row_step * 0.9)
                w_win = int(step_x * 0.9)
                
                x0_win = max(0, int(cx_orig - w_win // 2))
                x1_win = min(w_img, int(cx_orig + w_win // 2))
                y0_win = max(0, int(row_y_center - h_win // 2))
                y1_win = min(h_img, int(row_y_center + h_win // 2))
                
                if (x1_win - x0_win) > 5 and (y1_win - y0_win) > 5:
                    win_mask = cleaned_binary[y0_win:y1_win, x0_win:x1_win]
                    if win_mask.sum() >= 10:
                        ys_act, xs_act = np.where(win_mask)
                        new_x0 = x0_win + int(xs_act.min())
                        new_x1 = x0_win + int(xs_act.max()) + 1
                        new_y0 = y0_win + int(ys_act.min())
                        new_y1 = y0_win + int(ys_act.max()) + 1
                        
                        comps = [{
                            "x0": new_x0,
                            "x1": new_x1,
                            "y0": new_y0,
                            "y1": new_y1,
                            "width": new_x1 - new_x0,
                            "height": new_y1 - new_y0,
                            "center_x": (new_x0 + new_x1) / 2.0,
                            "center_y": (new_y0 + new_y1) / 2.0
                        }]
            
            # cv2 rescue: if cv2 is available, check match score and try adding nearby unassigned components
            if _CV2_AVAILABLE and char_label.strip() and comps:
                from trainerlib.cv2_verify_grid import compute_match_score
                
                def _build_crop(comps_list):
                    if not comps_list:
                        return None, None
                    _x0 = min(c["x0"] for c in comps_list)
                    _x1 = max(c["x1"] for c in comps_list)
                    _y0 = min(c["y0"] for c in comps_list)
                    _y1 = max(c["y1"] for c in comps_list)
                    m = 6
                    _x0 = max(0, _x0 - m)
                    _x1 = min(w_img, _x1 + m)
                    _y0 = max(0, _y0 - m)
                    _y1 = min(h_img, _y1 + m)
                    return cleaned_binary[_y0:_y1, _x0:_x1], (_x0, _y0, _x1, _y1)
                
                base_crop, _ = _build_crop(comps)
                if base_crop is not None:
                    base_score = compute_match_score(base_crop, char_label)
                    
                    # Gather nearby components in the same physical cell.
                    # Search around the actual center of the already-assigned components if available to handle shifted characters robustly.
                    if comps:
                        char_center_x_corr = sum(float(c.get("center_x_corr", c["center_x"])) for c in comps) / len(comps)
                    else:
                        char_center_x_corr = centroids[col_idx]
                        
                    near_comps = [
                        (i, c) for i, c in enumerate(char_comps)
                        if c not in comps
                        and row_top <= float(c["center_y"]) <= row_bottom
                        and abs(float(c["center_x_corr"]) - char_center_x_corr) < step_x * 0.62
                    ]
                    
                    # Try adding each nearby component individually to see if it improves score
                    improved = True
                    trial_comps = list(comps)
                    while improved:
                        improved = False
                        best_trial_score = base_score
                        best_trial_comp = None
                        for i, nc in near_comps:
                            if nc in trial_comps:
                                continue
                            trial_crop, _ = _build_crop(trial_comps + [nc])
                            if trial_crop is None:
                                continue
                            trial_score = compute_match_score(trial_crop, char_label)
                            if trial_score > best_trial_score + 0.01:
                                best_trial_score = trial_score
                                best_trial_comp = nc
                        if best_trial_comp is not None:
                            trial_comps.append(best_trial_comp)
                            base_score = best_trial_score
                            improved = True
                    comps = trial_comps
            
            row_col_comps[col_idx] = comps

        # Pass 2: Conflict Resolution
        # Enforce that each component in char_comps is assigned to at most one column in this row.
        # If multiple columns want the same component, assign it to the one with the closest DP-assigned component (anchor),
        # falling back to centroid distance if the column has no DP-assigned component.
        col_to_dp_comp = {col: comp for comp, col in assignments.items()}
        
        comp_claims = {i: [] for i in range(len(char_comps))}
        for col_idx, comps in row_col_comps.items():
            for c in comps:
                for i, cc in enumerate(char_comps):
                    if c is cc or (c["x0"] == cc["x0"] and c["y0"] == cc["y0"] and c["x1"] == cc["x1"] and c["y1"] == cc["y1"]):
                        comp_claims[i].append(col_idx)
                        break

        for i, col_indices in comp_claims.items():
            if len(col_indices) > 1:
                comp = char_comps[i]
                comp_cx = float(comp.get("center_x_corr", comp["center_x"]))
                # If this component was assigned by DP to one of the claiming columns, prefer that column
                dp_col = assignments.get(i)
                if dp_col in col_indices:
                    best_col = dp_col
                else:
                    # Assign to the column whose physical grid center is closest to this component.
                    def get_dist(col):
                        return abs(comp_cx - centroids[col])
                    best_col = min(col_indices, key=get_dist)
                if row_idx == 2:
                    print(f"  Conflict: Comp {i} (x0={comp['x0']}, cx={comp_cx:.1f}) claimed by {col_indices} ({[template[col] for col in col_indices]}). DP={dp_col} ({template[dp_col] if dp_col is not None else 'None'}). Selected={best_col} ({template[best_col]})")
                for col in col_indices:
                    if col != best_col:
                        row_col_comps[col] = [
                            c for c in row_col_comps[col]
                            if not (c is comp or (c["x0"] == comp["x0"] and c["y0"] == comp["y0"] and c["x1"] == comp["x1"] and c["y1"] == comp["y1"]))
                        ]

        # Pass 3: Assemble segments sequentially for this row using the resolved components
        row_col_bboxes = {}
        row_col_inks = {}
        
        for col_idx in range(15):
            char_label = template[col_idx]
            if not char_label or not char_label.strip():
                continue
                
            comps = row_col_comps.get(col_idx, [])
            if not comps:
                continue
                
            left_boundary_corr = (centroids[col_idx - 1] + centroids[col_idx]) / 2.0 if col_idx > 0 else 0.0
            right_boundary_corr = (centroids[col_idx] + centroids[col_idx + 1]) / 2.0 if col_idx < 14 else w_img
            row_y_center = sum(c["center_y"] for c in r) / len(r) if r else h_img / 2.0
            left_limit_calc = max(0, int(round(left_boundary_corr + (row_y_center - y_ref) * dx_dy)))
            right_limit_calc = min(w_img, int(round(right_boundary_corr + (row_y_center - y_ref) * dx_dy)))

            new_x0 = min(item["x0"] for item in comps)
            new_x1 = max(item["x1"] for item in comps)
            new_y0 = min(item["y0"] for item in comps)
            new_y1 = max(item["y1"] for item in comps)
            ink_x0, ink_y0, ink_x1, ink_y1 = int(new_x0), int(new_y0), int(new_x1), int(new_y1)
            
            # The expansion limits must not cut the original components
            left_limit = min(left_limit_calc, new_x0)
            right_limit = max(right_limit_calc, new_x1)
            
            if (new_x1 - new_x0) < 2 or (new_y1 - new_y0) < 2:
                continue
            
            # 1. Base margin padding to prevent boundary clipping & improve skeletonization, constrained by gutters
            margin = 6
            new_x0 = max(left_limit, new_x0 - margin)
            new_x1 = min(right_limit, new_x1 + margin)
            new_y0 = max(0, new_y0 - margin)
            new_y1 = min(h_img, new_y1 + margin)
            
            # 2. Dynamic expansion to capture strokes extending outside the box (run up to 3 times), constrained by gutters
            for _ in range(3):
                local_binary = binary[new_y0:new_y1, new_x0:new_x1]
                
                touch_left = np.any(local_binary[:, 0]) if new_x0 > left_limit else False
                touch_right = np.any(local_binary[:, -1]) if new_x1 < right_limit else False
                touch_top = np.any(local_binary[0, :]) if new_y0 > 0 else False
                touch_bottom = np.any(local_binary[-1, :]) if new_y1 < h_img else False
                
                if not (touch_left or touch_right or touch_top or touch_bottom):
                    break
                    
                expand_size = 10
                if touch_left:
                    new_x0 = max(left_limit, new_x0 - expand_size)
                if touch_right:
                    new_x1 = min(right_limit, new_x1 + expand_size)
                if touch_top:
                    new_y0 = max(0, new_y0 - expand_size)
                if touch_bottom:
                    new_y1 = min(h_img, new_y1 + expand_size)
                    
            row_col_bboxes[col_idx] = {"x0": new_x0, "x1": new_x1, "y0": new_y0, "y1": new_y1}
            row_col_inks[col_idx] = {"x0": ink_x0, "x1": ink_x1, "y0": ink_y0, "y1": ink_y1}

        # Resolve overlaps between adjacent non-blank columns
        nonblank_indices = [idx for idx in range(15) if template[idx] and template[idx].strip()]
        for i in range(len(nonblank_indices) - 1):
            curr_col = nonblank_indices[i]
            next_col = nonblank_indices[i + 1]
            
            curr_box = row_col_bboxes.get(curr_col)
            next_box = row_col_bboxes.get(next_col)
            curr_ink = row_col_inks.get(curr_col)
            next_ink = row_col_inks.get(next_col)
            
            if not curr_box or not next_box or not curr_ink or not next_ink:
                continue
                
            if curr_box["x1"] > next_box["x0"]:
                curr_ink_x1 = curr_ink["x1"]
                next_ink_x0 = next_ink["x0"]
                
                if curr_ink_x1 < next_ink_x0:
                    # Inks do not overlap. Clip to the midpoint.
                    split_x = (curr_ink_x1 + next_ink_x0) // 2
                    curr_box["x1"] = split_x
                    next_box["x0"] = split_x + 1
                else:
                    # Inks physically overlap. Clip bounding boxes to their respective ink boundaries to minimize overlap.
                    curr_box["x1"] = min(curr_box["x1"], curr_ink_x1)
                    next_box["x0"] = max(next_box["x0"], next_ink_x0)

        previous_row_bbox: dict | None = None
        for col_idx in range(15):
            char_label = template[col_idx]
            if not char_label or not char_label.strip():
                continue
                
            box = row_col_bboxes.get(col_idx)
            ink = row_col_inks.get(col_idx)
            
            if not box or not ink:
                # Truly missing segment, append placeholder to keep exact sequence align
                final_segments.append({
                    "bbox": None,
                    "trajectory": [],
                    "label": char_label,
                    "quality": None,
                    "extraction": {},
                })
                continue
                
            new_x0, new_x1, new_y0, new_y1 = box["x0"], box["x1"], box["y0"], box["y1"]
            ink_x0, ink_y0, ink_x1, ink_y1 = ink["x0"], ink["y0"], ink["x1"], ink["y1"]
            
            if previous_row_bbox and new_x0 < int(previous_row_bbox["x1"]) <= ink_x0:
                # Remove pure padding overlap without clipping the current ink.
                new_x0 = min(ink_x0, int(previous_row_bbox["x1"]) + 1)
            
            w = new_x1 - new_x0
            h = new_y1 - new_y0
                
            if w < 2 or h < 2:
                final_segments.append({
                    "bbox": None,
                    "trajectory": [],
                    "label": char_label,
                    "quality": None,
                    "extraction": {},
                })
                continue
                
            # Collect other assigned component IDs in the same row to mask them out
            comps = row_col_comps.get(col_idx, [])
            assigned_ids = set()
            for c in comps:
                if "ids" in c:
                    assigned_ids.update(c["ids"])
                elif c.get("id") is not None:
                    assigned_ids.add(c["id"])

            other_assigned_ids = set()
            for other_col, other_comps in row_col_comps.items():
                if other_col != col_idx:
                    for c in other_comps:
                        if "ids" in c:
                            other_assigned_ids.update(c["ids"])
                        elif c.get("id") is not None:
                            other_assigned_ids.add(c["id"])

            # Use a hybrid approach: guide mask from dilated cleaned_binary to filter grid lines, 
            # while keeping faint strokes from the original binary, but mask out components assigned to other columns!
            from scipy.ndimage import binary_dilation
            local_cleaned = cleaned_binary[new_y0:new_y1, new_x0:new_x1].copy()
            local_binary = binary[new_y0:new_y1, new_x0:new_x1].copy()
            local_labels = labels_im[new_y0:new_y1, new_x0:new_x1]
            
            if other_assigned_ids:
                exclude_mask = np.isin(local_labels, list(other_assigned_ids))
                local_cleaned[exclude_mask] = False
                local_binary[exclude_mask] = False

            guide_mask = binary_dilation(local_cleaned, structure=np.ones((5, 5), dtype=bool))
            char_mask = local_binary & guide_mask
            
            stroke, stroke_meta = _stroke_from_mask(char_mask, force_profile=None, min_comp_len=3)
            if not stroke:
                final_segments.append({
                    "bbox": None,
                    "trajectory": [],
                    "label": char_label,
                    "quality": None,
                    "extraction": {},
                })
                continue
 
            for p in stroke:
                p["x"] = int(p["x"]) + int(new_x0)
                p["y"] = int(p["y"]) + int(new_y0)
 
            stroke_quality = _sequence_quality_score(
                [
                    [
                        (float(p2["x"]) - float(p1["x"])) / 20.0,
                        (float(p2["y"]) - float(p1["y"])) / 20.0,
                        1.0 if p2.get("pen_state") == "down" else 0.0,
                        float(p2.get("width", 1)),
                    ]
                    for p1, p2 in zip(stroke[:-1], stroke[1:])
                ]
            )
 
            if char_label in _CHAR_OPTICAL_SIZE_BIAS:
                stroke = _apply_label_optical_bias(stroke, char_label, {"x0": new_x0, "y0": new_y0, "x1": new_x1, "y1": new_y1})
 
            match_score = _score_bbox_with_cv2(binary, cleaned_binary, (new_x0, new_y0, new_x1, new_y1), char_label)
            final_segments.append({
                "bbox": {"x0": new_x0, "y0": new_y0, "x1": new_x1, "y1": new_y1},
                "trajectory": stroke,
                "label": char_label,
                "quality": round(float(max(stroke_quality, stroke_meta.get("quality", float("-inf")))), 4),
                "extraction": {
                    "profile": stroke_meta.get("profile"),
                    "max_points": stroke_meta.get("max_points"),
                    "bbox_source": col_sources.get(col_idx, "fallback"),
                    "cv2_match_score": match_score,
                    "cv2_verified": None if match_score is None else bool(match_score >= 0.10),
                    "ink_bbox": {"x0": ink_x0, "y0": ink_y0, "x1": ink_x1, "y1": ink_y1},
                },
            })
            previous_row_bbox = {"x0": new_x0, "y0": new_y0, "x1": new_x1, "y1": new_y1}

    # Extract writer style vector
    writer_style = _extract_style_vector(final_segments)

    res = {
        "success": True,
        "reason_code": None,
        "segment_count": len(final_segments),
        "segments": final_segments,
        "writer_style": writer_style,
    }
    return _sanitize_numpy(res)


def _sanitize_numpy(val):
    if isinstance(val, dict):
        return {k: _sanitize_numpy(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_sanitize_numpy(v) for v in val]
    elif isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(val)
    elif isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return _sanitize_numpy(val.tolist())
    else:
        return val


def serialize_preprocess(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Writer style extraction helper
# ---------------------------------------------------------------------------

def _extract_style_vector(segments: list[dict]) -> dict:
    """Calculate a simple style vector for the whole writing.

    The vector aggregates statistics over all trajectories:
    * mean / std of stroke length (pixel)
    * mean / std of instantaneous speed (Δx/Δt, Δy/Δt approximated)
    * mean / std of pen width (if provided)
    * mean / std of curvature (angle change per step)
    """
    import math
    lengths = []
    speeds = []
    widths = []
    curvatures = []
    for seg in segments:
        traj = seg.get("trajectory", [])
        if not traj:
            continue
        for i in range(1, len(traj)):
            p0 = traj[i - 1]
            p1 = traj[i]
            dx = p1["x"] - p0["x"]
            dy = p1["y"] - p0["y"]
            dist = math.hypot(dx, dy)
            lengths.append(dist)
            speeds.append(dist)  # points are equally spaced in time
            widths.append(p1.get("width", 1))
            if i >= 2:
                p_prev = traj[i - 2]
                v1x = p0["x"] - p_prev["x"]
                v1y = p0["y"] - p_prev["y"]
                v2x = dx
                v2y = dy
                dot = v1x * v2x + v1y * v2y
                norm1 = math.hypot(v1x, v1y)
                norm2 = math.hypot(v2x, v2y)
                if norm1 > 0 and norm2 > 0:
                    cos_angle = max(-1.0, min(1.0, dot / (norm1 * norm2)))
                    angle = math.acos(cos_angle)
                    curvatures.append(angle)
    def _stats(arr):
        if not arr:
            return None, None
        a = np.array(arr, dtype=float)
        return float(a.mean()), float(a.std())
    mean_len, std_len = _stats(lengths)
    mean_spd, std_spd = _stats(speeds)
    mean_w, std_w = _stats(widths)
    mean_curv, std_curv = _stats(curvatures)
    return {
        "mean_stroke_length": mean_len,
        "std_stroke_length": std_len,
        "mean_speed": mean_spd,
        "std_speed": std_spd,
        "mean_width": mean_w,
        "std_width": std_w,
        "mean_curvature": mean_curv,
        "std_curvature": std_curv,
    }
