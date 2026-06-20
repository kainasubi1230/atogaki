from io import BytesIO
import json
from math import atan2, hypot
import numpy as np
from scipy.ndimage import binary_opening, binary_erosion
from PIL import Image
from trainer.public_dataset import _image_to_sequence


LOW_CONTRAST = "LOW_CONTRAST"
NO_TEXT_DETECTED = "NO_TEXT_DETECTED"
TOO_FEW_SEGMENTS = "TOO_FEW_SEGMENTS"

ROW_TEMPLATES = [
    list("あいうえおかきくけこさしすせそ"), # Row 1
    list("たちつてとなにぬねのはひふへほ"), # Row 2
    list("まみむめもや ゆ よらりるれろ"), # Row 3
    ["わ", "", "", "を", "", "", "ん"] + [""] * 8, # Row 4
    list("アイウエオカキクケコサシスセソ"), # Row 5
    list("タチツテトナニヌネノハヒフヘホ"), # Row 6
    list("マミムメモヤ ユ ヨラリルレロ"), # Row 7
    ["ワ", "", "", "ヲ", "", "", "ン"] + [""] * 8  # Row 8
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


def _stroke_from_mask(mask: np.ndarray, force_profile: str | None = None) -> tuple[list[dict], dict[str, object]]:
    if mask.ndim != 2:
        return [], {}
        
    # Smooth character contours using morphological operations to prevent skeletonization noise (branches/wiggles)
    from scipy.ndimage import binary_dilation, binary_erosion
    smoothed = binary_dilation(mask, structure=np.ones((2, 2), dtype=bool))
    smoothed = binary_erosion(smoothed, structure=np.ones((2, 2), dtype=bool))
    
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
                "x": int(round(x)),
                "y": int(round(y)),
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
    cleaned_binary = binary_opening(cleaned_binary, structure=np.ones((3, 3), dtype=bool))

    raw_components = get_connected_components(cleaned_binary)
    raw_components = [c for c in raw_components if c["width"] >= 2 and c["height"] >= 2 and (c["width"] * c["height"]) >= 6]

    if not raw_components:
        return _sanitize_numpy({"success": False, "reason_code": TOO_FEW_SEGMENTS})

    # Row clustering based on center_y
    all_ys = [c["center_y"] for c in raw_components]
    min_y, max_y = min(all_ys), max(all_ys)
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

    # Centroids initialization for 15 columns using 2%-98% percentile range
    all_xs = [c["center_x_corr"] for r in valid_rows for c in r]
    if all_xs:
        all_xs_sorted = sorted(all_xs)
        p1 = all_xs_sorted[int(len(all_xs_sorted) * 0.02)]
        p99 = all_xs_sorted[int(len(all_xs_sorted) * 0.98)]
        step_x = (p99 - p1) / 14.0
        offset_x = p1
        centroids = [offset_x + i * step_x for i in range(15)]
    else:
        centroids = [85.0 + i * 43.5 for i in range(15)]

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
        raw_char_comps = sorted(r, key=lambda x: x["x0"])
        
        # Split wide components (two characters merged) before DP alignment
        char_comps = []
        for comp in raw_char_comps:
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
                        "center_x": (s_x0 + s_x1) / 2.0
                    }
                    char_comps.append(sub_comp)
            else:
                char_comps.append(comp)

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
                    skip_cost = 0.0 if not template[j-1].strip() else 100.0
                    if dp[i][j-1] + skip_cost < dp[i][j]:
                        dp[i][j] = dp[i][j-1] + skip_cost
                        parent[(i, j)] = (i, j-1, -1)
                        
                if i > 0 and j > 0:
                    comp = char_comps[i-1]
                    dist = abs(comp["center_x_corr"] - centroids[j-1])
                    col_label = template[j-1]
                    
                    is_small = comp["width"] < 6 and comp["height"] < 6
                    label_penalty = 0.0
                    if not col_label.strip():
                        label_penalty = 50.0 if is_small else 250.0
                        
                    match_cost = dist + label_penalty
                    if dp[i-1][j-1] + match_cost < dp[i][j]:
                        dp[i][j] = dp[i-1][j-1] + match_cost
                        parent[(i, j)] = (i-1, j-1, j-1)
                        
                if i > 0:
                    comp = char_comps[i-1]
                    is_small = comp["width"] < 6 and comp["height"] < 6
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
            
        col_groups = {}
        for i, comp in enumerate(char_comps):
            if i in assignments:
                col_idx = assignments[i]
                if col_idx not in col_groups:
                    col_groups[col_idx] = []
                col_groups[col_idx].append(comp)
                
        if ignored_comps and assignments:
            for i in ignored_comps:
                comp = char_comps[i]
                best_col = None
                best_dist = 1e9
                for c_idx, col_idx in assignments.items():
                    dist = abs(comp["center_x_corr"] - centroids[col_idx])
                    if dist < best_dist:
                        best_dist = dist
                        best_col = col_idx
                if best_col is not None:
                    if best_col not in col_groups:
                        col_groups[best_col] = []
                    col_groups[best_col].append(comp)

        # Assemble segments sequentially for this row
        for col_idx in range(15):
            char_label = template[col_idx]
            if not char_label or not char_label.strip():
                continue
                
            comps = col_groups.get(col_idx, [])
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
                
                if not comps:
                    # Truly missing segment, append placeholder to keep exact sequence align
                    final_segments.append({
                        "bbox": None,
                        "trajectory": [],
                        "label": char_label,
                        "quality": None,
                        "extraction": {},
                    })
                    continue
                
            new_x0 = min(item["x0"] for item in comps)
            new_x1 = max(item["x1"] for item in comps)
            new_y0 = min(item["y0"] for item in comps)
            new_y1 = max(item["y1"] for item in comps)
            w = new_x1 - new_x0
            h = new_y1 - new_y0
            
            # Dynamic bbox expansion for faint / eroded component(s)
            if w < 16 or h < 16:
                center_x = (new_x0 + new_x1) / 2.0
                center_y = (new_y0 + new_y1) / 2.0
                half_w = max(18, w // 2 + 5)
                half_h = max(18, h // 2 + 5)
                new_x0 = max(0, int(center_x - half_w))
                new_x1 = min(w_img, int(center_x + half_w))
                new_y0 = max(0, int(center_y - half_h))
                new_y1 = min(h_img, int(center_y + half_h))
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
                
            # Use a hybrid approach: guide mask from dilated cleaned_binary to filter grid lines, 
            # while keeping faint strokes from the original binary.
            from scipy.ndimage import binary_dilation
            local_cleaned = cleaned_binary[new_y0:new_y1, new_x0:new_x1]
            guide_mask = binary_dilation(local_cleaned, structure=np.ones((5, 5), dtype=bool))
            char_mask = binary[new_y0:new_y1, new_x0:new_x1] & guide_mask
            
            stroke, stroke_meta = _stroke_from_mask(char_mask, force_profile="default")
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

            final_segments.append({
                "bbox": {"x0": new_x0, "y0": new_y0, "x1": new_x1, "y1": new_y1},
                "trajectory": stroke,
                "label": char_label,
                "quality": round(float(max(stroke_quality, stroke_meta.get("quality", float("-inf")))), 4),
                "extraction": {
                    "profile": stroke_meta.get("profile"),
                    "max_points": stroke_meta.get("max_points"),
                },
            })

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

