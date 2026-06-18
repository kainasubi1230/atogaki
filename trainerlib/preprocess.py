from io import BytesIO
import json
import numpy as np
from PIL import Image
from trainer.public_dataset import _image_to_sequence


LOW_CONTRAST = "LOW_CONTRAST"
NO_TEXT_DETECTED = "NO_TEXT_DETECTED"
TOO_FEW_SEGMENTS = "TOO_FEW_SEGMENTS"

ROW_TEMPLATES = [
    list("あいうえおかきくけこさしすせそ"), # Row 1
    list("たちつてとなにぬねのはひふへほ"), # Row 2
    list("まみむめもや ゆ よらりるれろ"), # Row 3
    list("アイウエオカキクケコサシスセソ"), # Row 4
    list("タチツテトナニヌネノハヒフヘホ"), # Row 5
    list("マミムメモヤ ユ ヨラリルレロ")  # Row 6
]


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
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
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


def _stroke_from_mask(mask: np.ndarray) -> list[dict]:
    if mask.ndim != 2:
        return []
    tile = np.where(mask, 0, 255).astype(np.uint8)
    seq = _image_to_sequence(tile, max_points=220, threshold=128, smooth_profile="default")
    if not seq:
        return []

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
        return []
    points[-1]["pen_state"] = "up"
    return points


def preprocess_scan(image_bytes: bytes) -> dict:
    image = Image.open(BytesIO(image_bytes)).convert("L")
    arr = np.array(image)
    h_img, w_img = arr.shape
    contrast = float(arr.std())
    if contrast < 9.0:
        return {"success": False, "reason_code": LOW_CONTRAST}

    threshold = 180
    binary = arr < threshold
    if float(binary.mean()) < 0.003:
        return {"success": False, "reason_code": NO_TEXT_DETECTED}

    # Clean notebook lines using robust line removal
    cleaned_binary = remove_notebook_lines_robust(binary, min_h_run=60, max_v_thickness=3)
    
    # Detect lines dynamically
    lines = _binary_projection_groups(cleaned_binary, axis=1)
    
    # Pre-extract components to judge if this is the 15-column kana chart
    raw_components = []
    for line_idx, (y0, y1) in enumerate(lines):
        # Limit kana search area for safety in notebook
        if y1 > 630 and h_img > 630:
            continue
        line_mask = cleaned_binary[y0:y1, :]
        v_sums = line_mask.sum(axis=0)
        blank_cols = np.where(v_sums <= 2)[0]
        line_mask_severed = line_mask.copy()
        line_mask_severed[:, blank_cols] = False
        
        chars = _binary_projection_groups(line_mask_severed, axis=0)
        for x0, x1 in chars:
            char_mask = line_mask_severed[:, x0:x1]
            h_char, w_char = char_mask.shape
            if w_char < 6 or h_char < 6 or (w_char * h_char) < 40:
                continue
            raw_components.append({
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "width": w_char,
                "height": h_char,
                "center_y": (y0 + y1) / 2.0,
                "center_x": (x0 + x1) / 2.0,
                "line_idx": line_idx,
            })

    if not raw_components:
        return {"success": False, "reason_code": TOO_FEW_SEGMENTS}

    # Conditional logic: If it looks like a standard kana table, use robust grid mapping.
    # Otherwise, fallback to plain sequential sorting (for tests / arbitrary uploads).
    use_grid_alignment = len(lines) >= 5 and len(raw_components) >= 30 and h_img > 600

    final_segments = []

    if use_grid_alignment:
        # Group raw components into exactly 6 rows (Kana table) based on line_idx or center_y
        rows = []
        for c in raw_components:
            cy = c["center_y"]
            fit_row = None
            for r in rows:
                r_cy = sum(item["center_y"] for item in r) / len(r)
                if abs(cy - r_cy) < 25:
                    fit_row = r
                    break
            if fit_row is not None:
                fit_row.append(c)
            else:
                rows.append([c])
                
        valid_rows = [r for r in rows if len(r) >= 5]
        valid_rows.sort(key=lambda r: sum(item["center_y"] for item in r) / len(r))
        
        # Grid X centers
        grid_start_x = 70.0
        grid_end_x = 715.0
        grid_step_x = (grid_end_x - grid_start_x) / 14.0
        
        # Align grid centers dynamically
        all_xs = [c["center_x"] for r in valid_rows for c in r]
        centroids = [85.0 + i * 43.5 for i in range(15)]
        xs_arr = np.array(all_xs)
        for _ in range(12):
            new_centroids = []
            for c in centroids:
                close_points = xs_arr[np.abs(xs_arr - c) < 21.0]
                if len(close_points) > 0:
                    new_centroids.append(float(close_points.mean()))
                else:
                    new_centroids.append(c)
            centroids = new_centroids

        for row_idx, r in enumerate(valid_rows[:6]):
            col_groups = {}
            for c in r:
                cx = c["center_x"]
                col_idx = int(np.argmin([abs(cx - cent) for cent in centroids]))
                if col_idx not in col_groups:
                    col_groups[col_idx] = []
                col_groups[col_idx].append(c)
                
            for col_idx, comps in col_groups.items():
                new_x0 = min(item["x0"] for item in comps)
                new_x1 = max(item["x1"] for item in comps)
                new_y0 = min(item["y0"] for item in comps)
                new_y1 = max(item["y1"] for item in comps)
                w = new_x1 - new_x0
                h = new_y1 - new_y0
                
                if w >= 10 and h >= 10 and (w * h) >= 120:
                    char_mask = binary[new_y0:new_y1, new_x0:new_x1]
                    stroke = _stroke_from_mask(char_mask)
                    if stroke:
                        for p in stroke:
                            p["x"] = int(p["x"]) + int(new_x0)
                            p["y"] = int(p["y"]) + int(new_y0)
                        
                        template = ROW_TEMPLATES[row_idx] if row_idx < len(ROW_TEMPLATES) else []
                        char_label = template[col_idx] if col_idx < len(template) else None
                        if char_label and char_label.strip():
                            final_segments.append(
                                {
                                    "bbox": {"x0": new_x0, "y0": new_y0, "x1": new_x1, "y1": new_y1},
                                    "trajectory": stroke,
                                    "label": char_label
                                }
                            )
    else:
        # Plain fallback flow: Sort sequentially row-by-row, col-by-col
        # Group by line_idx
        line_groups = {}
        for c in raw_components:
            l_idx = c["line_idx"]
            if l_idx not in line_groups:
                line_groups[l_idx] = []
            line_groups[l_idx].append(c)
            
        sorted_lines = sorted(line_groups.keys())
        for l_idx in sorted_lines:
            comps = sorted(line_groups[l_idx], key=lambda item: item["x0"])
            for c in comps:
                x0, y0, x1, y1 = c["x0"], c["y0"], c["x1"], c["y1"]
                char_mask = binary[y0:y1, x0:x1]
                stroke = _stroke_from_mask(char_mask)
                if stroke:
                    for p in stroke:
                        p["x"] = int(p["x"]) + int(x0)
                        p["y"] = int(p["y"]) + int(y0)
                    final_segments.append(
                        {
                            "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                            "trajectory": stroke,
                            # Label is assigned externally or left None for arbitrary drawings
                            "label": None 
                        }
                    )

    if len(final_segments) < 1:
        return {"success": False, "reason_code": TOO_FEW_SEGMENTS}

    return {
        "success": True,
        "reason_code": None,
        "segment_count": len(final_segments),
        "segments": final_segments,
    }


def serialize_preprocess(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False)
