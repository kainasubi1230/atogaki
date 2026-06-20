import io
import numpy as np
from PIL import Image
from api.app.storage import get_storage
from scipy.ndimage import binary_opening
from trainerlib.preprocess import _binary_projection_groups, _stroke_from_mask, get_connected_components

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

storage = get_storage()
object_key = "scans/u34/99df8937a7e04e1689d65980f3eded05-S__14434308.jpg"
image_bytes = storage.get_bytes(object_key)

image = Image.open(io.BytesIO(image_bytes)).convert("L")
arr = np.array(image)
h_img, w_img = arr.shape

mean = float(arr.mean())
std = float(arr.std())
threshold = max(120, mean - std)
binary = arr < threshold

# margins clear
binary[:, :int(w_img * 0.02)] = False
binary[:, int(w_img * 0.98):] = False
binary[:int(h_img * 0.01), :] = False
binary[int(h_img * 0.99):] = False

# We keep notebook lines removal disabled since we cluster CCs directly (no vertical slicing issues)
# and we want to preserve horizontal strokes like in "ナ"
cleaned_binary = binary_opening(binary, structure=np.ones((3, 3), dtype=bool))

raw_components = get_connected_components(cleaned_binary)
raw_components = [c for c in raw_components if c["width"] >= 2 and c["height"] >= 2 and (c["width"] * c["height"]) >= 6]

# Row clustering
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

# Calculate skew
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

all_xs = [c["center_x_corr"] for r in valid_rows for c in r]
all_xs_sorted = sorted(all_xs)
p1 = all_xs_sorted[int(len(all_xs_sorted) * 0.02)]
p99 = all_xs_sorted[int(len(all_xs_sorted) * 0.98)]
step_x = (p99 - p1) / 14.0
offset_x = p1
centroids = [offset_x + i * step_x for i in range(15)]

final_segments = []
for row_idx, r in enumerate(valid_rows[:8]):
    template = ROW_TEMPLATES[row_idx] if row_idx < len(ROW_TEMPLATES) else []
    
    char_comps = sorted(r, key=lambda x: x["x0"])
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
                    
    # Backtrack
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
            
    # Process segments
    for col_idx in range(15):
        char_label = template[col_idx] if col_idx < len(template) else None
        if not char_label or not char_label.strip():
            continue
            
        comps = col_groups.get(col_idx, [])
        if not comps:
            print(f"  Row {row_idx} Col {col_idx} ({char_label}): Missing!")
            continue
            
        new_x0 = min(item["x0"] for item in comps)
        new_x1 = max(item["x1"] for item in comps)
        new_y0 = min(item["y0"] for item in comps)
        new_y1 = max(item["y1"] for item in comps)
        w = new_x1 - new_x0
        h = new_y1 - new_y0
        
        # If the bounding box of assigned components is too small (meaning the character is faint or eroded)
        # we dynamically expand the bbox around its center to a standard character size (e.g. 36x36)
        # to ensure we cover the entire character in the original 'binary' image.
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
            print(f"  Row {row_idx} Col {col_idx} ({char_label}): BBox too small ({w}x{h})")
            continue
            
        # Cut char_mask from original 'binary' (before opening) to preserve faint strokes!
        from scipy.ndimage import binary_dilation
        local_cleaned = cleaned_binary[new_y0:new_y1, new_x0:new_x1]
        guide_mask = binary_dilation(local_cleaned, structure=np.ones((5, 5), dtype=bool))
        char_mask = binary[new_y0:new_y1, new_x0:new_x1] & guide_mask
        stroke, stroke_meta = _stroke_from_mask(char_mask)
        if not stroke:
            print(f"  Row {row_idx} Col {col_idx} ({char_label}): Stroke extraction failed")
            continue
            
        if char_label in ['あ', 'い', 'う']:
            print(f"\n[PLOT] Character: '{char_label}'")
            print(f"  Stroke length: {len(stroke)}")
            print(f"  Meta: {stroke_meta}")
            xs = [p['x'] for p in stroke if p.get('pen_state') == 'down']
            ys = [p['y'] for p in stroke if p.get('pen_state') == 'down']
            if xs and ys:
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                w_box = max(1e-5, max_x - min_x)
                h_box = max(1e-5, max_y - min_y)
                grid = [[" " for _ in range(30)] for _ in range(30)]
                for p in stroke:
                    if p.get('pen_state') == 'down':
                        gx = int((p['x'] - min_x) / w_box * 29)
                        gy = int((p['y'] - min_y) / h_box * 29)
                        if 0 <= gx < 30 and 0 <= gy < 30:
                            grid[gy][gx] = "*"
                for row in grid:
                    print("".join(row))
            else:
                print("  No pen-down points to plot!")
            
        final_segments.append({
            "label": char_label,
            "row": row_idx,
            "col": col_idx,
            "bbox": {"x0": new_x0, "y0": new_y0, "x1": new_x1, "y1": new_y1}
        })

print(f"\nTotal extracted segments with labels: {len(final_segments)}")
missing_labels = []
for r_idx in range(8):
    row_labels = [s["label"] for s in final_segments if s["row"] == r_idx]
    template = ROW_TEMPLATES[r_idx]
    expected_row = [ch for ch in template if ch.strip()]
    missing_in_row = [ch for ch in expected_row if ch not in row_labels]
    missing_labels.extend(missing_in_row)

print(f"Total missing: {len(missing_labels)}")
print(f"Missing: {missing_labels}")
