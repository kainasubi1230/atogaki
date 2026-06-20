import io
import numpy as np
from PIL import Image
from api.app.storage import get_storage
from trainerlib.preprocess import remove_notebook_lines_robust
from scipy.ndimage import binary_opening, binary_dilation, binary_erosion

from trainer.public_dataset import (
    _zhang_suen_thinning,
    _remove_singletons,
    _split_connected_components,
    _component_to_nonoverlap_paths,
    _resample_path,
    _smooth_polyline,
    _path_to_sequence_with_cursor
)

def print_grid_coords(coords, title=""):
    print(f"\n--- {title} ---")
    xs = [p[1] for p in coords]
    ys = [p[0] for p in coords]
    if not xs:
        print("Empty coords")
        return
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max(1, max_x - min_x)
    h = max(1, max_y - min_y)
    
    grid = [[' ' for _ in range(40)] for _ in range(40)]
    for y, x in coords:
        gx = int((x - min_x) / w * 39)
        gy = int((y - min_y) / h * 39)
        if 0 <= gx < 40 and 0 <= gy < 40:
            grid[gy][gx] = '*'
    for row in grid:
        print("".join(row))

def main():
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
    
    binary[:, :int(w_img * 0.02)] = False
    binary[:, int(w_img * 0.98):] = False
    binary[:int(h_img * 0.01), :] = False
    binary[int(h_img * 0.99):] = False
    
    cleaned_binary = remove_notebook_lines_robust(binary, min_h_run=65, max_v_thickness=3)
    cleaned_binary = binary_opening(cleaned_binary, structure=np.ones((3, 3), dtype=bool))
    
    new_x0, new_y0, new_x1, new_y1 = 159, 132, 198, 172
    local_cleaned = cleaned_binary[new_y0:new_y1, new_x0:new_x1]
    guide_mask = binary_dilation(local_cleaned, structure=np.ones((5, 5), dtype=bool))
    char_mask = binary[new_y0:new_y1, new_x0:new_x1] & guide_mask
    
    smoothed = binary_dilation(char_mask, structure=np.ones((2, 2), dtype=bool))
    smoothed = binary_erosion(smoothed, structure=np.ones((2, 2), dtype=bool))
    
    thinned = _zhang_suen_thinning(smoothed)
    cleaned_thinned = _remove_singletons(thinned)
    components = _split_connected_components(cleaned_thinned)
    
    # Sort components
    components.sort(key=lambda comp: (min(p[1] for p in comp), min(p[0] for p in comp)))
    
    paths = []
    raw_len = 0
    for comp in components:
        comp_paths = _component_to_nonoverlap_paths(comp, cleaned_thinned)
        for path in comp_paths:
            if len(path) < 2:
                continue
            paths.append(path)
            raw_len += len(path)
            
    print(f"Total paths to process: {len(paths)}, raw_len={raw_len}")
    
    keep_total = 240
    seq = []
    cursor = (0.0, 0.0)
    
    for idx, path in enumerate(paths):
        comp_keep = max(2, int(round((len(path) / max(1, raw_len)) * keep_total)))
        print(f"\nPath {idx}: length={len(path)}, comp_keep={comp_keep}")
        
        # 1. Resample Path (sampled_i)
        sampled_i = _resample_path(path, comp_keep)
        print_grid_coords(sampled_i, f"Path {idx} - After _resample_path (sampled_i)")
        
        # 2. Smooth polyline
        spline_raw = _smooth_polyline(sampled_i, passes=3, blend=0.65)
        print_grid_coords(spline_raw, f"Path {idx} - After _smooth_polyline (spline_raw)")
        
        # 3. Re-resample spline_raw (as in public_dataset.py)
        sampled_f = spline_raw
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
                resampled_f = []
                jj = 0
                for _t in targets_f:
                    while jj < len(sampled_f) - 1 and dists_f[jj + 1] < _t:
                        jj += 1
                    resampled_f.append(sampled_f[jj])
                sampled_f = resampled_f
        
        print_grid_coords(sampled_f, f"Path {idx} - After re-resampling (sampled_f)")
        
        # 4. _path_to_sequence_with_cursor
        part, cursor = _path_to_sequence_with_cursor(sampled_f, cursor, stroke_width=0.72)
        seq.extend(part)

    # Reconstruct whole seq back to absolute coords to see if the overall sequence is correct
    reconstructed = []
    rx, ry = 0.0, 0.0
    for row in seq:
        dx = row[0] * 20.0
        dy = row[1] * 20.0
        rx += dx
        ry += dy
        pen_down = row[2] > 0.5
        if pen_down:
            reconstructed.append((ry, rx))
            
    print_grid_coords(reconstructed, "FINAL RECONSTRUCTED \"あ\" FROM SEQ")

if __name__ == '__main__':
    main()
