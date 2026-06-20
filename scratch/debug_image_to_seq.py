import io
import numpy as np
from PIL import Image
from api.app.storage import get_storage
from trainerlib.preprocess import remove_notebook_lines_robust
from scipy.ndimage import binary_opening, binary_dilation, binary_erosion

# Import intermediate functions from public_dataset
from trainer.public_dataset import (
    _zhang_suen_thinning,
    _remove_singletons,
    _split_connected_components,
    _component_to_nonoverlap_paths,
    _resample_path,
    _smooth_polyline,
    _path_to_sequence_with_cursor
)

def print_grid(mask, title=""):
    print(f"\n--- {title} ---")
    for row in mask:
        print("".join(["*" if val else " " for val in row]))

def main():
    storage = get_storage()
    object_key = "scans/u34/99df8937a7e04e1689d65980f3eded05-S__14434308.jpg"
    image_bytes = storage.get_bytes(object_key)
    
    # Replicate preprocessing steps
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(image)
    h_img, w_img = arr.shape
    mean = float(arr.mean())
    std = float(arr.std())
    threshold = max(120, mean - std)
    binary = arr < threshold
    
    # clear margins
    binary[:, :int(w_img * 0.02)] = False
    binary[:, int(w_img * 0.98):] = False
    binary[:int(h_img * 0.01), :] = False
    binary[int(h_img * 0.99):] = False
    
    cleaned_binary = remove_notebook_lines_robust(binary, min_h_run=65, max_v_thickness=3)
    cleaned_binary = binary_opening(cleaned_binary, structure=np.ones((3, 3), dtype=bool))
    
    # "あ" bounding box from JSON
    # x0: 159, y0: 132, x1: 198, y1: 172
    new_x0, new_y0, new_x1, new_y1 = 159, 132, 198, 172
    
    local_cleaned = cleaned_binary[new_y0:new_y1, new_x0:new_x1]
    guide_mask = binary_dilation(local_cleaned, structure=np.ones((5, 5), dtype=bool))
    char_mask = binary[new_y0:new_y1, new_x0:new_x1] & guide_mask
    
    # Smooth
    smoothed = binary_dilation(char_mask, structure=np.ones((2, 2), dtype=bool))
    smoothed = binary_erosion(smoothed, structure=np.ones((2, 2), dtype=bool))
    
    # Step 1: Base Mask
    print_grid(smoothed, "Input Mask (smoothed)")
    
    # Step 2: Zhang-Suen Thinning
    thinned = _zhang_suen_thinning(smoothed)
    print_grid(thinned, "After Zhang-Suen Thinning")
    
    # Step 3: Remove Singletons
    cleaned_thinned = _remove_singletons(thinned)
    print_grid(cleaned_thinned, "After Removing Singletons")
    
    # Step 4: Connected Components
    components = _split_connected_components(cleaned_thinned)
    print(f"\nFound {len(components)} components.")
    
    # Step 5: Convert components to paths
    paths = []
    raw_len = 0
    for idx, comp in enumerate(components):
        comp_paths = _component_to_nonoverlap_paths(comp, cleaned_thinned)
        print(f"Component {idx}: size={len(comp)}, generated {len(comp_paths)} path(s)")
        for p_idx, path in enumerate(comp_paths):
            print(f"  Path {p_idx}: length={len(path)}")
            # Let's plot this individual path
            path_mask = np.zeros_like(cleaned_thinned, dtype=bool)
            for y, x in path:
                path_mask[y, x] = True
            print_grid(path_mask, f"Path {idx}-{p_idx}")
            paths.append(path)
            raw_len += len(path)

if __name__ == '__main__':
    main()
