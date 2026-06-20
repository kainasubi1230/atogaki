import io
import numpy as np
from PIL import Image
from api.app.storage import get_storage
from trainerlib.preprocess import remove_notebook_lines_robust
from scipy.ndimage import binary_opening, binary_dilation, binary_erosion

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
    
    print("ASCII Plot of local_cleaned mask:")
    for row in local_cleaned:
        print("".join(["*" if val else " " for val in row]))
        
    print("\nASCII Plot of guide_mask:")
    for row in guide_mask:
        print("".join(["*" if val else " " for val in row]))

    print("\nASCII Plot of smoothed (char_mask) mask:")
    for row in smoothed:
        print("".join(["*" if val else " " for val in row]))

if __name__ == '__main__':
    main()
