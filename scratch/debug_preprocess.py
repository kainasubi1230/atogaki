import io
import json
import numpy as np
from PIL import Image, ImageDraw
from api.app.storage import get_storage
from trainerlib.preprocess import preprocess_scan, remove_notebook_lines_robust

# Initialize storage
storage = get_storage()

# Download the latest scanned image bytes
object_key = "scans/u34/99df8937a7e04e1689d65980f3eded05-S__14434308.jpg"
print(f"Downloading {object_key}...")
image_bytes = storage.get_bytes(object_key)

print("Running preprocess_scan...")
# Run the preprocess scan
res = preprocess_scan(image_bytes)

if not res.get("success"):
    print(f"Preprocess failed! Reason code: {res.get('reason_code')}")
    exit(1)

print(f"Preprocess succeeded. Segment count: {res.get('segment_count')}")

# Load the original image to draw bounding boxes
image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
draw = ImageDraw.Draw(image)

segments = res.get("segments", [])
for seg in segments:
    bbox = seg.get("bbox")
    if not bbox:
        continue
    label = seg.get("label")
    # Draw green box for successfully mapped segments
    draw.rectangle([bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]], outline="green", width=3)
    if label:
        # Just write the label nearby
        draw.text((bbox["x0"], bbox["y0"] - 12), label, fill="red")

# Save debug image to storage
output_path = "storage/debug_preprocess_grid.png"
image.save(output_path)
print(f"Saved bounding box visualization to: {output_path}")

# Let's also save the raw binarized image after notebook line removal to see how it looks
arr = np.array(Image.open(io.BytesIO(image_bytes)).convert("L"))
mean = float(arr.mean())
std = float(arr.std())
threshold = max(120, mean - std)  # adaptive threshold matching preprocess_scan
binary = arr < threshold
cleaned = remove_notebook_lines_robust(binary, min_h_run=30, max_v_thickness=3)  # relaxed horizontal run length

# Convert binary back to PIL image for visualization (black for text, white for bg)
cleaned_img = Image.fromarray(np.where(cleaned, 0, 255).astype(np.uint8))
cleaned_output_path = "storage/debug_binary_cleaned.png"
cleaned_img.save(cleaned_output_path)
print(f"Saved cleaned binary image to: {cleaned_output_path}")

# Save full result JSON for debugging
json_output_path = "storage/debug_preprocess_result.json"
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(f"Saved preprocess result JSON to: {json_output_path}")
