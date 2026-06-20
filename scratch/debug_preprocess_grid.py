import sys, traceback, io
from PIL import Image, ImageDraw
from trainerlib.preprocess import preprocess_scan

# Create a simple blank white image (200x200)
img = Image.new('L', (200, 200), color=255)
buf = io.BytesIO()
img.save(buf, format='PNG')
image_bytes = buf.getvalue()

# Run preprocessing to obtain segments
result = preprocess_scan(image_bytes)

# Convert to RGB for drawing
vis = img.convert('RGB')
draw = ImageDraw.Draw(vis)

# Draw each segment's bounding box in green (if present)
for seg in result.get('segments', []):
    bbox = seg.get('bbox')
    if bbox:
        draw.rectangle([bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']], outline='green', width=2)

# Save debug image to artifact location
output_path = r'C:/Users/s65234kh/.gemini/antigravity-ide/brain/fea19543-00ac-4d91-be03-672d01f006fd/debug_preprocess_grid.png'
vis.save(output_path)
print('Debug image saved to', output_path)
