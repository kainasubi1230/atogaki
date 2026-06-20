import sys
import json
import numpy as np

sys.path.append("/app")

from api.app.database import SessionLocal
from api.app.models import StyleAdapter
from api.app.storage import get_storage

db = SessionLocal()
storage = get_storage()

adapter = db.query(StyleAdapter).filter(StyleAdapter.id == 170).first()
if not adapter:
    print("Adapter 170 not found!")
    sys.exit(1)

content = storage.get_text(adapter.adapter_key)
data = json.loads(content)

exemplars = data.get("user_char_exemplars_text", {})
print("Total exemplars in adapter:", len(exemplars))

# Let's inspect 'あ', 'い', 'う'
for char in ['あ', 'い', 'う']:
    print(f"\n--- Inspecting '{char}' ---")
    strokes = exemplars.get(char, [])
    print(f"Number of strokes: {len(strokes)}")
    if not strokes:
        print("No strokes!")
        continue
        
    print(f"First stroke points count: {len(strokes[0])}")
    print("Point format sample of first stroke first point:", strokes[0][0])
    
    # Accumulate points across all strokes
    x, y = 0.0, 0.0
    xs, ys = [], []
    
    for stroke in strokes:
        for p in stroke:
            dx = float(p[0]) * 20.0
            dy = float(p[1]) * 20.0
            pen_down = float(p[2]) > 0.5
            x += dx
            y += dy
            if pen_down:
                xs.append(x)
                ys.append(y)
            
    if xs and ys:
        print(f"X range: {min(xs):.2f} to {max(xs):.2f}")
        print(f"Y range: {min(ys):.2f} to {max(ys):.2f}")
        
        # Simple text representation of the stroke
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w = max(1e-5, max_x - min_x)
        h = max(1e-5, max_y - min_y)
        
        grid = [[" " for _ in range(30)] for _ in range(30)]
        x_accum, y_accum = 0.0, 0.0
        for stroke in strokes:
            for p in stroke:
                dx = float(p[0]) * 20.0
                dy = float(p[1]) * 20.0
                pen_down = float(p[2]) > 0.5
                x_accum += dx
                y_accum += dy
                if pen_down:
                    gx = int((x_accum - min_x) / w * 29)
                    gy = int((y_accum - min_y) / h * 29)
                    if 0 <= gx < 30 and 0 <= gy < 30:
                        grid[gy][gx] = "*"
                
        print("ASCII Plot:")
        for row in grid:
            print("".join(row))
    else:
        print("No pen_down points to plot!")

db.close()
