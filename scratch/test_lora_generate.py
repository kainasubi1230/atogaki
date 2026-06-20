import sys
import numpy as np
from pathlib import Path
import random

sys.path.append("/app")

from api.app.database import SessionLocal
from api.app.models import StyleAdapter
from api.app.storage import get_storage
from trainerlib.model import _parse_style_seed, _load_base_model, _pick_best_exemplar_sequence
from api.app.settings import settings

db = SessionLocal()
storage = get_storage()

adapter = db.query(StyleAdapter).filter(StyleAdapter.id == 174).first()
adapter_seed = storage.get_text(adapter.adapter_key)

style_payload = _parse_style_seed(adapter_seed)
user_char_exemplars_text = style_payload.get("user_char_exemplars_text")

char = "あ"
user_bucket = user_char_exemplars_text.get(char)

rng = random.Random(42)
seq_user = _pick_best_exemplar_sequence(user_bucket, rng, trials=28)
seq = seq_user # Original (167 points)

local = [(0.0, 0.0, "up", 2)]
lx, ly = 0.0, 0.0
for row in seq:
    dx = float(row[0]) * 20.0
    dy = float(row[1]) * 20.0
    lx += dx
    ly += dy
    pen = "down" if float(row[2]) > 0.5 else "up"
    width = int(round(max(0.2, min(1.2, float(row[3]))) * 4.0))
    local.append((lx, ly, pen, max(1, min(4, width))))

down_local = [(p[0], p[1]) for p in local if p[2] == "down"]
xs = [p[0] for p in down_local]
ys = [p[1] for p in down_local]
min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)
span_x = max(1e-6, max_x - min_x)
span_y = max(1e-6, max_y - min_y)

scale_x = 51.0 / span_x
scale_y = 51.0 / span_y

grid = [[" " for _ in range(30)] for _ in range(30)]
for lx_p, ly_p, pen, width in local:
    if pen == "down":
        tx = (lx_p - min_x) * scale_x
        ty = (ly_p - min_y) * scale_y
        gx = int(tx / 51.0 * 29)
        gy = int(ty / 51.0 * 29)
        if 0 <= gx < 30 and 0 <= gy < 30:
            grid[gy][gx] = "*"
            
print("ASCII Plot of ORIGINAL 'あ' (No Resampling):")
for row in grid:
    print("".join(row))

db.close()
