import sys
from pathlib import Path

sys.path.append("/app")

from api.app.database import SessionLocal
from api.app.models import StyleAdapter
from api.app.storage import get_storage
from trainerlib.model import generate_trajectory
from trainerlib.svg import trajectory_to_svg

db = SessionLocal()
storage = get_storage()

adapter = db.query(StyleAdapter).filter(StyleAdapter.id == 183).first()
adapter_seed = storage.get_text(adapter.adapter_key)

# Generate and save SVG for "い"
traj_i = generate_trajectory("い", adapter_seed, "/app/storage/models/base_model.pt")
svg_i = trajectory_to_svg(traj_i, "TEST_I")
Path("/app/scratch/test_i.svg").write_text(svg_i, encoding="utf-8")
print("Saved /app/scratch/test_i.svg")

# Generate and save SVG for "う"
traj_u = generate_trajectory("う", adapter_seed, "/app/storage/models/base_model.pt")
svg_u = trajectory_to_svg(traj_u, "TEST_U")
Path("/app/scratch/test_u.svg").write_text(svg_u, encoding="utf-8")
print("Saved /app/scratch/test_u.svg")

# Generate and save SVG for "お"
traj_o = generate_trajectory("お", adapter_seed, "/app/storage/models/base_model.pt")
svg_o = trajectory_to_svg(traj_o, "TEST_O")
Path("/app/scratch/test_o.svg").write_text(svg_o, encoding="utf-8")
print("Saved /app/scratch/test_o.svg")

db.close()
