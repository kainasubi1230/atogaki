import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from api.app.database import engine
from api.app.models import Base

print("Creating missing tables...")
Base.metadata.create_all(bind=engine)
print("Done!")
