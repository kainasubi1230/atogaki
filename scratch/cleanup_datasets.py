import sys
import json
import uuid
from pathlib import Path

sys.path.append("/app")

from api.app.database import SessionLocal
from api.app.models import Dataset, StyleAdapter, Job, User
from api.app.tasks import run_train_lora_job
from api.app.jsonutil import dumps

db = SessionLocal()

user_id = 34

# 1. Get active datasets for this user
active_datasets = db.query(Dataset).filter(Dataset.user_id == user_id, Dataset.active.is_(True)).order_by(Dataset.id.desc()).all()
print(f"Found {len(active_datasets)} active datasets.")

if len(active_datasets) > 1:
    # Keep the newest one, deactivate the rest
    newest = active_datasets[0]
    print(f"Keeping Dataset ID {newest.id} active. Deactivating the rest...")
    for ds in active_datasets[1:]:
        print(f"  Deactivating Dataset ID {ds.id}")
        ds.active = False
        db.add(ds)
    db.commit()
else:
    print("No multiple active datasets. Nothing to deactivate.")

# 2. Let's trigger a fresh LoRA train job using ONLY the newest dataset
# We create a new style adapter so it has a fresh ID
user = db.query(User).filter(User.id == user_id).first()
if not user:
    print("User not found!")
    sys.exit(1)

style = StyleAdapter(user_id=user_id, status="queued", adapter_key=None, disabled=False)
db.add(style)
db.commit()
db.refresh(style)
print(f"Created new StyleAdapter with ID: {style.id}")

# Create job
job_id = uuid.uuid4().hex
job = Job(
    id=job_id,
    user_id=user_id,
    kind="train_lora",
    status="queued",
    payload_json=dumps({"user_id": user_id, "style_id": style.id})
)
db.add(job)
db.commit()
print(f"Created Job with ID: {job_id}")

# Run the job synchronously to apply fixes immediately
print("Running LoRA train job synchronously...")
try:
    run_train_lora_job(job_id)
    print("Job completed successfully!")
    
    # Reload style to see details
    db.refresh(style)
    print(f"Style ID {style.id} status: {style.status}")
    print(f"Adapter key: {style.adapter_key}")
except Exception as e:
    print(f"Job failed with exception: {e}")

db.close()
