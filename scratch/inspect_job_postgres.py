import sys
sys.path.append("/app/api")
from app.database import SessionLocal
from app.models import Dataset, StyleAdapter, Job
from app.storage import get_storage
import json


db = SessionLocal()
storage = get_storage()

print("=== Datasets ===")
datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).limit(5).all()
for d in datasets:
    print(f"ID: {d.id} | User: {d.user_id} | Key: {d.object_key} | Status: {d.preprocess_status} | Artifact: {d.preprocess_artifact_key}")

print("\n=== Style Adapters ===")
adapters = db.query(StyleAdapter).order_by(StyleAdapter.created_at.desc()).limit(5).all()
for a in adapters:
    print(f"ID: {a.id} | User: {a.user_id} | Status: {a.status} | Key: {a.adapter_key} | Disabled: {a.disabled}")

print("\n=== Jobs ===")
jobs = db.query(Job).order_by(Job.updated_at.desc()).limit(5).all()
for j in jobs:
    print(f"ID: {j.id} | Kind: {j.kind} | Status: {j.status} | Error: {j.error_code}")
    if j.result_json:
        try:
            res = json.loads(j.result_json)
            print(f"  Result summary: similarity={res.get('similarity_score')} cer={res.get('cer')} adapter_key={res.get('adapter_key')} sample_count={res.get('sample_count')} char_coverage={res.get('char_coverage')} bootstrap={res.get('bootstrap_sample_count')}")
        except Exception as e:
            print(f"  Error parsing result: {e}")

# If we have ready adapters, read the content from MinIO
for a in adapters:
    if a.status == "ready" and a.adapter_key:
        print(f"\n=== Adapter Details ({a.adapter_key}) ===")
        try:
            content = storage.get_text(a.adapter_key)
            if content:
                data = json.loads(content)
                print(f"  sample_count: {data.get('sample_count')}")
                print(f"  style_sample_count: {data.get('style_sample_count')}")
                print(f"  bootstrap_sample_count: {data.get('bootstrap_sample_count')}")
                print(f"  effective_exemplar_count: {data.get('effective_exemplar_count')}")
                print(f"  char_coverage: {data.get('char_coverage')}")
                print(f"  has weights: {data.get('weights') is not None}")
                if data.get('weights'):
                    print(f"  weights type: {type(data['weights'])}")
                    if isinstance(data['weights'], list):
                        print(f"  weights length: {len(data['weights'])}")
                        # Check first few items
                        print(f"  first few weights: {data['weights'][:5]}")
                print(f"  style_profile: {data.get('style_profile')}")
                # Check user_char_exemplars keys
                ex = data.get("user_char_exemplars_text", {})
                print(f"  user_char_exemplars_text keys: {list(ex.keys())[:15]}... (total: {len(ex)})")
        except Exception as e:
            print(f"  Error reading adapter key: {e}")

db.close()
