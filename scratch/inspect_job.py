import json
import sqlite3
from pathlib import Path

# Connect to database
db_path = Path("app.db")
if not db_path.exists():
    print("app.db not found!")
    exit(1)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

print("=== Datasets ===")
cursor.execute("SELECT id, user_id, object_key, preprocess_status, preprocess_artifact_key FROM datasets ORDER BY created_at DESC LIMIT 5")
for r in cursor.fetchall():
    print(f"ID: {r[0]} | User: {r[1]} | Key: {r[2]} | Status: {r[3]} | Artifact: {r[4]}")

print("\n=== Style Adapters ===")
cursor.execute("SELECT id, user_id, status, adapter_key, disabled FROM style_adapters ORDER BY created_at DESC LIMIT 5")
adapters = cursor.fetchall()
for r in adapters:
    print(f"ID: {r[0]} | User: {r[1]} | Status: {r[2]} | Key: {r[3]} | Disabled: {r[4]}")

print("\n=== Jobs ===")
cursor.execute("SELECT id, kind, status, payload_json, result_json, error_code FROM jobs ORDER BY updated_at DESC LIMIT 5")
for r in cursor.fetchall():
    print(f"ID: {r[0]} | Kind: {r[1]} | Status: {r[2]} | Error: {r[5]}")
    try:
        res = json.loads(r[4])
        print(f"  Result summary: similarity={res.get('similarity_score')} cer={res.get('cer')} adapter_path={res.get('adapter_path')}")
    except Exception:
        pass

# Check the latest ready adapter content if it exists
for adapter in adapters:
    adapter_id, user_id, status, adapter_key, disabled = adapter
    if status == "ready" and adapter_key:
        print(f"\n=== Adapter Details ({adapter_key}) ===")
        # We need to read from MinIO (or filesystem fallback if storage is mock/local).
        # Wait, since the docker environment uses S3_ENDPOINT_URL=http://minio:9000,
        # we can look for it in the local minio data directory!
        # The docker-compose volumes mount minio to the docker volume minio.
        # But wait, local pytest/development might use filesystem storage or minio container.
        # Let's see if we can find the adapter file in local storage.
        # Let's search for the adapter_key on the filesystem under storage/ or elsewhere.
        print(f"Searching for adapter file on host filesystem for: {adapter_key}")
        # adapter_key is like "adapters/u1/styles/154/adapter.json"
        local_path = Path("storage") / adapter_key
        if local_path.exists():
            try:
                data = json.loads(local_path.read_text(encoding="utf-8"))
                print(f"Found file at: {local_path}")
                print(f"  sample_count: {data.get('sample_count')}")
                print(f"  style_sample_count: {data.get('style_sample_count')}")
                print(f"  bootstrap_sample_count: {data.get('bootstrap_sample_count')}")
                print(f"  effective_exemplar_count: {data.get('effective_exemplar_count')}")
                print(f"  char_coverage: {data.get('char_coverage')}")
                print(f"  has weights: {data.get('weights') is not None}")
                if data.get('weights'):
                    print(f"  weights length: {len(data['weights'])}")
                print(f"  style_profile: {data.get('style_profile')}")
            except Exception as e:
                print(f"Error reading file: {e}")
        else:
            print(f"File not found on host filesystem: {local_path}")
conn.close()
