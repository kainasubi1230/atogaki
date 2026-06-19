import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient
from api.app.main import app
from api.tests.test_api import _signup

client = TestClient(app)
user = _signup(client, "local_test@example.com")
headers = {"Authorization": f"Bearer {user['access_token']}"}

res_kata = client.post(
    "/generate",
    headers=headers,
    json={
        "user_id": user["user_id"],
        "style_id": 0,
        "text": "カタカナ",
        "purpose": "accessibility",
    },
)

print("Katakana Status:", res_kata.status_code)
if res_kata.status_code == 200:
    svg = res_kata.json().get("svg", "")
    print(f"Katakana SVG length: {len(svg)}")
    print(f"Has <path: {'<path' in svg}")
    if "<path" not in svg:
        print("SVG is EMPTY or MISSING PATHS!")
else:
    print("Error:", res_kata.json())
