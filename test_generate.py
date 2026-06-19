import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from api.app.main import app
from api.app.models import User
from api.app.database import SessionLocal

client = TestClient(app)

try:
    print("Testing /generate endpoint directly...")
    
    # We need a user to test this, so let's mock one or use signup
    response = client.post("/auth/signup", json={"email": "test_generate@example.com", "password": "password"})
    if response.status_code == 409:
        response = client.post("/auth/login", json={"email": "test_generate@example.com", "password": "password"})
    
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    user_id = response.json()["user_id"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test Katakana
    res_kata = client.post("/generate", headers=headers, json={"user_id": user_id, "style_id": 0, "text": "カタカナ", "purpose": "accessibility"})
    assert res_kata.status_code == 200
    svg_kata = res_kata.json()["svg"]
    print("Katakana SVG Length:", len(svg_kata))
    print("Katakana has <path>?", "<path" in svg_kata)
    if not "<path" in svg_kata:
        print("FAIL: No <path> in Katakana SVG!")
        print(svg_kata[:500])
        
    # Test Kanji
    res_kanji = client.post("/generate", headers=headers, json={"user_id": user_id, "style_id": 0, "text": "漢字", "purpose": "accessibility"})
    assert res_kanji.status_code == 200
    svg_kanji = res_kanji.json()["svg"]
    print("Kanji SVG Length:", len(svg_kanji))
    print("Kanji has <path>?", "<path" in svg_kanji)
    if not "<path" in svg_kanji:
        print("FAIL: No <path> in Kanji SVG!")
        print(svg_kanji[:500])
        
except Exception as e:
    print("Exception:", e)
