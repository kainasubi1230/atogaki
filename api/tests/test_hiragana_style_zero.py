from fastapi.testclient import TestClient
from api.app.main import app
from api.tests.test_api import _signup

def test_hiragana_style_zero():
    client = TestClient(app)
    user = _signup(client, "zero@example.com")
    headers = {"Authorization": f"Bearer {user['access_token']}"}

    # Test Hiragana
    res_hira = client.post(
        "/generate",
        headers=headers,
        json={
            "user_id": user["user_id"],
            "style_id": 0,
            "text": "ひらがな",
            "purpose": "accessibility",
        },
    )
    print("Hiragana Status:", res_hira.status_code)
    if res_hira.status_code != 200:
        print("Hiragana Error:", res_hira.json())
        
    # Test Katakana
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
    if res_kata.status_code != 200:
        print("Katakana Error:", res_kata.json())

    assert res_hira.status_code == 200
    assert "svg" in res_hira.json()
    print("Hiragana SVG:", res_hira.json()["svg"])
    assert "<path" in res_hira.json()["svg"]

    assert res_kata.status_code == 200
    assert "svg" in res_kata.json()
    print("Katakana SVG:", res_kata.json()["svg"])
    assert "<path" in res_kata.json()["svg"]
