import io
import os
import shutil
from pathlib import Path

from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
import pytest

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["LOCAL_STORAGE_DIR"] = "./test_storage"
os.environ["RQ_INLINE"] = "true"
os.environ["STORAGE_BACKEND"] = "local"

from api.app.main import app  # noqa: E402
from api.app.database import Base, engine  # noqa: E402
from api.app.settings import settings  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    storage_path = Path("test_storage")
    if storage_path.exists():
        shutil.rmtree(storage_path)
    with TestClient(app) as test_client:
        yield test_client


def _signup(client: TestClient, email: str, password: str = "password123") -> dict:
    response = client.post("/auth/signup", json={"email": email, "password": password})
    assert response.status_code == 200
    return response.json()


def _scan_image_bytes(low_contrast: bool = False) -> bytes:
    image = Image.new("L", (320, 100), color=220 if low_contrast else 255)
    draw = ImageDraw.Draw(image)
    if low_contrast:
        draw.text((10, 30), "aaa", fill=216)
    else:
        draw.text((10, 30), "漢字かなABC", fill=20)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_auth_required_for_preprocess(client: TestClient) -> None:
    response = client.post("/datasets/1/preprocess")
    assert response.status_code == 401


def test_owner_checks_and_generation_guardrails(client: TestClient) -> None:
    user1 = _signup(client, "u1@example.com")
    user2 = _signup(client, "u2@example.com")

    headers1 = {"Authorization": f"Bearer {user1['access_token']}"}
    headers2 = {"Authorization": f"Bearer {user2['access_token']}"}

    upload = client.post(
        "/datasets/upload-scan",
        headers=headers1,
        files={"file": ("sample.png", _scan_image_bytes(), "image/png")},
        data={"consent": "true"},
    )
    assert upload.status_code == 200

    preprocess = client.post(f"/datasets/{user1['user_id']}/preprocess", headers=headers1)
    assert preprocess.status_code == 200
    preprocess_job_id = preprocess.json()["job_id"]

    preprocess_status = client.get(f"/jobs/{preprocess_job_id}", headers=headers1)
    assert preprocess_status.status_code == 200
    assert preprocess_status.json()["status"] == "completed"

    train = client.post(f"/styles/{user1['user_id']}/train-lora", headers=headers1)
    assert train.status_code == 200
    style_id = train.json()["style_id"]
    train_job_id = train.json()["job_id"]

    train_status = client.get(f"/jobs/{train_job_id}", headers=headers1)
    assert train_status.status_code == 200
    assert train_status.json()["status"] == "completed"

    forbidden_purpose = client.post(
        "/generate",
        headers=headers1,
        json={"user_id": user1["user_id"], "style_id": style_id, "text": "テスト", "purpose": "assignment"},
    )
    assert forbidden_purpose.status_code == 403

    forbidden_owner = client.post(
        "/generate",
        headers=headers2,
        json={"user_id": user2["user_id"], "style_id": style_id, "text": "テスト", "purpose": "accessibility"},
    )
    assert forbidden_owner.status_code == 403

    generated = client.post(
        "/generate",
        headers=headers1,
        json={"user_id": user1["user_id"], "style_id": style_id, "text": "漢字テスト", "purpose": "accessibility"},
    )
    assert generated.status_code == 200
    data = generated.json()
    assert "AI生成（アクセシビリティ支援）" in data["svg"]
    assert len(data["trajectory"]) > 0

    generated_latin = client.post(
        "/generate",
        headers=headers1,
        json={"user_id": user1["user_id"], "style_id": style_id, "text": "Abc", "purpose": "accessibility"},
    )
    assert generated_latin.status_code == 200
    data_latin = generated_latin.json()
    assert "AI生成（アクセシビリティ支援）" in data_latin["svg"]
    assert len(data_latin["trajectory"]) > 0

    output = client.get(f"/outputs/{data['output_id']}", headers=headers1)
    assert output.status_code == 200
    assert output.json()["output_id"] == data["output_id"]
    assert output.json()["trajectory"]

    logs = client.get("/audit/logs", headers=headers1)
    assert logs.status_code == 200
    assert any(log["event_type"] == "generation_created" for log in logs.json())



def test_preprocess_failure_reason_code(client: TestClient) -> None:
    user = _signup(client, "quality@example.com")
    headers = {"Authorization": f"Bearer {user['access_token']}"}

    upload = client.post(
        "/datasets/upload-scan",
        headers=headers,
        files={"file": ("low.png", _scan_image_bytes(low_contrast=True), "image/png")},
        data={"consent": "true"},
    )
    assert upload.status_code == 200

    preprocess = client.post(f"/datasets/{user['user_id']}/preprocess", headers=headers)
    assert preprocess.status_code == 200
    job = client.get(f"/jobs/{preprocess.json()['job_id']}", headers=headers)
    assert job.status_code == 200
    body = job.json()
    assert body["status"] == "completed"
    assert body["result"]["failure_codes"]["LOW_CONTRAST"] >= 1


def test_inference_only_shared_style_flow(client: TestClient) -> None:
    owner = _signup(client, "owner@example.com")
    guest = _signup(client, "guest@example.com")
    owner_headers = {"Authorization": f"Bearer {owner['access_token']}"}
    guest_headers = {"Authorization": f"Bearer {guest['access_token']}"}

    upload = client.post(
        "/datasets/upload-scan",
        headers=owner_headers,
        files={"file": ("owner.png", _scan_image_bytes(), "image/png")},
        data={"consent": "true"},
    )
    assert upload.status_code == 200
    preprocess = client.post(f"/datasets/{owner['user_id']}/preprocess", headers=owner_headers)
    assert preprocess.status_code == 200
    train = client.post(f"/styles/{owner['user_id']}/train-lora", headers=owner_headers)
    assert train.status_code == 200
    shared_style_id = train.json()["style_id"]

    original_inference_only = settings.inference_only
    original_shared_style_id = settings.shared_style_id
    object.__setattr__(settings, "inference_only", True)
    object.__setattr__(settings, "shared_style_id", shared_style_id)
    try:
        shared_train = client.post(f"/styles/{guest['user_id']}/train-lora", headers=guest_headers)
        assert shared_train.status_code == 200
        shared_train_body = shared_train.json()
        assert shared_train_body["style_id"] == shared_style_id
        assert shared_train_body["job_id"] == "inference-only"
        assert shared_train_body["status"] == "ready"

        generated = client.post(
            "/generate",
            headers=guest_headers,
            json={
                "user_id": guest["user_id"],
                "style_id": 999999,
                "text": "shared style test",
                "purpose": "accessibility",
            },
        )
        assert generated.status_code == 200
    finally:
        object.__setattr__(settings, "inference_only", original_inference_only)
        object.__setattr__(settings, "shared_style_id", original_shared_style_id)


def test_canvas_mode_style_zero_katakana_kanji(client: TestClient) -> None:
    user = _signup(client, "canvas@example.com")
    headers = {"Authorization": f"Bearer {user['access_token']}"}

    response = client.post(
        "/generate",
        headers=headers,
        json={
            "user_id": user["user_id"],
            "style_id": 0,
            "text": "カタカナ漢字テスト",
            "purpose": "accessibility",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "AI生成（アクセシビリティ支援）" in data["svg"]
