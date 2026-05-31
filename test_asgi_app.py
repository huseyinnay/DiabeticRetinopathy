import io
import pytest
from fastapi.testclient import TestClient
from asgi_app import app
from PIL import Image

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}

def test_metadata():
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "threshold" in data
    assert "image_size" in data

def test_predict_invalid_mime():
    response = client.post(
        "/predict", 
        files={"file": ("test.txt", b"dummy content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Geçersiz dosya tipi" in response.json()["detail"]

def test_predict_large_file():
    large_content = b"0" * (11 * 1024 * 1024)  # 11 MB
    response = client.post(
        "/predict", 
        files={"file": ("large.jpg", large_content, "image/jpeg")}
    )
    assert response.status_code == 400
    assert "Dosya çok büyük" in response.json()["detail"]

def test_predict_invalid_image():
    response = client.post(
        "/predict", 
        files={"file": ("fake.jpg", b"not an image", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "Geçersiz görsel dosyası" in response.json()["detail"]

def test_predict_valid_image():
    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    response = client.post(
        "/predict", 
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "prob" in data
    assert "label" in data
