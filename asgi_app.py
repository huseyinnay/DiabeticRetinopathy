"""
Diabetic Retinopathy (DR) Binary Classifier API
------------------------------------------------
FastAPI tabanlı ASGI uygulaması:
- /predict : Görsel yükleyip DR tahmini al
- /health  : Servis durumu
- /metadata: Model ve ayar bilgileri

Notlar:
- Ağırlıklar (.onnx) ve summary.json yolu .env dosyasından okunur.
- Eğitimde kullanılan ön-işleme ayarları (.env veya summary.json) ile eşleşmelidir.
"""

import io
import json
import math
from pathlib import Path
from typing import Optional, List

import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from loguru import logger

# Ayarlar (Pydantic Settings)
class Settings(BaseSettings):
    dr_weights: Path = Field(default=Path("runs/binary_efficientnet_b0/best.onnx"))
    dr_summary: Path = Field(default=Path("runs/binary_efficientnet_b0/summary.json"))
    dr_image_size: int = Field(default=512)
    dr_mean: List[float] = Field(default=[0.485, 0.456, 0.406])
    dr_std: List[float] = Field(default=[0.229, 0.224, 0.225])
    dr_model_name: str = Field(default="efficientnet_b0")
    dr_threshold: Optional[float] = Field(default=None)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()

def _read_threshold() -> float:
    """Threshold değerini ENV -> summary.json -> default(0.5) sırasıyla oku."""
    if settings.dr_threshold is not None:
        return settings.dr_threshold
    
    if settings.dr_summary.exists():
        try:
            with settings.dr_summary.open("r", encoding="utf-8") as f:
                s = json.load(f)
            if isinstance(s, dict) and "threshold" in s:
                return float(s["threshold"])
        except Exception as e:
            logger.warning(f"Could not read threshold from {settings.dr_summary}: {e}")
    return 0.5

THRESHOLD = _read_threshold()

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def preprocess_image(img_rgb: Image.Image) -> np.ndarray:
    """NumPy kullanarak görsel ön işleme."""
    # PyTorch transforms.Resize varsayılan olarak BILINEAR kullanır.
    img = img_rgb.resize((settings.dr_image_size, settings.dr_image_size), Image.Resampling.BILINEAR)
    img_np = np.array(img, dtype=np.float32) / 255.0
    
    mean = np.array(settings.dr_mean, dtype=np.float32)
    std = np.array(settings.dr_std, dtype=np.float32)
    
    img_np = (img_np - mean) / std
    img_np = img_np.transpose(2, 0, 1)  # HWC to CHW
    img_np = np.expand_dims(img_np, axis=0)  # Add batch dimension
    return img_np

class DRModelONNX:
    """ONNX Runtime ile DR binary sınıflandırma."""

    def __init__(self, weights_path: Path):
        if not weights_path.exists():
            logger.error(f"Ağırlık dosyası bulunamadı: {weights_path.resolve()}")
            raise FileNotFoundError(f"Ağırlık dosyası bulunamadı: {weights_path.resolve()}")
        
        logger.info(f"ONNX Model yükleniyor: {weights_path}")
        
        # ONNX oturumu oluştur
        self.session = ort.InferenceSession(str(weights_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        logger.info(f"Model başarıyla yüklendi. Eşik değeri (Threshold): {THRESHOLD}")

    def predict_proba(self, img_rgb: Image.Image) -> float:
        """Görselden DR+ olasılığını (sigmoid) döndür."""
        x = preprocess_image(img_rgb)
        logits = self.session.run([self.output_name], {self.input_name: x})[0]
        logit = float(logits[0][0])
        prob = sigmoid(logit)
        return prob

    def classify(self, img_rgb: Image.Image) -> dict:
        """Olasılığı eşiğe göre ikili sınıfa dönüştür."""
        p = self.predict_proba(img_rgb)
        label = int(p >= THRESHOLD)
        return {"prob": p, "label": label, "threshold": THRESHOLD}

# Singleton/önbellek
_model: Optional[DRModelONNX] = None
def get_model() -> DRModelONNX:
    global _model
    if _model is None:
        _model = DRModelONNX(settings.dr_weights)
    return _model


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlarken modeli RAM'e yükle."""
    logger.info("FastAPI uygulaması başlıyor, ONNX modeli belleğe yükleniyor...")
    get_model()
    yield

# ASGI Uygulaması

app = FastAPI(
    title="DR Binary Classifier API",
    version="1.0.0",
    description="Fundus fotoğraflarından DR (binary) tahmini (ONNX Backend)",
    lifespan=lifespan
)

# CORS ayarları 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    """Servis durumu."""
    return {"ok": True}

@app.get("/metadata")
def metadata():
    """Model ve konfigürasyon bilgilerini döndür."""
    return {
        "model_name": settings.dr_model_name,
        "weights_path": str(settings.dr_weights),
        "image_size": settings.dr_image_size,
        "normalize": {"mean": settings.dr_mean, "std": settings.dr_std},
        "threshold": THRESHOLD,
        "device": "cpu (onnxruntime)",
    }

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Fundus görselinden DR tahmini yap."""
    if file.content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"Geçersiz dosya tipi gönderildi: {file.content_type}")
        raise HTTPException(
            status_code=400, 
            detail=f"Geçersiz dosya tipi: {file.content_type}. Desteklenen tipler: {ALLOWED_MIME_TYPES}"
        )
        
    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE:
        logger.warning(f"Dosya çok büyük: {len(raw)} bytes")
        raise HTTPException(
            status_code=400, 
            detail=f"Dosya çok büyük. Maksimum izin verilen boyut {MAX_FILE_SIZE // (1024*1024)} MB."
        )
        
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        logger.error(f"Görsel açılamadı: {e}")
        raise HTTPException(status_code=400, detail="Geçersiz görsel dosyası.")
        
    try:
        model = get_model()
        out = model.classify(img)
        logger.info(f"Tahmin başarılı. Olasılık: {out['prob']:.4f}, Etiket: {out['label']}")
        return {"ok": True, **out}
    except Exception as e:
        logger.exception("Çıkarım (inference) sırasında hata oluştu")
        raise HTTPException(status_code=500, detail="İç sunucu hatası (Inference error).")

# Frontend için static dosyaları sunma
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
