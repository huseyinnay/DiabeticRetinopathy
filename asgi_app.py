"""
Diabetic Retinopathy (DR) Binary Classifier API
------------------------------------------------
FastAPI tabanlı ASGI uygulaması:
- /predict : Görsel yükleyip DR tahmini al
- /health  : Servis durumu
- /metadata: Model ve ayar bilgileri

Notlar:
- Ağırlıklar (.pt) ve summary.json yolu .env dosyasından okunur.
- Eğitimde kullanılan ön-işleme ayarları (.env veya summary.json) ile eşleşmelidir.
"""

import io
import json
from pathlib import Path
from typing import Optional, List

import torch
import timm
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from loguru import logger

# Ayarlar (Pydantic Settings)
class Settings(BaseSettings):
    dr_weights: Path = Field(default=Path("runs/binary_efficientnet_b0/best.pt"))
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

# Görüntü ön-işleme pipeline 
TFM = transforms.Compose([
    transforms.Resize((settings.dr_image_size, settings.dr_image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=settings.dr_mean, std=settings.dr_std),
])

# Model 

class DRModel:
    """PyTorch timm modeli ile DR binary sınıflandırma."""

    def __init__(self, weights_path: Path, model_name: str):
        if not weights_path.exists():
            logger.error(f"Ağırlık dosyası bulunamadı: {weights_path.resolve()}")
            raise FileNotFoundError(f"Ağırlık dosyası bulunamadı: {weights_path.resolve()}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Model {model_name} yükleniyor. Cihaz: {self.device}")

        # Modeli oluştur
        self.model = timm.create_model(model_name, pretrained=False, num_classes=1)

        # Ağırlıkları yükle 
        try:
            state = torch.load(str(weights_path), map_location=self.device, weights_only=True)
        except TypeError:  # eski PyTorch sürümleri için fallback
            state = torch.load(str(weights_path), map_location=self.device)

        # Eğer checkpoint 'state_dict' içeriyorsa onu al
        if isinstance(state, dict) and "state_dict" in state:
            state = {k.replace("model.", ""): v for k, v in state["state_dict"].items()}

        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model başarıyla yüklendi. Eşik değeri (Threshold): {THRESHOLD}")

    @torch.inference_mode()
    def predict_proba(self, img_rgb: Image.Image) -> float:
        """Görselden DR+ olasılığını (sigmoid) döndür."""
        x = TFM(img_rgb).unsqueeze(0).to(self.device)
        logit = self.model(x).squeeze(0).float()
        prob = torch.sigmoid(logit).item()
        return float(prob)

    def classify(self, img_rgb: Image.Image) -> dict:
        """Olasılığı eşiğe göre ikili sınıfa dönüştür."""
        p = self.predict_proba(img_rgb)
        label = int(p >= THRESHOLD)
        return {"prob": p, "label": label, "threshold": THRESHOLD}

# Singleton/önbellek
_model: Optional[DRModel] = None
def get_model() -> DRModel:
    global _model
    if _model is None:
        _model = DRModel(settings.dr_weights, settings.dr_model_name)
    return _model


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlarken modeli RAM'e yükle."""
    logger.info("FastAPI uygulaması başlıyor, model belleğe yükleniyor...")
    get_model()
    yield

# ASGI Uygulaması

app = FastAPI(
    title="DR Binary Classifier API",
    version="1.0.0",
    description="Fundus fotoğraflarından DR (binary) tahmini",
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
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    }


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Fundus görselinden DR tahmini yap."""
    # Girdi Doğrulama: MIME type kontrolü
    if file.content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"Geçersiz dosya tipi gönderildi: {file.content_type}")
        raise HTTPException(
            status_code=400, 
            detail=f"Geçersiz dosya tipi: {file.content_type}. Desteklenen tipler: {ALLOWED_MIME_TYPES}"
        )
        
    # Girdi Doğrulama: Dosya boyutu kontrolü
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
