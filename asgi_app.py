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
import os
from pathlib import Path
from typing import Optional

import torch
import timm
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from dotenv import load_dotenv

load_dotenv()


# Ayarlar

WEIGHTS_PATH = Path(os.getenv("DR_WEIGHTS", "runs/binary_efficientnet_b0/best.pt"))
SUMMARY_PATH = Path(os.getenv("DR_SUMMARY", "runs/binary_efficientnet_b0/summary.json"))

IMAGE_SIZE = int(os.getenv("DR_IMAGE_SIZE", "512"))
MEAN = json.loads(os.getenv("DR_MEAN", "[0.485, 0.456, 0.406]"))
STD  = json.loads(os.getenv("DR_STD",  "[0.229, 0.224, 0.225]"))

MODEL_NAME = os.getenv("DR_MODEL_NAME", "efficientnet_b0")

def _read_threshold() -> float:
    """Threshold değerini ENV -> summary.json -> default(0.5) sırasıyla oku."""
    env_thr = os.getenv("DR_THRESHOLD")
    if env_thr:
        try:
            return float(env_thr)
        except ValueError:
            pass
    if SUMMARY_PATH.exists():
        try:
            with SUMMARY_PATH.open("r", encoding="utf-8") as f:
                s = json.load(f)
            if isinstance(s, dict) and "threshold" in s:
                return float(s["threshold"])
        except Exception:
            pass
    return 0.5

THRESHOLD = _read_threshold()

# Görüntü ön-işleme pipeline 
TFM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Model 

class DRModel:
    """PyTorch timm modeli ile DR binary sınıflandırma."""

    def __init__(self, weights_path: Path, model_name: str):
        if not weights_path.exists():
            raise FileNotFoundError(f"Ağırlık dosyası bulunamadı: {weights_path.resolve()}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    @torch.inference_mode()
    def predict_proba(self, img_rgb: Image.Image) -> float:
        """Görselden DR+ olasılığını (sigmoid) döndür."""
        x = TFM(img_rgb).unsqueeze(0).to(self.device)
        logit = self.model(x).squeeze().float()
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
        _model = DRModel(WEIGHTS_PATH, MODEL_NAME)
    return _model


# ASGI Uygulaması

app = FastAPI(
    title="DR Binary Classifier API",
    version="1.0.0",
    description="Fundus fotoğraflarından DR (binary) tahmini"
)

# CORS ayarları 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _load_on_startup():
    """Uygulama başlarken modeli RAM'e yükle."""
    get_model()

@app.get("/health")
def health():
    """Servis durumu."""
    return {"ok": True}

@app.get("/metadata")
def metadata():
    """Model ve konfigürasyon bilgilerini döndür."""
    return {
        "model_name": MODEL_NAME,
        "weights_path": str(WEIGHTS_PATH),
        "image_size": IMAGE_SIZE,
        "normalize": {"mean": MEAN, "std": STD},
        "threshold": THRESHOLD,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Fundus görselinden DR tahmini yap."""
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        model = get_model()
        out = model.classify(img)
        return {"ok": True, **out}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
