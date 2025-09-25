# Diabetic Retinopathy (DR) Binary Classifier API

Fundus göz fotoğraflarından **diyabetik retinopati** (DR) tespiti yapan **uçtan uca makine öğrenimi projesi**.  
PyTorch ile eğitilmiş **EfficientNet-B0** tabanlı bir model, FastAPI tabanlı bir REST servisi üzerinden kullanılabilir.

## Proje Özeti
- **Model:** EfficientNet-B0 (timm)  
- **Görev:** İkili sınıflandırma → `0 = Sağlıklı`, `1–4 = Refer` 
- **Veri Düzeni:** APTOS benzeri; `ImageFolder` yapısı (aşağıdaki ağaç)  
- **Performans (örnek):** Test AUC ≈ **0.95**, F1@0.5 ≈ **0.90** (detaylar `summary.json`)

## Proje Yapısı 
```
.
├── data
│   ├── test
│   │   ├── 0
│   │   ├── 1
│   │   ├── 2
│   │   ├── 3
│   │   └── 4
│   ├── train
│   │   ├── 0
│   │   ├── 1
│   │   ├── 2
│   │   ├── 3
│   │   └── 4
│   └── val
│       ├── 0
│       ├── 1
│       ├── 2
│       ├── 3
│       └── 4
├── runs/                    # Eğitim çıktıları (best.pt, summary.json vb.)
├── .env
├── .gitignore
├── asgi_app.py              # FastAPI servis
├── recognizing_data.ipynb   # (opsiyonel) keşif/notebook
├── requirements.txt
├── train_dr_binary.py       # Eğitim scripti
├── .venv/                   # (opsiyonel) sanal ortam
└── __pycache__/
```


## Kurulum
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


## Eğitim
```bash
python train_dr_binary.py --data data --model efficientnet_b0 --imgsz 224 --epochs 20 --bs 32 --amp
```
- En iyi model ağırlıkları: `runs/binary_efficientnet_b0/best.pt`  
- Özet metrikler: `runs/binary_efficientnet_b0/summary.json`


## API Kullanımı

### .env örneği
```
DR_WEIGHTS=runs/binary_efficientnet_b0/best.pt
DR_SUMMARY=runs/binary_efficientnet_b0/summary.json
DR_MODEL_NAME=efficientnet_b0
DR_IMAGE_SIZE=224
# Opsiyonel: DR_THRESHOLD=0.09
```

### Servisi başlat
```bash
uvicorn asgi_app:app --host 0.0.0.0 --port 8000
```

### Endpointler
- `GET /health` → servis durumu  
- `GET /metadata` → model/cihaz/normalize/threshold bilgileri  
- `POST /predict` → `multipart/form-data` içinde `file` alanı ile görsel gönder

**curl örneği**
```bash
curl -X POST http://localhost:8000/predict -F "file=@/path/to/fundus.jpg"
```

Örnek çıktı:
```json
{
  "ok": true,
  "prob": 0.8234,
  "label": 1,
  "threshold": 0.09
}
```

## Notlar
- Görseli API kendisi `IMAGE_SIZE x IMAGE_SIZE` boyutuna ölçekler ve ImageNet mean/std ile normalize eder.
- Eşik (threshold) `.env` > `summary.json` > default(0.5) sırasıyla okunur.

