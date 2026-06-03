# Diabetic Retinopathy (DR) Classifier API & Web UI

Fundus göz fotoğraflarından **diyabetik retinopati** (DR) tespiti yapan, baştan uca (end-to-end) bir makine öğrenimi projesi.
PyTorch kullanılarak eğitilmiş EfficientNet-B0 tabanlı modelin `ONNX` formatına dönüştürülüp, Gunicorn, Docker ve FastAPI kullanılarak yüksek performanslı ve modern bir arayüzle sunulduğu *Production Ready* bir versiyondur.

## Proje Özeti
- **Model:** EfficientNet-B0 (timm ile eğitilip, ONNXRuntime ile çalıştırılır)
- **Görev:** İkili sınıflandırma (Binary Classification) → `0 = Sağlıklı`, `1 = Refer (Riskli)` 
- **Veri Düzeni:** APTOS benzeri; `ImageFolder` yapısı.
- **Performans:** Yüksek hız (ONNX) ve eşzamanlı destek (Gunicorn + 4 Worker). 
- **Arayüz:** "Glassmorphism" ve "Dark Mode" destekli, etkileşimli Web UI.

## Proje Yapısı 
```
.
├── Dockerfile               # Production için optimize edilmiş Docker imajı
├── docker-compose.yml       # Hızlı kurulum için Compose yapılandırması
├── gunicorn_conf.py         # Multi-worker Gunicorn ayarları
├── .env                     # Ortam değişkenleri (DR_WEIGHTS vb.)
├── asgi_app.py              # FastAPI servis ve ONNX backend
├── export_onnx.py           # PyTorch (.pt) modelini ONNX formatına çeviren betik
├── train_dr_binary.py       # (Opsiyonel) Eğitim betiği
├── static/                  # Web Arayüzü (Frontend)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── runs/                    # Model çıktıları (best.pt, best.onnx, summary.json)
└── requirements.txt         # Gerekli kütüphaneler
```

## Kurulum ve Çalıştırma (Docker - Önerilen)
Canlı ortama (production) en yakın deneyim ve en kolay kurulum için Docker kullanın.
Proje kök dizininde aşağıdaki komutu çalıştırmanız yeterlidir:

```bash
docker-compose up --build
```
- API ve Web Arayüzü `http://localhost:8000` adresinde aktif olacaktır.
- (Eğer yüklü değilse, arka planda gereksiz PyTorch vb. dev bağımlılıkları silinir ve sadece ONNXRuntime içeren hafif bir imaj derlenir.)

## Web Arayüzü (Frontend)
Docker (veya Gunicorn) çalışırken doğrudan `http://localhost:8000` adresine giderek sistemi test edebilirsiniz.
- Fundus fotoğraflarını **Sürükle & Bırak** yapabilir veya tıklayarak seçebilirsiniz.
- **Analyze Image** diyerek saniyeler içerisinde olasılık değerlerini ve analiz sonucunu görebilirsiniz.

## Kurulum ve Çalıştırma (Local - Geliştirici)
Geliştirme yapmak isterseniz yerel bir python ortamında çalıştırabilirsiniz.

```bash
python -m venv .venv 
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Modeli ONNX Formatına Çevirme
ONNX modeli henüz mevcut değilse PyTorch (.pt) modelini dışa aktarmak için:
```bash
python export_onnx.py
```
> Bu işlem `runs/binary_efficientnet_b0/best.onnx` dosyasını yaratır. `.env` içerisinden `DR_WEIGHTS` yolunu buna göre güncelleyin.

### Servisi Başlat (Gunicorn)
```bash
gunicorn -c gunicorn_conf.py asgi_app:app
```
(Windows kullanıcıları alternatif olarak `uvicorn asgi_app:app --host 0.0.0.0 --port 8000` komutunu kullanabilir.)

## Eğitim Süreci
Eğer modeli kendi veri setinizle yeniden eğitmek isterseniz:
```bash
python train_dr_binary.py --data data --model efficientnet_b0 --imgsz 512 --epochs 20 --bs 32 --amp
```

## API Endpointler
- `GET /` → Web Arayüzü sayfasına gider.
- `GET /health` → Servisin çalışıp çalışmadığını kontrol eder.
- `GET /metadata` → Model, eşik değeri (threshold) ve yapılandırma bilgilerini döndürür.
- `POST /predict` → `multipart/form-data` kullanarak görsel (file) gönderilir ve ONNX tahmini (prob, label) JSON olarak döner.

## Notlar
- `asgi_app.py` artık `torch` bağımlılığı barındırmaz. Görsel ön-işleme işlemleri tamamen `numpy` ile, inference ise CPU tabanlı `onnxruntime` ile gerçekleştirilir.
- Eşik (threshold) değeri sırasıyla; `.env` dosyası -> `summary.json` -> varsayılan üzerinden okunur.
