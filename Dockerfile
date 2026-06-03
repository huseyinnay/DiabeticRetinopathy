FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Remove heavy training libraries and testing libs from production install
RUN grep -vE 'torch|torchvision|timm|pytest|httpx|albumentations|scikit-learn' requirements.txt > req_prod.txt && \
    pip install --no-cache-dir -r req_prod.txt

COPY . .

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["gunicorn", "-c", "gunicorn_conf.py", "asgi_app:app"]
