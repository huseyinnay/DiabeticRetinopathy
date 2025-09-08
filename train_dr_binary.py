"""
Diabetic Retinopathy İkili Sınıflandırma Eğitimi
================================================================
Hedef: 0 = sağlıklı, 1–4 = doktora görünmeli (binary)

Klasör Yapısı (ImageFolder):
    data/
      train/0,1,2,3,4
      val/0,1,2,3,4
      test/0,1,2,3,4

Notlar:
- {1,2,3,4} sınıfları tek bir "refer(1)" sınıfına eşlenir (BinaryWrapper).
- Kayıp fonksiyonu: BCEWithLogitsLoss (+pos_weight ile sınıf dengesizliği telafisi).
- Değerlendirme: AUC, AP, F1/ACC/Sens/Spec (@0.5), ve VAL üzerinde F2 ile eşik seçimi.
- En iyi VAL AUC'a sahip ağırlıklar `best.pt` olarak kaydedilir; testte VAL-optimal eşik kullanılır.

Kullanım:
    python train_dr_binary.py --data data --model efficientnet_b0 --imgsz 224 --epochs 20 --bs 32 --amp

Gereksinimler:
    pip install torch torchvision timm scikit-learn albumentations opencv-python
"""

import os
import math
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import timm

# ----- Görüntü normalizasyonu (ImageNet) -----
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    """Komut satırı argümanlarını ayrıştır."""
    ap = argparse.ArgumentParser(
        description="APTOS-benzeri DR ikili sınıflandırma (0: sağlıklı, 1–4: refer)."
    )
    ap.add_argument("--data", type=str, default="data",
                    help="train/val/test alt klasörlerini içeren kök dizin")
    ap.add_argument("--model", type=str, default="efficientnet_b0",
                    help="timm model adı (örn: efficientnet_b0, convnext_tiny, mobilenetv3_large_100)")
    ap.add_argument("--imgsz", type=int, default=224,
                    help="giriş görüntü boyutu (kare)")
    ap.add_argument("--epochs", type=int, default=20,
                    help="epoch sayısı")
    ap.add_argument("--bs", type=int, default=32,
                    help="mini-batch boyutu")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="öğrenme oranı (AdamW)")
    ap.add_argument("--wd", type=float, default=1e-4,
                    help="weight decay (AdamW)")
    ap.add_argument("--amp", action="store_true",
                    help="Mixed precision (AMP) etkin")
    ap.add_argument("--workers", type=int, default=4,
                    help="DataLoader worker sayısı")
    ap.add_argument("--out", type=str, default="runs/binary_efficientnet_b0",
                    help="çıktı dizini")
    return ap.parse_args()


class BinaryWrapper(Dataset):
    """
    Çok-sınıflı (0..4) veri setini ikili hedefe dönüştürür:
      - '0' klasörü -> 0 (healthy)
      - '1','2','3','4' klasörleri -> 1 (refer)
    """
    def __init__(self, base: Dataset):
        self.base = base
        # ImageFolder -> label id -> klasör adı ('0','1',...)
        self.idx_to_name = {v: k for k, v in base.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        y_bin = 0 if self.idx_to_name[y] == '0' else 1
        return x, y_bin


def get_transforms(imgsz: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Eğitim ve değerlendirme dönüştürmeleri."""
    train_tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, test_tf


def build_loaders(data_root: str, imgsz: int, bs: int, workers: int
                  ) -> Tuple[DataLoader, DataLoader, DataLoader, float]:
    """
    DataLoader'ları oluştur ve pozitif sınıf için pos_weight hesapla.

    pos_weight = negatif/pozitif oranı (BCEWithLogitsLoss için)
    """
    train_tf, test_tf = get_transforms(imgsz)
    train_ds = BinaryWrapper(datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf))
    val_ds   = BinaryWrapper(datasets.ImageFolder(os.path.join(data_root, "val"), transform=test_tf))
    test_ds  = BinaryWrapper(datasets.ImageFolder(os.path.join(data_root, "test"), transform=test_tf))

    # Sınıf dengesizliği: pos_weight tahmini (neg/pos)
    y_train: List[int] = []
    for _, y in DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=workers):
        # y bir tensordür; CPU numpy listesine aktar
        y_train.extend(y.numpy().tolist())
    y_train = np.array(y_train)
    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    pos_weight = (neg_count / max(pos_count, 1)) if len(y_train) > 0 else 1.0

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, test_loader, float(pos_weight)


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """timm modeli oluştur (çıktı boyutu 1: ikili lojit)."""
    model = timm.create_model(name, pretrained=pretrained, num_classes=1, in_chans=3)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str
             ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Bir DataLoader üzerinde değerlendirme yap.

    Dönüş:
        metrics: {'auc','ap','f1@0.5','acc@0.5','sens@0.5','spec@0.5'}
        probs: sigmoid olasılıklar (np.ndarray)
        y_true: gerçek etiketler (np.ndarray)
    """
    model.eval()
    logits_list, y_list = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device)
        logits = model(x).squeeze(1)  # (B,)
        logits_list.append(logits.detach().cpu())
        y_list.append(y.detach().cpu())

    logits = torch.cat(logits_list).numpy()
    y_true = torch.cat(y_list).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))

    auc = roc_auc_score(y_true, probs)
    ap  = average_precision_score(y_true, probs)

    # Varsayılan 0.5 eşiğiyle özet metrikler
    y_pred = (probs >= 0.5).astype(int)
    f1  = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn + 1e-9)  # recall (refer sınıfı)
    spec = tn / (tn + fp + 1e-9)

    metrics = {
        "auc": auc, "ap": ap,
        "f1@0.5": f1, "acc@0.5": acc,
        "sens@0.5": sens, "spec@0.5": spec
    }
    return metrics, probs, y_true


def find_best_threshold(probs: np.ndarray, y_true: np.ndarray, beta: float = 2.0
                        ) -> Tuple[float, float]:
    """
    VAL üzerinde F-beta (varsayılan F2) maksimize eden eşiği ara.
    Klinik senaryo için recall’a daha çok ağırlık verir (beta>1).
    """
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (probs >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        pp = (y_pred == 1).sum()
        p1 = (y_true == 1).sum()
        precision = tp / max(pp, 1)
        recall    = tp / max(p1, 1)

        if precision == 0 and recall == 0:
            f = 0.0
        else:
            b2 = beta * beta
            f = (1 + b2) * precision * recall / (b2 * precision + recall + 1e-12)

        if f > best_f:
            best_f, best_t = f, t
    return float(best_t), float(best_f)


def main() -> None:
    args = parse_args()

    # Cihaz seçimi ve küçük hız/tekrar üretilebilirlik ayarları
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Veri ve model
    train_loader, val_loader, test_loader, pos_w = build_loaders(
        args.data, args.imgsz, args.bs, args.workers
    )
    model = build_model(args.model).to(device)

    # Optimizasyon
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # CosineAnnealingLR'yi adım-başı kullanıyoruz
    total_steps = args.epochs * math.ceil(len(train_loader.dataset) / args.bs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # AMP: güncel API
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # Sınıf ağırlıklı BCE
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))

    best_auc, best_path = -1.0, os.path.join(args.out, "best.pt")
    step = 0

    for epoch in range(args.epochs):
        model.train()
        running = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.float().to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                logits = model(xb).squeeze(1)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item()
            step += 1

        # VAL metrikleri ve "en iyi" ağırlık kaydı
        val_metrics, val_probs, val_true = evaluate(model, val_loader, device)
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"loss {running/len(train_loader):.4f} | "
            f"AUC {val_metrics['auc']:.4f} "
            f"F1@0.5 {val_metrics['f1@0.5']:.4f} "
            f"Sens {val_metrics['sens@0.5']:.4f} "
            f"Spec {val_metrics['spec@0.5']:.4f}"
        )

    # En iyi modeli yükle
    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])

    # VAL üzerinde F2 ile eşik belirle ve TEST'te uygula
    val_metrics, val_probs, val_true = evaluate(model, val_loader, device)
    best_t, f2 = find_best_threshold(val_probs, val_true, beta=2.0)
    print(f"Best threshold (F2) on VAL: {best_t:.3f} | F2 {f2:.4f}")

    test_metrics, test_probs, test_true = evaluate(model, test_loader, device)
    y_pred = (test_probs >= best_t).astype(int)

    print("== TEST metrics (using VAL-optimal threshold) ==")
    print("AUC:", test_metrics["auc"])
    print("AP :", test_metrics["ap"])
    print("ACC:", accuracy_score(test_true, y_pred))
    print("F1 :", f1_score(test_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(test_true, y_pred).ravel()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    print(f"Sensitivity (Recall, referable): {sens:.4f} | Specificity: {spec:.4f}")
    print(classification_report(test_true, y_pred, digits=4, target_names=["healthy(0)", "refer(1)"]))

    # Özet JSON
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(
            {
                "val": val_metrics,
                "test_auc": test_metrics["auc"],
                "threshold": best_t,
                "model": args.model,
                "imgsz": args.imgsz,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
